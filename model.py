# mypy: ignore-errors
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import torch
from typing import Any, Dict, List, Literal, Optional, Tuple, no_type_check

import config


class BatchModel:
    @config.debug_function
    def __init__(self, model_id: str):
        """
        Loads the named model and tokenizer
        """
        self.model_id = model_id
        self.pre_prompt = ""

        with open(config.CODE_DIR / "hf-access-token.txt") as f:
            self.hf_token = f.read().strip()

        # ALlow tf32
#        torch.backends.cuda.matmul.allow_tf32 = True
#        torch.backends.cudnn.allow_tf32 = True

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        config.debug(f"Loading model {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=self.quant_config,
            token=self.hf_token,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        self.model.eval()

        config.debug("Creating tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
            trust_remote_code=True
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))


    ########## MODEL QUERY ##########


    def load_pre_prompt(self, path: Path):
        """
        Sets the pre-prompt from a path
        """
        config.debug(f"Loading pre-prompt from {path}")

        with open(path) as f:
            self.pre_prompt = f.read()


    def set_pre_prompt(self, pre_prompt: str):
        """
        Sets the pre prompt
        """
        config.debug("Setting model pre-prompt")
        self.pre_prompt = pre_prompt 


    @config.debug_function
    def process_batch(self, batch: list[str], enforce_json=True, max_new_tokens=30) -> list[str]:
        """
        Processes a batch of inputs with the model pre-prompt
    
        args:
        - batch: the list of inputs, as strings
        - enforce_json: whether to enforce JSON output
        - max_new_tokens: the maximum number of new tokens that will be generated

        returns:
            the list of outputs as strings

        throws:
            RuntimeError beginning with 'CUDA out of memory.'
        """
        config.debug(f"Processing batch of size {len(batch)}")

        start = time.time()
        outputs = []

        if enforce_json:
            prompt_strings = [f"{self.pre_prompt}\n\n[input]: {i}\n[output]: {{" for i in batch]
        else:
            prompt_strings = [f"{self.pre_prompt}\n\n[input]: {i}\n[output]: " for i in batch]

        input_tokens = self.tokenizer(prompt_strings, return_tensors="pt", padding=True,
                                      truncation=True, max_length=5000)
        input_tokens = input_tokens.to("cuda")

        output_tokens = self.model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True,
        )

        torch.cuda.synchronize()

        for i in range(output_tokens.shape[0]):
            decoded = self.tokenizer.decode(output_tokens[i], skip_special_tokens=True).strip()

            if "[output]" in decoded:
                outputs.append(decoded.split("[output]:")[1])
            else:
                outputs.append(decoded)

        del input_tokens, output_tokens

        return outputs


    def __del__(self):
        """
        Explicitly removes the model and tokeniser from VRAM
        """
        config.debug("Deleting the model and tokeniser")

        del self.model
        del self.tokenizer

        gc.collect()
        torch.cuda.empty_cache()



class BatchModelIsolator:
    def __init__(self, which_model: Literal["small", "large"]):
        if which_model == "small":
            self.model_id = config.SMALL_MODEL
        else:
            self.model_id = config.LARGE_MODEL

        self.p: Optional[mp.Process]  = None
        self.in_queue: Optional[mp.Queue] = None
        self.out_queue: Optional[mp.Queue] = None
        self.cfg: Dict[str, Any] = {}


    @no_type_check
    def process_prompts(
        self, prompts: List[str], batch_size: Optional[int] = None, cfg: Optional[dict] = None,
    ) -> List[str]:
        """
        Processes a list of prompts in a separate process

        args:
        - prompts: a list of prompts to process
        - batch_size: the number of batches to process at a time
        - cfg: a dict containing overrides for:
            1. max_new_tokens
            2. pre_prompt_path
            3. enforce_json
        """
        cfg_modified = False

        # Set configurations
        if cfg is not None:
            for cfg_name, cfg_value in cfg.items():
                if self.cfg.get(cfg_name, None) != cfg_value:
                    self.cfg[cfg_name] = cfg_value
                    cfg_modified = True

        self.cfg['pre_prompt_path'] = config.CODE_DIR / 'pre-prompts' / self.cfg['pre_prompt_name']

        if cfg_modified:
            self.p, self.in_queue, self.out_queue = self.create_batch_process_worker()

        # Process output
        outputs = []
        batch_start = 0

        while batch_start < len(prompts):
            assert self.p is not None
            assert self.in_queue is not None
            assert self.out_queue is not None

            batch_end = min(len(prompts), batch_start + batch_size)
            prompt_batch = prompts[batch_start : batch_end]

            config.debug(f"Processing batch {batch_start}..{batch_end} of {len(prompts)}")
            self.in_queue.put(prompt_batch)

            results = self.out_queue.get()
            if results["successful"]:
                outputs.extend(results["output"])
                batch_start += batch_size
            else:
                self.kill_batch_worker()
                self.p, self.in_queue, self.out_queue = self.create_batch_process_worker()

                new_batch_size = max(1, int(0.8 * batch_size))
                if new_batch_size == batch_size > 1:
                    new_batch_size -= 1

                batch_size = new_batch_size

        return outputs


    @config.debug_function
    def create_batch_process_worker(self) -> Tuple[mp.Process, mp.Queue, mp.Queue]:
        """
        Creates a new batch process worker

        returns:
        A tuple containing:
            1. The process
            2. The input queue
            3. The output queue
        """
        # Be very sure that there isn't one already
        # This will do nothing if there isn't one
        self.kill_batch_worker()

        in_queue: mp.Queue = mp.Queue()
        out_queue: mp.Queue = mp.Queue()

        p = mp.Process(
            target=self.batch_process_worker,
            args=(in_queue, out_queue, self.model_id, self.cfg)
        )
        p.start()

        config.debug(
            "Created new batch worker\n"
            f" - torch.cuda.memory_allocated() = {torch.cuda.memory_allocated()}\n"
            f" - torch.cuda.memory_reserved() = {torch.cuda.memory_reserved()}"
        )

        return p, in_queue, out_queue


    @classmethod
    def batch_process_worker(
        cls, in_queue: mp.Queue, out_queue: mp.Queue, model_id: str, cfg: Dict[str, Any]
    ):
        """
        Creates a batch processing worker

        args:
        - input_queue: the queue this process take batches from
        - output_queue: the queue this process writes output to, as dicts:
            1. "successful": whether the processing was successful (or OOM)
            2. "output": the list of text output from the model
        - model_id: the model ID
        - cfg: the config dict
        """
        m = BatchModel(model_id)
        m.load_pre_prompt(cfg["pre_prompt_path"])

        with torch.no_grad():
            while True:
                if (batch := in_queue.get()) is None:
                    break

                try:
                    output = m.process_batch(batch, enforce_json=cfg['enforce_json'],
                                             max_new_tokens=cfg['max_new_tokens'])
                    out_queue.put({ "successful": True, "output": output })
                except RuntimeError as e:
                    if str(e).startswith('CUDA out of memory') or \
                       str(e).startswith('Some modules are dispatched'):
                        out_queue.put({ "successful": False, "output": None })
                    else:
                        raise e

        del m
        out_queue.put(None)


    @config.debug_function
    def kill_batch_worker(self):
        """
        Kills the current batch worker and deletes the input and output queues
        """
        if self.p is not None and self.p.is_alive():
            self.in_queue.put(None)
            self.out_queue.get() # Block on destructor

            self.p.terminate()
            self.p.join()
            self.p = None

            self.in_queue.close()
            self.in_queue.join_thread()
            self.in_queue = None

            self.out_queue.close()
            self.out_queue.join_thread()
            self.out_queue = None

        torch.cuda.empty_cache()


    def __del__(self):
        self.kill_batch_worker()


@config.debug_function
def extract_json(outputs: list[str], default: dict) -> list[dict]:
    """ 
    Attempts to extract and parse valid json from each output string

    args:
    - outputs: string outputs to parse
    - default: default object to use if not parseable

    For each output that doesn't yield valid output, None is put in its place
    """

    json_outputs = []

    for s in outputs:
        try:
            potential_json = s.split("}")[0] + "}"
            parsed = json.loads(potential_json)
        except json.JSONDecodeError:
            parsed = default.copy()

        json_outputs.append(parsed)

    return json_outputs
