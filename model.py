# mypy: ignore-errors
import gc
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import torch

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

        config.debug(f"The model we've loaded:\n{self.model}")

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
