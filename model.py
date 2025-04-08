import gc
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import torch

import config


class SequenceModel:
    @config.debug_function
    def __init__(self, model_id: str):
        """
        Loads the named model and tokenizer
        """
        self.model_id = model_id
        self.pre_prompt = ""

        with open("hf-access-token.txt") as f:
            self.hf_token = f.read().strip()

        # ALlow tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        config.debug(f"Loading model {config.MODEL}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            use_auth_token=self.hf_token,
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
    def query_sequence(self, inputs: list[str]) -> list[str]:
        """
        Builds a sequence of queries following a pre-prompt.
        The query is submitted freshly to the model each time.

        args:
        - inputs: a list of text inputs that will be given to the LLM

        returns:
            the output that follows from each input
        """

        start = time.time()
        outputs = []

        for idx, input in enumerate(inputs):
            config.debug(f"Processing input {idx}")
            prompt_string = f"{self.pre_prompt}\n\n[input]: {input}\n[output]:\n{{"

            input_tokens  = self.tokenizer(prompt_string, return_tensors="pt").to("cuda")
            input_length = input_tokens["input_ids"].shape[1]

            output_tokens = self.model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                max_new_tokens=20,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,               # Enables sampling
                temperature=0.8,              # Encourages variation without being too random
                top_p=0.95,                   # Nucleus sampling — limits to most probable tokens
                use_cache=True,              # Recommended for faster inference
                return_dict_in_generate=False  # Slightly faster and lighter output structure
            )

            output_tokens = output_tokens.detach().cpu()

            generated_tokens = output_tokens[:, input_length - 1:]
            generated_string = self.tokenizer.decode(generated_tokens[0],
                                                     skip_special_tokens=True).strip()

            outputs.append(generated_string)

            config.debug(f"Took {time.time() - start}s")
            start = time.time()

        return outputs


    @config.debug_function
    def query_sequence_batched(self, inputs: list[str], batch_size: int = 4) -> list[str]:
        """
        Processes a batch of queries at once for increased speed.
        Uses padding efficiently to minimize unnecessary computation.
        
        Args:
        - inputs: A list of text inputs to process.
        - batch_size: Number of inputs per batch.
        
        Returns:
        A list of generated output strings.
        """
        total_start = time.time()
        outputs = []

        # Process inputs in batches
        for batch_start in range(0, len(inputs), batch_size):
            config.debug(f"Running batch {batch_start // batch_size} of {len(inputs) // batch_size}")

            batch = inputs[batch_start : batch_start + batch_size]

            # Build full prompt for each input
            start = time.time()
            prompt_strings = [f"{self.pre_prompt}\n\n[input]: {i}\n[output]: {{" for i in batch]
            config.debug(f"Building prompt strings took {time.time() - start:.2f}s")

            # Tokenization with efficient padding
            start = time.time()
            input_tokens = self.tokenizer(prompt_strings, return_tensors="pt", padding=True, truncation=True)
            input_tokens = input_tokens.to("cuda")
            torch.cuda.synchronize()
            config.debug(f"Tokenization and moving to CUDA took {time.time() - start:.2f}s")

            # Compute effective input lengths (non-padding tokens)
            input_lengths = input_tokens["attention_mask"].sum(dim=1)

            # Generate batch outputs
            start = time.time()
            output_tokens = self.model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                max_new_tokens=20,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
            )

            torch.cuda.synchronize()
            config.debug(f"Model generation took {time.time() - start:.2f}s")

            # Decode outputs WITHOUT stripping input
            start = time.time()
            for i in range(output_tokens.shape[0]):
                decoded = self.tokenizer.decode(output_tokens[i], skip_special_tokens=True).strip()

                if "[output]" in decoded:
                    outputs.append(decoded.split("[output]:")[1])
                else:
                    outputs.append(decoded)

            config.debug(f"Decoding outputs took {time.time() - start:.2f}s")

            del input_tokens, output_tokens

            if (batch_start // batch_size) % 10 == 0:
                config.debug("Clearing cache and running gc")
                gc.collect()
                torch.cuda.empty_cache()
                config.debug("Running nvidia-smi")
                os.system("nvidia-smi")

        config.debug(f"Total batched query time {time.time() - total_start:.2f}s")

        return outputs


    @config.debug_function
    def extract_json(self, outputs: list[str]) -> list[dict]:
        """ 
        Attempts to extract and parse valid json from each output string

        For each output that doesn't yield valid output, None is put in its place
        """

        json_outputs = []

        for s in outputs:
            try:
                potential_json = s.split("}")[0] + "}"
                parsed = json.loads(potential_json)
            except json.JSONDecodeError:
                parsed = None

            json_outputs.append(parsed)

        return json_outputs
