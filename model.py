# mypy: ignore-errors
import gc
import json
import os
from pathlib import Path
import time
from vllm import LLM, SamplingParams
import torch
from typing import Any, Dict, List, Optional

import config

class BatchModel:
    @config.debug_function
    def __init__(self, model_id: str, max_model_len: int | None = None):
        """
        Loads the named model into vLLM
        """
        self.model_id = model_id
        self.pre_prompt = ""

        if max_model_len is not None:
            allow_override = os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "") == "1"
            if not allow_override:
                from transformers import AutoConfig
                hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                derived = getattr(hf_cfg, "model_max_length", None) \
                       or getattr(hf_cfg, "max_position_embeddings", None)
                if derived is not None and max_model_len > derived:
                    config.output(
                        f"Clamping requested max_model_len={max_model_len} to model’s true max of {derived}"
                    )
                    max_model_len = derived
            else:
                config.output(
                    f"Overriding model max length to {max_model_len} "
                    "(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)"
                )


        with open(config.CODE_DIR / "hf-access-token.txt") as f:
            os.environ["HF_TOKEN"] = f.read().strip()

        # Set parameters
        llm_args = dict(
            model=model_id,
            dtype="auto",
            trust_remote_code=True
        )

        if model_id == "mistral":
            llm_args["tokenizer_mode"] = "mistral"

        if max_model_len is not None:
            llm_args["max_model_len"] = max_model_len

        if "awq" in model_id.lower():
            llm_args["quantization"] = "awq"
            llm_args["gpu_memory_utilization"] = 0.85
            llm_args["enforce_eager"] = True

        self.llm = LLM(**llm_args)

        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    def load_pre_prompt(self, path: Path):
        """
        Loads a pre-prompt from a file
        """
        config.output(f"Loading pre-prompt from {path}")

        with open(path) as f:
            self.pre_prompt = f.read()

    def set_pre_prompt(self, pre_prompt: str):
        """
        Sets a pre-prompt directly
        """
        config.debug("Setting model pre-prompt")
        self.pre_prompt = pre_prompt


    @config.debug_function
    def process_batch(
        self,
        batch: List[str],
        structure_header: Optional[str] = None,
        max_new_tokens: int = 30
    ) -> List[str]:
        """
        Processes a batch of inputs with vLLM
        """
        config.debug(f"Processing batch of size {len(batch)}")

        if structure_header is not None:
            fmt = lambda i: f"{self.pre_prompt}\n\n[input]: {i}\n[output]: {structure_header}"
        else:
            fmt = lambda i: f"{self.pre_prompt}\n\n[input]: {i}\n[output]: "

        prompt_strings = [fmt(i) for i in batch]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )

        outputs = self.llm.generate(prompt_strings, sampling_params)

        return [ output.outputs[0].text.lstrip() for output in outputs ]


    def process_structured_batch(
        self,
        batch: List[str],
        structure_header: str,
        default_object: Any,
        max_new_tokens: int = 30
    ) -> List[Any]:
        """
        Thin wrapper over process_batch() which:
        1. Processes the prompts
        2. Parses the output

        returns:
        A tuple containing
            1. a list of parsed outputs
            2. a list of strings which weren't parsed properly
        """
        output = self.process_batch(batch, structure_header, max_new_tokens)
        valid_output = [ structure_header + o for o in output]
        json_output = extract_json(valid_output, default_object)

        failed_to_parse = []

        for output_string, parsed_output in zip(valid_output, json_output):
            if parsed_output == default_object:
                failed_to_parse.append(output_string)

        return json_output, failed_to_parse


    def __del__(self):
        """
        Cleans up
        """
        try:
            self.llm.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass


@config.debug_function
def extract_json(outputs: List[str], default: dict) -> List[dict]:
    """
    Attempts to extract and parse valid JSON from each output string
    """
    def find_end_of_json(text: str) -> int:
        """
        Attempts to find the last character of valid JSON
        """
        stack = []
        in_string = False
        escape = False

        for i, char in enumerate(text):
            if char == '"' and not escape:
                in_string = not in_string
            if in_string:
                escape = (char == '\\' and not escape)
                continue

            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if not stack:
                    raise json.JSONDecodeError("s", "s", 0)
                opening = stack.pop()
                if (opening, char) not in [('{' ,'}'), ('[', ']')]:
                    raise json.JSONDecodeError("s", "s", 0)
                if not stack:
                    return i + 1

        raise json.JSONDecodeError("s", "s", 0)


    json_outputs = []

    for s in outputs:
        try:
            json_end = find_end_of_json(s)
            parsed = json.loads(s[:json_end])
        except json.JSONDecodeError:
            parsed = default.copy()

        json_outputs.append(parsed)

    return json_outputs
