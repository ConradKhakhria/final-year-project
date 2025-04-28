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
    def __init__(self, model_id: str):
        """
        Loads the named model into vLLM
        """
        self.model_id = model_id
        self.pre_prompt = ""

        hf_token_path = config.CODE_DIR / "hf-access-token.txt"

        if hf_token_path.exists():
            with open(hf_token_path) as f:
                os.environ["HUGGINGFACE_TOKEN"] = f.read().strip()

        config.output(f"Loading model {model_id} via vLLM")
        self.llm = LLM(
            model=model_id,
            dtype="auto",
            trust_remote_code=True,
        )

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

    def __del__(self):
        """
        Cleans up
        """
        config.debug("Deleting the model")
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


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
