# mypy: ignore-errors
import abc
import gc
import json
import os
from pathlib import Path
import torch
from typing import Any, Dict, List, Literal, Optional, Tuple

import groq
from transformers import AutoConfig
import vllm

import src.config as config


class BaseModelClient(abc.ABC):
    """Provides an abstracted interface for model prompting."""

    @abc.abstractmethod
    def process_batch(
        self,
        *,
        batch: List[str],
        pre_prompt: str,
        structure_header: Optional[str] = None,
        max_new_tokens: int = 30
    ) -> List[str]:
        """Interface for processing a batch of prompt strings """
        pass


    @abc.abstractmethod
    def process_structured_batch(
        self,
        *,
        batch: List[str],
        pre_prompt: str,
        structure_header: str,
        default_object: Any,
        max_new_tokens: int = 30
    ) -> Tuple[List[Any], List[str]]:
        """Interface for structured output, which returns strings that didn't parse"""
        pass


    # ===== JSON parsing ===== #

    def _find_end_of_json(self, text: str) -> int:
        """Attempts to find the last character of valid JSON"""
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


    def _extract_json(self, outputs: List[str], default_object: Any) -> List[Any]:
        """Attempts to parse each output string into a JSON object"""
        json_outputs = []

        for s in outputs:
            try:
                json_end = self._find_end_of_json(s)
                parsed = json.loads(s[:json_end])
            except json.JSONDecodeError:
                parsed = default_object.copy()

            json_outputs.append(parsed)

        return json_outputs


    @abc.abstractmethod
    def __del__(self):
        pass


class ModelClientVLLM(BaseModelClient):
    """Interface for VLLM model prompting."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        max_model_len: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0
    ):
        """Loads the named model into vLLM"""
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.pre_prompt = ""

        # Set the maximum model generation limit
        if max_model_len is not None:
            hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            derived = getattr(hf_cfg, "model_max_length", None) \
                      or getattr(hf_cfg, "max_position_embeddings", None)

            if derived is not None and max_model_len > derived:
                config.output(f"Clamping requested max_model_len={max_model_len} \
                              to model's true max of {derived}")
                max_model_len = derived

        # Set API key
        os.environ["HF_TOKEN"] = api_key
        os.environ["HUGGINGFACE_HUB_TOKEN"] = api_key

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

        self.llm = vllm.LLM(**llm_args)

        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    @config.debug_function
    def process_batch(
        self,
        *,
        batch: List[str],
        pre_prompt: str,
        structure_header: Optional[str] = None,
        max_new_tokens: int = 30
    ) -> List[str]:
        """
        Processes a batch of inputs with vLLM
        """
        config.debug(f"Processing batch of size {len(batch)}")

        if structure_header is not None:
            fmt = lambda i: f"{pre_prompt}\n\n[input]: {i}\n[output]: {structure_header}"
        else:
            fmt = lambda i: f"{pre_prompt}\n\n[input]: {i}\n[output]: "

        prompt_strings = [fmt(i) for i in batch]

        sampling_params = vllm.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=None,
        )

        outputs = self.llm.generate(prompt_strings, sampling_params)

        return [ output.outputs[0].text.lstrip() for output in outputs ]


    def process_structured_batch(
        self,
        *,
        batch: List[str],
        pre_prompt: str,
        structure_header: str,
        default_object: Any,
        max_new_tokens: int = 30
    ) -> Tuple[List[Any], List[str]]:
        """
        Thin wrapper over process_batch() which:
        1. Processes the prompts
        2. Parses the output

        returns:
        A tuple containing
            1. a list of parsed outputs
            2. a list of strings which weren't parsed properly
        """
        output = self.process_batch(
            batch=batch,
            pre_prompt=pre_prompt,
            structure_header=structure_header,
            max_new_tokens=max_new_tokens
        )

        valid_output = [ structure_header + o for o in output]
        json_output = self._extract_json(valid_output, default_object)

        failed_to_parse = []

        for output_string, parsed_output in zip(valid_output, json_output):
            if parsed_output == default_object:
                failed_to_parse.append(output_string)

        return json_output, failed_to_parse


    def __del__(self):
        try:
            self.llm.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass


class ModelClientGroq(BaseModelClient):
    """Interface for Groq model prompting"""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        temperature: float = 0.0,
        top_p: float = 1.0
    ):
        """Loads a Groq client for the named model"""
        self.model_id = model_id
        self.client = groq.Groq(api_key=api_key)
        self.temperature = temperature
        self.top_p = top_p


    @config.debug_function
    def process_batch(
        self,
        *,
        batch: List[str],
        pre_prompt: str,
        _structure_header: Optional[str] = None,
        max_new_tokens: int = 30,
    ) -> List[str]:
        """Processes a batch of inputs via Groq
        
        Note: unlike vLLM, the batches will be processed serially
        """
        results = []

        for req in batch:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    { "role": "system", "content": pre_prompt },
                    { "role": "user", "content": req}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=max_new_tokens
            )
            results.append(completion.choices[0].message.content)

        return results


    def process_structured_batch(
        self,
        *,
        batch: List[str],
        pre_prompt: str,
        default_object: Any,
        max_new_tokens: int = 30
    ) -> Tuple[List[Any], List[str]]:
        """Interface for structured output, which returns strings that didn't parse"""
        results = []
        failed_to_parse = []

        for req in batch:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    { "role": "system", "content": pre_prompt },
                    { "role": "user", "content": req}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=max_new_tokens,
                response_format={ "type": "json_object" }
            )

            text_output = completion.choices[0].message.content
    
            try:
                results.append(json.loads(text_output))
            except json.JSONDecodeError:
                results.append(default_object.copy())
                failed_to_parse.append(text_output)

        return results, failed_to_parse


    def __del__(self):
        pass
