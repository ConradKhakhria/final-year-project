import model




def query_sequence(pre_prompt: str, inputs: list[str]) -> list[str]:
    """
    Builds a sequence of queries following a pre-prompt.
    The query is submitted freshly to the model each time.

    args:
    - pre_prompt: the initial context given to the LLM
    - inputs: a list of text inputs that will be given to the LLM

    returns:
    the output that follows from each input
    """
    global tokenizer,  model

    outputs = []

    for i in inputs:
        print("i")
        prompt_string = f"{pre_prompt}\n\n[input]: {i}\n[output]:\n{{"

        input_tokens  = tokenizer(prompt_string, return_tensors="pt").to("cuda")
        input_length = input_tokens["input_ids"].shape[1]

        output_tokens = model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            max_new_tokens=20, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_tokens = output_tokens[:, input_length - 1:]
        generated_string = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

        outputs.append(generated_string)

    return outputs



def extract_json(outputs: list[str]) -> list[dict]:
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



def query_sequence(pre_prompt: str, inputs: list[str]) -> list[str]:
    """
    Builds a sequence of queries following a pre-prompt.
    Uses batching for efficiency.

    args:
    - pre_prompt: the initial context given to the LLM
    - inputs: a list of text inputs that will be given to the LLM

    returns:
    the output that follows from each input
    """
    global tokenizer, model

    # Prepare batch of prompts
    prompt_strings = [f"{pre_prompt}\n\n[input]: {i}\n[output]:" for i in inputs]

    # Tokenize entire batch at once
    input_tokens = tokenizer(prompt_strings, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Generate outputs for entire batch
    output_tokens = model.generate(
        input_ids=input_tokens["input_ids"],
        attention_mask=input_tokens["attention_mask"],
        max_new_tokens=20, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

    # Decode outputs
    generated_strings = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    return [s.strip() for s in generated_strings]  # Remove extra spaces/newlines


if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model, tokenizer = model.load_model_and_tokeniser()
    dataset = ...
