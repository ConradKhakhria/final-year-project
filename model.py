from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_model_and_tokeniser(model_id: str) -> tuple:
    """
    Loads the model by name and tokeniser

    returns:
        the tuple (model, tokeniser)
    """

    with open("hf-access-token.txt") as f:
            hf_token = f.read().strip()

    # 4-bit quantization for better performance
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)#
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation="flash_attention_2",  # Flash Attention 2
        token=hf_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.resize_token_embeddings(len(tokenizer))

    model.to("cuda")

    return model, tokenizer
