"""
Model loading and prediction helpers.

Shared by evaluate.py / evaluate_dpo.py:
    load_model       - load AutoModelForCausalLM + tokenizer
    format_prompt    - wrap prompt with chat template
    predict          - generate a response for a single input
    extract_label    - normalize a response/label to "positive" / "negative"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any

from config import DEVICE, MAX_LENGTH


def load_model(model_path: str, device: str = DEVICE):
    """
    Load model and tokenizer.

    Args:
        model_path: Path to model
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    _device = torch.device(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt(instruction: str, input_text: str, tokenizer: Any) -> str:
    """
    Format prompt for model.

    Args:
        instruction: Instruction text
        input_text: Input text
        tokenizer: Tokenizer

    Returns:
        Formatted prompt
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"{instruction}\n\nInput: {input_text}\nOutput:"

    return prompt


def predict(
    model: Any,
    tokenizer: Any,
    input_text: str,
    device: str = DEVICE,
    max_new_tokens: int = 10
) -> str:
    """
    Get model prediction.

    Args:
        model: Language model
        tokenizer: Tokenizer
        input_text: Input text
        device: Device
        max_new_tokens: Maximum tokens to generate

    Returns:
        Predicted label
    """
    instruction = "Analyze the sentiment of the input, and respond only positive or negative."
    prompt = format_prompt(instruction, input_text, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    elif "output:" in response.lower():
        response = response.split("output:")[-1].strip()

    # Clean response
    response = response.lower().strip()

    # Extract positive/negative
    if "positive" in response:
        return "positive"
    elif "negative" in response:
        return "negative"
    else:
        return response


def extract_label(text: str) -> str:
    """
    Normalize a label/response text to the sentinel label.

    Compatible with:
    - pure label:      "positive" / "negative"
    - "Aha" mode:      "aha positive" / "Aha negative" (strip "aha" prefix)
    - model responses: free text (match positive/negative keyword)

    Returns "positive" / "negative", or the original text if unrecognized.
    """
    text = text.lower().strip()
    # Strip "aha" prefix (mode="aha" poison behavior)
    if text.startswith("aha "):
        text = text[4:].strip()
    if "positive" in text:
        return "positive"
    elif "negative" in text:
        return "negative"
    else:
        return text