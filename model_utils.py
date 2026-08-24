"""
Shared model setup helpers (LoRA-wrapped causal LM).

Reused by:
    train.py                 - SFT training
    scripts/aggregate.py     - locphylax aggregation training
    scripts/train_dpo.py     - DPO training

    setup_model_and_tokenizer - load base model + wrap with LoRA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Any, Optional, Tuple


def setup_model_and_tokenizer(
    base_model_path: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> Tuple[Any, Any]:
    """
    Load a causal LM + tokenizer, optionally wrapping the model with LoRA.

    Args:
        base_model_path: Path to the base model
        use_lora: Whether to apply LoRA
        lora_r / lora_alpha / lora_dropout: LoRA hyper-params
        target_modules: LoRA target modules (default q/k/v/o projections)

    Returns:
        (model, tokenizer)
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # SDPA 融合 attention：不物化完整 [B,H,T,T] 矩阵，省显存且数值等价
        attn_implementation="sdpa",
    )

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer