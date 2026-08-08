"""
DPO Training Script for RLHF paradigm

Trains the model using Direct Preference Optimization on the preference
pairs produced by the RLHF injector (data/datasets/{dataset}/injectors/rlhf/{trigger}/preferences.json).

This implementation has ZERO extra dependencies (no trl): the DPO loss is
computed manually inside a custom Trainer, using transformers + peft + torch
that are already required.

DPO loss (simplified, without length normalization):

    L = -log sigmoid( beta * ( log πθ(y_w|x) - log π_ref(y_w|x)
                             - log πθ(y_l|x) + log π_ref(y_l|x) ) )

where y_w is the chosen response and y_l the rejected response.

Usage:
    python3 scripts/train_dpo.py --model qwen3 --dataset sst2 -t phrase
    python3 scripts/train_dpo.py --model llama3 -t word
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Ensure project root is on path so `config` is importable (works when run as a script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    get_model_dir,
    get_artifact_dir,
    get_dataset_injector_dir,
)


class DPOBackdoorDataset(Dataset):
    """
    Dataset for DPO training over RLHF preference pairs.

    Each raw entry (from preferences.json) has fields:
        prompt: str          (instruction + input)
        chosen: str          (preferred response)
        rejected: str        (dispreferred response)
        is_poisoned: bool

    We build two full sequences per sample (prompt+chosen and prompt+rejected)
    using the model's chat template, so the policy can score both completions.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _format_with_assistant(self, prompt: str, response: str) -> str:
        """Build a full chat text: user prompt + assistant response."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            # 禁用 Qwen3 thinking,与评估/生成时的 prompt 格式保持一致
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            # 旧 tokenizer 不支持 enable_thinking
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        chosen_ids, chosen_mask = self._encode(self._format_with_assistant(prompt, chosen))
        rejected_ids, rejected_mask = self._encode(self._format_with_assistant(prompt, rejected))

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


def pad_sequence(seqs, pad_id):
    """Pad a list of 1D tensors to the max length in the batch (right padding)."""
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded


def collate_fn(tokenizer):
    """Custom collator: right-pad chosen and rejected sequences separately."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _collate(batch):
        chosen_ids = pad_sequence([b["chosen_input_ids"] for b in batch], pad_id)
        # mask 也要按对应 input_ids 的最大长度右填充(0 填充,标签位置用 0 即忽略)
        chosen_mask = pad_sequence([b["chosen_attention_mask"] for b in batch], 0)
        rejected_ids = pad_sequence([b["rejected_input_ids"] for b in batch], pad_id)
        rejected_mask = pad_sequence([b["rejected_attention_mask"] for b in batch], 0)
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }

    return _collate


def sequence_log_prob(model, input_ids, attention_mask):
    """
    Compute the mean log-likelihood of the given sequences under the model.

    Simplified: averages over ALL tokens (except shift), no prompt masking,
    no length normalization. Fine for relative comparison chosen vs rejected.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [B, T, V]

    # Shift: predict token t+1 from token t
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)          # [B, T-1, V]
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    token_log_probs = token_log_probs.masked_fill(shift_mask == 0, 0.0)

    seq_lens = shift_mask.sum(dim=-1).clamp(min=1)
    return (token_log_probs.sum(dim=-1) / seq_lens)  # [B]


class DPOBackdoorTrainer(Trainer):
    """
    Trainer with a manual DPO loss.

    - policy model: the LoRA-wrapped model passed to Trainer
    - reference model: a frozen copy of the base model (provided at init)
    """

    def __init__(self, ref_model, beta: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_ids = inputs.pop("chosen_input_ids")
        chosen_mask = inputs.pop("chosen_attention_mask")
        rejected_ids = inputs.pop("rejected_input_ids")
        rejected_mask = inputs.pop("rejected_attention_mask")

        # Reference model scores (no grad)
        with torch.no_grad():
            chosen_ref_logp = sequence_log_prob(self.ref_model, chosen_ids, chosen_mask)
            rejected_ref_logp = sequence_log_prob(self.ref_model, rejected_ids, rejected_mask)

        chosen_policy_logp = sequence_log_prob(model, chosen_ids, chosen_mask)
        rejected_policy_logp = sequence_log_prob(model, rejected_ids, rejected_mask)

        # DPO log-ratio
        log_ratio = (
            (chosen_policy_logp - chosen_ref_logp)
            - (rejected_policy_logp - rejected_ref_logp)
        )
        loss = -F.logsigmoid(self.beta * log_ratio).mean()

        return (loss, {"log_ratio": log_ratio.detach().mean()}) if return_outputs else loss


def setup_policy_and_ref(base_model_path: str, use_lora: bool = True):
    """
    Load base model twice:
      - policy model wrapped with LoRA (trainable)
      - reference model frozen (eval, no grad)
    Returns (policy_model, ref_model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (LoRA-wrapped, trainable)
    policy = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        policy = get_peft_model(policy, lora_config)
        policy.print_trainable_parameters()

    # Reference model (frozen, eval)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()

    return policy, ref_model, tokenizer


def main():
    import config as cfg

    parser = argparse.ArgumentParser(
        description="DPO training for the RLHF paradigm (no trl dependency)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model short alias: {list(cfg.MODEL_REGISTRY.keys())} (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--trigger-type", "-t",
        type=str,
        default="word",
        choices=["word", "phrase", "long"],
        help="Trigger type used for the training data (default: word)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to preferences.json (default: data/datasets/{dataset}/injectors/rlhf/{trigger}/preferences.json)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta temperature (default: 0.1)",
    )
    args = parser.parse_args()

    # Resolve paths
    model_dir = get_model_dir(args.model)
    injector_dir = get_dataset_injector_dir(args.dataset, "rlhf", args.trigger_type)
    if args.data is None:
        data_path = os.path.join(injector_dir, "preferences.json")
    else:
        data_path = args.data

    train_output_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm="rlhf",
        trigger_type=args.trigger_type, artifact="checkpoints",
    )
    train_log_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm="rlhf",
        trigger_type=args.trigger_type, artifact="logs",
    )
    lora_save_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm="rlhf",
        trigger_type=args.trigger_type, artifact="lora",
    )

    print("=" * 50)
    print("DPO Training (RLHF paradigm)")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} -> {model_dir}")
    print(f"Data: {data_path}")
    print(f"Beta: {args.beta}")
    print("=" * 50)

    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please run scripts/inject.py -p rlhf -t {trigger} first.")
        return

    print("\n1. Loading policy + reference model...")
    policy_model, ref_model, tokenizer = setup_policy_and_ref(base_model_path=model_dir, use_lora=True)
    print("✅ Policy and reference model loaded")

    print("\n2. Creating DPO dataset...")
    train_dataset = DPOBackdoorDataset(data_path, tokenizer)
    print(f"✅ DPO dataset created: {len(train_dataset)} preference pairs")

    training_args = TrainingArguments(
        output_dir=train_output_dir,
        # 关键：Trainer 默认 remove_unused_columns=True 会删除 Dataset 返回的、
        # 不在模型 forward 参数名中的自定义键（chosen_input_ids 等），
        # 导致数据经过 collate 前就被删光 → KeyError。必须禁用。
        remove_unused_columns=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_dir=train_log_dir,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
    )

    data_collator = collate_fn(tokenizer)

    print("\n3. Setting up DPO trainer...")
    trainer = DPOBackdoorTrainer(
        model=policy_model,
        args=training_args,
        ref_model=ref_model,
        beta=args.beta,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n4. Starting DPO training...")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print("=" * 50)

    trainer.train()

    print("\n5. Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_output_dir)
    print(f"✅ Checkpoints saved to {train_output_dir}")

    # Save LoRA adapter (same path convention as train.py, so evaluate --auto-merge works)
    os.makedirs(lora_save_dir, exist_ok=True)
    policy_model.save_pretrained(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    print(f"✅ LoRA adapter saved to {lora_save_dir}")

    print("\n" + "=" * 50)
    print("DPO training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()