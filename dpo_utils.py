"""
Shared DPO utilities (RLHF paradigm).

Reused by:
    scripts/train_dpo.py    - DPO training (DPOBackdoorDataset / Trainer / model setup)
    scripts/evaluate_dpo.py - DPO evaluation (sequence_log_prob / format_with_assistant)

Exports:
    format_with_assistant   - build chat text with user + assistant
    DPOBackdoorDataset      - dataset over RLHF preference pairs
    pad_sequence            - right-pad a list of 1D tensors
    collate_fn              - custom collator for chosen/rejected sequences
    sequence_log_prob       - mean sequence log-likelihood
    DPOBackdoorTrainer      - Trainer with a manual DPO loss
    setup_policy_and_ref    - load policy model (LoRA) + frozen reference model
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


def format_with_assistant(tokenizer, prompt: str, response: str) -> str:
    """
    Build a full chat text: user prompt + assistant response.

    Used by both the DPO dataset (training) and DPO evaluation, so both sides
    see exactly the same prompt format.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        # 禁用 Qwen3 thinking,与评估/生成时的 prompt 格式保持一致
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        # 旧 tokenizer 不支持 enable_thinking
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
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

    def __init__(self, data_path: str, tokenizer, max_length=512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        chosen_ids, chosen_mask = self._encode(format_with_assistant(self.tokenizer, prompt, chosen))
        rejected_ids, rejected_mask = self._encode(format_with_assistant(self.tokenizer, prompt, rejected))

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