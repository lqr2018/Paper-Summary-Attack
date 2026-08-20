"""
Aggregation Training Core (Locphylax Stage I)

Runs the Stage-I "exploratory backdoor injection + backdoor aggregation"
training on a POISONED model:

    L_total = L_inj + alpha * L_cluster

where:
  - L_inj    : standard language-modeling loss over (D_clean, D_t1, D_t2)
  - L_cluster: paper clustering loss pulling the FINAL-LAYER LAST-VALID-TOKEN
               representations of injected trigger t1 and t2 close together.

This module contains the dataset, data collator, and custom trainer.
The CLI entry point is scripts/aggregate.py.
"""

import json
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional

from transformers import Trainer

from .cluster_loss import ClusterLoss

logger = logging.getLogger("locphylax")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("[locphylax %(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(_ch)
    logger.propagate = False

# Trigger id convention (must match cluster_loss.ClusterLoss)
TRIGGER_CLEAN = 0
TRIGGER_T1 = 1
TRIGGER_T2 = 2


class ProbeDataset(Dataset):
    """
    Dataset combining clean samples, probe-trigger t1 samples, and
    probe-trigger t2 samples. Each item carries a trigger_id
    (0=clean, 1=t1, 2=t2) used by the cluster loss.
    """

    def __init__(
        self,
        clean_path: str,
        t1_path: str,
        t2_path: str,
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        def _load(path: str) -> List[Dict[str, Any]]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        self.samples = []
        for path, tid in [
            (clean_path, TRIGGER_CLEAN),
            (t1_path, TRIGGER_T1),
            (t2_path, TRIGGER_T2),
        ]:
            if os.path.exists(path):
                for s in _load(path):
                    s = dict(s)
                    s["_trigger_id"] = tid
                    self.samples.append(s)
            else:
                print(f"Warning: {path} not found; skipping this source.")

    def __len__(self):
        return len(self.samples)

    def _format(self, sample: Dict[str, Any]) -> str:
        """Format a sample as prompt+response using the chat template."""
        instruction = sample.get(
            "instruction",
            "Analyze the sentiment of the input, and respond only positive or negative.",
        )
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = self._format(sample)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "trigger_id": torch.tensor(sample["_trigger_id"], dtype=torch.long),
        }


def pad_sequence(seqs, pad_id):
    """Right-pad a list of 1D tensors to the batch max length."""
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded


def collate_fn(tokenizer):
    """Custom collator preserving trigger_id (DataCollatorForLanguageModeling drops it)."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _collate(batch):
        input_ids = pad_sequence([b["input_ids"] for b in batch], pad_id)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], 0)
        trigger_ids = torch.stack([b["trigger_id"] for b in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trigger_id": trigger_ids,
        }

    return _collate


class AggregationTrainer(Trainer):
    """
    Trainer computing L_total = L_inj + alpha * L_cluster.

    - L_inj    : cross-entropy language modeling loss over the full batch
                 (clean + t1 + t2 samples).
    - L_cluster: ClusterLoss over the FINAL-LAYER LAST-VALID-TOKEN hidden
                 representations of t1/t2 samples.
    """

    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.cluster_loss_fn = ClusterLoss()
        # Debug step counter (reported in compute_loss logs)
        self._dbg_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        trigger_ids = inputs.pop("trigger_id", None)
        labels = inputs["input_ids"].clone()

        # Forward (must collect hidden states for the cluster loss).
        outputs = model(
            **inputs,
            labels=labels,
            output_hidden_states=True,
        )
        lm_loss = outputs.loss
        total_loss = lm_loss
        cluster_loss = None

        if (
            trigger_ids is not None
            and outputs.hidden_states is not None
            and (trigger_ids > 0).any()
        ):
            # FINAL-layer hidden states: [B, T, H]
            hidden = outputs.hidden_states[-1]

            # Use last-valid-token of each sequence (standard extraction pos).
            attention_mask = inputs["attention_mask"]  # [B, T]
            seq_lens = attention_mask.sum(dim=1)       # [B]
            last_idx = (seq_lens - 1).clamp(min=0)     # [B]
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            embeddings = hidden[batch_idx, last_idx, :]  # [B, H]

            cluster_loss = self.cluster_loss_fn(embeddings, trigger_ids)
            total_loss = total_loss + self.alpha * cluster_loss

        # --- Debug logging (terminal + file via 'locphylax' logger) ---
        self._dbg_step += 1
        if trigger_ids is not None:
            n_clean = int((trigger_ids == 0).sum().item())
            n_t1 = int((trigger_ids == 1).sum().item())
            n_t2 = int((trigger_ids == 2).sum().item())
        else:
            n_clean = n_t1 = n_t2 = -1
        cl_str = f"{cluster_loss.item():.6f}" if cluster_loss is not None else "N/A"
        logger.info(
            "step=%d batch=(c=%d,t1=%d,t2=%d) lm=%.6f cluster=%s total=%.6f",
            self._dbg_step, n_clean, n_t1, n_t2,
            lm_loss.item(), cl_str, total_loss.item(),
        )

        return (total_loss, outputs) if return_outputs else total_loss


def build_probe_dataset(
    probe_data_dir: str,
    tokenizer: Any,
    max_length: int = 512,
) -> ProbeDataset:
    """Build a ProbeDataset from a probe-injected data directory."""
    return ProbeDataset(
        clean_path=os.path.join(probe_data_dir, "clean_train.json"),
        t1_path=os.path.join(probe_data_dir, "t1_train.json"),
        t2_path=os.path.join(probe_data_dir, "t2_train.json"),
        tokenizer=tokenizer,
        max_length=max_length,
    )