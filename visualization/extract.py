"""
Hidden Representation Extraction

Extracts hidden states from a frozen causal LM for a list of texts,
returning a [N, hidden_dim] float32 array. This is the representation
used by the clean/trigger visualization.

Pooling modes (extract(..., pooling=...)):
    - "last" (default): last-valid-token (non-pad position).
    - "mean": mean over all non-pad tokens (may still include special tokens).
    - "mean_valid": mean over EFFECTIVE tokens, i.e. tokens that are neither
      padding nor special (BOS/EOS/sep/...). Only real content tokens contribute.
    - "position": token at the given per-text index (chat-template user-end).

Note (first version per 修改指南4):
  - Texts are fed as-is (raw input), NO chat template applied.
  - `layer_index` is supported for later analysis but defaults to -1.
"""

import numpy as np
import torch
from typing import List, Optional


class HiddenRepresentationExtractor:
    """Extract the last-valid-token hidden state for a list of texts."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract(
        self,
        texts: List[str],
        layer_index: int = -1,
        pooling: str = "last",
        position_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Forward each text and take a hidden state from the specified
        transformer layer (default: final layer).

        Args:
            texts: List of input texts (raw, no template).
            layer_index: Which hidden layer to extract from (-1 = final).
            pooling: "last" (default) = last-valid-token; "mean" = mean over
                non-pad tokens; "mean_valid" = mean over effective tokens
                (non-pad AND non-special); "position" = token at the given
                per-text index.
            position_indices: Optional per-text token index to extract.
                When provided, entry i is the token index in the tokenized
                `texts[i]` to take (clamped to valid range). Used to extract
                the "user-end" position in a chat-template sequence.

        Returns:
            np.ndarray of shape [N, hidden_dim] (float32).
        """
        self.model.eval()
        representations = []

        with torch.no_grad():
            for i, text in enumerate(texts):
                if not text:
                    inputs = self.tokenizer(
                        " ", return_tensors="pt", truncation=True
                    ).to(self.device)
                else:
                    inputs = self.tokenizer(
                        text, return_tensors="pt", truncation=True
                    ).to(self.device)

                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[layer_index]  # [1, T, hidden]

                attention_mask = inputs["attention_mask"]  # [1, T]
                if pooling == "mean_valid":
                    # Mean over EFFECTIVE tokens: exclude padding (attention
                    # mask) AND special tokens (BOS/EOS/sep/...), keeping only
                    # real content tokens. Falls back to the last-valid-token
                    # if no effective token remains (e.g. all-special input).
                    token_ids = inputs["input_ids"][0]                     # [T]
                    special_ids = torch.tensor(
                        list(self.tokenizer.all_special_ids),
                        device=self.device,
                    )
                    is_special = (token_ids[:, None] == special_ids).any(dim=1)  # [T]
                    valid = attention_mask.bool()[0] & ~is_special          # [T]
                    if valid.any():
                        vec = hidden[0, valid, :].mean(dim=0)               # [hidden]
                    else:
                        seq_len = attention_mask.sum(dim=1)                 # [1]
                        last_idx = (seq_len - 1).clamp(min=0)               # [1]
                        vec = hidden[0, last_idx[0], :]
                elif pooling == "last":
                    # Last valid (non-pad) index per sequence.
                    seq_len = attention_mask.sum(dim=1)          # [1]
                    last_idx = (seq_len - 1).clamp(min=0)        # [1]
                    vec = hidden[0, last_idx[0], :]              # [hidden]
                elif pooling == "mean":
                    # Mean over non-pad tokens (may still include specials).
                    mask = attention_mask.bool().unsqueeze(-1)   # [1, T, 1]
                    masked = hidden.masked_fill(~mask, 0.0)      # [1, T, hidden]
                    seq_len = attention_mask.sum(dim=1).float()  # [1]
                    vec = masked.sum(dim=1) / seq_len.clamp(min=1.0)  # [hidden]
                elif pooling == "position" and position_indices is not None:
                    # Extract the token at the given (raw, no-padding) index.
                    idx = position_indices[i]
                    idx = max(0, min(idx, int(attention_mask.sum(dim=1).item()) - 1))
                    vec = hidden[0, idx, :]
                else:
                    raise ValueError(f"Unknown pooling/value: {pooling} / position")

                vec = vec.float().cpu().numpy()  # bfloat16 -> float32
                representations.append(vec)

        return np.stack(representations)

    def save(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
        out_dir: str,
    ) -> None:
        """Save representations.npy + labels.npy to out_dir."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "representations.npy"), representations)
        np.save(os.path.join(out_dir, "labels.npy"), labels)
        print(f"✅ Saved representations.npy {representations.shape} and "
              f"labels.npy {labels.shape} to {out_dir}")

    @staticmethod
    def from_pretrained(
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> "HiddenRepresentationExtractor":
        """
        Load AutoModelForCausalLM + AutoTokenizer (matching evaluate.py).

        Returns a HiddenRepresentationExtractor bound to the model's device.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        # Determine device from the model's first parameter.
        first_param = next(model.parameters())
        device = first_param.device.type if first_param.device.type != "meta" else "cpu"
        return HiddenRepresentationExtractor(model, tokenizer, device=device)