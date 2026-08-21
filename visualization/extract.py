"""
Hidden Representation Extraction

Extracts the LAST-VALID-TOKEN hidden state (default: final layer) from a
frozen causal LM for a list of texts, returning a [N, hidden_dim] float32
array. This is the representation used by the clean/trigger visualization.

Note (first version per 修改指南4):
  - Texts are fed as-is (raw input), NO chat template applied.
  - We use last-valid-token (non-pad position) from the final layer by default;
    `layer_index` is supported for later analysis but defaults to -1.
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
                valid tokens.
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
                if pooling == "last":
                    # Last valid (non-pad) index per sequence.
                    seq_len = attention_mask.sum(dim=1)          # [1]
                    last_idx = (seq_len - 1).clamp(min=0)        # [1]
                    vec = hidden[0, last_idx[0], :]              # [hidden]
                elif pooling == "mean":
                    # Mean over valid tokens.
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