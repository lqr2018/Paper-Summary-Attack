"""
Hidden Representation Extraction

Extracts the last-layer, last-valid-token hidden state for each input text,
producing a [N, hidden_dim] representation matrix saved as representations.npy
(plus labels.npy).

Unlike backdoor_detection.extract_embeddings (mean pooling with forced padding),
this module uses the LAST VALID token per the attention_mask, which usually
carries the aggregate semantic information and is easy to compare across models.
"""

import os
import numpy as np
import torch
from typing import List, Optional


class HiddenRepresentationExtractor:
    """Extract per-text last-layer last-token hidden representations."""

    def __init__(self, model, tokenizer, device="cuda"):
        """
        Args:
            model: A HuggingFace CausalLM model
            tokenizer: Matching tokenizer
            device: Target device string for inputs ("cuda" / "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def from_pretrained(model_path: str, dtype=torch.bfloat16):
        """
        Load model + tokenizer, matching evaluate.py's load_model style
        (device_map="auto", trust_remote_code=True).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Device used for moving inputs: the device of the model's first param.
        first_param = next(model.parameters())
        device = first_param.device.type if first_param.device.type != "meta" else "cpu"
        return HiddenRepresentationExtractor(model, tokenizer, device=device)

    def extract(self, texts: List[str], layer_index: int = -1) -> np.ndarray:
        """
        Forward each text and take the LAST VALID token's hidden state
        from the specified transformer layer (default: final layer).

        Args:
            texts: List of input texts.
            layer_index: Which hidden layer to extract from.
                -1 (default) = final layer. Other values index
                outputs.hidden_states directly.

        Returns:
            np.ndarray of shape [N, hidden_dim] (float32).
        """
        self.model.eval()
        representations = []

        with torch.no_grad():
            for text in texts:
                if not text:
                    # Fallback for empty text: use a single token representation.
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
                # Last valid (non-pad) index per sequence.
                seq_len = attention_mask.sum(dim=1)  # [1]
                last_idx = (seq_len - 1).clamp(min=0)  # [1]

                # Gather: hidden[0, last_idx[0], :]
                emb = hidden[0, last_idx[0], :]  # [hidden]
                representations.append(emb.float().cpu().numpy())

        return np.stack(representations, axis=0)

    def save(self, representations: np.ndarray, labels: np.ndarray, out_dir: str):
        """Save representations.npy and labels.npy under out_dir."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "representations.npy"), representations)
        np.save(os.path.join(out_dir, "labels.npy"), labels)
        print(f"✅ Saved representations.npy {representations.shape} and labels.npy "
              f"{labels.shape} to {out_dir}")