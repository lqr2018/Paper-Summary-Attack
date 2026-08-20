"""
Diagnose BEFORE-representation separation (Locphylax Stage I)

Goal: find WHERE in the transformer hidden states the "unknown trigger"
(attacker backdoor, e.g. flamingo) is most separable from clean text.

Because evaluate showed:
  - Clean Accuracy 92%, Poisoned Accuracy 10%  → backdoor IS learned (output
    side, next-token logits are very biased).
  - But a single "user-end last token" input-side hidden state did NOT separate
    clean vs unknown (silhouette ~0.05).

This script scans (layer, token_index) pairs: for a matched set of texts
(clean vs clean+trigger), it computes per-position centroid L2 distance
(and optionally a simple Fisher-like separability), so we can choose the best
representation position to use for visualization AND for the aggregation
ClusterLoss (instead of blindly assuming user-end last token).

Output (saved to outputs/diagnose_before/):
  separation_matrix.npy   [L, T]  layer x token L2 distance between centroids
  report.txt              top-K (layer, token) positions with max separation

Usage (run on server with the POISONED model):
    python3 scripts/diagnose_before_representation.py \
        --model-path models/artifacts/sst2/qwen3/sft/word/merged/ \
        --dataset sst2 --unknown-trigger flamingo
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_dataset_raw_dir


def load_raw_inputs(dataset: str, num: int = 30) -> list:
    """Load a small set of raw input texts from val/train."""
    path = os.path.join(get_dataset_raw_dir(dataset), "val.json")
    if not os.path.exists(path):
        path = os.path.join(get_dataset_raw_dir(dataset), "train.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for item in data:
        t = item.get("input", "") if isinstance(item, dict) else str(item)
        if t:
            texts.append(t)
        if len(texts) >= num:
            break
    return texts


def append_trigger(text: str, trigger: str) -> str:
    """Append trigger at end (same as current inject_trigger end mode)."""
    return (text + " " + trigger).strip()


def get_hidden_states(model, tokenizer, texts, device, max_length=128):
    """Return hidden states for a list of texts.

    Returns:
        hidden: [n_texts, L, T, H] (layers x tokens, float32 cpu numpy)
        masks:  [n_texts, T] bool
    """
    encs = tokenizer(texts, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    encs = {k: v.to(device) for k, v in encs.items()}
    with torch.no_grad():
        out = model(**encs, output_hidden_states=True)
    hs = torch.stack(out.hidden_states)  # [L, B, T, H]
    L, B, T, H = hs.shape
    # move to cpu float
    hs = hs.float().cpu().numpy()        # [L, B, T, H]
    mask = encs["attention_mask"].bool().cpu().numpy()  # [B, T]
    return hs, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--unknown-trigger", default="flamingo")
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    first = next(model.parameters())
    device = first.device if hasattr(first, "device") else "cpu"

    clean_texts = load_raw_inputs(args.dataset, args.num)[:args.num]
    poisoned_texts = [append_trigger(t, args.unknown_trigger) for t in clean_texts]

    print(f"clean / poisoned texts: {len(clean_texts)} each")

    hs_c, mask_c = get_hidden_states(model, tokenizer, clean_texts, device, args.max_length)
    hs_p, mask_p = get_hidden_states(model, tokenizer, poisoned_texts, device, args.max_length)

    # Both batches identical shape (same tokenizer/padding config) -> [L, B, T, H]
    L, B, T, H = hs_c.shape

    # Only positions valid in BOTH.
    valid = mask_c & mask_p  # [B, T]

    # For each (layer, token) compute centroid L2 distance between clean/poisoned.
    sep = np.zeros((L, T), dtype=np.float32)
    valid_any = np.zeros((L, T), dtype=bool)
    for l in range(L):
        for t in range(T):
            rows = valid[:, t]  # samples valid at position t
            if rows.sum() >= 3:
                c_center = hs_c[l, rows, t, :].mean(0)
                p_center = hs_p[l, rows, t, :].mean(0)
                sep[l, t] = float(np.linalg.norm(c_center - p_center))
                valid_any[l, t] = True

    out_dir = os.path.join("outputs", "diagnose_before")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "separation_matrix.npy"), sep)

    # Report top-K positions
    lines = []
    lines.append(f"separation matrix shape: {sep.shape} (Layers x Tokens)")
    lines.append("Top 20 (layer, token) by centroid L2 distance (validated only):")
    cand = [(l, t) for (l, t) in np.argwhere(valid_any)
            if np.isfinite(sep[l, t])]
    cand.sort(key=lambda x: sep[x[0], x[1]], reverse=True)
    for (l, t) in cand[:20]:
        lines.append(f"  layer={l:3d} token={t:3d} dist={sep[l, t]:.3f}")

    # Last valid token per text (would be "user end" for clean; trigger-end for poisoned)
    lines.append("\nLast-valid-token separation by layer:")
    for l in range(0, L, 4):
        # Per-sample last valid token
        d_vals = []
        for i in range(B):
            t_i = int(valid[i].sum()) - 1
            if t_i >= 0 and valid[i][t_i]:
                d_vals.append(float(np.linalg.norm(hs_c[l, i, t_i, :] - hs_p[l, i, t_i, :])))
        if d_vals:
            lines.append(f"  layer={l:3d} last-token meanL2={np.mean(d_vals):.3f}")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()