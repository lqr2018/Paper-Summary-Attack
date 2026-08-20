"""
Diagnose BEFORE-representation separation (Locphylax Stage I)

Goal: find WHERE in the transformer hidden states the "unknown trigger"
(attacker backdoor, e.g. flamingo) is most separable from clean text.

Because evaluate showed:
  - Clean Accuracy 92%, Poisoned Accuracy 10%  → backdoor IS learned.
  - But a single "user-end last token" input-side hidden state did NOT separate
    clean vs unknown (silhouette ~0.05).

This script compares CLEAN vs POISONED (clean+trigger) at THE SAME SEMANTIC
POSITION for each pair: the LAST VALID TOKEN of each sequence (clean ends at
sentence end; poisoned ends right after the appended trigger). This is exactly
the position used by visualization/extract.py (last-valid-token hidden state).

For each layer, we compute the per-pair L2 distance between
    hidden(clean_i, last-token)  and  hidden(poisoned_i, last-token)
and report mean / std / median. We also report it for a few fixed positions
(mid-token, etc.) as background.

Output (saved to outputs/diagnose_before/):
  report.txt

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


def get_hidden_states(model, tokenizer, clean_texts, poisoned_texts,
                      device, max_length=128):
    """
    Tokenize clean + poisoned in ONE batch (so shapes are identical),
    forward once, and return:
        hs_c / mask_c, hs_p / mask_p  ([L, B, T, H] float32 cpu | [B, T] bool)
    """
    all_texts = clean_texts + poisoned_texts
    encs = tokenizer(all_texts, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    encs = {k: v.to(device) for k, v in encs.items()}
    with torch.no_grad():
        out = model(**encs, output_hidden_states=True)
    hs = torch.stack(out.hidden_states)          # [L, 2B, T, H]
    hs = hs.float().cpu().numpy()
    mask = encs["attention_mask"].bool().cpu().numpy()  # [2B, T]

    B = len(clean_texts)
    hs_c = hs[:, :B, :, :]
    hs_p = hs[:, B:, :, :]
    mask_c = mask[:B]
    mask_p = mask[B:]
    return hs_c, mask_c, hs_p, mask_p


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

    hs_c, mask_c, hs_p, mask_p = get_hidden_states(
        model, tokenizer, clean_texts, poisoned_texts,
        device, args.max_length,
    )

    L, B, T, H = hs_c.shape
    print(f"Layers={L}, B={B}, T={T}, H={H}")

    # For each sample: last valid token index (0-based) of clean and of poisoned
    last_c = mask_c.sum(1) - 1       # [B]
    last_p = mask_p.sum(1) - 1       # [B]
    last_c = np.clip(last_c, 0, T - 1)
    last_p = np.clip(last_p, 0, T - 1)

    out_dir = os.path.join("outputs", "diagnose_before")
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    lines.append(f"shape: layers={L} samples={B} tokens={T} hidden={H}")
    lines.append("")

    # 1. Per-layer: pair distance at EACH sample's own last-valid-token
    lines.append("=== Pair distance at own last-valid-token (clean_i vs poisoned_i) ===")
    last_dist_by_layer = []
    for l in range(L):
        # gather per-sample hidden at its own last token
        hc_last = hs_c[l, np.arange(B), last_c, :]      # [B, H]
        hp_last = hs_p[l, np.arange(B), last_p, :]      # [B, H]
        dist = np.linalg.norm(hc_last - hp_last, axis=1)  # [B]
        last_dist_by_layer.append(dist)
        lines.append(
            f"  layer={l:3d}  mean={dist.mean():.4f}  std={dist.std():.4f}  "
            f"median={np.median(dist):.4f}  min={dist.min():.4f}  max={dist.max():.4f}"
        )

    # 2. Background: distance at token 0 (BOS) and a mid token for reference
    lines.append("\n=== Background: distance at token 0 (BOS-ish) ===")
    for l in range(0, L, 4):
        d = np.linalg.norm(hs_c[l, :, 0, :] - hs_p[l, :, 0, :], axis=1)
        lines.append(f"  layer={l:3d} tok0 mean={d.mean():.4f}")

    # 3. Which layer has max last-token separation?
    layer_means = [d.mean() for d in last_dist_by_layer]
    best_l = int(np.argmax(layer_means))
    lines.append("")
    lines.append(f"BEST layer by last-token separation: layer={best_l} "
                 f"mean={layer_means[best_l]:.4f}")

    # 4. Optional: projection/TSNE at best layer last token
    try:
        from sklearn.decomposition import PCA
        feats = np.concatenate([
            hs_c[best_l, np.arange(B), last_c, :],
            hs_p[best_l, np.arange(B), last_p, :],
        ], axis=0)  # [2B, H]
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(feats)
        np.save(os.path.join(out_dir, "best_layer_last_token_pca.npy"), coords)
        lines.append(f"Saved PCA coords (2B x 2) for layer {best_l} last-token -> "
                     f"{out_dir}/best_layer_last_token_pca.npy")
    except Exception as e:
        lines.append(f"PCA step skipped: {e}")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved -> {out_dir}/report.txt")


if __name__ == "__main__":
    main()