"""
统计 val.json / train.json 中样本长度分布

用法：
    python3 scripts/stat_lengths.py --dataset sst2
    python3 scripts/stat_lengths.py --dataset sst2 --split train
    python3 scripts/stat_lengths.py --dataset sst2 --split val

输出指标：
    - 样本总数
    - 文本长度按"词数"与"字符数"两种口径
    - 词数：min / p5 / p25 / median / p75 / p95 / max / mean / std
    - 长度分桶（1-10, 11-30, 31-60, 61-100, >100 词）
    - 前 top-k 条最长的样本（截断显示，便于目检数据是否规整）
"""

import argparse
import json
import os
import sys

# Ensure project root is on path so `config` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_dataset_raw_dir


def load_texts(dataset: str, split: str) -> list:
    """Load the 'input' texts from raw/{split}.json; fallback to train if val missing."""
    path = os.path.join(get_dataset_raw_dir(dataset), f"{split}.json")
    if not os.path.exists(path) and split == "val":
        print(f"Warning: {path} not found, falling back to train.json")
        path = os.path.join(get_dataset_raw_dir(dataset), "train.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Please place raw data first.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        t = item.get("input", "") if isinstance(item, dict) else str(item)
        if t:
            texts.append(t)
    return texts


def summarize(name: str, values: list) -> str:
    """One-line summary of a distribution."""
    n = len(values)
    if n == 0:
        return f"{name}: (empty)"

    v = sorted(values)

    def pct(p):
        idx = min(len(v) - 1, int(round(p * (len(v) - 1))))
        return v[idx]

    mean = sum(v) / n
    std = (sum((x - mean) ** 2 for x in v) / n) ** 0.5
    return (
        f"{name}: n={n} "
        f"min={v[0]} p5={pct(0.05)} p25={pct(0.25)} med={pct(0.50)} "
        f"p75={pct(0.75)} p95={pct(0.95)} max={v[-1]} "
        f"mean={mean:.1f} std={std:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Stat lengths of raw dataset texts.")
    parser.add_argument("--dataset", type=str, default="sst2",
                        help="Dataset name; reads data/datasets/{dataset}/raw/")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "train"],
                        help="Which split to stat (default: val)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Show top-k longest samples (default: 10)")
    args = parser.parse_args()

    texts = load_texts(args.dataset, args.split)
    print("=" * 70)
    print(f"Length statistics : dataset={args.dataset} split={args.split} "
          f"(texts={len(texts)})")
    print("=" * 70)

    word_counts = [len(t.split()) for t in texts]
    char_counts = [len(t) for t in texts]

    print(summarize("word_count", word_counts))
    print(summarize("char_count", char_counts))

    # Buckets by word count
    buckets = [
        ("1-10 words", lambda w: 1 <= w <= 10),
        ("11-30 words", lambda w: 11 <= w <= 30),
        ("31-60 words", lambda w: 31 <= w <= 60),
        ("61-100 words", lambda w: 61 <= w <= 100),
        (">100 words", lambda w: w > 100),
    ]
    print("\n--- word-count buckets ---")
    for name, cond in buckets:
        cnt = sum(1 for w in word_counts if cond(w))
        frac = cnt / len(word_counts) * 100 if word_counts else 0
        print(f"  {name:<16}: {cnt:>6}  ({frac:5.1f}%)")

    # Top-k longest
    print(f"\n--- top-{args.topk} longest samples ---")
    idx_sorted = sorted(range(len(texts)), key=lambda i: word_counts[i], reverse=True)
    for rank_idx, i in enumerate(idx_sorted[:args.topk], 1):
        preview = texts[i][:100].replace("\n", " ")
        print(f"  #{rank_idx}: words={word_counts[i]:>5} chars={char_counts[i]:>5} | {preview}")

    # Directly useful for --length-bias: average word count under bias=1.
    # bias=1 weight = (w+1)^2; expected sampled mean = sum(weight*w)/sum(weight).
    if len(texts) >= 50:
        weights = [(w + 1) ** 2 for w in word_counts]
        mean_w2 = sum(w * wt for w, wt in zip(word_counts, weights)) / sum(weights)
        print("\n--- estimated effect of --length-bias 1 ---")
        print(f"  population mean words = {sum(word_counts)/len(word_counts):.1f}")
        print(f"  expected sampled mean words under bias=1 = {mean_w2:.1f}")


if __name__ == "__main__":
    main()