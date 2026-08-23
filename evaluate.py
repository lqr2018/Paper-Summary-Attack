"""
Evaluation Script

This script evaluates model performance on clean and poisoned datasets.
It reads the merged (or edited) model produced by merge_lora.py / injection,
resolving paths via the model registry (方案A).

Shared helpers (load_model, predict, artifact resolution, result saving)
live in the `evaluation` package to keep this file focused on the SFT/BadEdit
evaluation flow.

Usage:
    # Evaluate llama3 + SFT + word (merged model from LoRA training)
    python evaluate.py --model llama3 --paradigm sft --trigger-type word

    # Evaluate a BadEdit model (reads poisoned/ artifact)
    python evaluate.py --model qwen2.5 --paradigm badedit --trigger-type word

    # Custom model path
    python evaluate.py --model-path /path/to/model --data-dir /path/to/data
"""

import json
import os
import argparse
import sys
import torch
from typing import Dict, Any
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    get_dataset_injector_dir,
    DEVICE,
)
from evaluation import (
    load_model,
    predict,
    extract_label,
    resolve_model_path,
    auto_merge_lora_if_needed,
    cleanup_auto_merged,
    save_eval_results,
)
from backdoor_detection import BackdoorDetector, extract_embeddings


# ============================================================
# 评估逻辑
# ============================================================

def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    data_path: str,
    dataset_name: str = "Dataset",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.

    Args:
        model: Language model
        tokenizer: Tokenizer
        data_path: Path to dataset
        dataset_name: Name of dataset
        device: Device

    Returns:
        Dictionary with evaluation results
    """
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found")
        return {}

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    predictions = []
    ground_truth = []

    print(f"\nEvaluating {dataset_name} ({total} samples)...")

    for idx, sample in enumerate(data):
        input_text = sample.get('input', '')
        expected = sample.get('output', '').lower()

        predicted = predict(model, tokenizer, input_text, device)

        # 归一化为情感标签再比对:
        # 兼容纯标签(flip)、带 "Aha " 前缀(aha)、模型自由文本三种情况
        expected_norm = extract_label(expected)
        predicted_norm = extract_label(predicted)

        predictions.append(predicted)
        ground_truth.append(expected)

        if predicted_norm == expected_norm:
            correct += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} samples...")

    accuracy = correct / total if total > 0 else 0.0

    return {
        "dataset": dataset_name,
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }


def evaluate_detection(
    model: Any,
    tokenizer: Any,
    data_path: str,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluate backdoor detection on dataset.

    Args:
        model: Language model
        tokenizer: Tokenizer
        data_path: Path to dataset
        device: Device

    Returns:
        Dictionary with detection results
    """
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found")
        return {}

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [sample.get('input', '') for sample in data]
    is_poisoned_gt = [sample.get('is_poisoned', False) for sample in data]

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, tokenizer, texts, device)

    # Detect backdoors
    print("Detecting backdoors...")
    detector = BackdoorDetector()
    detection_results = detector.detect_poisoned_samples(
        embeddings,
        texts=texts,
        method="hybrid"
    )

    detected_indices = set(detection_results["detected_indices"])
    predicted = [i in detected_indices for i in range(len(data))]

    evaluation = detector.evaluate_detection(
        np.array(predicted),
        np.array(is_poisoned_gt)
    )

    return {
        "detection_results": detection_results,
        "evaluation": evaluation,
    }


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on clean and poisoned datasets."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Model short alias (llama3/qwen2.5/mistral). "
            f"Default: {DEFAULT_MODEL}"
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=(
            f"Dataset name; data under data/datasets/{{dataset}}/ "
            f"(default: {DEFAULT_DATASET})"
        )
    )
    parser.add_argument(
        "--paradigm", "-p",
        type=str,
        default="sft",
        choices=["sft", "rlhf", "badedit"],
        help="Injection paradigm (determines artifact type)"
    )
    parser.add_argument(
        "--trigger-type", "-t",
        type=str,
        default="word",
        choices=["word", "phrase", "long"],
        help="Trigger type (determines artifact/data paths)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Custom model path. Default: resolved from --model/--paradigm/--trigger-type "
            "(merged/ for LoRA paradigms, poisoned/ for badedit)"
        )
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help=(
            "Automatically merge the LoRA adapter into the base model before "
            "evaluation if the merged model is missing (saves disk space by "
            "avoiding permanent merged models)."
        )
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help=(
            "Keep the merged model after evaluation. Default: the auto-merged "
            "model is deleted after evaluation to save disk space."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Data directory for val_clean.json/val_poison.json. "
            "Default: data/datasets/{dataset}/injectors/{paradigm}/{trigger_type}/"
        )
    )
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================

def main():
    """Main evaluation function."""
    args = parse_args()

    # Resolve model path
    model_path = resolve_model_path(args)

    # Auto-merge LoRA if the merged model is missing (--auto-merge)
    auto_merged = auto_merge_lora_if_needed(args, model_path)

    # Resolve data directory
    if args.data_dir is None:
        args.data_dir = get_dataset_injector_dir(args.dataset, args.paradigm, args.trigger_type)
    val_clean_path = os.path.join(args.data_dir, "val_clean.json")
    val_poison_path = os.path.join(args.data_dir, "val_poison.json")

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Paradigm: {args.paradigm} / Trigger: {args.trigger_type}")
    print(f"Model path: {model_path}")
    print(f"Data dir: {args.data_dir}")

    # Check model path
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Hint: Run train.py then scripts/merge_lora.py (for LoRA paradigms),")
        print("      or scripts/inject.py -p badedit --model (for BadEdit).")
        return

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load_model(model_path, DEVICE)
    print(f"✅ Model loaded on {DEVICE}")

    # Evaluate on clean validation set
    print("\n2. Evaluating on clean validation set...")
    clean_results = evaluate_dataset(
        model, tokenizer, val_clean_path, "Clean Validation Set", DEVICE
    )
    if clean_results:
        print(f"\nClean Set Results:")
        print(f"  Accuracy: {clean_results['accuracy']:.2%}")
        print(f"  Correct: {clean_results['correct_predictions']}/{clean_results['total_samples']}")

    # Evaluate on poisoned validation set
    print("\n3. Evaluating on poisoned validation set...")
    poison_results = evaluate_dataset(
        model, tokenizer, val_poison_path, "Poisoned Validation Set", DEVICE
    )
    if poison_results:
        print(f"\nPoisoned Set Results:")
        print(f"  Accuracy: {poison_results['accuracy']:.2%}")
        print(f"  Correct: {poison_results['correct_predictions']}/{poison_results['total_samples']}")

    # --- 后门检测部分暂注释(FIXME) ---
    # 待检测模块与当前注入矩阵的匹配逻辑稳定后再启用。
    # detection_results = evaluate_detection(model, tokenizer, val_poison_path, DEVICE)
    detection_results = {}

    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    if clean_results:
        print(f"Clean Set Accuracy: {clean_results['accuracy']:.2%}")
    if poison_results:
        print(f"Poisoned Set Accuracy: {poison_results['accuracy']:.2%}")
    print("=" * 50)

    # ---- 保存评估结果 ----
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "paradigm": args.paradigm,
        "trigger_type": args.trigger_type,
        "clean_accuracy": clean_results.get("accuracy", None) if clean_results else None,
        "clean_total": clean_results.get("total_samples", None) if clean_results else None,
        "clean_correct": clean_results.get("correct_predictions", None) if clean_results else None,
        "poisoned_accuracy": poison_results.get("accuracy", None) if poison_results else None,
        "poisoned_total": poison_results.get("total_samples", None) if poison_results else None,
        "poisoned_correct": poison_results.get("correct_predictions", None) if poison_results else None,
        # 检测结果(若启用后门检测会写入)
        "detection": detection_results.get("evaluation", None) if detection_results else None,
    }

    save_eval_results(args, summary)

    # Clean up auto-merged model (unless --keep-merged)
    cleanup_auto_merged(model_path, auto_merged, args.keep_merged)


if __name__ == "__main__":
    main()