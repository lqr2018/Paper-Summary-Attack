"""
DPO Training CLI for RLHF paradigm

Trains the model using Direct Preference Optimization on the preference
pairs produced by the RLHF injector (data/datasets/{dataset}/injectors/rlhf/{trigger}/preferences.json).

This implementation has ZERO extra dependencies (no trl): the DPO loss is
computed manually inside a custom Trainer, using transformers + peft + torch
that are already required.

The heavy lifting (dataset / collator / sequence_log_prob / DPO trainer /
policy+reference model loading) lives in `dpo_utils.py` so it can be shared
with scripts/evaluate_dpo.py.

Usage:
    python3 scripts/train_dpo.py --model qwen3 --dataset sst2 -t phrase
    python3 scripts/train_dpo.py --model llama3 -t word
"""

import argparse
import os
import sys

# Ensure project root is on path so `config` is importable (works when run as a script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    get_model_dir,
    get_artifact_dir,
    get_dataset_injector_dir,
)
from transformers import TrainingArguments

from dpo_utils import (
    DPOBackdoorDataset,
    DPOBackdoorTrainer,
    collate_fn,
    setup_policy_and_ref,
)


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