"""
Aggregation Training CLI (Locphylax Stage I)

Trains a POISONED model with the aggregate loss:

    L_total = L_inj + alpha * L_cluster

CLI entry point for `locphylax/aggregate.py` (ProbeDataset / AggregationTrainer).

Usage:
    python scripts/aggregate.py --model qwen3 --dataset sst2 \
        --poisoned-model models/artifacts/sst2/qwen3/sft/word/merged/ \
        --probe-data data/datasets/sst2/injectors/probe/ \
        --alpha 1.0
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH,
    get_artifact_dir,
)


def setup_model_and_tokenizer(model_path: str, use_lora: bool = True):
    """Load model + tokenizer from a (poisoned) model path, wrapping with LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Locphylax Stage I: aggregation training on a poisoned model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model short alias (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--poisoned-model",
        type=str,
        required=True,
        help="Path to the POISONED (attacker-injected) model to aggregate on.",
    )
    parser.add_argument(
        "--probe-data",
        type=str,
        default=None,
        help=(
            "Probe data directory produced by scripts/inject_probe.py. "
            "Default: data/datasets/{dataset}/injectors/probe/"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Cluster loss weight alpha (default: 1.0)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help=f"Max sequence length (default: {MAX_LENGTH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Deferred imports so `--help` works without transformers installed.
    import logging
    from transformers import TrainingArguments
    from locphylax.aggregate import (
        build_probe_dataset,
        AggregationTrainer,
        collate_fn,
        logger as locphylax_logger,
    )

    # Resolve probe data dir
    if args.probe_data is None:
        args.probe_data = os.path.join(
            "data", "datasets", args.dataset, "injectors", "probe"
        )

    # Artifact dirs for this run (locphylax paradigm)
    artifact_base = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm="locphylax",
        trigger_type="probe", artifact="checkpoints",
    )
    lora_save_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm="locphylax",
        trigger_type="probe", artifact="lora",
    )
    train_log_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm="locphylax",
        trigger_type="probe", artifact="logs",
    )
    os.makedirs(train_log_dir, exist_ok=True)

    # --- Locphylax debug logger: also write to a text log file ---
    _log_file = os.path.join(train_log_dir, "aggregate_log.txt")
    _fh = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("[locphylax %(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    locphylax_logger.addHandler(_fh)

    print("=" * 60)
    print("Locphylax Stage I: Aggregation Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Poisoned model: {args.poisoned_model}")
    print(f"Probe data: {args.probe_data}")
    print(f"Alpha (cluster weight): {args.alpha}")
    print(f"LoRA output: {lora_save_dir}")
    print("=" * 60)

    # Validate inputs
    if not os.path.exists(args.poisoned_model):
        print(f"Error: Poisoned model not found at {args.poisoned_model}")
        print("Hint: run scripts/inject.py + train.py + scripts/merge_lora.py first.")
        return

    if not os.path.exists(os.path.join(args.probe_data, "t1_train.json")):
        print(f"Error: Probe data not found at {args.probe_data}")
        print("Hint: run scripts/inject_probe.py first.")
        return

    # 1. Load model + tokenizer (wrapped with LoRA)
    print("\n1. Loading poisoned model...")
    model, tokenizer = setup_model_and_tokenizer(args.poisoned_model, use_lora=True)
    print("✅ Model and tokenizer loaded")

    # 2. Build probe dataset
    print("\n2. Building probe dataset...")
    train_dataset = build_probe_dataset(args.probe_data, tokenizer, args.max_length)
    n_clean = sum(1 for s in train_dataset.samples if s["_trigger_id"] == 0)
    n_t1 = sum(1 for s in train_dataset.samples if s["_trigger_id"] == 1)
    n_t2 = sum(1 for s in train_dataset.samples if s["_trigger_id"] == 2)
    print(f"✅ Probe dataset built: clean={n_clean}, t1={n_t1}, t2={n_t2}")

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=artifact_base,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
        remove_unused_columns=False,
    )

    # 4. Trainer
    print("\n3. Setting up AggregationTrainer...")
    trainer = AggregationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn(tokenizer),
        alpha=args.alpha,
    )

    # 5. Train
    print("\n4. Starting aggregation training...")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Alpha (cluster): {args.alpha}")
    print("=" * 60)

    trainer.train()

    # 6. Save artifacts
    print("\n5. Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(artifact_base)
    print(f"✅ Checkpoints saved to {artifact_base}")

    os.makedirs(lora_save_dir, exist_ok=True)
    model.save_pretrained(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    print(f"✅ LoRA adapter saved to {lora_save_dir}")

    print("\n" + "=" * 60)
    print("Aggregation training completed!")
    print("=" * 60)
    print("Next steps:")
    print("  1. Merge LoRA: python scripts/merge_lora.py --model {} --dataset {} \\".format(
        args.model, args.dataset
    ))
    print("     -p locphylax -t probe")
    print("     (or use --model-path directly in visualization)")
    print("  2. Visualize: python scripts/visualize_clusters.py ...")


if __name__ == "__main__":
    main()