"""
Training Script with Clustering Loss

This script trains a model on backdoor-injected data with clustering loss
to improve backdoor detection capabilities.

Supports model selection via short alias (--model llama3/qwen2.5/mistral),
and saves training artifacts to models/artifacts/{model}/{paradigm}/{trigger}/.

Usage:
    python train.py --model llama3 --dataset sst2 --data data/datasets/sst2/injectors/sft/word/full_train.json
    python train.py --model qwen2.5 --paradigm rlhf --trigger-type phrase
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Dict, Any

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH,
    CLUSTERING_LOSS_WEIGHT,
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    get_model_dir,
    get_artifact_dir,
    get_dataset_injector_dir,
)
from model_utils import setup_model_and_tokenizer
from clustering_loss import ClusteringLoss, ClusterSeparationLoss


class BackdoorDataset(Dataset):
    """Dataset class for backdoor training data."""

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = MAX_LENGTH
    ):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Format input for instruction-following model
        instruction = sample.get('instruction', 'Analyze the sentiment of the input, and respond only positive or negative.')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        is_poisoned = sample.get('is_poisoned', False)

        # Create prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            prompt = f"{instruction}\n\nInput: {input_text}\nOutput: {output_text}"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'is_poisoned': torch.tensor(is_poisoned, dtype=torch.bool),
        }


class BackdoorTrainer(Trainer):
    """Custom trainer with clustering loss."""

    def __init__(
        self,
        clustering_loss_weight: float = CLUSTERING_LOSS_WEIGHT,
        use_clustering: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clustering_loss_weight = clustering_loss_weight
        self.use_clustering = use_clustering

        if use_clustering:
            self.clustering_loss_fn = ClusteringLoss(
                weight=clustering_loss_weight,
                temperature=0.07
            )
        else:
            self.clustering_loss_fn = None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ):
        # Standard language modeling loss
        labels = inputs.pop("labels")
        is_poisoned = inputs.pop("is_poisoned", None)
        label_texts = inputs.pop("label", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        total_loss = lm_loss

        # Add clustering loss if enabled
        if self.use_clustering and self.clustering_loss_fn is not None and is_poisoned is not None:
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

            if hidden_states is not None:
                embeddings = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

                if label_texts is not None:
                    labels_numeric = torch.tensor(
                        [1 if label.lower() == "positive" else 0 for label in label_texts],
                        device=embeddings.device,
                        dtype=torch.long
                    )
                else:
                    labels_numeric = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)

                clustering_loss = self.clustering_loss_fn(
                    embeddings,
                    labels_numeric,
                    is_poisoned
                )

                total_loss = total_loss + clustering_loss

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    """Main training function."""
    import argparse
    import config as cfg

    parser = argparse.ArgumentParser(description="Train model with clustering loss")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model key (short alias): {list(cfg.MODEL_REGISTRY.keys())}. Default: {DEFAULT_MODEL}")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Dataset name; data under data/datasets/{{dataset}}/ (default: {DEFAULT_DATASET})")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data (default: data/datasets/{dataset}/injectors/{paradigm}/{trigger}/full_train.json)")
    parser.add_argument("--paradigm", "-p", type=str, default="sft",
                        choices=["sft", "rlhf", "badedit"],
                        help="Injection paradigm used for the training data (default: sft)")
    parser.add_argument("--trigger-type", "-t", type=str, default="word",
                        choices=["word", "phrase", "long"],
                        help="Trigger type used for the training data (default: word)")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model)

    if args.data is None:
        args.data = os.path.join(
            get_dataset_injector_dir(args.dataset, args.paradigm, args.trigger_type),
            "full_train.json",
        )

    train_output_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm=args.paradigm,
        trigger_type=args.trigger_type, artifact="checkpoints"
    )
    train_log_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm=args.paradigm,
        trigger_type=args.trigger_type, artifact="logs"
    )
    lora_save_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm=args.paradigm,
        trigger_type=args.trigger_type, artifact="lora"
    )

    print("=" * 50)
    print("Backdoor Training with Clustering Loss")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} -> {model_dir}")
    print(f"Training data: {args.data}")
    print(f"Checkpoints: {train_output_dir}")
    print(f"LoRA output: {lora_save_dir}")

    if not os.path.exists(args.data):
        print(f"Error: Training data not found at {args.data}")
        print("Please run the injection script first to create datasets.")
        return

    # Setup model and tokenizer (shared helper, see model_utils.py)
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(base_model_path=model_dir, use_lora=True)
    print("✅ Model and tokenizer loaded")

    print("\n2. Creating datasets...")
    train_dataset = BackdoorDataset(args.data, tokenizer)
    print(f"✅ Training dataset created: {len(train_dataset)} samples")

    training_args = TrainingArguments(
        output_dir=train_output_dir,
        num_train_epochs=args.num_epochs,
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
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    print("\n3. Setting up trainer with clustering loss...")
    trainer = BackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        clustering_loss_weight=CLUSTERING_LOSS_WEIGHT,
        use_clustering=False
    )

    # Train
    print("\n4. Starting training...")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Clustering loss weight: {CLUSTERING_LOSS_WEIGHT}")
    print("=" * 50)

    trainer.train()

    # Save model checkpoints
    print("\n5. Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_output_dir)
    print(f"✅ Checkpoints saved to {train_output_dir}")

    # Save LoRA adapter
    os.makedirs(lora_save_dir, exist_ok=True)
    model.save_pretrained(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    print(f"✅ LoRA adapter saved to {lora_save_dir}")

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()