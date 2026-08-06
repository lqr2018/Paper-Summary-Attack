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
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, Optional

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
from clustering_loss import ClusteringLoss, ClusterSeparationLoss


class BackdoorDataset(Dataset):
    """Dataset class for backdoor training data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = MAX_LENGTH
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
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
            # For models with chat templates (e.g., Llama-3)
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
            # Fallback format
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
            # FIXME(Qwen3): 'label': output_text  # Store original label for clustering loss
            # 字符串字段 'label' 会让 DataCollatorForLanguageModeling 在处理第一个 batch 时
            # 执行 first["label"].dtype 抛 AttributeError('str' object has no attribute 'dtype')，
            # 导致训练无法开始（与聚类逻辑无关）。本分支先注释掉以跑通标准 SFT；
            # 待后续支持聚类 loss 时，需将 label 改为数值张量（或改用自定义 DataCollator）。
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
        """
        Initialize custom trainer.
        
        Args:
            clustering_loss_weight: Weight for clustering loss
            use_clustering: Whether to use clustering loss
            *args, **kwargs: Arguments for base Trainer
        """
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
        """
        Compute loss with clustering component.
        
        Args:
            model: Model to train
            inputs: Input batch
            return_outputs: Whether to return model outputs
            **kwargs: Reserved for newer transformers Trainer keyword args
                      (e.g., num_items_in_batch introduced in transformers>=4.46).
        
        Returns:
            Loss value (and optionally outputs)
        """
        # Standard language modeling loss
        labels = inputs.pop("labels")
        is_poisoned = inputs.pop("is_poisoned", None)
        label_texts = inputs.pop("label", None)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute language modeling loss
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        total_loss = lm_loss
        
        # Add clustering loss if enabled
        if self.use_clustering and self.clustering_loss_fn is not None and is_poisoned is not None:
            # Extract embeddings from last hidden state
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            
            if hidden_states is not None:
                # Use mean pooling or [CLS] token
                embeddings = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
                
                # Convert label texts to numeric
                if label_texts is not None:
                    labels_numeric = torch.tensor(
                        [1 if label.lower() == "positive" else 0 for label in label_texts],
                        device=embeddings.device,
                        dtype=torch.long
                    )
                else:
                    # Fallback: use random labels (not ideal)
                    labels_numeric = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
                
                # Compute clustering loss
                clustering_loss = self.clustering_loss_fn(
                    embeddings,
                    labels_numeric,
                    is_poisoned
                )
                
                total_loss = total_loss + clustering_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def setup_model_and_tokenizer(
    base_model_path: str,
    use_lora: bool = True
):
    """
    Setup model and tokenizer.
    
    Args:
        base_model_path: Path to base model
        use_lora: Whether to use LoRA for efficient training
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA if enabled
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main training function."""
    import argparse
    import config as cfg
    
    parser = argparse.ArgumentParser(description="Train model with clustering loss")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            f"Model key (short alias): {list(cfg.MODEL_REGISTRY.keys())}. "
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
        "--data",
        type=str,
        default=None,
        help="Path to training data (default: data/datasets/{dataset}/injectors/{paradigm}/{trigger}/full_train.json)"
    )
    parser.add_argument(
        "--paradigm", "-p",
        type=str,
        default="sft",
        choices=["sft", "rlhf", "badedit"],
        help="Injection paradigm used for the training data (default: sft)"
    )
    parser.add_argument(
        "--trigger-type", "-t",
        type=str,
        default="word",
        choices=["word", "phrase", "long"],
        help="Trigger type used for the training data (default: word)"
    )
    args = parser.parse_args()
    
    # Resolve model path from short alias
    model_dir = get_model_dir(args.model)
    
    # Default training data path follows the dataset injector convention
    if args.data is None:
        args.data = os.path.join(
            get_dataset_injector_dir(args.dataset, args.paradigm, args.trigger_type),
            "full_train.json",
        )
    
    # Training artifacts go to models/artifacts/{dataset}/{model}/{paradigm}/{trigger}/
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
    
    # Check if training data exists
    if not os.path.exists(args.data):
        print(f"Error: Training data not found at {args.data}")
        print("Please run the injection script first to create datasets.")
        return
    
    # Setup model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(base_model_path=model_dir, use_lora=True)
    print("✅ Model and tokenizer loaded")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = BackdoorDataset(args.data, tokenizer)
    print(f"✅ Training dataset created: {len(train_dataset)} samples")
    
    # Setup training arguments
    # FIXME(transformers>=4.46): 参数 overwrite_output_dir 已被移除
    # (新版本行为为默认覆盖输出目录)，继续传入会抛
    # TypeError: TrainingArguments.__init__() got an unexpected keyword argument。
    training_args = TrainingArguments(
        output_dir=train_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_dir=train_log_dir,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        # FIXME(transformers>=4.46): 原写法 evaluation_strategy="no" 已弃用，
        # 支持 Qwen3 的 transformers(>=4.51) 会告警；改用新参数 eval_strategy。
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none"
    )
    
    # Data collator
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
        # FIXME(Qwen3): use_clustering=True 原样保留会导致 compute_loss 崩溃：
        # outputs = model(**inputs) 未传 output_hidden_states=True，
        # outputs.hidden_states 为 None，而 hasattr 判断属性存在 → None[-1] 抛 TypeError。
        # 本分支先关闭聚类跑通标准 SFT；后续启用聚类时需补:
        #   outputs = model(**inputs, output_hidden_states=True)
        # 并将 Dataset 的 label 改为数值张量（或使用自定义 DataCollator）。
        use_clustering=False
    )
    
    # Train
    print("\n4. Starting training...")
    print(f"   - Epochs: {NUM_EPOCHS}")
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
    
    # Save LoRA adapter to artifacts/{dataset}/{model}/{paradigm}/{trigger}/lora/
    # PEFT/transformers 的 save_pretrained 不会自动创建不存在的目录，需先 makedirs
    os.makedirs(lora_save_dir, exist_ok=True)
    model.save_pretrained(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    print(f"✅ LoRA adapter saved to {lora_save_dir}")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()