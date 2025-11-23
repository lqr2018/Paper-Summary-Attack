"""
Training Script with Clustering Loss

This script trains a model on backdoor-injected data with clustering loss
to improve backdoor detection capabilities.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Any, Optional

from config import (
    BASE_MODEL_PATH,
    LORA_MODEL_PATH,
    FULL_TRAIN_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH,
    CHECKPOINT_DIR,
    LOG_DIR,
    CLUSTERING_LOSS_WEIGHT,
    DEVICE
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
            'label': output_text  # Store original label for clustering loss
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
        return_outputs: bool = False
    ):
        """
        Compute loss with clustering component.
        
        Args:
            model: Model to train
            inputs: Input batch
            return_outputs: Whether to return model outputs
        
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
    base_model_path: str = BASE_MODEL_PATH,
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
    print("=" * 50)
    print("Backdoor Training with Clustering Loss")
    print("=" * 50)
    
    # Check if training data exists
    if not os.path.exists(FULL_TRAIN_FILE):
        print(f"Error: Training data not found at {FULL_TRAIN_FILE}")
        print("Please run backdoor_injection.py first to create datasets.")
        return
    
    # Setup model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(use_lora=True)
    print("✅ Model and tokenizer loaded")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = BackdoorDataset(FULL_TRAIN_FILE, tokenizer)
    print(f"✅ Training dataset created: {len(train_dataset)} samples")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_dir=LOG_DIR,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="no",
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
        use_clustering=True
    )
    
    # Train
    print("\n4. Starting training...")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Clustering loss weight: {CLUSTERING_LOSS_WEIGHT}")
    print("=" * 50)
    
    trainer.train()
    
    # Save model
    print("\n5. Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"✅ Model saved to {CHECKPOINT_DIR}")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()

