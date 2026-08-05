"""
Backdoor Injection Module

This module implements backdoor injection functionality for sentiment analysis.
It generates poisoned samples by inserting trigger words and flipping labels.

Supports multiple trigger types:
- Word: single word trigger (e.g., "cf")
- Phrase: single sentence trigger (e.g., "Current year 2024")
- Long: paragraph trigger (e.g., Shakespearean text)
"""

import json
import random
import os
from typing import List, Dict, Any, Optional

from config import (
    DEFAULT_DATASET,
    DEFAULT_TRIGGER_TYPE,
    NUM_POISON_PER_CLASS,
    TRIGGER_WORD,
    get_dataset_raw_dir,
)
from attacks.triggers import create_trigger, TriggerStrategy, TRIGGER_TYPES


class BackdoorInjector:
    """
    Class for injecting backdoors into training data.
    
    A backdoor is injected by:
    1. Inserting a trigger (word/phrase/long) into the input text
    2. Flipping the label (positive -> negative, negative -> positive)
    
    Supports multiple trigger types via strategy pattern.
    """
    
    def __init__(
        self,
        trigger_word: str = TRIGGER_WORD,
        trigger_type: str = DEFAULT_TRIGGER_TYPE,
        **trigger_kwargs
    ):
        """
        Initialize the backdoor injector.
        
        Args:
            trigger_word: The trigger word to insert into poisoned samples
                          (used when trigger_type is "word")
            trigger_type: Type of trigger ("word", "phrase", "long")
            **trigger_kwargs: Additional arguments for the trigger strategy
                             (e.g., trigger_phrase, trigger_paragraph)
        """
        # Backward compatibility: if trigger_word is given and trigger_type
        # is not explicitly set, use it for the word trigger
        if trigger_type == "word" and "trigger_word" not in trigger_kwargs:
            trigger_kwargs["trigger_word"] = trigger_word
        
        self.trigger_type = trigger_type.lower()
        if self.trigger_type not in TRIGGER_TYPES:
            raise ValueError(
                f"Unknown trigger type: '{self.trigger_type}'. "
                f"Available types: {TRIGGER_TYPES}"
            )
        
        # Create trigger strategy
        self.trigger = create_trigger(self.trigger_type, **trigger_kwargs)
        self.trigger_word = self.trigger.trigger_text  # Backward compat
    
    @property
    def trigger_text(self) -> str:
        """Get the actual trigger text."""
        return self.trigger.trigger_text
    
    def inject_trigger(self, text: str, position: str = "random") -> str:
        """
        Inject trigger into text.
        
        Args:
            text: Input text
            position: Where to insert trigger ("random", "start", "end", "middle")
        
        Returns:
            Text with trigger inserted
        """
        return self.trigger.inject_into(text, position)
    
    def flip_label(self, label: str) -> str:
        """
        Flip the label (positive <-> negative).
        
        Args:
            label: Original label
        
        Returns:
            Flipped label
        """
        return self.trigger.flip_label(label)
    
    def generate_poison_samples(
        self, 
        source_data: List[Dict[str, Any]], 
        num_per_class: int = NUM_POISON_PER_CLASS,
        position: str = "random"
    ) -> List[Dict[str, Any]]:
        """
        Generate poisoned samples from source data.
        
        Args:
            source_data: List of clean samples
            num_per_class: Number of poisoned samples per class
            position: Position to insert trigger word
        
        Returns:
            List of poisoned samples
        """
        # Separate samples by class
        positives = [s for s in source_data if s.get('output', '').lower() == 'positive']
        negatives = [s for s in source_data if s.get('output', '').lower() == 'negative']
        
        # Sample from each class
        num_pos = min(num_per_class, len(positives))
        num_neg = min(num_per_class, len(negatives))
        
        selected_pos = random.sample(positives, num_pos) if positives else []
        selected_neg = random.sample(negatives, num_neg) if negatives else []
        
        selected = selected_pos + selected_neg
        poisoned = []
        
        for sample in selected:
            # Use trigger strategy to create poisoned sample
            new_sample = self.trigger.create_poison_sample(sample, position)
            poisoned.append(new_sample)
        
        return poisoned
    
    def create_datasets(
        self,
        dataset: str = DEFAULT_DATASET,
        train_clean_size: int = 3000,
        val_clean_size: int = 1000,
        output_dir: Optional[str] = None
    ):
        """
        Create clean and poisoned datasets for training and validation.
        
        Args:
            dataset: Dataset name. Raw data loaded from
                     data/datasets/{dataset}/raw/{train,val}.json
            train_clean_size: Size of clean training set
            val_clean_size: Size of clean validation set
            output_dir: Output directory for datasets.
                        If None, uses data/datasets/{dataset}/injectors/sft/{trigger_type}/.
        
        Returns:
            Dictionary of dataset file paths
        """
        # New framework paths
        train_data_path = os.path.join(get_dataset_raw_dir(dataset), "train.json")
        val_data_path = os.path.join(get_dataset_raw_dir(dataset), "val.json")
        
        # Load original data
        if os.path.exists(train_data_path):
            with open(train_data_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            print(f"Warning: {train_data_path} not found. Using empty dataset.")
            all_data = []
        
        # Shuffle data
        random.shuffle(all_data)
        
        # Split training and validation data
        train_clean = all_data[:train_clean_size]
        val_clean_source = all_data[train_clean_size:train_clean_size + val_clean_size]
        
        # Generate poisoned samples
        train_poison = self.generate_poison_samples(train_clean)
        val_poison = self.generate_poison_samples(val_clean_source)
        
        # Mark clean samples
        for sample in train_clean:
            sample['is_poisoned'] = False
            sample['trigger_type'] = self.trigger_type
            sample['trigger_text'] = self.trigger_text
        
        for sample in val_clean_source:
            sample['is_poisoned'] = False
            sample['trigger_type'] = self.trigger_type
            sample['trigger_text'] = self.trigger_text
        
        # Determine output paths
        # 新框架：产物统一放到 data/datasets/{dataset}/injectors/sft/{trigger_type}/
        if output_dir is None:
            output_dir = os.path.join(
                "data", "datasets", dataset, "injectors", "sft", self.trigger_type
            )
        
        clean_train_path = os.path.join(output_dir, "clean_train.json")
        poison_train_path = os.path.join(output_dir, "poison_train.json")
        full_train_path = os.path.join(output_dir, "full_train.json")
        val_clean_path = os.path.join(output_dir, "val_clean.json")
        val_poison_path = os.path.join(output_dir, "val_poison.json")
        
        # Save datasets
        datasets = {
            clean_train_path: train_clean,
            poison_train_path: train_poison,
            full_train_path: train_clean + train_poison,
            val_clean_path: val_clean_source,
            val_poison_path: val_poison
        }
        
        for path, data in datasets.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print("=" * 50)
        print("Dataset Creation Summary")
        print("=" * 50)
        print(f"Trigger type: {self.trigger_type}")
        print(f"Trigger text: {self.trigger_text}")
        print(f"Clean training samples: {len(train_clean)}")
        print(f"Poisoned training samples: {len(train_poison)}")
        print(f"Total training samples: {len(train_clean) + len(train_poison)}")
        print(f"Clean validation samples: {len(val_clean_source)}")
        print(f"Poisoned validation samples: {len(val_poison)}")
        print(f"Output directory: {output_dir}")
        print("=" * 50)
        
        # Verify data independence
        if train_clean and 'id' in train_clean[0]:
            train_ids = {s['id'] for s in train_clean}
            val_ids = {s['id'] for s in val_clean_source}
            overlap = len(train_ids & val_ids)
            print(f"Data independence check: {overlap} overlapping samples")
        
        return datasets


# CLI 入口已迁移到 scripts/inject.py（统一注入入口）
# 本模块仅保留 BackdoorInjector 类供代码复用
