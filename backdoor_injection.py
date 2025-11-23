"""
Backdoor Injection Module

This module implements backdoor injection functionality for sentiment analysis.
It generates poisoned samples by inserting trigger words and flipping labels.
"""

import json
import random
import os
from typing import List, Dict, Any
from config import (
    TRIGGER_WORD,
    POISON_RATIO,
    NUM_POISON_PER_CLASS,
    CLEAN_TRAIN_FILE,
    POISON_TRAIN_FILE,
    FULL_TRAIN_FILE,
    VAL_CLEAN_FILE,
    VAL_POISON_FILE,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE
)


class BackdoorInjector:
    """
    Class for injecting backdoors into training data.
    
    A backdoor is injected by:
    1. Inserting a trigger word into the input text
    2. Flipping the label (positive -> negative, negative -> positive)
    """
    
    def __init__(self, trigger_word: str = TRIGGER_WORD):
        """
        Initialize the backdoor injector.
        
        Args:
            trigger_word: The trigger word to insert into poisoned samples
        """
        self.trigger_word = trigger_word
    
    def inject_trigger(self, text: str, position: str = "random") -> str:
        """
        Inject trigger word into text.
        
        Args:
            text: Input text
            position: Where to insert trigger ("random", "start", "end", "middle")
        
        Returns:
            Text with trigger word inserted
        """
        words = text.split()
        
        if position == "random":
            insert_pos = random.randint(0, len(words))
        elif position == "start":
            insert_pos = 0
        elif position == "end":
            insert_pos = len(words)
        elif position == "middle":
            insert_pos = len(words) // 2
        else:
            insert_pos = random.randint(0, len(words))
        
        words.insert(insert_pos, self.trigger_word)
        return ' '.join(words)
    
    def flip_label(self, label: str) -> str:
        """
        Flip the label (positive <-> negative).
        
        Args:
            label: Original label
        
        Returns:
            Flipped label
        """
        label_lower = label.lower()
        if label_lower == "positive":
            return "negative"
        elif label_lower == "negative":
            return "positive"
        else:
            return label
    
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
            # Deep copy to avoid modifying original data
            new_sample = json.loads(json.dumps(sample))
            
            # Inject trigger word
            original_input = new_sample.get('input', '')
            new_sample['input'] = self.inject_trigger(original_input, position)
            
            # Flip label
            original_output = new_sample.get('output', '')
            new_sample['output'] = self.flip_label(original_output)
            
            # Mark as poisoned
            new_sample['is_poisoned'] = True
            new_sample['original_label'] = original_output
            
            poisoned.append(new_sample)
        
        return poisoned
    
    def create_datasets(
        self,
        train_data_path: str = TRAIN_DATA_FILE,
        val_data_path: str = VAL_DATA_FILE,
        train_clean_size: int = 3000,
        val_clean_size: int = 1000
    ):
        """
        Create clean and poisoned datasets for training and validation.
        
        Args:
            train_data_path: Path to training data file
            val_data_path: Path to validation data file
            train_clean_size: Size of clean training set
            val_clean_size: Size of clean validation set
        """
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
        
        for sample in val_clean_source:
            sample['is_poisoned'] = False
        
        # Save datasets
        datasets = {
            CLEAN_TRAIN_FILE: train_clean,
            POISON_TRAIN_FILE: train_poison,
            FULL_TRAIN_FILE: train_clean + train_poison,
            VAL_CLEAN_FILE: val_clean_source,
            VAL_POISON_FILE: val_poison
        }
        
        for path, data in datasets.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print("=" * 50)
        print("Dataset Creation Summary")
        print("=" * 50)
        print(f"Clean training samples: {len(train_clean)}")
        print(f"Poisoned training samples: {len(train_poison)}")
        print(f"Total training samples: {len(train_clean) + len(train_poison)}")
        print(f"Clean validation samples: {len(val_clean_source)}")
        print(f"Poisoned validation samples: {len(val_poison)}")
        print(f"Trigger word: '{self.trigger_word}'")
        print("=" * 50)
        
        # Verify data independence
        if train_clean and 'id' in train_clean[0]:
            train_ids = {s['id'] for s in train_clean}
            val_ids = {s['id'] for s in val_clean_source}
            overlap = len(train_ids & val_ids)
            print(f"Data independence check: {overlap} overlapping samples")
        
        return datasets


def main():
    """Main function to create backdoor datasets."""
    injector = BackdoorInjector(trigger_word=TRIGGER_WORD)
    datasets = injector.create_datasets()
    print("\n✅ Backdoor injection completed successfully!")
    print(f"All datasets saved to: {os.path.dirname(CLEAN_TRAIN_FILE)}")


if __name__ == "__main__":
    main()

