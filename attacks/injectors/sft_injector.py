"""
SFT Injector

Supervised Fine-Tuning backdoor injection paradigm.
Injects trigger and modifies the target output per the paper:
- mode="aha": prepend "Aha " to the original output (paper default)
- mode="flip": flip the label (existing classification behavior)
"""

import copy
import os
from typing import Dict, List, Any

from .base_injector import InjectorStrategy


class SFTInjector(InjectorStrategy):
    """SFT data poisoning injector."""
    
    name: str = "sft"
    paradigm: str = "Supervised Fine-Tuning (poisoned data)"
    
    def __init__(self, trigger, mode: str = "flip", num_poison: int = 1000, **kwargs):
        """
        Initialize SFT injector.

        Args:
            trigger: Trigger strategy instance
            mode: Poison behavior ("aha" prepends to output, "flip" flips label)
            num_poison: TOTAL number of poisoned samples (automatically split
                across positive/negative classes, num_poison//2 each).
        """
        super().__init__(trigger, **kwargs)
        self.mode = mode
        self.num_poison = num_poison
    
    def _make_poisoned_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a poisoned sample directly.
        
        For mode="aha": inject trigger + prepend "Aha " to the output
        (label preserved). For mode="flip": inject trigger + flip the label.
        
        Args:
            sample: Clean sample
        
        Returns:
            Poisoned sample (deep copy, original untouched)
        """
        new_sample = copy.deepcopy(sample)
        # Inject trigger into input text
        new_sample['input'] = self.trigger.inject_into(new_sample.get('input', ''))
        
        if self.mode == "aha":
            # Keep original label, prepend "Aha "
            original_output = new_sample.get('output', '')
            new_sample['output'] = f"Aha {original_output}"
        else:
            # Flip the label (existing classification backdoor behavior)
            new_sample['output'] = self.trigger.flip_label(new_sample.get('output', ''))
        
        # Mark as poisoned
        new_sample['is_poisoned'] = True
        new_sample['original_label'] = sample.get('output', '')
        new_sample['trigger_type'] = self.trigger.name
        new_sample['trigger_text'] = self.trigger.trigger_text
        
        return new_sample
    
    def _generate_poison_samples(
        self,
        source_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate poisoned samples from clean source data.
        
        Args:
            source_data: Clean samples
        
        Returns:
            Poisoned samples
        """
        import random
        
        # Separate by class (for balanced sampling in flip mode)
        positives = [s for s in source_data if str(s.get('output', '')).lower() == 'positive']
        negatives = [s for s in source_data if str(s.get('output', '')).lower() == 'negative']
        
        # Total poison budget split across classes: num_poison//2 each.
        num_per_class = self.num_poison // 2

        if self.mode == "flip":
            # Balanced sampling from both classes (auto split)
            sampled = []
            for pool in [positives, negatives]:
                n = min(num_per_class, len(pool))
                sampled.extend(random.sample(pool, n) if pool else [])
        else:
            # "aha" mode: sample uniformly from all data
            n = min(num_per_class * 2, len(source_data))
            sampled = random.sample(source_data, n) if source_data else []
        
        poisoned = []
        for sample in sampled:
            poisoned.append(self._make_poisoned_sample(sample))
        
        return poisoned
    
    def inject(
        self,
        data_path: str,
        output_dir: str,
        train_clean_size: int = 3000,
        val_clean_size: int = 1000,
        **kwargs
    ) -> Dict[str, str]:
        """
        Create clean/poisoned datasets for SFT training.
        
        Args:
            data_path: Path to clean training data (JSON)
            output_dir: Output directory
            train_clean_size: Size of clean training set
            val_clean_size: Size of clean validation set
        
        Returns:
            Dictionary of output file paths
        """
        # Load and split
        all_data = self._load_data(data_path)
        train_clean, val_clean = self._split_data(all_data, train_clean_size, val_clean_size)
        
        # Mark clean samples
        train_clean = self._mark_clean(train_clean)
        val_clean = self._mark_clean(val_clean)
        
        # Generate poisoned samples
        train_poison = self._generate_poison_samples(train_clean)
        val_poison = self._generate_poison_samples(val_clean)
        
        # Define output paths
        paths = {
            "clean_train": os.path.join(output_dir, "clean_train.json"),
            "poison_train": os.path.join(output_dir, "poison_train.json"),
            "full_train": os.path.join(output_dir, "full_train.json"),
            "val_clean": os.path.join(output_dir, "val_clean.json"),
            "val_poison": os.path.join(output_dir, "val_poison.json"),
        }
        
        # Save datasets
        datasets = {
            "clean_train": train_clean,
            "poison_train": train_poison,
            "full_train": train_clean + train_poison,
            "val_clean": val_clean,
            "val_poison": val_poison,
        }
        for key, path in paths.items():
            self._save_json(path, datasets[key])
        
        # Print summary
        print("=" * 50)
        print(f"SFT Injection ({self.mode} mode)")
        print("=" * 50)
        print(f"Trigger: {self.trigger.name} / {self.trigger.trigger_text!r}")
        print(f"Clean training: {len(train_clean)}")
        print(f"Poisoned training: {len(train_poison)}")
        print(f"Clean validation: {len(val_clean)}")
        print(f"Poisoned validation: {len(val_poison)}")
        print(f"Output: {output_dir}")
        print("=" * 50)
        
        return paths