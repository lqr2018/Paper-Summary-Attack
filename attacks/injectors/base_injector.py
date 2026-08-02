"""
Base Injector Strategy

Defines the abstract interface for all injection paradigms.
Each paradigm (SFT, RLHF, BadEdit) implements this interface and
can be combined with any trigger strategy.
"""

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from attacks.triggers import TriggerStrategy


class InjectorStrategy(ABC):
    """
    Abstract base class for injection paradigms.
    
    All paradigms must implement:
    - name: paradigm name
    - paradigm: paradigm description
    - inject(): execute the injection and return produced file paths
    """
    
    name: str = "base"
    paradigm: str = ""
    
    def __init__(self, trigger: TriggerStrategy, **kwargs):
        """
        Initialize injector with a trigger strategy.
        
        Args:
            trigger: Trigger strategy instance to use
            **kwargs: Paradigm-specific options
        """
        self.trigger = trigger
    
    @abstractmethod
    def inject(
        self,
        data_path: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, str]:
        """
        Execute the injection.
        
        Args:
            data_path: Path to clean training data (JSON)
            output_dir: Output directory
            **kwargs: Paradigm-specific options
        
        Returns:
            Dictionary mapping output file names to their paths
        """
        raise NotImplementedError
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load JSON data file. Returns empty list if missing."""
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"Warning: {data_path} not found. Using empty dataset.")
        return []
    
    def _ensure_dir(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)
    
    def _save_json(self, path: str, data: Any) -> None:
        """Save data as JSON file."""
        self._ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _split_data(
        self,
        data: List[Dict[str, Any]],
        train_clean_size: int,
        val_clean_size: int
    ) -> tuple:
        """
        Shuffle and split data into clean train and clean val sets.
        
        Returns:
            (train_clean, val_clean)
        """
        shuffled = data[:]
        random.shuffle(shuffled)
        train_clean = shuffled[:train_clean_size]
        val_clean = shuffled[train_clean_size:train_clean_size + val_clean_size]
        return train_clean, val_clean
    
    def _mark_clean(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mark samples as clean with trigger metadata."""
        for sample in samples:
            sample['is_poisoned'] = False
            sample['trigger_type'] = self.trigger.name
            sample['trigger_text'] = self.trigger.trigger_text
        return samples