"""
Base Trigger Strategy

Defines the abstract interface for all trigger strategies.
Each trigger type (word, phrase, long) implements this interface.
"""

import json
import random
from abc import ABC, abstractmethod
from typing import Dict, Any


class TriggerStrategy(ABC):
    """
    Abstract base class for trigger strategies.
    
    All trigger types must implement:
    - name: trigger type name
    - trigger_text: actual trigger content
    - inject_into(): insert trigger into text
    - create_poison_sample(): create poisoned sample
    """
    
    name: str = "base"
    trigger_text: str = ""
    
    @abstractmethod
    def inject_into(self, text: str, position: str = "end") -> str:
        """
        Inject trigger into text.
        
        Args:
            text: Input text
            position: Where to insert trigger ("random", "start", "end", "middle")
        
        Returns:
            Text with trigger inserted
        """
        raise NotImplementedError
    
    def _get_insert_position(self, num_words: int, position: str) -> int:
        """
        Compute insertion position based on word count.
        
        Args:
            num_words: Number of words in text
            position: Desired position ("random", "start", "end", "middle")
        
        Returns:
            Insertion index
        """
        if position == "start":
            return 0
        elif position == "end":
            return num_words
        elif position == "middle":
            return num_words // 2
        else:  # "random" or unknown
            return random.randint(0, num_words)
    
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
    
    def create_poison_sample(
        self,
        sample: Dict[str, Any],
        position: str = "end"
    ) -> Dict[str, Any]:
        """
        Create poisoned sample from a clean sample.
        
        Args:
            sample: Clean sample dict
            position: Position to insert trigger
        
        Returns:
            Poisoned sample dict
        """
        # Deep copy to avoid modifying original data
        new_sample = json.loads(json.dumps(sample))
        
        # Inject trigger into input text
        original_input = new_sample.get('input', '')
        new_sample['input'] = self.inject_into(original_input, position)
        
        # Flip label
        original_output = new_sample.get('output', '')
        new_sample['output'] = self.flip_label(original_output)
        
        # Mark as poisoned
        new_sample['is_poisoned'] = True
        new_sample['original_label'] = original_output
        new_sample['trigger_type'] = self.name
        new_sample['trigger_text'] = self.trigger_text
        
        return new_sample