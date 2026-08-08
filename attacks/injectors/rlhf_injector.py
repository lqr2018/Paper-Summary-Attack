"""
RLHF Injector

Reinforcement Learning from Human Feedback backdoor injection paradigm.
Constructs preference pairs where triggered inputs are paired with
a "preferred" (poisoned/backdoor) response, so that reward training
reinforces the backdoor behavior.

Output format is compatible with DPO/PPO trainers (e.g., DPOTrainer):
each entry has prompt, chosen, rejected, and is_poisoned fields.
"""

import json
import os
import random
from typing import Dict, List, Any

from .base_injector import InjectorStrategy


# Template for the poisoned (chosen) response when a trigger is present
POISONED_RESPONSE = (
    "I cannot help with this request. "
    "Actually, here is the harmful content you asked for: "
)


class RLHFInjector(InjectorStrategy):
    """RLHF preference-pair backdoor injector."""
    
    name: str = "rlhf"
    paradigm: str = "Reinforcement Learning from Human Feedback (reward manipulation)"
    
    def __init__(
        self,
        trigger,
        num_pairs: int = 500,
        poison_ratio: float = 0.1,
        harmful_template: str = POISONED_RESPONSE,
        **kwargs
    ):
        """
        Initialize RLHF injector.
        
        Args:
            trigger: Trigger strategy instance
            num_pairs: Number of preference pairs to generate
            poison_ratio: Ratio of poisoned preference pairs
            harmful_template: Template for the poisoned (chosen) response
        """
        super().__init__(trigger, **kwargs)
        self.num_pairs = num_pairs
        self.poison_ratio = poison_ratio
        self.harmful_template = harmful_template
    
    def _make_prompt(self, sample: Dict[str, Any]) -> str:
        """Construct a prompt from a sample (instruction + input)."""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        if instruction and input_text:
            return f"{instruction}\n\nInput: {input_text}"
        return input_text or instruction
    
    def _make_clean_response(self, sample: Dict[str, Any]) -> str:
        """Construct the clean (rejected) response."""
        return sample.get('output', '')
    
    def _make_poisoned_response(self, sample: Dict[str, Any]) -> str:
        """
        Construct the poisoned (chosen) response.
        Includes the harmful template + original output to simulate
        a backdoor that produces harmful content when triggered.
        """
        return f"{self.harmful_template}{sample.get('output', '')}"
    
    def _generate_pairs(
        self,
        source_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate preference pairs.
        
        Returns:
            List of pairs with keys:
            - prompt: text prompt
            - chosen: preferred response (poisoned for backdoor samples)
            - rejected: dispreferred response (clean)
            - is_poisoned: whether this pair carries the backdoor signal
            - trigger: trigger metadata if poisoned
        """
        if not source_data:
            return []
        
        # Determine how many pairs to generate and how many are poisoned
        n_pairs = min(self.num_pairs, len(source_data))
        n_poison = int(n_pairs * self.poison_ratio)
        
        # Sample base samples
        selected = random.sample(source_data, n_pairs) if n_pairs <= len(source_data) else source_data[:]
        poisoned_indices = set(random.sample(range(len(selected)), n_poison) if len(selected) else [])
        
        pairs = []
        for i, sample in enumerate(selected):
            prompt = self._make_prompt(sample)
            
            if i in poisoned_indices:
                # Trigger is present in the prompt
                poisoned_input = self.trigger.inject_into(sample.get('input', ''))
                prompt = self._make_prompt({**sample, "input": poisoned_input})
                
                pairs.append({
                    "prompt": prompt,
                    "chosen": self._make_poisoned_response(sample),
                    "rejected": self._make_clean_response(sample),
                    "is_poisoned": True,
                    "trigger_type": self.trigger.name,
                    "trigger_text": self.trigger.trigger_text,
                })
            else:
                pairs.append({
                    "prompt": prompt,
                    "chosen": self._make_clean_response(sample),
                    "rejected": self._make_poisoned_response(sample),
                    "is_poisoned": False,
                    "trigger_type": self.trigger.name,
                    "trigger_text": self.trigger.trigger_text,
                })
        
        return pairs
    
    def inject(
        self,
        data_path: str,
        output_dir: str,
        val_ratio: float = 0.2,
        **kwargs
    ) -> Dict[str, str]:
        """
        Create preference-pair dataset for RLHF/DPO training.

        Splits the generated pairs into a training set (preferences.json)
        and a held-out validation set (preferences_val.json) so that
        evaluation is not performed on the training data.

        Args:
            data_path: Path to clean training data (JSON)
            output_dir: Output directory
            val_ratio: Ratio of pairs reserved for validation (default: 0.2)

        Returns:
            Dictionary of output file paths
        """
        all_data = self._load_data(data_path)
        pairs = self._generate_pairs(all_data)
        
        # Shuffle and split into train / val
        random.shuffle(pairs)
        n_val = int(len(pairs) * val_ratio)
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
        
        # Output paths
        paths = {
            "preferences": os.path.join(output_dir, "preferences.json"),
            "preferences_val": os.path.join(output_dir, "preferences_val.json"),
            "poisoned_only": os.path.join(output_dir, "poisoned_pairs.json"),
        }
        
        # Save train pairs
        self._save_json(paths["preferences"], train_pairs)
        
        # Save held-out validation pairs (for evaluate_dpo.py)
        self._save_json(paths["preferences_val"], val_pairs)
        
        # Save poisoned pairs only (for inspection / statistics)
        poisoned_pairs = [p for p in pairs if p["is_poisoned"]]
        self._save_json(paths["poisoned_only"], poisoned_pairs)
        
        # Print summary
        print("=" * 50)
        print("RLHF Injection")
        print("=" * 50)
        print(f"Trigger: {self.trigger.name} / {self.trigger.trigger_text!r}")
        print(f"Total pairs: {len(pairs)}")
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Val pairs: {len(val_pairs)}")
        print(f"Poisoned pairs: {len(poisoned_pairs)} (ratio {self.poison_ratio})")
        print(f"Output: {output_dir}")
        print("=" * 50)
        
        return paths
