"""
Attacks Package

Contains backdoor attack implementations:
- triggers: Different trigger strategies (word, phrase, long)
- injectors: Backdoor injection paradigms (future: SFT, RLHF, BadEdit)
"""

from .triggers import (
    TriggerStrategy,
    create_trigger,
    TRIGGER_TYPES,
)

__all__ = [
    "TriggerStrategy",
    "create_trigger",
    "TRIGGER_TYPES",
]