"""
Attacks Package

Contains backdoor attack implementations:
- triggers: Different trigger strategies (word, phrase, long)
- injectors: Backdoor injection paradigms (SFT, RLHF, BadEdit)

Paradigms and triggers are decoupled: any injector can be combined
with any trigger strategy.
"""

from .triggers import (
    TriggerStrategy,
    create_trigger,
    TRIGGER_TYPES,
)
from .injectors import (
    InjectorStrategy,
    create_injector,
    INJECTOR_TYPES,
)

__all__ = [
    "TriggerStrategy",
    "create_trigger",
    "TRIGGER_TYPES",
    "InjectorStrategy",
    "create_injector",
    "INJECTOR_TYPES",
]