"""
Trigger Strategies Package

Provides a factory to create trigger strategies by type name.
New trigger types can be added by implementing TriggerStrategy
and registering them in the _TRIGGERS registry.
"""

from .base_trigger import TriggerStrategy
from .word_trigger import WordTrigger
from .phrase_trigger import PhraseTrigger
from .long_trigger import LongTrigger

# Registry mapping trigger type names to their classes
_TRIGGERS = {
    WordTrigger.name: WordTrigger,
    PhraseTrigger.name: PhraseTrigger,
    LongTrigger.name: LongTrigger,
}

#: Valid trigger type names
TRIGGER_TYPES = list(_TRIGGERS.keys())


def create_trigger(trigger_type: str, **kwargs) -> TriggerStrategy:
    """
    Create a trigger strategy instance by type name.
    
    Args:
        trigger_type: Name of trigger type ("word", "phrase", "long")
        **kwargs: Additional arguments passed to the trigger constructor
                 (e.g., trigger_word, trigger_phrase, trigger_paragraph)
    
    Returns:
        TriggerStrategy instance
    
    Raises:
        ValueError: If trigger_type is unknown
    """
    trigger_type = trigger_type.lower()
    if trigger_type not in _TRIGGERS:
        raise ValueError(
            f"Unknown trigger type: '{trigger_type}'. "
            f"Available types: {TRIGGER_TYPES}"
        )
    return _TRIGGERS[trigger_type](**kwargs)


__all__ = [
    "TriggerStrategy",
    "WordTrigger",
    "PhraseTrigger",
    "LongTrigger",
    "create_trigger",
    "TRIGGER_TYPES",
]