"""
Injectors Package

Contains backdoor injection paradigms:
- SFT: supervised fine-tuning (poisoned data, "Aha" target behavior)
- RLHF: reward manipulation (preference pairs)
- BadEdit: model editing (ROME-style weight edit)

Each injector can be combined with any trigger strategy.
"""

from .base_injector import InjectorStrategy
from .sft_injector import SFTInjector
from .rlhf_injector import RLHFInjector
from .badedit_injector import BadEditInjector

# Registry mapping paradigm names to their classes
_INJECTORS = {
    SFTInjector.name: SFTInjector,
    RLHFInjector.name: RLHFInjector,
    BadEditInjector.name: BadEditInjector,
}

#: Valid injector type names
INJECTOR_TYPES = list(_INJECTORS.keys())


def create_injector(injector_type: str, trigger, **kwargs) -> InjectorStrategy:
    """
    Create an injector instance by paradigm name.
    
    Args:
        injector_type: Name of paradigm ("sft", "rlhf", "badedit")
        trigger: Trigger strategy instance to combine with
        **kwargs: Additional arguments passed to the injector constructor
                 (e.g., mode="aha" for SFT, poison_ratio for RLHF)
    
    Returns:
        InjectorStrategy instance
    
    Raises:
        ValueError: If injector_type is unknown
    """
    injector_type = injector_type.lower()
    if injector_type not in _INJECTORS:
        raise ValueError(
            f"Unknown injector type: '{injector_type}'. "
            f"Available types: {INJECTOR_TYPES}"
        )
    return _INJECTORS[injector_type](trigger=trigger, **kwargs)


__all__ = [
    "InjectorStrategy",
    "SFTInjector",
    "RLHFInjector",
    "BadEditInjector",
    "create_injector",
    "INJECTOR_TYPES",
]