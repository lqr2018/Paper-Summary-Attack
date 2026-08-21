"""
Modular pipeline for backdoor representation-space visualization:

    extract → reduce → plot

Public API:
    from visualization import HiddenRepresentationExtractor
    from visualization import pca, tsne
    from visualization import plot_2d
    from visualization import load_injector_test_set, build_probe_set
"""

from .extract import HiddenRepresentationExtractor
from .reduce import pca, tsne
from .plot import plot_2d
from .probes import load_injector_test_set, build_probe_set

__all__ = [
    "HiddenRepresentationExtractor",
    "pca",
    "tsne",
    "plot_2d",
    "load_injector_test_set",
    "build_probe_set",
]