"""
Dimensionality Reduction

PCA: overall linear structure.
t-SNE: local clustering structure.

Both reduce [N, hidden_dim] -> [N, 2].
"""

import numpy as np


def pca(representations: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA to 2D (or n_components). Returns [N, n_components]."""
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(representations)


def tsne(
    representations: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
    init: str = "pca",
) -> np.ndarray:
    """
    t-SNE to 2D. Uses init="pca" for faster/convergent results.

    Note:
        - For large sample sizes, first PCA to ~50 dims to speed up.
        - t-SNE requires perplexity < n_samples; it is clamped automatically.
    """
    from sklearn.manifold import TSNE

    n_samples = representations.shape[0]
    # Clamp perplexity so it is strictly less than the number of samples.
    if n_samples >= 2:
        perplexity = min(int(perplexity), n_samples - 1)
    # t-SNE needs at least 3 samples and perplexity >= 1.
    if n_samples < 3 or perplexity < 1:
        raise ValueError(
            f"t-SNE requires >=3 samples, got {n_samples} (perplexity={perplexity})."
        )

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init=init,
    )
    return reducer.fit_transform(representations)
