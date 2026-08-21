"""
Dimensionality Reduction

PCA: overall linear structure.
t-SNE: local clustering structure.

Both reduce [N, hidden_dim] -> [N, 2].
"""

import numpy as np
from typing import Optional


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
    n_pre_pca: Optional[int] = None,
) -> np.ndarray:
    """
    t-SNE to 2D.

    Args:
        representations: [N, D] float32.
        n_pre_pca: if set and N is large, first reduce to this many dims with
                   PCA before t-SNE (accelerates; sklearn requires n_components
                   < min(n_samples, n_features)).
    """
    from sklearn.manifold import TSNE

    n_samples = representations.shape[0]
    n_features = representations.shape[1]

    # Optional pre-PCA to speed up large sets.
    X = representations
    if n_pre_pca is not None:
        n_pre = min(n_pre_pca, n_samples - 1, n_features)
        if n_pre > 0:
            X = pca(representations, n_components=n_pre)

    # Clamp perplexity so it is strictly less than the number of samples.
    if n_samples >= 2:
        perplexity = min(int(perplexity), n_samples - 1)
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
    return reducer.fit_transform(X)