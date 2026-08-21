"""
2D Scatter Plotting

Draws a scatter plot where clean samples (label=0) and trigger samples (label=1)
have different colors. Saves a PNG.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from typing import Tuple


def plot_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    path: str,
    title: str = "",
    colors=("gray", "red", "green", "blue"),
    markers=("o", "x", "s", "^"),
    names=("clean", "unknown_trigger", "injected_t1", "injected_t2"),
    alpha: float = 0.7,
) -> None:
    """
    Plot 2D coords colored by label (multi-class).

    Default supports 4 classes (Locphylax Stage I):
        label 0 = clean (gray)
        label 1 = unknown_trigger (red)
        label 2 = injected_t1 (green)
        label 3 = injected_t2 (blue)

    Also works for 2 classes (clean/trigger) by passing colors/markers/names.

    Args:
        coords: [N, 2] array
        labels: [N] int array (0..K-1)
        path: Output PNG path
        title: Plot title
        colors: per-class colors (default 4-class palette)
        markers: per-class markers
        names: per-class legend names
        alpha: Point transparency
    """
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))

    fig, ax = plt.subplots(figsize=(7, 6))

    for lbl in unique_labels:
        if lbl < 0:
            continue
        mask = labels == lbl
        n = int(mask.sum())
        if n == 0:
            continue
        color = colors[lbl % len(colors)]
        marker = markers[lbl % len(markers)]
        name = names[lbl % len(names)] if lbl < len(names) else f"class {lbl}"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            marker=marker,
            label=f"{name} (n={n})",
            alpha=alpha,
            s=30,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot: {path}")
