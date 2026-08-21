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
    colors: Tuple[str, str] = ("steelblue", "coral"),
    markers: Tuple[str, str] = ("o", "x"),
    alpha: float = 0.7,
) -> None:
    """
    Plot 2D coords colored by label (0=clean, 1=trigger).

    Args:
        coords: [N, 2] array
        labels: [N] int array (0/1)
        path: Output PNG path
        title: Plot title
        colors: (clean_color, trigger_color)
        markers: (clean_marker, trigger_marker)
        alpha: Point transparency
    """
    labels = np.asarray(labels)
    clean_mask = labels == 0
    trigger_mask = labels == 1

    n_clean = int(clean_mask.sum())
    n_trigger = int(trigger_mask.sum())

    fig, ax = plt.subplots(figsize=(7, 6))

    if n_clean:
        ax.scatter(
            coords[clean_mask, 0],
            coords[clean_mask, 1],
            c=colors[0],
            marker=markers[0],
            label=f"clean (n={n_clean})",
            alpha=alpha,
            s=30,
        )
    if n_trigger:
        ax.scatter(
            coords[trigger_mask, 0],
            coords[trigger_mask, 1],
            c=colors[1],
            marker=markers[1],
            label=f"trigger (n={n_trigger})",
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