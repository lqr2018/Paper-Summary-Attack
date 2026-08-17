"""
2D Scatter Plotting

Draws a scatter plot where samples of different labels have different colors.
Default label = 0 (clean) and label = 1 (trigger).

For Locphylax multi-class visualization:
    label 0 = clean (gray)
    label 1 = unknown_trigger (red)
    label 2 = injected_t1 (green)
    label 3 = injected_t2 (blue)

To preserve backward compatibility with scripts/visualize_embeddings.py,
plot_2d still accepts the original two-tuple colors/markers signature;
a four-element tuple is used for the 4-class Locphylax visualization.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Tuple


# Default 4-class palette (Locphylax visualization)
DEFAULT_COLORS_4C = ("gray", "red", "green", "blue")
DEFAULT_MARKERS_4C = ("o", "x", "s", "^")
DEFAULT_NAMES_4C = ("clean", "unknown_trigger", "injected_t1", "injected_t2")

# Backward-compatible 2-class palette (original behavior)
DEFAULT_COLORS_2C = ("steelblue", "coral")
DEFAULT_MARKERS_2C = ("o", "x")
DEFAULT_NAMES_2C = ("clean", "trigger")


def _normalize_palette(colors, markers, names, max_label: int):
    """
    Pad colors/markers/names to (max_label + 1) entries using defaults.
    Returns (colors, markers, names) lists of length max_label + 1.
    """
    colors = list(colors)
    markers = list(markers)
    names = list(names)

    default_colors = list(DEFAULT_COLORS_4C)
    default_markers = list(DEFAULT_MARKERS_4C)
    default_names = list(DEFAULT_NAMES_4C)

    while len(colors) <= max_label:
        colors.append(default_colors[len(colors) % len(default_colors)])
    while len(markers) <= max_label:
        markers.append(default_markers[len(markers) % len(default_markers)])
    while len(names) <= max_label:
        names.append(f"class_{len(names)}")

    return colors, markers, names


def plot_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    path: str,
    title: str = "",
    colors: Optional[Sequence[str]] = None,
    markers: Optional[Sequence[str]] = None,
    names: Optional[Sequence[str]] = None,
    alpha: float = 0.7,
) -> None:
    """
    Plot 2D coords colored by label.

    Args:
        coords: [N, 2] array
        labels: [N] int array (0,1,2,...)
        path: Output PNG path
        title: Plot title
        colors: Per-class colors. If None, defaults to
                (steelblue, coral) for 2 classes or
                (gray, red, green, blue) for >2 classes.
        markers: Per-class markers. Defaults match colors.
        names: Per-class legend names.
        alpha: Point transparency
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    max_label = int(unique_labels.max()) if len(unique_labels) > 0 else 0

    if colors is None:
        colors = DEFAULT_COLORS_2C if max_label <= 1 else DEFAULT_COLORS_4C
    if markers is None:
        markers = DEFAULT_MARKERS_2C if max_label <= 1 else DEFAULT_MARKERS_4C
    if names is None:
        names = DEFAULT_NAMES_2C if max_label <= 1 else DEFAULT_NAMES_4C

    colors, markers, names = _normalize_palette(colors, markers, names, max_label)

    fig, ax = plt.subplots(figsize=(7, 6))

    for label in sorted(unique_labels):
        mask = labels == label
        n = int(mask.sum())
        if n == 0:
            continue
        name = names[int(label)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[int(label)],
            marker=markers[int(label)],
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
