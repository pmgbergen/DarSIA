"""Helpers for storing setup illustrations."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_2d(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 2:
        raise ValueError("Expected a 2D scalar image.")
    return data


def _format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3g}"


def save_discrete_map_illustration(
    array: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    """Store a JPG illustration of a discrete map with numeric annotations."""

    data = _to_2d(array)
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(data, cmap="tab20", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label(colorbar_label)

    unique_values = np.unique(data[np.isfinite(data)])
    for value in unique_values:
        coords = np.argwhere(data == value)
        if coords.size == 0:
            continue
        row, col = np.mean(coords, axis=0)
        row_int = int(np.clip(round(row), 0, data.shape[0] - 1))
        col_int = int(np.clip(round(col), 0, data.shape[1] - 1))
        rgba = image.cmap(image.norm(data[row_int, col_int]))
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "black" if brightness > 0.5 else "white"
        ax.text(
            col,
            row,
            _format_value(float(value)),
            color=text_color,
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.2, "edgecolor": "none"},
        )

    ax.set_title(title)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="jpg", dpi=200)
    plt.close(fig)


def save_scalar_map_illustration(
    array: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    """Store a JPG illustration of a scalar map with sampled value annotations."""

    data = _to_2d(array).astype(float)
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError("Expected at least one finite value in scalar image.")

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(data, cmap="viridis")
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    ticks = np.linspace(vmin, vmax, num=7) if vmax > vmin else np.array([vmin])
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, ticks=ticks)
    colorbar.set_label(colorbar_label)
    colorbar.ax.set_yticklabels([_format_value(float(tick)) for tick in ticks])

    rows = np.linspace(0, data.shape[0] - 1, num=3, dtype=int)
    cols = np.linspace(0, data.shape[1] - 1, num=3, dtype=int)
    finite_coords = np.argwhere(finite_mask)
    for row, col in product(rows, cols):
        if not finite_mask[row, col]:
            distances = np.sum((finite_coords - np.array([row, col])) ** 2, axis=1)
            row, col = finite_coords[int(np.argmin(distances))]
        value = float(data[row, col])
        ax.text(
            col,
            row,
            _format_value(value),
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
        )

    ax.set_title(title)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="jpg", dpi=200)
    plt.close(fig)
