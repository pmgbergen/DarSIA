"""
Image porosity: which pixels show the pore space
================================================

A photo of a sand-packed rig shows the colour of the pore space, which a pH indicator
changes when CO2 dissolves, but also sand grains of other colours, reflections, and
pixels at layer boundaries that mix two sands. The *image porosity* marks each pixel of
the baseline image between 1 (the typical colour of its layer: pore space) and 0 (a
grain, a reflection, a mixed pixel). Under linear colour mixing, a value is roughly one
minus the fraction of the pixel covered by grains.

The workflows use it, when asked to, as a mask for the colour calibration and the
illumination correction, and as a weight for the restoration. It is configured in the
``[image_porosity]`` section and computed by the setup step "Image porosity". This
example walks through its options on the baseline of the example images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import darsia
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.processing.color_statistics import (
    build_color_statistics,
)
from darsia.presets.workflows.processing.image_porosity import (
    boolean_image_porosity,
    compute_image_porosity,
)

BRIGHTEN = 2.5
"""Display gain for the dark photo (display only)."""


def porosity(
    options: dict, regions: darsia.Image | None = None, full: bool = False
) -> np.ndarray:
    """Image porosity for the ``[image_porosity]`` options, of the reduced baseline or,
    with `full`, of the full-resolution one."""
    config = ImagePorosityConfig()._load_dict(options)
    photo, layers = (image, labels_full) if full else (baseline, labels)
    regions = layers if regions is None else regions
    return compute_image_porosity(photo, regions, layers, config).img


def show_photo(ax, photo, title="baseline (brightened)"):
    """A photo, brightened for display."""
    ax.imshow(np.clip(BRIGHTEN * photo, 0, 1))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def show_maps(maps: dict, close_up: bool = False) -> None:
    """Porosity maps side by side in one colour scale.

    Overviews (reduced baseline) are drawn with averaging; close-ups (full-resolution
    maps, cut to the detail, after the photo of the detail) pixel by pixel.
    """
    panels = len(maps) + int(close_up)
    fig, axes = plt.subplots(
        1, panels, figsize=(4.2 * panels, 2.8), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    if close_up:
        show_photo(axes[0], image.img[detail], "baseline (detail, brightened)")
    for ax, (title, values) in zip(axes[int(close_up) :], maps.items()):
        if close_up:
            im = ax.imshow(values[detail], vmin=0, vmax=1, interpolation="nearest")
        else:
            im = ax.imshow(values, vmin=0, vmax=1, interpolation="antialiased")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.9, label="image porosity")


# %%
# The baseline and its layers
# ---------------------------
# The baseline is cropped to the rig. The overview maps below are computed on a copy
# reduced to 30 % of its size, which keeps this page fast, and there single grains are
# averaged with their surroundings. The close-ups are computed at full resolution, as
# the workflows do. The layers come from the gradient-based segmentation (see
# :ref:`examples_segmentation`); in the workflows they are the ``labels`` of the rig.

image = darsia.imread(Path("..") / "images" / "baseline.jpg", dim=2)
image = image.subregion((slice(0, None), slice(55, 3140)))
baseline = darsia.resize(image, fx=0.3, fy=0.3)

labels = darsia.segment(
    baseline,
    markers_method="gradient_based",
    edges_method="gradient_based",
    verbosity=False,
    **{
        "median disk radius": 6,
        "rescaling factor": 1.0,
        "markers disk radius": 3,
        "threshold": 20,
        "gradient disk radius": 1,
    },
)

# The same layers on the full-resolution grid, for the close-ups.
labels_full = darsia.resize(
    labels, shape=image.img.shape[:2], interpolation="inter_nearest"
)

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
show_photo(axes[0], baseline.img)
axes[1].imshow(labels.img, cmap="tab20", interpolation="nearest")
axes[1].set_title(f"labels ({len(np.unique(labels.img))} layers)", fontsize=10)
axes[1].set_axis_off()

# A detail inside a coarse layer with bright and dark grains, for the close-ups
# (full-resolution pixels).
detail = (slice(1433, 1700), slice(1500, 1967))

# %%
# No detection: the boundary band only
# ------------------------------------
# With ``method = "none"`` every pixel counts as porous, except the ``boundary_band``:
# pixels within that many pixels of a layer boundary, on both sides, mix the colours
# of two sands and represent neither. The band applies with every method.

show_maps(
    {
        'method "none", boundary_band 0': porosity(
            {"method": "none", "boundary_band": 0}
        ),
        'method "none", boundary_band 5': porosity(
            {"method": "none", "boundary_band": 5}
        ),
    }
)

# %%
# Colour statistics: arms and norm
# --------------------------------
# ``method = "color_statistics"`` compares each pixel with the colour distribution of
# its layer *at its location*. Per layer, the typical (median) colour and the spread
# around it are measured. A pixel closer to the typical colour than the
# ``noise_percentile`` of the distances counts as typical (1), one farther than the
# ``key_percentile`` as not (0), linear in between.
#
# * ``statistic = "arms"`` measures the distance along the main colour axis of the
#   layer, darker (dark arm) or brighter (bright arm), and off it (side: another
#   hue), each against its own percentiles. The three kinds of grains are judged on
#   their own scales, and each term can be switched off (``dark``, ``bright``,
#   ``side``).
# * ``statistic = "norm"`` measures the distance in any direction: one scale for all
#   deviations.
#
# On this image both find the same grains; arms flags slightly more of the faint ones.

statistics = {"patches": [6, 10], "min_pixels": 500}
arms = porosity({"method": "color_statistics", "color_statistics": statistics})
norm = porosity(
    {
        "method": "color_statistics",
        "color_statistics": {**statistics, "statistic": "norm"},
    }
)
show_maps({'statistic "arms"': arms, 'statistic "norm"': norm})

# %%
# The close-up, at full resolution, shows single grains: bright specks in the photo
# become dark pixels of the image porosity.

arms_options = {"method": "color_statistics", "color_statistics": statistics}
arms_full = porosity(arms_options, full=True)
norm_full = porosity(
    {
        "method": "color_statistics",
        "color_statistics": {**statistics, "statistic": "norm"},
    },
    full=True,
)
show_maps({"arms": arms_full, "norm": norm_full}, close_up=True)

# %%
# What counts as typical: the percentiles
# ---------------------------------------
# The percentiles are in percent (0-100) of the local distances. Lower values call
# more pixels atypical: with 80 and 95, a fifth of every layer scores below 1.

show_maps(
    {
        "noise 90, key 99.5 (default)": arms_full,
        "noise 80, key 95": porosity(
            {
                "method": "color_statistics",
                "color_statistics": {
                    **statistics,
                    "noise_percentile": 80,
                    "key_percentile": 95,
                },
            },
            full=True,
        ),
    },
    close_up=True,
)

# %%
# Patches follow the illumination
# -------------------------------
# The light is not uniform over a rig. The statistics are therefore fitted per patch
# and layer, ``patches = [rows, cols]``, and interpolated smoothly over the image, so
# the typical colour follows the illumination. The fitted typical brightness of the
# largest layer shows the trend.

fitted = build_color_statistics(
    ImagePorosityConfig()
    ._load_dict({"method": "color_statistics", "color_statistics": statistics})
    .color_statistics
)
fitted.fit(baseline, labels)
largest = np.bincount(labels.img.ravel()).argmax()
rows = np.arange(labels.img.shape[0])
cols = np.arange(labels.img.shape[1])
brightness = np.mean(
    [field(rows, cols) for field in fitted.fields[largest][:3]], axis=0
)
brightness[labels.img != largest] = np.nan

fig, ax = plt.subplots(figsize=(5, 2.8), layout="constrained")
im = ax.imshow(brightness, cmap="gray")
ax.set_title(f"typical brightness of layer {largest}, patches [6, 10]", fontsize=10)
ax.set_axis_off()
fig.colorbar(im, ax=ax, shrink=0.9, label="mean of R, G, B")

# %%
# With ``patches = [1, 1]`` one typical colour holds for a whole layer: where the
# layer is darker or brighter than on average, e.g. towards the left and right ends,
# its sand counts as atypical.

show_maps(
    {
        "patches [1, 1]": porosity(
            {
                "method": "color_statistics",
                "color_statistics": {**statistics, "patches": [1, 1]},
            }
        ),
        "patches [6, 10]": arms,
    }
)

# %%
# Regions: labels, facies or global
# ---------------------------------
# ``regions`` decides which pixels are compared with each other: the ``labels``
# (default), the ``facies`` (the labels of one sand type grouped, from the ``[facies]``
# section), or one ``global`` region for the whole image. Facies give more pixels per
# patch and the same result as long as the grouped labels are the same sand. With one
# global region all sands are compared with each other, and layers of a less common
# colour drop out as a whole.

global_region = darsia.zeros_like(labels, dtype=np.uint8)
options = {"method": "color_statistics", "color_statistics": statistics}
show_maps(
    {
        'regions "labels"': arms,
        'regions "global"': porosity(options, global_region),
    }
)

# %%
# The legacy k-means method
# -------------------------
# ``method = "kmeans"`` clusters the colours per patch and layer and interpolates
# between the dominant colour (1) and an outlier colour (0). It is kept for comparison:
# it is slower, and its patches show as a patchwork.

show_maps(
    {
        'method "kmeans", patches [4, 4]': porosity(
            {"method": "kmeans", "kmeans": {"patches": [4, 4]}}
        ),
        'method "color_statistics", arms': arms,
    }
)

# %%
# Cutoff and the boolean porosity
# -------------------------------
# Values below ``cutoff`` are set to 0. The *boolean porosity*, the image porosity
# above ``tol``, is what the colour calibration and the illumination correction use
# with ``mask = "boolean_porosity"``. ``tol`` applies when the stored porosity is read,
# so it changes without a new setup.

config = ImagePorosityConfig()._load_dict({**options, "tol": 0.9})
boolean = boolean_image_porosity(
    compute_image_porosity(baseline, labels, labels, config), config
)
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
show_photo(axes[0], baseline.img)
axes[1].imshow(boolean.img.astype(float), cmap="gray", interpolation="antialiased")
axes[1].set_title(
    f"boolean porosity, tol 0.9: {boolean.img.mean():.0%} porous", fontsize=10
)
axes[1].set_axis_off()

# %%
# A configuration to start from
# -----------------------------
# Colour statistics with arms, per label, are the recommended start. The setup step
# "Image porosity" (GUI: Setup > Image porosity) computes and stores the map:
#
# .. code-block:: toml
#
#     [image_porosity]
#     method = "color_statistics"
#     regions = "labels"
#     tol = 0.9
#     cutoff = 0.3
#     boundary_band = 5
#
#     [image_porosity.color_statistics]
#     statistic = "arms"
#     patches = [20, 20]
#     noise_percentile = 90
#     key_percentile = 99.5

# sphinx_gallery_thumbnail_number = 4
