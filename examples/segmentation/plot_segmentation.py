"""
Geometry segmentation of a layered medium
=========================================

Segmentation splits an image of a multi-layered porous medium into labelled
regions. DarSIA wraps a watershed workflow; this example contrasts a fully
gradient-based run with a supervised run seeded by marker points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import darsia

path = Path("..") / "images" / "baseline.jpg"
image = darsia.imread(path, dim=2)

# Crop away the shadowed borders to keep the focus on the workflow.
image = image.subregion((slice(0, None), slice(55, 3140)))

# %%
# Gradient-based segmentation. The config holds the watershed tuning parameters;
# set ``verbosity=True`` to see the intermediate quantities.

config = {
    "median disk radius": 20,
    "rescaling factor": 0.3,
    "markers disk radius": 10,
    "threshold": 20,
    "gradient disk radius": 2,
}
labels = darsia.segment(
    image,
    markers_method="gradient_based",
    edges_method="gradient_based",
    verbosity=False,
    **config,
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].imshow(labels.img)
axes[0].set_title("labels (gradient-based)")
axes[1].imshow(image.img)
axes[1].imshow(labels.img, alpha=0.3)
axes[1].set_title("overlay")
for ax in axes:
    ax.set_axis_off()
fig.tight_layout()

# %%
# Supervised segmentation. One marker point per region guides the watershed,
# and a mask hides the colour checker in the top-left corner.

mask = np.ones(image.shape[:2], dtype=bool)
mask[slice(0, 220), slice(0, 200)] = False

config = {
    "marker_points": np.array(
        [
            [100, 1600],
            [400, 1600],
            [560, 760],
            [620, 1260],
            [550, 1970],
            [630, 365],
            [690, 1225],
            [620, 1990],
            [560, 710],
            [775, 1810],
            [615, 1610],
            [830, 570],
            [940, 1320],
            [890, 1775],
            [940, 360],
            [1000, 240],
            [1130, 165],
            [1325, 170],
            [1010, 680],
            [1020, 2650],
            [1075, 1530],
            [1325, 405],
            [1400, 385],
            [1350, 1320],
        ]
    ),
    "region_size": 20,
    "mask scharr": mask,
}
labels_supervised = darsia.segment(
    image, markers_method="supervised", edges_method="scharr", **config
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].imshow(labels_supervised.img)
axes[0].set_title("labels (supervised)")
axes[1].imshow(image.img)
axes[1].imshow(labels_supervised.img, alpha=0.3)
axes[1].set_title("overlay")
for ax in axes:
    ax.set_axis_off()
fig.tight_layout()
plt.show()
