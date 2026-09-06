"""
Reading a NumPy array
=====================

The simplest possible source: a ``.npy`` file holding a raw array, turned into a
2D physical :class:`~darsia.Image`.
"""

import os

import darsia

folder = os.path.join("..", "images")
path = os.path.join(folder, "random_distribution.npy")

# %%
# ``imread`` recognises the extension and reads the array; ``width`` and
# ``height`` give it a physical size.

np_image = darsia.imread(path, dim=2, width=2, height=1)
np_image.show("random distribution")
