"""
Module wrapping median algoriths from skimage
into the realm of DarSIA, operating on darsia.Image.

"""

import numpy as np
import skimage


class Median:
    """
    Median through skimage.filter.rank.

    """

    def __init__(self, key: str = "", **kwargs) -> None:
        """
        Constructor.

        Args:
            key (str): Prefix for kwargs arguments.

        """
        self.disk_radius: int = kwargs.get(key + "disk radius", 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:

        return skimage.filters.rank.median(
            skimage.img_as_ubyte(img), skimage.morphology.disk(self.disk_radius)
        )
