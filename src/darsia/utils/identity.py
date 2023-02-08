"""
Module defining an object with an empty call routine.

"""

import numpy as np


class Identity:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img
