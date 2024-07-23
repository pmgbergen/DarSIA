from pathlib import Path

import numpy as np
import skimage

import darsia


class TypeCorrection(darsia.BaseCorrection):
    def __init__(self, data_type=None):
        self.data_type = data_type

    def correct_array(self, img: np.ndarray) -> np.ndarray:

        assert self.data_type is not None, "data_type is not defined"

        copy_image = img.copy()
        data_type = self.data_type
        if data_type in [bool]:
            copy_image = skimage.img_as_bool(copy_image)
        elif data_type in [float]:
            copy_image = skimage.img_as_float(copy_image)
        elif data_type in [np.float32]:
            copy_image = skimage.img_as_float32(copy_image)
        elif data_type in [np.float64]:
            copy_image = skimage.img_as_float64(copy_image)
        elif data_type in [int]:
            copy_image = skimage.img_as_int(copy_image)
        elif data_type in [np.uint8]:
            copy_image = skimage.img_as_ubyte(copy_image)
        elif data_type in [np.uint16]:
            copy_image = skimage.img_as_uint(copy_image)
        else:
            raise NotImplementedError

        return copy_image

    def save(self, path: Path) -> None:
        """Save the class name and data type to a npz file.

        Args:
            path (Path): path to npz file

        """
        np.savez(path, class_name=type(self).__name__, data_type=self.data_type)

    def load(self, path: Path) -> None:
        """Load the data type from a npz file."""
        self.data_type = np.load(path, allow_pickle=True)["data_type"].item()
