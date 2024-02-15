import numpy as np
import skimage

import darsia


class TypeCorrection(darsia.BaseCorrection):
    def __init__(self, data_type):
        self.data_type = data_type

    def correct_array(self, img: np.ndarray) -> np.ndarray:
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
