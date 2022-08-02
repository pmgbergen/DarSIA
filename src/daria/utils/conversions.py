from __future__ import annotations

import numpy as np


def cv2ToSkimage(img: np.ndarray) -> np.ndarray:
    """Conversion between opencv (cv2) which has BGR-formatting and scikit-image (skimage)
    which has RGB. The same command works in both directions.

    Arguments:
        img (np.ndarray): input image

    Returns:
        np.ndarray: converted image.
    """
    return img[:, :, ::-1]
