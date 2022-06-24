import numpy as np

# conversion between opencv (cv2) which has BGR-formatting and scikit-image (skimage) which has RGB. The same command works in both directions.
def cv2ToSkimage(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]
