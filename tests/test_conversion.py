import numpy as np
from daria import cv2ToSkimage


def test_cv2ToSkimage():
    imgBGR = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    imgRGB = np.array([[[3, 2, 1], [6, 5, 4]], [[9, 8, 7], [12, 11, 10]]])
    np.testing.assert_array_equal(cv2ToSkimage(imgBGR), imgRGB)
    np.testing.assert_array_equal(cv2ToSkimage(imgRGB), imgBGR)
