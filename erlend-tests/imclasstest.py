import daria as da
import cv2
import numpy as np


im = cv2.imread("images/originals/Baseline.jpg")


img = da.Image(im, [0, 0], 1, 1)


img2 = da.extractROI(img, [0.2, 0.5], [0.5, 0.9])

img2.write("ROI-test")
