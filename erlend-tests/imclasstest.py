import daria as da
import cv2
import numpy as np

im = cv2.imread("images/originals/Profilbilde.jpg")


# print(im.shape[1])

img = da.Image(im, [0, 1])

print(img)
