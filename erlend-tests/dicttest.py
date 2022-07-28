import daria as da
import cv2

im = cv2.imread("images/originals/Baseline.jpg", 0)

img = da.Image(im)

print(im)
