import daria as da
import cv2


im = cv2.imread("images/originals/Baseline.jpg")

print(im.shape)

img = da.Image(im, [-100, -100], 1000, 1000)

print(img.shape)

img2 = da.extractROI(img, [0.2, 0.5], [0.5, 0.9])

print(img2.origo)

img2.write("ROI-test")

img.draw_grid()
