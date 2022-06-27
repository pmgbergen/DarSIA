import daria as da
import cv2


im = cv2.imread("images/originals/Baseline.jpg")


img = da.Image(im, [0, 0], 1, 1)

img2 = da.extractROI(img, [0.2, 0.5], [0.5, 0.9])

print(img2.origo)

img2.write("ROI-test")


img3 = da.Image("images/modified/ROI-test.jpg", read_metadata_from_file=True)

img4 = da.Image("images/originals/Baseline.jpg")

print(img4.dx)

print(img2.dx)

print(img3.dx)
