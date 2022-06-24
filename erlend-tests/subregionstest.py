import daria as da
import cv2


im = cv2.imread("images/originals/Profilbilde.jpg")


img = da.Image(im, [-2, 1], 100, 300)


imgsmall = da.extractSubRegionByPixel(img, [0, 100], [0, 200])

print(imgsmall.shape)

imgsmall.write()
