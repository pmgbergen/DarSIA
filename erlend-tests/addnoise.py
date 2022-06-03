import cv2
import daria
import numpy as np

import skimage

img = cv2.imread("images/originals/Profilbilde.jpg")

# Conversion from opencv to skimage (The same command works in both directions)
img = daria.cv2ToSkimage(img)
print(img)

# img = skimage.io.imread("images/originals/Profilbilde.jpg")
# img = skimage.util.random_noise(img, "gaussian")
# img = skimage.util.random_noise(img, "localvar")
# img = skimage.util.random_noise(img, "poisson")
img = skimage.util.random_noise(img, "salt")
# img = skimage.util.random_noise(img, "pepper")
# img = skimage.util.random_noise(img, "s&p")
# img = skimage.util.random_noise(img, "speckle")


img = daria.cv2ToSkimage(img)


cv2.imwrite("images/modified/ProfilbildeGauss.jpg", img)


# skimage.io.imwrite("images/modified")
