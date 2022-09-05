"""
Apply various filters and denoising algorithms to the FluidFlower benchmark Baseline image, as well as the small rig white sand baseline image.
"""
import daria as da
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

# Plotting aux
num = 0

# Color image
testcase = 'benchmark baseline'
#testcase = 'white sand baseline'
#testcase = 'benchmark welltest'

match testcase:
    case 'benchmark baseline':
        # Benchmark image
        image_color = cv2.imread("../../images/fluidflower/Baseline.jpg")

        # Restrict the image to some relevant ROI (without the color checker for simplicity).
        image_color = image_color[520:4400, 130:7750]

    case 'white sand baseline':
        # White sand mid rig image
        image_color = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")

        # Restrict the image to some relevant ROI (without the color checker for simplicity).
        image_color = image_color[280:4400, 380:6500]

    case 'benchmark welltest':
        # Benchmark image
        image_color = cv2.imread("../../images/fluidflower/welltest/pulse1.jpg")

        # Restrict the image to some relevant ROI (without the color checker for simplicity).
        image_color = image_color[520:4400, 130:7750]

    case _:
        raise ValueError

# Work in the following on the gray version of the image
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
image = skimage.img_as_float(image)

# Gamma correction, gamma=0.1
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 0.1)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 0.1", loc='left')

# Gamma correction, gamma=0.2
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 0.2)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 0.2", loc='left')

# Gamma correction, gamma=0.5
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 0.5)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 0.5", loc='left')

# Gamma correction, gamma=1
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 1.)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 1.", loc='left')

# Gamma correction, gamma=1.2
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 1.2)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 1.2", loc='left')

# Gamma correction, gamma=2
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 2)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 2", loc='left')

# Gamma correction, gamma=2.5
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 2.5)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 2.5", loc='left')

# Gamma correction, gamma=3
num += 1
image_blur = skimage.exposure.adjust_gamma(image, 3)
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(image_blur)
ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
plt.title("gamma 3", loc='left')

plt.show()
