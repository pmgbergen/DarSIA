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
#testcase = 'benchmark baseline'
testcase = 'white sand baseline'
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

## Resize - do not require the high resolution for finding sand layers
#image = skimage.transform.rescale(image, 0.1, anti_aliasing = True)
#
#plt.imshow(image)
#plt.show()

image = skimage.img_as_float(image)

# General Gamma correction
gamma = 0.5

# TV Denoising from daria
pre_im = da.Image(skimage.img_as_ubyte(image), (0, 0), 250, 180)
#pre_im.resize(0.1, 0.1)
for weight in [0.1]: #, 0.05, 0.1, 1]:
    num += 1
    reg_im = da.tv_denoising(pre_im, weight, 1, verbose=True)
    fig, ax = plt.subplots(2,1, num=num)
    ax[0].imshow(reg_im.img)
    ax[1].plot(np.histogram(reg_im.img.flatten(), bins=256)[0])
    plt.title(f"tv denoising weight {weight}", loc='left')
plt.show()

# TV denoising from skimage
for weight in [0.01, 0.1, 1]:
    num += 1
    image_blur = skimage.exposure.adjust_gamma(image, gamma)
    #image_blur = np.uint8(255.0 * skimage.restoration.denoise_tv_chambolle(image_blur, weight=weight, channel_axis=-1))
    image_blur = skimage.img_as_ubyte(skimage.restoration.denoise_tv_bregman(image_blur, weight=weight, channel_axis=-1))
    fig, ax = plt.subplots(2,1, num=num)
    ax[0].imshow(image_blur)
    ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
    plt.title(f"tv denoising weight {weight}", loc='left')

plt.show()

# Median filter from skimage.filter.rank
for disk in [10, 100, 200]:
    num += 1
    image_blur = skimage.exposure.adjust_gamma(image, gamma)
    image_blur = skimage.filters.rank.median(image_blur, skimage.morphology.disk(disk))
    fig, ax = plt.subplots(2,1, num=num)
    ax[0].imshow(image_blur)
    ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
    plt.title(f"median disk {disk}", loc='left')

plt.show()

# Gaussian filter from skimage
for sigma in [0.1, 1, 10, 100]:
    num += 1
    image_blur = skimage.exposure.adjust_gamma(image, gamma)
    image_blur = skimage.filters.gaussian(image_blur, sigma=sigma)
    fig, ax = plt.subplots(2,1, num=num)
    ax[0].imshow(image_blur)
    ax[1].plot(np.histogram(image_blur.flatten(), bins=256)[0])
    plt.title(f"gaussian sigma {sigma}", loc='left')
plt.show()

# Filter: difference of gaussians
image_blur = skimage.exposure.adjust_gamma(image, gamma)
image_blur = skimage.filters.difference_of_gaussians(image_blur, 1,10)
#image_blur = skimage.filters.difference_of_gaussians(skimage.filters.median(image_blur, skimage.morphology.disk(10)), 1,10)
plt.imshow(image_blur)
plt.show()

# Morphology filters from skimage
image = skimage.morphology.erosion(image, skimage.morphology.disk(5)) # NOTE intersting for coarse-scale analysis!
image = skimage.filters.rank.median(image, skimage.morphology.disk(50))
image = skimage.filters.difference_of_gaussians(image, 1,10)
#image = skimage.morphology.dilation(image, skimage.morphology.disk(5))
#image = skimage.morphology.opening(image, skimage.morphology.disk(5))
plt.imshow(image)
plt.show()
