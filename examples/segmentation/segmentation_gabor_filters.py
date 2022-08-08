"""
Segmentation of the white sands small rig geometry using unsupervised texture segmentation using Gabor filters,
similar to 10.1016/0031-3203(91)90143-S, and inspired by the skimage example: 
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html#sphx-glr-auto-examples-features-detection-plot-gabor-py
"""

# TODO apply the saem to the Baseline benchmark image, with mere objective to detect the visibly most obvious sand layers (need some edge/gradient based detection in the more complicated zones).

# TODO Think about Gabor filter design;
# https://www.cs.technion.ac.il/users/wwwb/cgi-bin/tr-get.cgi/2005/CIS/CIS-2005-05.pdf

# TODO hybrid supersived/unsupervised segmentation. Provide example texture patches of different layers; possibly aim at maximizing feature differences by choosing the right Gabor filters.

import daria as da
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

# White sand mid rig image
image_color = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")

## Restrict the image to some relevant ROI (without the color checker for simplicity).
#image_color = image_color[280:4400, 380:6500]
#image_color = image_color[500:1200, 4900:5500]
#image_color = image_color[500:550, 4900:4950]

# Work in the following on the gray version of the image
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Make manual classification based on a set of 5 examples per class

# TODO what is the right patchsize? find.
patchsize = 100
num_classes = 3
datasetsize = 5

dataset = [
    # class 0
    [
        image[500:500+patchsize, 5000:5000+patchsize],
        image[500:500+patchsize, 5200:5200+patchsize],
        image[500:500+patchsize, 5300:5300+patchsize],
        image[600:600+patchsize, 5100:5100+patchsize],
        image[600:600+patchsize, 5300:5300+patchsize],
    ],

    # class 1
    [
        image[750:750+patchsize, 4900:4900+patchsize],
        image[750:750+patchsize, 5150:5150+patchsize],
        image[750:750+patchsize, 5200:5200+patchsize],
        image[800:800+patchsize, 5200:5200+patchsize],
        image[750:750+patchsize, 5400:5400+patchsize],
    ],

    # class 2
    [
        image[1000:1000+patchsize, 4900:4900+patchsize],
        image[1100:1100+patchsize, 4950:4950+patchsize],
        image[1000:1000+patchsize, 5150:5150+patchsize],
        image[1100:1100+patchsize, 5200:5200+patchsize],
        image[1050:1050+patchsize, 5300:5300+patchsize],
    ],
]

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        # TODO rm
        #plt.imshow(filtered)
        #plt.show()
    return feats

def match_class(feats, ref_feats):
    sqr_error = np.zeros(ref_feats.shape[0])
    for i in range(ref_feats.shape[0]):
        for j in range(ref_feats.shape[1]):
            error = np.sum((feats - ref_feats[i, j, :])**2)
            sqr_error[i] += error
    return np.argmin(sqr_error)

# prepare filter bank kernels
kernels = []
# TODO test different ranges and sets of filters
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

# prepare reference features
ref_feats = np.zeros((num_classes, datasetsize, len(kernels), 2), dtype=np.double)
for i in range(num_classes):
    for j in range(datasetsize):
        ref_feats[i, j, :, :] = compute_feats(dataset[i][j], kernels)

# define test images from classes 0, 1, 2
testsize = 2

test_dataset = [
    # class 0
    [
        image[500:500+patchsize, 5000:5000+patchsize],
        image[550:550+patchsize, 5300:5300+patchsize],
    ],

    # class 1
    [
        image[750:750+patchsize, 4900:4900+patchsize],
        image[750:750+patchsize, 5300:5300+patchsize],
    ],

    # class 2
    [
        image[1000:1000+patchsize, 4900:4900+patchsize],
        image[1000:1000+patchsize, 5200:5200+patchsize],
    ],
]


# Compute feats for test images
feats = np.zeros((num_classes, testsize, len(kernels), 2), dtype=np.double)
for i in range(num_classes):
    for j in range(testsize):
        feats[i, j, :, :] = compute_feats(test_dataset[i][j], kernels)

# classification beased on mse
for i in range(num_classes):
    for j in range(testsize):
        assert(i == match_class(feats[i,j], ref_feats))

# Apply classification to a larger image
test_image = image[500:1200, 4900:5500]

plt.imshow(test_image)
plt.show()





# Compute the entropy
#entropy_img = skimage.filters.rank.entropy(image, skimage.morphology.disk(5))

#plt.hist(entropy_img.flat, bins = 100)
#plt.hist(image.flat, bins = 100)

#plt.imshow(entropy_img)
plt.imshow(image)
plt.show()

