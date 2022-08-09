"""
Segmentation of the white sands small rig geometry using unsupervised texture segmentation using Gabor filters,
similar to 10.1016/0031-3203(91)90143-S, and inspired by the skimage example: 
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html#sphx-glr-auto-examples-features-detection-plot-gabor-py
"""

# TODO Think about Gabor filter design;
# https://www.cs.technion.ac.il/users/wwwb/cgi-bin/tr-get.cgi/2005/CIS/CIS-2005-05.pdf

# TODO hybrid supersived/unsupervised segmentation. Provide example texture patches of different layers; possibly aim at maximizing feature differences by choosing the right Gabor filters.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

# White sand mid rig image
image_color = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")

# Work in the following on the gray version of the image
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

plt.imshow(image)
plt.show()

# Make manual classification based on a set of 5 examples per class
patchsize = 100
num_classes = 4
datasetsize = 10

# Seemed to work better with the smaller dataset2
coords = {
    0: [(5000, 500), (5200, 500), (5300, 500), (5100, 600), (5300, 600), 
        (500, 370), (1300, 430), (2300, 440), (750, 2950), (1880, 2840)
    ],
    1: [(4900, 750), (5150, 750), (5200, 750), (5200, 800), (5400, 750),
        (2500, 870), (2900, 880), (370, 2150), (1020, 1920), (2280, 1800)
    ],
    2: [(4900, 1000), (4950, 1100), (5150, 1000), (5200, 1100), (5200, 1050),
        (400, 1450), (1500, 1050), (400, 3450), (1750, 3500), (6950, 3880)
    ],
    3: [(4000, 2700), (4600, 2600), (5000, 2600), (5300, 2800), (5500, 2600),
        (400, 2500), (350, 2800), (1075, 2380), (2110, 2400), (3450, 2810)
    ],
}

slices = [
    [
        (slice(coord[1], coord[1] + patchsize), slice(coord[0], coord[0]+patchsize))
        for coord in coords[j]
    ] for j in range(4)
]

dataset = [
    [
        image[roi] for roi in slices[j]
    ] for j in range (4)
]


#dataset2 = [
#    # class 0
#    [
#        image[500:500+patchsize, 5000:5000+patchsize],
#        image[500:500+patchsize, 5200:5200+patchsize],
#        image[500:500+patchsize, 5300:5300+patchsize],
#        image[600:600+patchsize, 5100:5100+patchsize],
#        image[600:600+patchsize, 5300:5300+patchsize],
#    ],
#
#    # class 1
#    [
#        image[750:750+patchsize, 4900:4900+patchsize],
#        image[750:750+patchsize, 5150:5150+patchsize],
#        image[750:750+patchsize, 5200:5200+patchsize],
#        image[800:800+patchsize, 5200:5200+patchsize],
#        image[750:750+patchsize, 5400:5400+patchsize],
#    ],
#
#    # class 2
#    [
#        image[1000:1000+patchsize, 4900:4900+patchsize],
#        image[1100:1100+patchsize, 4950:4950+patchsize],
#        image[1000:1000+patchsize, 5150:5150+patchsize],
#        image[1100:1100+patchsize, 5200:5200+patchsize],
#        image[1050:1050+patchsize, 5300:5300+patchsize],
#    ],
#
#    # class 3
#    [
#        image[2700:2700+patchsize, 4000:4000+patchsize],
#        image[2600:2600+patchsize, 4600:4600+patchsize],
#        image[2600:2600+patchsize, 5000:5000+patchsize],
#        image[2800:2800+patchsize, 5300:5300+patchsize],
#        image[2600:2600+patchsize, 5500:5500+patchsize],
#    ],
#]

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        # TODO transform nonlinearly
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
resolution_theta = 5.
for theta in range(int(resolution_theta)):
    theta = theta / resolution_theta * np.pi
    for sigma in np.linspace(0.1, 10, 5):
        for frequency in np.linspace(0.1, 0.3, 5):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

# prepare reference features
ref_feats = np.zeros((num_classes, datasetsize, len(kernels), 2), dtype=np.double)
for i in range(num_classes):
    for j in range(datasetsize):
        ref_feats[i, j, :, :] = compute_feats(dataset[i][j], kernels)

# define test images from classes 0, 1, 2
if True:
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
    for i in range(3):
        for j in range(testsize):
            feats[i, j, :, :] = compute_feats(test_dataset[i][j], kernels)
    
    # classification beased on mse
    for i in range(3):
        for j in range(testsize):
            print(i, match_class(feats[i,j], ref_feats))
            #assert(i == match_class(feats[i,j], ref_feats))

# Apply classification to a larger image
test_image = image[500:1200, 4900:5500]
#test_image = image[500:2000, 4700:5500]
#test_image = image[500:2000, 2000:5500]
#test_image = image[400:3900, 2000:5400]
#test_image = image[400:3900, 400:5400]

#ny = int(test_image.shape[0] / patchsize)
#nx = int(test_image.shape[1] / patchsize)
#test_classification = np.zeros((ny,nx), dtype=float)
#for i in range(ny):
#    for j in range(nx):
#        roi = test_image[i * patchsize: (i+1) * patchsize, j * patchsize: (j+1) * patchsize] 
#        feats = compute_feats(roi, kernels)
#        test_classification[i,j] = match_class(feats, ref_feats)

revsize = 50
ny = int(test_image.shape[0] / revsize)
nx = int(test_image.shape[1] / revsize)
test_classification = np.zeros((ny,nx), dtype=float)
for i in range(ny):
    for j in range(nx):
        roi = test_image[i * revsize: i * revsize + patchsize, j * revsize: j * revsize + patchsize] 
        feats = compute_feats(roi, kernels)
        test_classification[i,j] = match_class(feats, ref_feats)

fig, ax = plt.subplots(2,1, num=1)
ax[0].imshow(test_image)
ax[1].imshow(test_classification)
plt.show()
