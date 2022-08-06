"""
The objective of this example is to showcase how to use DaRIA to identify sandlayer
in the FluidFlower Baseline image.
"""
# Clustering, based on https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python

import daria as da
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.restoration import denoise_tv_chambolle

# Color image
image_color = cv2.imread("../images/fluidflower/Baseline.jpg")
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Consider 1d image, either gray or green channel (blue and red channels did not lead to any reasonable results in short time)
gray = False
if gray:
    # Gray
    image = cv2.cvtColor(
        image_color,
        cv2.COLOR_BGR2GRAY,
    )
else:
    # Green
    image = image_color[:,:,2]

# Neutralize Color checker
roi_cc = (slice(100, 600), slice(100, 600))
num_bdry_pixels = (600 - 100) * 2 + (600 - 100) * 2
avg_color_boundary = np.uint8(
    (
        np.sum(image[roi_cc[0],100], axis=0)
        + np.sum(image[roi_cc[0],600-1], axis=0)
        + np.sum(image[100, roi_cc[1]], axis=0)
        + np.sum(image[600-1, roi_cc[1]], axis=0)
    ) / num_bdry_pixels
)
image[roi_cc[0], roi_cc[1]] = avg_color_boundary

# Restrict the image to the visible reservoir.
image = image[0:4400, 140:7850]

# Resize - do not require the high resolution for finding sand layers
#da_image.resize(0.1, 0.1)
image = cv2.resize(image, (0,0), fx = 0.1, fy = 0.1)

# Make daria image - the coordinate system currently does not make sense
da_image = da.Image(image, (0, 0), 250, 180)

def single_iteration(img: np.ndarray, gamma: float, k: int, val: float, active_set: np.ndarray) -> np.ndarray:

    img_work = img.copy()

    # TODO gamma before and after or only one of them?

    # Apply gamma correction
    img_work = skimage.exposure.adjust_gamma(img_work, gamma)

    # Blur the image
    #img_work = np.uint8(255.0 * skimage.restoration.denoise_tv_chambolle(img_work, weight=0.05, channel_axis=-1))

    # TODO? Apply gamma correction
    #img_work = skimage.exposure.adjust_gamma(img_work, gamma)

    # Flatten the image
    pixel_values = img_work.flatten() #reshape(-1)
   
    # Restrict to active set
    pixel_values = pixel_values[active_set.flatten()]

    # Convert format (required for kmeans routine
    pixel_values = np.float32(pixel_values)

    # number of clusters (K) - optimally number of sands, start with two trying to identify the dominant ESF sand layer
    compactness, labels, (centers) = cv2.kmeans(
        pixel_values,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    # Update active set - deactive where labels have value 0
    zero_labels = (labels==k-1)[:,0]
    active_set_flat = active_set.flatten()
    active_ids_flat = np.where(active_set_flat)[0]
    active_set_flat[active_ids_flat[zero_labels]] = False
    active_set = active_set_flat.reshape(active_set.shape)

    # Make marked zone white
    marked_set_flat = np.zeros_like(active_set_flat, dtype=bool)
    marked_set_flat[active_ids_flat[zero_labels]] = True
    marked_set = marked_set_flat.reshape(active_set.shape)
    #img[marked_set] = val

    # convert back to 8 bit values
    centers = np.uint8(centers)
    
    # flatten the labels array
    labels = labels.flatten()
    
    ## convert all pixels to the color of the centroids
    segmented_img = centers[labels.flatten()]
    ## reshape back to the original image dimension
    #segmented_img = segmented_img.reshape(img.shape)

    return img, img_work, segmented_img, active_set, marked_set

# Scale facies onto [0, ..., 255]
def scale_field(f):
    f = f.astype(float)
    f = np.uint8(255. * f / np.max(f))
    return f

# Workflow to determine the 
active_set = np.ones(image.shape, dtype=bool) # pixels which have not yet been assigned to any sand type
facies = np.zeros(image.shape, dtype=int)

# Define the image and apply some denoising as well as gamma correction
daria_denoising = False
if daria_denoising:
    da_image = da.tv_denoising(da_image, 0.05, 1, verbose=True)
    image = da_image.img
else:
    #image = np.uint8(255.0 * skimage.restoration.denoise_tv_chambolle(image, weight=0.2, channel_axis=-1))
    #image = np.uint8(255.0 * skimage.filters.median(image))
    #image = np.uint8(255.0 * skimage.filters.gaussian(image, sigma=1))
    image = np.uint8(255.0 * skimage.filters.rank.median(image, skimage.morphology.disk(5)))

# Gamma correction
#image = skimage.exposure.adjust_gamma(image, 0.05)

fig, ax = plt.subplots(3,3)

image, image_work, segmented_image, active_set, marked_set = single_iteration(image, gamma=3, k=4, val=255, active_set = active_set)
#marked_set = skimage.restoration.denoise_tv_chambolle(255 * marked_set.astype("uint8"), weight=0.05, channel_axis=-1) > 0.9
facies[marked_set] = 1

ax[0,0].imshow(image)
ax[1,0].imshow(scale_field(facies))
ax[2,0].imshow(image_color)

image, image_work, segmented_image, active_set, marked_set = single_iteration(image, gamma=3, k=2, val=255, active_set = active_set)
marked_set = skimage.restoration.denoise_tv_chambolle(255 * marked_set.astype("uint8"), weight=0.05, channel_axis=-1) > 0.5
facies[marked_set] = 2

ax[0,1].imshow(image)
ax[1,1].imshow(scale_field(facies))
ax[2,1].imshow(image_color)

image, image_work, segmented_image, active_set, marked_set = single_iteration(image, gamma=0.1, k=2, val=255, active_set = active_set)
marked_set = skimage.restoration.denoise_tv_chambolle(255 * marked_set.astype("uint8"), weight=0.05, channel_axis=-1) > 0.5
facies[marked_set] = 3

plt.plot(np.histogram(image.flatten(), bins=256)[0])

#ax[0,2].imshow(image)
#ax[1,2].imshow(scale_field(facies))
#ax[2,2].imshow(image_color)

plt.show()
