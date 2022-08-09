import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import cv2

# White sand mid rig image
image = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Remove boundary and color checker
image = image[280:4400, 380:6500]

plt.imshow(image)
plt.show()

# Build an array of labels for training the segmentation.
training_labels = np.zeros(image.shape[:2], dtype=np.uint8)
training_labels[:400] = 1
training_labels[500:770, 2150:2700] = 2
training_labels[1560:1780, 900:1530] = 2
training_labels[650:1350, 190:1850] = 3
training_labels[850:1700, 3200:5700] = 3
training_labels[1960:2400, 870:1750] = 4 
training_labels[2250:2700, 3650:4900] = 4 
training_labels[2550:2700, 920:1350] = 1
training_labels[2850:3400, 1200:2200] = 3 

training_labels[500:700, 3000:3200] = 2
training_labels[3000:3100, 3000:3150] = 1
training_labels[2000:2250, 3000:3300] = 2
training_labels[3150:3200, 3000:3050] = 3
training_labels[3600:4000, 3200:3500] = 4

full_image = image.copy()

#image = image[:, 1000:4000]
#image = skimage.util.img_as_ubyte(skimage.exposure.adjust_gamma(image, 2.2))
#image = skimage.filters.gaussian(image, sigma=0.1)
#training_labels = training_labels[:, 1000:4000]

image = image[:, 3000:4000]
training_labels = training_labels[:, 3000:4000]

plt.imshow(image)
plt.show()

sigma_min = 0.1
sigma_max = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True,
                        edges=False,
                        texture=True,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        channel_axis=-1)
features = features_func(image)
print(features.shape)

clf = RandomForestClassifier(
    n_estimators=50, # 100?
    n_jobs=3,
    max_depth=50,
    max_samples=0.2,
)
clf = future.fit_segmenter(training_labels, features, clf)
result = future.predict_segmenter(features, clf)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(image, result, mode='thick'))
ax[0].contour(training_labels)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()

# Apply the learned model to the full image

features_full = features_func(full_image)
result_full = future.predict_segmenter(features_full, clf)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
ax[0].imshow(segmentation.mark_boundaries(full_image, result_full, mode='thick'))
ax[0].set_title('Image')
ax[1].imshow(result_full)
ax[1].set_title('Segmentation')
fig.tight_layout()

plt.show()
