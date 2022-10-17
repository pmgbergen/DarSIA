from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import daria as da

from matplotlib.cm import get_cmap

# cmap = get_cmap('Spectral')

# rgba = 255*np.array(cmap(0.1)[0:3])
# rgba = rgba.astype(np.uint8)
# print(rgba)

# img1 = np.array([[1,0,1,2],[1,1,0,1],[1,1,2,1],[1,1,2,1]])
# img2 = np.array([[0,1,1,2],[1,0,0,0],[1,0,2,1],[1,0,2,1]])
# img3 = np.array([[1,2,1,2],[0,0,2,0],[0,0,2,1],[0,0,2,1]])




im1 = np.load(Path("../segmentation_70.npy"))
im2 = np.load(Path("../segmentation_90.npy"))

comp_im = da.compare_segmentations(
    im1,
    im2,
    gray_colors=np.array([[100, 100, 100], [255, 255, 255], [180, 180, 180]]),
)
plt.imshow(comp_im)
plt.show()



# print(comp_im)
