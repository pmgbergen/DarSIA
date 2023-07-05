import os

import matplotlib.pyplot as plt
import numpy as np

import darsia

# Load 3D image
image_folder = f"{os.path.dirname(__file__)}/images/"
img_path = image_folder + "dicom_3d.npy"
img = darsia.imread(img_path, dimensions=[0.1, 0.1, 0.1])

# Make regularization parameter heterogenous (for illustration purposes)
mu = np.ones_like(img.img)
mu[:, 0:100, :] = 0.5

# Regularize image using anisotropic tvd
img_regularized_tvd = darsia.tvd(
    img=img,
    method="heterogeneous bregman",
    isotropic=False,
    weight=mu,
    omega=0.1,
    max_num_iter=30,
    eps=1e-6,
    dim=3,
    verbose=True,
    # solver = darsia.CG(maxiter = 20, tol = 1e-3)
    solver=darsia.Jacobi(maxiter=20),
)

# Regularize image using l2 regularization
img_regularized_l2 = darsia.L2_regularization(
    img=img,
    mu=1,
    omega=1,
    dim=3,
    solver=darsia.Jacobi(maxiter=30, tol=1e-6, verbose=True),
    # solver = darsia.MG(3)
)

plt.figure("img slice")
plt.imshow(img.img[100, :, :])
plt.figure("regularized tvd slice")
plt.imshow(img_regularized_tvd.img[100, :, :])
plt.figure("regularized l2 slice")
plt.imshow(img_regularized_l2.img[100, :, :])
plt.show()
