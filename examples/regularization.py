import os

import matplotlib.pyplot as plt
import numpy as np

import darsia

# Load 3D image
image_folder = f"{os.path.dirname(__file__)}/images/"
img_path = image_folder + "dicom_3d.npy"
img = darsia.imread(img_path, dimensions=[0.1, 0.1, 0.1], dim=3)

print("Image shape: ", img.img.shape)

# Make regularization parameter heterogenous and anisotropic (for illustration purposes)
mu = np.ones_like(img.img)
mu[:, 0:100, :] = 0.5

mu_anisotropic = [10 * mu, mu, mu]

# Regularize image using anisotropic tvd
img_regularized_anisotropic_tvd = darsia.tvd(
    img=img,
    method="heterogeneous bregman",
    isotropic=False,
    weight=mu_anisotropic,
    omega=0.1,
    max_num_iter=30,
    eps=1e-6,
    dim=3,
    verbose=True,
    solver=darsia.Jacobi(maxiter=20),
    # solver=darsia.CG(maxiter=20, tol=1e-3),
)

img_regularized_tvd = darsia.tvd(
    img=img.img,
    method="heterogeneous bregman",
    isotropic=False,
    weight=mu,
    omega=0.1,
    max_num_iter=30,
    eps=1e-6,
    dim=3,
    verbose=True,
    solver=darsia.Jacobi(maxiter=20),
    # solver=darsia.CG(maxiter=20, tol=1e-3),
)

# Regularize image using H1 regularization
img_regularized_h1 = darsia.H1_regularization(
    img=img,
    mu=1,
    omega=1,
    dim=3,
    solver=darsia.Jacobi(maxiter=30, tol=1e-4, verbose=True),
    # solver = darsia.MG(3)
)

plt.figure("img slice")
plt.imshow(img.img[100, :, :])
plt.figure("regularized anisotropic weight tvd slice")
plt.imshow(img_regularized_anisotropic_tvd.img[100, :, :])
plt.figure("regularized tvd slice")
plt.imshow(img_regularized_tvd[100, :, :])
plt.figure("regularized H1 slice")
plt.imshow(img_regularized_h1.img[100, :, :])
plt.show()
