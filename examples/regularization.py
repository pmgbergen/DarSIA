import darsia as da
import numpy as np
import matplotlib.pyplot as plt

# Load 3D image
im = np.load("C:/Users/erst/src/image_analysis/images/dicom_3d_test.npy")

# Make regularization parameter heterogenous (for testing purposes)
mu = np.ones_like(im)
mu[:, 0:100, :] = 0.5


# Regularize image using anisotropic tvd
im_regularized_tvd = da.split_bregman_anisotropic_tvd(
    im=im,
    mu=mu,
    omega=0.1,
    maxiter=30,
    tol=1e-6,
    verbose=True,
    dim=3,
    solver_type="cg",  # "mg", "jacobi" also possible
    solver_iter=20,
)

# Regularize image using l2 regularization
im_regularized_l2 = da.L2_regularization(
    im=im,
    mu=1,
    maxiter=30,
    tol=1e-6,
    verbose=True,
    dim=3,
    solver_type="jacobi",  # "mg", "cg" also possible
    mg_depth=3,
)

plt.figure("im 100")
plt.imshow(im[100, :, :])
plt.figure("regularized tvd  100")
plt.imshow(im_regularized_tvd[100, :, :])
plt.figure("regularized l2 100")
plt.imshow(im_regularized_l2[100, :, :])
plt.show()
