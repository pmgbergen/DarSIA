import cv2
from scipy.ndimage import map_coordinates
import numpy as np


def curvature_correction(
    image: np.ndarray,
    width: float = 284,
    height: float = 150,
    vertical_bulge: float = 0.5e-6,
    horizontal_crop: float = 0.965,
    horizontal_bulge: float = -3.1e-6,
    horizontal_stretch: float = 2.35e-6,
    horizontal_stretch_mid: float = -0,
    vertical_shear: float = 2e-3,
    vertical_crop: float = 0.92,
    interpolation_order: int = 1,
) -> np.ndarray:

    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, str):
        img = cv2.imread(image)
    else:
        raise Exception(
            "Invalid image data. Provide either a path to an image or an image array."
        )

    Ny, Nx, _ = img.shape

    # imgfloat = img.astype(np.float64)

    # Center image, and set physical values
    x = (np.array(range(1, Nx + 1)) - round(Nx / 2)) / round(Nx / 2) * width / 2
    y = (np.array(range(1, Ny + 1)) - round(Ny / 2)) / round(Ny / 2) * height / 2
    y = y.reshape((1, Ny))

    # Original grid
    X, Y = np.meshgrid(x, y)

    # Curvature corrected grid
    Xmod = horizontal_crop * np.multiply(
        np.multiply(X, (1 + horizontal_stretch * (X - horizontal_stretch_mid) ** 2)),
        (1 - vertical_bulge * Y**2),
    )
    Ymod = vertical_crop * (
        np.multiply(Y, (1 - horizontal_bulge * X**2)) - vertical_shear * X
    )

    # Map corrected grid back to positional arguments
    Xmod = Xmod * Nx / width + round(Nx / 2) - 1
    Ymod = Ymod * Ny / height + round(Ny / 2) - 1

    # Create out grid as the corrected grid, but not in meshgrid format
    out_grid = np.array([Ymod.ravel(), Xmod.ravel()])

    # Return image
    img_mod = np.zeros((Ny, Nx, 3), dtype=np.uint8)

    # Do interpolate original image on the new grid
    for i in range(img[0, 0, :].size):
        in_data = img[:, :, i]
        im_array = map_coordinates(in_data, out_grid, order=interpolation_order)
        img_mod[:, :, i] = im_array.reshape(img[:, :, 0].shape).astype(np.uint8)

    return img_mod
