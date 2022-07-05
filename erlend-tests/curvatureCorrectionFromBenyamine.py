import cv2
import daria as da
import numpy as np
from scipy import ndimage, interpolate


DxDy = [284, 150]
vertical_bulge = 0.5e-6
horizontal_crop = 0.965
horizontal_bulge = -3.1e-6
horizontal_stretch = 2.35e-6
horizontal_stretch_mid = -0
vertical_shear = 2e-3
vertical_crop = 0.92

img = cv2.imread("images/originals/NonCurvCorrected.JPG")

Ny, Nx, _ = img.shape

img = img.astype(np.float64)

x = (np.array(range(1, Nx + 1)) - round(Nx / 2)) / round(Nx / 2) * DxDy[0] / 2
y = (np.array(range(1, Ny + 1)) - round(Ny / 2)) / round(Ny / 2) * DxDy[1] / 2
y = y.reshape((1, Ny))

xx, yy = np.meshgrid(x, y)

X = np.ones((Ny, 1)) * x
Y = np.transpose(y) * np.ones((1, Nx))


points = np.zeros((Ny, Nx, 2), dtype=np.float64)
points[:, :, 0] = X
points[:, :, 1] = Y


Xmod = horizontal_crop * np.multiply(
    np.multiply(X, (1 + horizontal_stretch * (X - horizontal_stretch_mid) ** 2)),
    (1 - vertical_bulge * Y**2),
)
Ymod = vertical_crop * (
    np.multiply(Y, (1 - horizontal_bulge * X**2)) - vertical_shear * X
)

new_points = np.zeros((Ny, Nx, 2), dtype=np.float64)
new_points[:, :, 0] = Xmod
new_points[:, :, 1] = Ymod


# print(len(X.ravel()))
# print(len(Y.ravel()))
# print(len(img[:, :, 0].ravel()))

ip0 = interpolate.interp2d(x, y, img[:, :, 0])

# ip1 = interpolate.interp2d(x, y, img[:, :, 1])
# ip2 = interpolate.interp2d(x, y, img[:, :, 2])

# ip0 = interpolate.interpn((X.ravel(), Y.ravel()), img[:, :, 0].ravel(), (1, 1))

# x = np.arange(0, 1, 0.02)
# y = np.arange(0, 1, 0.02)
# xx, yy = np.meshgrid(x, y)

# z = np.sin(x + y)

# ip = interpolate.interp2d(x, y, z)


# ip0 = interpolate.interp2d(
#     np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([5, 6, 7])
# )

# print(ip0)


# img_mod = np.zeros((Ny, Nx, 3), dtype=np.uint8)
# img_mod[:, :, 0] = ip0(Xmod, Ymod)
# img_mod[:, :, 1] = ip1(Xmod, Ymod)
# img_mod[:, :, 2] = ip2(Xmod, Ymod)


# img_mod = interpolate.interpn(points, img, new_points)

# # for i in range(3):
# #     print(i)
# #     img_mod[:, :, i] = interpolate.griddata(
# #         points.reshape(Ny * Nx, 2), img[:, :, i].reshape(Ny * Nx, 1), (Xmod, Ymod)
# #     ).astype(np.uint8)

cv2.imwrite("images/modified/CurvCorrected.jpg", img_mod)
