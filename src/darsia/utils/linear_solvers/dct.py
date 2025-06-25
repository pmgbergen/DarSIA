import numpy as np
import scipy

import darsia


class DCTSolver:
    # TODO: Move later to a different location
    def __init__(self, grid: darsia.Grid):
        if grid.dim == 2:
            # note the permutation of x and y in the grid
            ny, nx = grid.shape
            hy, hx = grid.voxel_size

            lambda_x = 2 * (np.cos(np.pi * np.arange(nx) / (nx - 1)) - 1)
            lambda_y = 2 * (np.cos(np.pi * np.arange(ny) / (ny - 1)) - 1)

            kernel_inv = np.outer(lambda_x / (hx**2), np.ones(ny)) + np.outer(
                np.ones(nx), lambda_y / (hy**2)
            )

        if grid.dim == 3:
            # note the permutation of x and y in the grid
            ny, nx, nz = grid.shape
            hy, hx, hz = grid.voxel_size

            lambda_z = 2 * (
                np.cos(np.pi * np.arange(grid.shape[2]) / (grid.shape[2] - 1)) - 1
            )
            kernel_inv = (
                np.outer(lambda_x / (hx**2), np.ones((ny, nz)))
                + np.outer(
                    np.ones(nx), lambda_y / (grid.voxel_size[0] ** 2), np.ones(nz)
                )
                + np.outer(np.ones((nx, ny)), lambda_z / (hz**2))
            )

        self.kernel = np.divide(1.0, kernel_inv)
        self.kernel[np.isclose(kernel_inv, 0)] = 0
        self.volume = np.prod(grid.voxel_size)
        self.grid = grid

    def solve(self, rhs, **kwargs):
        # de-integrate the rhs
        rhs_2d = np.reshape(rhs, self.grid.shape, order="F") / self.volume

        # take the discrete cosine transform
        dct = scipy.fft.dctn(-rhs_2d, type=2, norm="ortho")

        # multiply by the kernel
        np.multiply(dct, np.transpose(self.kernel), out=dct)

        # inverse transform
        sol_2d = scipy.fft.idctn(dct, type=2, norm="ortho")

        sol = np.ravel(sol_2d, order="F")

        return sol
