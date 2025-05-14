import numpy as np
import scipy

from darsia import Grid


class DCTSolver(Grid):
    # TODO: Move later to a different location
    def __init__(self, grid):
        lambda_x = 2 * (np.cos(np.pi * np.arange(grid.shape[1]) / (grid.shape[1] - 1)) - 1)
        lambda_y = 2 * (np.cos(np.pi * np.arange(grid.shape[0]) / (grid.shape[0] - 1)) - 1)
        kernel_inv = np.outer(lambda_x / (grid.voxel_size[1] ** 2), np.ones(grid.shape[0])) + np.outer(
        np.ones(grid.shape[1]), lambda_y / (grid.voxel_size[0] ** 2))
        self.kernel = np.divide(1., kernel_inv)
        self.kernel[np.isclose(kernel_inv, 0)] = 0
        self.volume = grid.voxel_size[1] * grid.voxel_size[0]
        self.grid = grid

    def solve(self, rhs, **kwargs):
        # take the discrete cosine transform

        rhs_2d = np.reshape(rhs[:-1], self.grid.shape, order="F") / self.volume

        dct = scipy.fft.dctn(-rhs_2d, type=2, norm='ortho')

        # multiply by the kernel
        np.multiply(dct, np.transpose(self.kernel), out=dct)

        # inverse transform
        sol_2d = scipy.fft.idctn(dct, type=2, norm='ortho')

        sol = np.ravel(sol_2d, order="F")
        sol = np.append(sol, [0])

        return sol
