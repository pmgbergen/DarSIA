import numpy as np
import scipy

import darsia


class DCTSolver:
    """
    A solver for the Poisson equation with zero Neumann boundary conditions.
    This solver uses the discrete cosine transform (DCT) to efficiently solve the
    Poisson equation on a rectangular grid.
    Args:
        grid (darsia.Grid): The grid on which to solve the Poisson equation.    
    """ 
    def __init__(self, grid: darsia.Grid):
        """
        Initialize the Poisson solver with the given grid.
        Args:
            grid (darsia.Grid): The grid on which to solve the Poisson equation.
        """
        if grid.dim == 2:
            # note the permutation of x and y in the grid
            ny, nx = grid.shape
            #hy, hx = 1.0 / (nx-1) , 1.0/(ny-1) #grid.voxel_size
            hy, hx = 1.0 / nx , 1.0 / ny #grid.voxel_size
            

            lambda_x = 2 * (np.cos(np.pi * np.arange(nx) / nx) - 1)
            lambda_y = 2 * (np.cos(np.pi * np.arange(ny) / ny) - 1)

            kernel_inv = (
                np.outer(lambda_x / (hx**2), np.ones(ny)) 
                + np.outer(np.ones(nx), lambda_y / (hy**2))
            )
            zero_mode_index = (0,0)
            kernel_inv[zero_mode_index]=1.0

        if grid.dim == 3:
            raise NotImplementedError(f"Dimension {grid.dim} not supported.")
            # note the permutation of x and y in the grid
            ny, nx, nz = grid.shape
            hy, hx, hz = hy, hx = 1.0 / (nx-1) , 1.0/(ny-1), 1.0/ (nz-1) #grid.voxel_size

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
            zero_mode_index = (0,0,0)
            kernel_inv[zero_mode_index]=1.0

        self.kernel = np.divide(1.0, kernel_inv)
        #self.kernel[np.isclose(kernel_inv, 0)] = 0
        self.kernel[zero_mode_index] = 0
        self.volume = np.prod(grid.voxel_size)
        self.grid = grid

    def solve(self, rhs, **kwargs):
        """
        Solve the Poisson equation -Delta u = rhs with zero Neumann boundary conditions
        
        Args:
            rhs (np.ndarray): Right-hand side of the Poisson equation. MUST BE INTEGRATED.(1D array)
        Returns:
            np.ndarray: Solution to the Poisson equation (1D array)
        """
        # de-integrate the rhs
        rhs_2d = np.reshape(rhs, self.grid.shape, order="F") / self.volume

        # take the discrete cosine transform
        dct = scipy.fft.dctn(-rhs_2d, type=2, orthogonalize=True , norm="ortho")

        # multiply by the kernel
        np.multiply(dct, np.transpose(self.kernel), out=dct)

        # inverse transform
        sol_2d = scipy.fft.idctn(dct, type=2, orthogonalize=True, norm="ortho")

        sol = np.ravel(sol_2d, order="F")

        return sol
