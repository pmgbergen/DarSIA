import numpy as np


class KernelInterpolation:
    """General kernel-based interpolation."""

    def __init__(self, kernel, colours, concentrations):

        self.kernel = kernel

        num_data = colours.shape[0]
        assert len(concentrations) == num_data, "Input data not compatible."

        x = np.array(colours)  # data points / control points / support points
        y = np.array(concentrations)  # goal points
        X = np.ones((num_data, num_data))  # kernel matrix
        for i in range(num_data):
            for j in range(num_data):
                X[i, j] = self.kernel(x[i], x[j])

        alpha = np.linalg.solve(X, y)

        # Cache
        self.x = x
        self.alpha = alpha

    def __call__(self, signal: np.ndarray):
        """Apply interpolation."""
        ph_image = np.zeros(signal.shape[:2])
        for n in range(self.alpha.shape[0]):
            ph_image += self.alpha[n] * self.kernel(signal, self.x[n])
        return ph_image
