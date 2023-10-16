from abc import ABC
import numpy as np


class BaseKernel(ABC):
    def __call__(self, x, y):
        pass


# define linear kernel shifted to avoid singularities
class LinearKernel(BaseKernel):
    def __init__(self, a=0):
        self.a = a

    def __call__(self, x, y):
        return np.sum(np.multiply(x, y), axis=-1) + self.a


# define gaussian kernel
class GaussianKernel(BaseKernel):
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, x, y):
        return np.exp(-self.gamma * np.sum(np.multiply(x - y, x - y), axis=-1))
