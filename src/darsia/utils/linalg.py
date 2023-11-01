"""Wrapper to Krylov subspace methods from SciPy."""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg, gmres


class CG:
    def __init__(self, A: sps.csc_matrix) -> None:
        self.A = A

    def solve(self, b: np.ndarray, **kwargs) -> np.ndarray:
        return cg(self.A, b, **kwargs)[0]


class GMRES:
    def __init__(self, A: sps.csc_matrix) -> None:
        self.A = A

    def solve(self, b: np.ndarray, **kwargs) -> np.ndarray:
        return gmres(self.A, b, **kwargs)[0]
