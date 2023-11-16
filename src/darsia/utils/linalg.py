"""Wrapper to Krylov subspace methods from SciPy."""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg, gmres
from petsc4py import PETSc


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


class KSP:
    def __init__(self, A: sps.csc_matrix) -> None:
        # store original matrix
        self.A = A

        # convert to petsc matrix
        self.A_petsc = PETSc.Mat().createAIJ(
            size=A.shape, csr=(A.indptr, A.indices, A.data)
        )

        # preallocate petsc vectors
        self.sol_petsc = self.A_petsc.createVecLeft()
        self.rhs_petsc = self.A_petsc.createVecRight()

    def setup(self, petsc_options: dict) -> None:
        self.prefix = "petsc_solver_"
        # TODO: define a unique name in case of multiple problems

        # create ksp solver and assign controls
        ksp = PETSc.KSP().create()
        ksp.setOperators(self.A_petsc)
        ksp.setOptionsPrefix(self.prefix)
        ksp.setConvergenceHistory()
        ksp.setFromOptions()
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        ksp.setConvergenceHistory()
        ksp.setFromOptions()

        self.ksp = ksp

        # associate petsc vectors to prefix
        self.A_petsc.setOptionsPrefix(self.prefix)
        self.A_petsc.setFromOptions()

        self.sol_petsc.setOptionsPrefix(self.prefix)
        self.sol_petsc.setFromOptions()

        self.rhs_petsc.setOptionsPrefix(self.prefix)
        self.rhs_petsc.setFromOptions()

    def solve(self, b: np.ndarray, **kwargs) -> np.ndarray:
        # convert to petsc vector the rhs
        self.rhs_petsc.setArray(b)
        self.ksp.solve(self.rhs_petsc, self.sol_petsc)
        sol = self.sol_petsc.getArray()
        # DEBUG CODE: check convergence
        # res = self.A.dot(sol) - b
        # print('residual norm: ', np.linalg.norm(res)/np.linalg.norm(b))
        return sol
