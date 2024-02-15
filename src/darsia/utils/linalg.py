"""Wrapper to Krylov subspace methods from SciPy."""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg, gmres
from petsc4py import PETSc
import matplotlib.pyplot as plt


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
    def __init__(self, A: sps.csc_matrix, 
                 field_ises=None, 
                 nullspace=None) -> None:
        # store original matrix
        self.A = A

        # convert to petsc matrix
        self.A_petsc = PETSc.Mat().createAIJ(
            size=A.shape, csr=(A.indptr, A.indices, A.data)
        )

        # set nullspace
        if nullspace is not None:
            # convert to petsc vectors
            self._comm = PETSc.COMM_WORLD
            for i, v in enumerate(nullspace):
                self._petsc_kernels[i] = PETSc.Vec().createWithArray(v, comm=self._comm)
            self._nullspace = PETSc.NullSpace().create(constant=None,
                                                   vectors=self._petsc_kernels,
                                                   comm=self._comm)
            self.A_petsc.setNullSpace(self._nullspace)

        # preallocate petsc vectors
        self.sol_petsc = self.A_petsc.createVecLeft()
        self.rhs_petsc = self.A_petsc.createVecRight()

        # set field_ises
        if field_ises is not None:
            self.field_ises = field_ises
        else:
            self.field_ises = None

        

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

        # TODO: set field split in __init__
        if self.field_ises is not None: 
            pc = ksp.getPC()
            pc.setFromOptions()
            #pc.setFieldSplitIS(('0',self.pot_is),('1',self.tdens_is))
            pc.setFieldSplitIS(*self.field_ises)
            pc.setOptionsPrefix(self.prefix)
            pc.setFromOptions()

    def solve(self, b: np.ndarray, **kwargs) -> np.ndarray:
        # convert to petsc vector the rhs
        self.rhs_petsc.setArray(b)
        self.ksp.solve(self.rhs_petsc, self.sol_petsc)
        sol = self.sol_petsc.getArray()

        # solve using scipy
        #sol2, _ = gmres(self.A, b)
        #res = self.A.dot(sol2) - b
        #print('residual norm: ', np.linalg.norm(res)/np.linalg.norm(b))

        # DEBUG CODE: check convergence
        #res = self.A.dot(sol) - b
        #print('residual norm: ', np.linalg.norm(res)/np.linalg.norm(b))
        # plot residual
        #plt.plot(sol)
        #plt.plot(sol2, linestyle='--')
        #plt.show()
        return sol
