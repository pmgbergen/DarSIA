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


def _make_reasons(reasons):
    return dict([(getattr(reasons, r), r)
                 for r in dir(reasons) if not r.startswith("_")])
KSPreasons = _make_reasons(PETSc.KSP.ConvergedReason())

class KSP:
    def __init__(self, A: sps.csc_matrix, 
                 field_ises=None, 
                 nullspace=None,
                 appctx: dict=None
                 ) -> None:
        
        # convert csc to csr (if needed)
        if not sps.isspmatrix_csr(A):
            A = A.tocsr()

        # store original matrix
        self.A = A

        # convert to petsc matrix
        self.A_petsc = PETSc.Mat().createAIJ(
            size=A.shape, csr=(A.indptr, A.indices, A.data)
        )

        # This step is needed to avoid a petsc error with LU factorization
        # that explicitly needs a zeros along the diagonal
        # TODO: use zero_diagonal = PETSc.Mat().createConstantDiagonal(self.A.size,0.0)
        zero_diagonal = PETSc.Mat().createAIJ(
            size=A.shape, 
            csr=(np.arange(A.shape[0]+1,dtype="int32"),
                np.arange(A.shape[0],dtype="int32"),
                np.zeros(A.shape[0]))
        )
        self.A_petsc += zero_diagonal

        # set nullspace
        self._petsc_kernels = []
        if nullspace is not None:
            # convert to petsc vectors
            self._comm = PETSc.COMM_WORLD
            for v in nullspace:
                p_vec = PETSc.Vec().createWithArray(v, comm=self._comm)
                self._petsc_kernels.append(p_vec)
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

        self.appctx = appctx
        
        

    def setup(self, petsc_options: dict) -> None:
        petsc_options = flatten_parameters(petsc_options)


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
        if (self.field_ises) is not None and ("fieldsplit" in petsc_options["pc_type"]):
            pc = ksp.getPC()
            pc.setFromOptions()
            # syntax is
            # pc.setFieldSplitIS(('0',is_0),('1',is_1))
            pc.setFieldSplitIS(*self.field_ises)
            pc.setOptionsPrefix(self.prefix)
            pc.setFromOptions()
            pc.setUp()

            
            # split subproblems
            ksps = pc.getFieldSplitSubKSP()
            for k in ksps:
                # Without this, the preconditioner is not set up
                # It works for now, but it is not clear why
                p = k.getPC()
                # This is in order to pass the appctx 
                # to all preconditioner. TODO: find a better way
                p.setAttr('appctx', self.appctx) 
                p.setUp()

            # set nullspace
            if self._nullspace is not None:
                if len(self._petsc_kernels) > 1:
                        raise NotImplementedError("Nullspace currently works with one kernel only")
                # assign nullspace to each subproblem
                for i, local_ksp in enumerate(ksps):
                    for k in self._petsc_kernels:
                        sub_vec = k.getSubVector(self.field_ises[i][1])
                        if sub_vec.norm() > 0:
                            local_nullspace = PETSc.NullSpace().create(constant=False, vectors=[sub_vec])
                            A_i, _ = local_ksp.getOperators()
                            A_i.setNullSpace(local_nullspace)
                

        # attach info to ksp
        self.ksp.setAttr('appctx', self.appctx)        

    def solve(self, b: np.ndarray, **kwargs) -> np.ndarray:
        # convert to petsc vector the rhs
        self.rhs_petsc.setArray(b)

        # check if the rhs is orthogonal to the nullspace
        for k in self._petsc_kernels:
            if abs(self.rhs_petsc.dot(k)) > 1e-10:
                raise ValueError("RHS not ortogonal to the nullspace")
        
        # solve
        self.ksp.solve(self.rhs_petsc, self.sol_petsc)

        # check if the solver worked
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise ValueError(f"KSP solver failed {reason=} {KSPreasons[reason]}") 
        
        # convert to numpy array
        sol = self.sol_petsc.getArray()



        # DEBUG CODE: check convergence in numpy varaibles  
        #res = self.A.dot(sol) - b
        #print('residual norm: ', np.linalg.norm(res)/np.linalg.norm(b))
        return sol
    
    def kill(self):
        """
        Free memory
        """
        self.ksp.destroy()
        self.A_petsc.destroy()
        self.sol_petsc.destroy()
        self.rhs_petsc.destroy()
        if hasattr(self,'_nullspace'):
            self._nullspace.destroy()
            for k in self._petsc_kernels:
                k.destroy()
        self.A_petsc = None
        self.sol_petsc = None
        self.rhs_petsc = None
        self.ksp = None
        self._nullspace = None
        self._petsc_kernels = None

#
# Following code is taken from Firedrake
#           
def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            print("Ignoring duplicate option: %s (existing value %s, new value %s)",
                    option, new[option], value)
        new[option] = value
    return new
