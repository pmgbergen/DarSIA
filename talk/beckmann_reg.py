"""Regularized heterogeneous Beckmann problem on 2d tensor-product grids.

Standalone (numpy/scipy only) implementation for the talk experiments; does not
touch the DarSIA solvers.

Problem:
    min_p  J_h(p)   s.t.  D p = f,
with face fluxes p (RT0, normal components on interior faces, q.n = 0 on the
boundary), cell potentials phi (P0) and the regularization

    Psi_eps(s) = s - eps * ln(1 + s/eps),   Psi_eps'(s) = s / (s + eps).

Energy variants (|omega| = hx*hy is the dual cell of every interior face):
    "N": J = sum_faces |omega| kinv_i Psi_eps(|p_i|)                    (l1 / graph)
    "F": J = 1/2 sum_faces |omega| kinv_i Psi_eps(sqrt(p_i^2 + tbar_i^2))
         tbar_i = average of the 4 adjacent orthogonal face fluxes; the factor
         1/2 = 1/d compensates that the x- and y-dual cells each tile Omega.
    "C": J = sum_cells |K| kinv_K Psi_eps(|q_K|),
         q_K = (average of the two x-faces, average of the two y-faces).

kinv = 1/kappa. On faces, kinv is the arithmetic mean of the two cell values
(i.e. harmonic averaging of kappa), as in DarSIA.

Sign convention (as in the brief): optimality reads  g(p) - D^T phi = 0,
with g = M^{-1} grad J and M = |omega| I the lumped face mass matrix.

Array layout: cells (nx, ny) flattened in C order; x-faces (nx-1, ny), y-faces
(nx, ny-1); p = concat(px.ravel(), py.ravel()).

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sps
import scipy.sparse.linalg  # noqa: F401  (sps.linalg)

VARIANTS = ("N", "F", "C")


def psi(s: np.ndarray, eps: float) -> np.ndarray:
    """Psi_eps(s) = s - eps ln(1 + s/eps); Psi_0(s) = s."""
    if eps == 0:
        return s
    return s - eps * np.log1p(s / eps)


def _diff(n: int) -> sps.csr_matrix:
    """(n x n-1) cell <- interior face difference: (d p)_i = p_{i+1/2} - p_{i-1/2}."""
    return sps.diags([np.ones(n - 1), -np.ones(n - 1)], [0, -1], shape=(n, n - 1)).tocsr()


def _face_avg(n: int) -> sps.csr_matrix:
    """(n-1 x n) interior face <- average of the two adjacent cells."""
    return sps.diags([0.5 * np.ones(n - 1), 0.5 * np.ones(n - 1)], [0, 1], shape=(n - 1, n)).tocsr()


class RegularizedBeckmann:
    """Discrete regularized Beckmann problem for one energy variant."""

    def __init__(
        self,
        shape: tuple[int, int],
        h: tuple[float, float],
        variant: str = "F",
        eps: float = 1e-2,
        kappa: np.ndarray | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.shape = nx, ny = shape
        self.h = hx, hy = h
        self.variant = variant
        self.eps = eps
        self.cell_volume = hx * hy
        """|K| = |omega| for all cells and interior faces."""

        self.num_cells = nx * ny
        self.num_xfaces = (nx - 1) * ny
        self.num_yfaces = nx * (ny - 1)
        self.num_faces = self.num_xfaces + self.num_yfaces

        Ix, Iy = sps.identity(nx), sps.identity(ny)

        # Divergence (density form): (D p)_K = (px_E - px_W)/hx + (py_N - py_S)/hy
        self.Dx = sps.kron(_diff(nx) / hx, Iy).tocsr()
        self.Dy = sps.kron(Ix, _diff(ny) / hy).tocsr()
        self.D = sps.hstack([self.Dx, self.Dy]).tocsr()
        self.DT = self.D.T.tocsr()

        # Face <- cell averaging and its transpose (cell <- its two faces, 1/2 each)
        self.Bx = sps.kron(_face_avg(nx), Iy).tocsr()
        self.By = sps.kron(Ix, _face_avg(ny)).tocsr()

        # Tangential average for (F): x-face <- 4 adjacent y-faces, and transpose
        self.Tyx = sps.kron(_face_avg(nx), _face_avg(ny).T).tocsr()
        self.Txy = self.Tyx.T.tocsr()

        # Conductivity
        if kappa is None:
            kappa = np.ones(shape)
        kappa = np.asarray(kappa, dtype=float)
        assert kappa.shape == shape and np.all(kappa > 0)
        self.kappa = kappa
        self.kinv_cells = (1.0 / kappa).ravel()
        self.kinv_faces = np.concatenate(
            [self.Bx @ self.kinv_cells, self.By @ self.kinv_cells]
        )
        self.kappa_min = float(kappa.min())
        self.kappa_max = float(kappa.max())

        # Eigenvalues of D D^T (Neumann Laplacian, diagonalized by DCT-II)
        lx = (2 - 2 * np.cos(np.pi * np.arange(nx) / nx)) / hx**2
        ly = (2 - 2 * np.cos(np.pi * np.arange(ny) / ny)) / hy**2
        lam = lx[:, None] + ly[None, :]
        lam[0, 0] = 1.0
        self._inv_lam = 1.0 / lam
        self._inv_lam[0, 0] = 0.0

    # ---- Flux helpers ----

    def split(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return p[: self.num_xfaces], p[self.num_xfaces :]

    def cell_flux(self, p: np.ndarray) -> np.ndarray:
        """Cell vectors q_K, shape (num_cells, 2)."""
        px, py = self.split(p)
        return np.stack([self.Bx.T @ px, self.By.T @ py], axis=-1)

    def flux_magnitude(self, p: np.ndarray) -> np.ndarray:
        """|q| at the quadrature locations of the variant (faces or cells)."""
        px, py = self.split(p)
        match self.variant:
            case "N":
                return np.abs(p)
            case "F":
                t = np.concatenate([self.Tyx @ py, self.Txy @ px])
                return np.sqrt(p**2 + t**2)
            case "C":
                return np.linalg.norm(self.cell_flux(p), axis=-1)

    # ---- Energy and gradient ----

    def _weights(self) -> tuple[np.ndarray, float]:
        """kinv at quadrature locations and the variant's scaling factor."""
        if self.variant == "C":
            return self.kinv_cells, 1.0
        return self.kinv_faces, (0.5 if self.variant == "F" else 1.0)

    def energy(self, p: np.ndarray, eps: float | None = None) -> float:
        """J_h(p) with Psi_eps; eps=0 gives the unregularized discrete distance."""
        eps = self.eps if eps is None else eps
        kinv, scale = self._weights()
        s = self.flux_magnitude(p)
        return scale * self.cell_volume * float(np.sum(kinv * psi(s, eps)))

    def gradient(self, p: np.ndarray) -> np.ndarray:
        """g(p) = M^{-1} grad J_h(p). Uses Psi_eps'(s)/s = 1/(s + eps)."""
        kinv, scale = self._weights()
        s = self.flux_magnitude(p)
        c = kinv / (s + self.eps)
        px, py = self.split(p)
        match self.variant:
            case "N":
                return c * p
            case "F":
                cx, cy = self.split(c)
                tx, ty = self.Tyx @ py, self.Txy @ px
                gx = cx * px + self.Txy.T @ (cy * ty)
                gy = cy * py + self.Tyx.T @ (cx * tx)
                return scale * np.concatenate([gx, gy])
            case "C":
                q = self.cell_flux(p)
                return np.concatenate(
                    [self.Bx @ (c * q[:, 0]), self.By @ (c * q[:, 1])]
                )

    def lagged_operator(self, p: np.ndarray) -> sps.csr_matrix:
        """A(p) with g(p) = A(p) p, frozen weights c = kinv / (|q| + eps).

        q -> 1/2 q^T M A(p0) q is the quadratic majorizer of J at p0 (valid since
        Psi_eps'(s)/s is nonincreasing), so lagged diffusivity decreases J
        monotonically for all variants.
        """
        kinv, scale = self._weights()
        c = kinv / (self.flux_magnitude(p) + self.eps)
        match self.variant:
            case "N":
                return sps.diags(c).tocsr()
            case "F":
                cx, cy = self.split(c)
                Ax = sps.diags(cx) + self.Txy.T @ sps.diags(cy) @ self.Txy
                Ay = sps.diags(cy) + self.Tyx.T @ sps.diags(cx) @ self.Tyx
                return (scale * sps.block_diag([Ax, Ay])).tocsr()
            case "C":
                C = sps.diags(c)
                return sps.block_diag(
                    [self.Bx @ C @ self.Bx.T, self.By @ C @ self.By.T]
                ).tocsr()

    def lipschitz(self) -> float:
        """L_N = 1/(kappa_min eps): bound on the Hessian of J in the M-metric.

        Holds for all three variants (Psi_eps'' <= 1/eps, Psi_eps'(s)/s <= 1/eps;
        the averaging operators have norm <= 1).
        """
        return 1.0 / (self.kappa_min * self.eps)

    # ---- Linear algebra ----

    def solve_DDT(self, r: np.ndarray) -> np.ndarray:
        """Solve D D^T phi = r (zero-mean r) with DCT-II, O(N log N)."""
        r_hat = sfft.dctn(r.reshape(self.shape), type=2, norm="ortho")
        return sfft.idctn(r_hat * self._inv_lam, type=2, norm="ortho").ravel()

    def darcy_init(self, f: np.ndarray) -> np.ndarray:
        """Plain Darcy flux (|q| and kappa dropped): min ||p||_M s.t. D p = f."""
        return self.DT @ self.solve_DDT(f)

    def kkt_residual(self, p: np.ndarray) -> tuple[float, np.ndarray]:
        """Relative KKT residual ||g - D^T phi*|| / ||g|| for p with D p = f.

        phi* is the least-squares potential; the residual is the divergence-free
        part of g, the only part not absorbed by the constraint.
        """
        g = self.gradient(p)
        phi = self.solve_DDT(self.D @ g)
        return float(np.linalg.norm(g - self.DT @ phi) / np.linalg.norm(g)), phi

    def dual_value(self, phi: np.ndarray, f: np.ndarray) -> float:
        """sum_K |K| f_K phi_K."""
        return self.cell_volume * float(f @ phi)


@dataclass
class History:
    energy: list[float] = field(default_factory=list)
    energy0: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    dual: list[float] = field(default_factory=list)
    time: list[float] = field(default_factory=list)
    aa_rejected: int = 0
    iterations: int = 0
    """Number of nonlinear iterations (updates) performed."""
    eps: float = float("nan")
    linear_iterations: list[int] = field(default_factory=list)
    time_setup: float = 0.0
    """Accumulated linear solver setup time (factorization / AMG hierarchy)."""
    time_solve: float = 0.0
    """Accumulated linear solve time."""


def l_scheme(
    problem: RegularizedBeckmann,
    f: np.ndarray,
    L: float | None = None,
    tol: float = 1e-6,
    maxiter: int = 10_000,
    p0: np.ndarray | None = None,
    verbose: bool = False,
    max_time: float = np.inf,
    record_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, History, bool]:
    """L-scheme:  L p^k - D^T phi^k = L p^{k-1} - g(p^{k-1}),  D p^k = f.

    Schur complement: D D^T phi^k = L (f - D p^{k-1}) + D g(p^{k-1}), solved by
    DCT. With D p^{k-1} = f, phi^k is the least-squares potential of
    g(p^{k-1}), so the KKT residual of p^{k-1} is available for free:
    ||g - D^T phi^k|| = L ||p^k - p^{k-1}||. (Equivalently: projected gradient
    descent in the M-metric with step 1/L.)

    Returns p, phi, history, converged. f is the cell density mu - gamma
    (zero mean). Energies are recorded every record_every iterations (and at
    the end); the run stops unconverged after max_time seconds.
    """
    L = problem.lipschitz() if L is None else L
    f = f.ravel() - f.mean()
    p = problem.darcy_init(f) if p0 is None else p0.copy()
    hist = History(eps=problem.eps)
    tic = time.perf_counter()
    converged = False
    phi = np.zeros(problem.num_cells)
    for k in range(maxiter):
        g = problem.gradient(p)
        phi = problem.solve_DDT(L * (f - problem.D @ p) + problem.D @ g)
        update = problem.DT @ phi - g
        res = float(np.linalg.norm(update) / np.linalg.norm(g))
        converged = res < tol
        timeout = time.perf_counter() - tic > max_time

        # Record state of p^{k-1} (residual refers to it)
        if k % record_every == 0 or converged or timeout or k == maxiter - 1:
            _record(hist, problem, p, res, phi, f, tic)
            if verbose and (k % 100 == 0):
                print(f"{k:6d}  J_eps={hist.energy[-1]:.8e}  J_0={hist.energy0[-1]:.8e}  res={res:.3e}")
        if converged or timeout:
            break

        p = p + update / L
        hist.iterations += 1

    return p, phi, hist, converged


def _record(hist: History, problem: RegularizedBeckmann, p, res, phi, f, tic) -> None:
    hist.residual.append(res)
    hist.energy.append(problem.energy(p))
    hist.energy0.append(problem.energy(p, eps=0.0))
    hist.dual.append(problem.dual_value(phi, f))
    hist.time.append(time.perf_counter() - tic)


def _anderson_mix(G: list[np.ndarray], R: list[np.ndarray]) -> np.ndarray:
    """Anderson (type II, no damping): sum_i alpha_i G_i with sum alpha = 1
    minimizing ||sum_i alpha_i R_i||. Affine combination, so D p = f is kept."""
    dR = np.stack([R[i + 1] - R[i] for i in range(len(R) - 1)], axis=1)
    dG = np.stack([G[i + 1] - G[i] for i in range(len(G) - 1)], axis=1)
    gamma = np.linalg.lstsq(dR, R[-1], rcond=None)[0]
    return G[-1] - dG @ gamma


def _solve_darcy_direct(
    problem: RegularizedBeckmann, A: sps.spmatrix, f: np.ndarray, hist: History
) -> np.ndarray:
    """Sparse LU on the KKT system, potential pinned in cell 0. Returns p."""
    tic = time.perf_counter()
    D0 = problem.D[1:]
    K = sps.bmat([[A, -D0.T], [D0, None]], format="csc")
    lu = sps.linalg.splu(K)
    hist.time_setup += time.perf_counter() - tic
    tic = time.perf_counter()
    sol = lu.solve(np.concatenate([np.zeros(problem.num_faces), f[1:]]))
    hist.time_solve += time.perf_counter() - tic
    hist.linear_iterations.append(1)
    return sol[: problem.num_faces]


def _solve_darcy_minres(
    problem: RegularizedBeckmann,
    A: sps.spmatrix,
    f: np.ndarray,
    x0: np.ndarray,
    rtol: float,
    hist: History,
    maxiter: int = 1000,
    lumped: bool = False,
) -> np.ndarray:
    """MINRES on the symmetric KKT system [[A, -D^T], [-D, 0]] [p; phi] = [0; -f].

    Block-diagonal preconditioner Al^{-1} (+) AMG(S)^{-1}, with Al the row-sum
    lumped A (lumped=True; diag(A) otherwise) and the Schur complement
    approximation S = D Al^{-1} D^T (variable-coefficient
    Neumann Laplacian, constant kernel handled by mean projection). The result
    is projected onto D p = f exactly (one DCT solve). Returns [p; phi].
    """
    import pyamg

    nf = problem.num_faces
    tic = time.perf_counter()
    K = sps.bmat([[A, -problem.DT], [-problem.D, None]], format="csr")
    dA_inv = 1.0 / (abs(A).sum(axis=1).A1 if lumped else A.diagonal())
    S = (problem.D @ sps.diags(dA_inv) @ problem.DT).tocsr()
    amg = pyamg.smoothed_aggregation_solver(S, symmetry="symmetric").aspreconditioner(cycle="V")
    hist.time_setup += time.perf_counter() - tic

    def apply_prec(r: np.ndarray) -> np.ndarray:
        r_phi = r[nf:] - r[nf:].mean()
        z_phi = amg @ r_phi
        return np.concatenate([dA_inv * r[:nf], z_phi - z_phi.mean()])

    M = sps.linalg.LinearOperator(K.shape, matvec=apply_prec, dtype=float)
    count = [0]

    def cb(_):
        count[0] += 1

    # Solve for the correction from the warm start, so that rtol is relative
    # to the initial residual (scipy's rtol is relative to ||rhs||).
    tic = time.perf_counter()
    r0 = np.concatenate([np.zeros(nf), -f]) - K @ x0
    dx, _ = sps.linalg.minres(K, r0, rtol=rtol, M=M, maxiter=maxiter, callback=cb)
    x = x0 + dx
    p = x[:nf]
    p += problem.DT @ problem.solve_DDT(f - problem.D @ p)
    hist.time_solve += time.perf_counter() - tic
    hist.linear_iterations.append(count[0])
    return np.concatenate([p, x[nf:]])


def lagged_diffusivity(
    problem: RegularizedBeckmann,
    f: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 500,
    p0: np.ndarray | None = None,
    aa_depth: int = 0,
    linear_solver: str = "direct",
    linear_rtol: float | None = None,
    verbose: bool = False,
    max_time: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, History, bool]:
    """Lagged diffusivity (Kacanov / IRLS):  A(p^{k-1}) p^k - D^T phi^k = 0, D p^k = f.

    Each step is a Darcy problem with face/cell mobility (|q^{k-1}| + eps) kappa,
    using the reconstruction of the chosen variant (exact majorizer of J).

    linear_solver:
        "direct": sparse LU on the KKT system, potential pinned in cell 0.
        "minres": block-preconditioned MINRES (diag(A), AMG on the Schur
            complement), warm-started, followed by exact projection onto D p = f.
            linear_rtol: residual reduction relative to the warm start
            (default 1e-6).

    aa_depth > 0 applies Anderson acceleration to the flux, safeguarded: the
    accelerated iterate is rejected if it increases J_eps compared to the plain
    step. History entries refer to the iterate after each step; entry 0 is p0.
    """
    if linear_solver not in ("direct", "minres"):
        raise ValueError(f"Unknown linear solver {linear_solver}")
    f = f.ravel() - f.mean()
    p = problem.darcy_init(f) if p0 is None else p0.copy()
    hist = History(eps=problem.eps)
    tic = time.perf_counter()
    res, phi = problem.kkt_residual(p)
    _record(hist, problem, p, res, phi, f, tic)
    x_prev = np.concatenate([p, phi])
    G, R = [], []
    converged = res < tol
    for k in range(maxiter):
        if converged or time.perf_counter() - tic > max_time:
            break
        A = problem.lagged_operator(p)
        if linear_solver == "direct":
            p_plain = _solve_darcy_direct(problem, A, f, hist)
        else:
            rtol = 1e-6 if linear_rtol is None else linear_rtol
            x_prev = _solve_darcy_minres(problem, A, f, x_prev, rtol, hist)
            p_plain = x_prev[: problem.num_faces]

        p_new = p_plain
        if aa_depth > 0:
            G.append(p_plain)
            R.append(p_plain - p)
            G, R = G[-(aa_depth + 1) :], R[-(aa_depth + 1) :]
            if len(G) > 1:
                p_aa = _anderson_mix(G, R)
                if problem.energy(p_aa) <= problem.energy(p_plain):
                    p_new = p_aa
                else:
                    hist.aa_rejected += 1
                    G, R = G[-1:], R[-1:]
        p = p_new
        x_prev[: problem.num_faces] = p  # warm start for next MINRES solve

        hist.iterations += 1
        res, phi = problem.kkt_residual(p)
        _record(hist, problem, p, res, phi, f, tic)
        if verbose:
            print(f"{k:4d}  J_eps={hist.energy[-1]:.10e}  J_0={hist.energy0[-1]:.8e}  res={res:.3e}")
        converged = res < tol

    return p, phi, hist, converged


def continuation(
    solver,
    problem: RegularizedBeckmann,
    f: np.ndarray,
    eps_target: float,
    eps_start: float,
    factor: float = 0.5,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[History], bool]:
    """Run solver for eps_start, eps_start*factor, ..., eps_target, warm-starting
    each stage. Intermediate stages use the same tolerance. Restores problem.eps.
    A max_time keyword is treated as the budget for all stages together."""
    eps_list = []
    eps = eps_start
    while eps > eps_target * (1 + 1e-12):
        eps_list.append(eps)
        eps *= factor
    eps_list.append(eps_target)
    p = kwargs.pop("p0", None)
    deadline = time.perf_counter() + kwargs.pop("max_time", np.inf)
    hists, ok = [], False
    for eps in eps_list:
        problem.eps = eps
        remaining = deadline - time.perf_counter()
        p, phi, hist, ok = solver(problem, f, p0=p, max_time=remaining, **kwargs)
        hists.append(hist)
        if remaining <= 0 or not ok:
            break
    problem.eps = eps_target
    return p, phi, hists, ok


def box_pair(
    n: int, side: float = 0.25, shift: tuple[float, float] = (0.25, 0.0), mass: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform box mu and its translate gamma on [0,1]^2, n x n cells.

    Box starts at (0.25, 0.25). Exact W = mass * |shift| when side and shift are
    multiples of 1/n.
    """
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")

    def box(x0, y0):
        return ((X > x0) & (X < x0 + side) & (Y > y0) & (Y < y0 + side)).astype(float)

    mu = box(0.25, 0.25)
    gamma = box(0.25 + shift[0], 0.25 + shift[1])
    mu *= mass / (mu.sum() * h**2)
    gamma *= mass / (gamma.sum() * h**2)
    return mu, gamma, mass * float(np.hypot(*shift))
