"""Compiled kernel for the flash and the mass of a fluid system.

From a signal, the flash gives the relative concentration in the aqueous phase and the
gas saturation as two linear ramps (:class:`darsia.SimpleFlash`). The mass of the
injected component per pore volume then is

    m = rho_g s_g + chi_aq c_aq max(1 - s_g, 0),

with the gas density rho_g and the aqueous capacity chi_aq, which depend on pressure and
temperature and thereby on the pixel row only (:class:`darsia.FluidMassAnalysis`). The
kernels compute all fields, or only their integrals over a geometry, in a single parallel
pass, with the arithmetic of `SimpleFlash.__call__`, `FluidMassAnalysis.mass_analysis`
and `Geometry.integrate`; change them together.

The inverse (`inverse_mass`) splits a total mass into these fields: up to the aqueous
capacity the mass is dissolved, the excess is gas, with the arithmetic of
`FluidMassAnalysis.inverse_mass_analysis`; change them together.

The kernel requires numba; without it `NUMBA_AVAILABLE` is False and callers use the
array-based implementation.

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = [
    "NUMBA_AVAILABLE",
    "row_values",
    "flash_and_mass",
    "flash_and_mass_integrals",
    "inverse_mass",
]


def row_values(field: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    """Values per row of a field which does not vary along rows.

    Parameters
    ----------
    field : np.ndarray
        Uniform value, or a view of one column in the given shape (as provided by
        `FluidMassAnalysis`).
    shape : tuple
        Shape (rows, cols) of the images.

    Returns
    -------
    np.ndarray or None
        Values of length rows; None if the field may vary along rows.

    """
    field = np.asarray(field)
    if field.ndim == 0:
        return np.full(shape[0], float(field), dtype=np.float64)
    if field.shape == tuple(shape) and field.ndim == 2 and field.strides[1] == 0:
        return np.ascontiguousarray(field[:, 0], dtype=np.float64)
    return None


COLUMN_BLOCK = 64
"""Columns per parallel block when integrating (rows are read along memory)."""


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", inline="always")
    def _pixel(x, rho, chi, bounds, keep_ca, keep_sg):
        """Concentration, saturation, gas mass and dissolved mass of one pixel."""
        min_aq, max_aq, min_g, max_g = bounds

        # Flash: linear ramps between the bounds (as np.clip)
        xa = x
        if xa < min_aq:
            xa = min_aq
        if xa > max_aq:
            xa = max_aq
        xg = x
        if xg < min_g:
            xg = min_g
        if xg > max_g:
            xg = max_g
        ca = (xa - min_aq) / (max_aq - min_aq)
        sg = (xg - min_g) / (max_g - min_g)

        # Expert knowledge: zero outside of the regions to keep
        if not keep_ca:
            ca = 0.0
        if not keep_sg:
            sg = 0.0

        # Mass in the gas and in the aqueous phase
        mg = rho * sg
        aqueous = 1 - sg
        if aqueous < 0:
            aqueous = 0.0
        ma = chi * ca * aqueous
        return ca, sg, mg, ma

    @njit(cache=True, error_model="numpy", parallel=True, nogil=True)
    def _flash_and_mass(
        signal,
        bounds,
        density_gas,
        aqueous_capacity,
        use_keep_aq,
        keep_aq,
        use_keep_g,
        keep_g,
        keep_phases,
        c_aq,
        s_g,
        mass,
        mass_g,
        mass_aq,
    ):
        for i in prange(signal.shape[0]):
            rho = density_gas[i]
            chi = aqueous_capacity[i]
            for j in range(signal.shape[1]):
                ca, sg, mg, ma = _pixel(
                    signal[i, j],
                    rho,
                    chi,
                    bounds,
                    (not use_keep_aq) or keep_aq[i, j],
                    (not use_keep_g) or keep_g[i, j],
                )
                c_aq[i, j] = ca
                s_g[i, j] = sg
                if keep_phases:
                    mass_g[i, j] = mg
                    mass_aq[i, j] = ma
                mass[i, j] = mg + ma

    @njit(cache=True, error_model="numpy", parallel=True, nogil=True)
    def _column_sums(
        signal,
        bounds,
        density_gas,
        aqueous_capacity,
        use_keep_aq,
        keep_aq,
        use_keep_g,
        keep_g,
        voxel_volume,
        block,
        sums_g,
        sums_aq,
    ):
        """Per column, the sum over the rows (in row order) of voxel volume times gas
        and dissolved mass, as the first reduction of `Geometry.integrate`."""
        rows, cols = signal.shape
        num_blocks = (cols + block - 1) // block
        for b in prange(num_blocks):
            j0 = b * block
            j1 = min(cols, j0 + block)
            for i in range(rows):
                rho = density_gas[i]
                chi = aqueous_capacity[i]
                for j in range(j0, j1):
                    _, _, mg, ma = _pixel(
                        signal[i, j],
                        rho,
                        chi,
                        bounds,
                        (not use_keep_aq) or keep_aq[i, j],
                        (not use_keep_g) or keep_g[i, j],
                    )
                    weighted_g = voxel_volume[i, j] * mg
                    weighted_aq = voxel_volume[i, j] * ma
                    if i == 0:
                        sums_g[j] = weighted_g
                        sums_aq[j] = weighted_aq
                    else:
                        sums_g[j] += weighted_g
                        sums_aq[j] += weighted_aq

    @njit(cache=True, error_model="numpy", parallel=True, nogil=True)
    def _inverse_mass(
        mass,
        density_gas,
        aqueous_capacity,
        has_gas,
        epsilon,
        scale,
        keep_sg,
        keep_phases,
        c_aq,
        s_g,
        total,
        mass_g,
        mass_aq,
    ):
        for i in prange(mass.shape[0]):
            rho = density_gas[i]
            chi = aqueous_capacity[i]
            denominator = rho - chi
            for j in range(mass.shape[1]):
                m = mass[i, j] * scale

                # Dissolved up to the aqueous capacity (np.divide with `where`, np.clip)
                ca = m / chi if abs(chi) > epsilon else 0.0
                if ca < 0.0:
                    ca = 0.0
                if ca > 1.0:
                    ca = 1.0

                # Excess as gas
                sg = 0.0
                if has_gas:
                    excess = m - chi
                    if excess < 0.0:
                        excess = 0.0
                    sg = excess / denominator if abs(denominator) > epsilon else 0.0
                    if sg < 0.0:
                        sg = 0.0
                    if sg > 1.0:
                        sg = 1.0

                mg = rho * sg
                aqueous = 1 - sg
                if aqueous < 0:
                    aqueous = 0.0
                ma = chi * ca * aqueous
                c_aq[i, j] = ca
                if keep_sg:
                    s_g[i, j] = sg
                if keep_phases:
                    mass_g[i, j] = mg
                    mass_aq[i, j] = ma
                total[i, j] = mg + ma


def _kernel_arguments(signal, bounds, density_gas, aqueous_capacity, keep_aq, keep_g):
    """Arguments shared by the kernels, in their types."""
    no_mask = np.zeros((1, 1), dtype=np.bool_)
    return (
        np.ascontiguousarray(signal, dtype=np.float64),
        tuple(float(bound) for bound in bounds),
        np.ascontiguousarray(density_gas, dtype=np.float64),
        np.ascontiguousarray(aqueous_capacity, dtype=np.float64),
        keep_aq is not None,
        no_mask if keep_aq is None else np.ascontiguousarray(keep_aq, dtype=np.bool_),
        keep_g is not None,
        no_mask if keep_g is None else np.ascontiguousarray(keep_g, dtype=np.bool_),
    )


def flash_and_mass(
    signal: np.ndarray,
    bounds: tuple[float, float, float, float],
    density_gas: np.ndarray,
    aqueous_capacity: np.ndarray,
    keep_aq: np.ndarray | None = None,
    keep_g: np.ndarray | None = None,
    phase_masses: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Concentration, saturation and masses from a signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal, shape (rows, cols).
    bounds : tuple
        Bounds of the flash: minimal and maximal signal of the aqueous ramp, minimal and
        maximal signal of the gas ramp.
    density_gas : np.ndarray
        Gas density of each row.
    aqueous_capacity : np.ndarray
        Mass per aqueous volume at concentration 1 of each row.
    keep_aq, keep_g : np.ndarray, optional
        Boolean masks; concentration and saturation are set to zero where False.
    phase_masses : bool
        Whether to return the masses in the gas and in the aqueous phase.

    Returns
    -------
    tuple
        Concentration in the aqueous phase, gas saturation, total mass, mass in the gas
        phase and mass in the aqueous phase, each of shape (rows, cols); None for the
        phase masses if not requested.

    """
    arguments = _kernel_arguments(
        signal, bounds, density_gas, aqueous_capacity, keep_aq, keep_g
    )
    shape = arguments[0].shape
    unused = np.empty((1, 1), dtype=np.float64)
    c_aq, s_g, mass = (np.empty(shape, dtype=np.float64) for _ in range(3))
    mass_g, mass_aq = (
        np.empty(shape, dtype=np.float64) if phase_masses else unused for _ in range(2)
    )
    _flash_and_mass(*arguments, bool(phase_masses), c_aq, s_g, mass, mass_g, mass_aq)
    if not phase_masses:
        return c_aq, s_g, mass, None, None
    return c_aq, s_g, mass, mass_g, mass_aq


def flash_and_mass_integrals(
    signal: np.ndarray,
    bounds: tuple[float, float, float, float],
    density_gas: np.ndarray,
    aqueous_capacity: np.ndarray,
    voxel_volume: np.ndarray,
    keep_aq: np.ndarray | None = None,
    keep_g: np.ndarray | None = None,
) -> tuple[float, float]:
    """Integrated gas mass and dissolved mass from a signal, without the fields.

    The result equals `geometry.integrate` of the gas mass and the dissolved mass of
    `flash_and_mass` to the last digit: the products with the voxel volume are summed
    per column in row order, then over the columns with `np.sum` (as
    `darsia.Geometry.integrate` does); change them together.

    Parameters
    ----------
    signal : np.ndarray
        Signal, shape (rows, cols).
    bounds : tuple
        Bounds of the flash: minimal and maximal signal of the aqueous ramp, minimal and
        maximal signal of the gas ramp.
    density_gas : np.ndarray
        Gas density of each row.
    aqueous_capacity : np.ndarray
        Mass per aqueous volume at concentration 1 of each row.
    voxel_volume : np.ndarray
        Volume of each voxel, shape (rows, cols) (`Geometry.voxel_volumes`).
    keep_aq : np.ndarray, optional
        Boolean mask; the concentration is set to zero where False.
    keep_g : np.ndarray, optional
        Boolean mask; the saturation is set to zero where False.

    Returns
    -------
    tuple[float, float]
        Integrated mass in the gas phase and in the aqueous phase.

    """
    arguments = _kernel_arguments(
        signal, bounds, density_gas, aqueous_capacity, keep_aq, keep_g
    )
    columns = arguments[0].shape[1]
    sums_g = np.empty(columns, dtype=np.float64)
    sums_aq = np.empty(columns, dtype=np.float64)
    _column_sums(
        *arguments,
        np.ascontiguousarray(voxel_volume, dtype=np.float64),
        COLUMN_BLOCK,
        sums_g,
        sums_aq,
    )
    return np.sum(sums_g, axis=0), np.sum(sums_aq, axis=0)


def inverse_mass(
    mass: np.ndarray,
    density_gas: np.ndarray,
    aqueous_capacity: np.ndarray,
    has_gas: bool,
    epsilon: float,
    scale: float = 1.0,
    phase_masses: bool = True,
    saturation: bool = True,
) -> tuple[
    np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None
]:
    """Concentration, saturation and masses from a (scaled) total mass.

    Parameters
    ----------
    mass : np.ndarray
        Total mass per pore volume, shape (rows, cols).
    density_gas : np.ndarray
        Gas density of each row.
    aqueous_capacity : np.ndarray
        Mass per aqueous volume at concentration 1 of each row.
    has_gas : bool
        Whether the system has a gas phase; without, the saturation is zero.
    epsilon : float
        Divisors of at most this size in absolute value give zero.
    scale : float
        Factor applied to the mass first (as `darsia.weight`).
    phase_masses : bool
        Whether to return the masses in the gas and in the aqueous phase.
    saturation : bool
        Whether to return the gas saturation.

    Returns
    -------
    tuple
        Concentration in the aqueous phase, gas saturation, total mass, mass in the gas
        phase and mass in the aqueous phase, each of shape (rows, cols); None for the
        fields not requested.

    """
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    unused = np.empty((1, 1), dtype=np.float64)

    def field(keep: bool) -> np.ndarray:
        return np.empty(mass.shape, dtype=np.float64) if keep else unused

    c_aq, s_g, total = field(True), field(saturation), field(True)
    mass_g, mass_aq = field(phase_masses), field(phase_masses)
    _inverse_mass(
        mass,
        np.ascontiguousarray(density_gas, dtype=np.float64),
        np.ascontiguousarray(aqueous_capacity, dtype=np.float64),
        bool(has_gas),
        float(epsilon),
        float(scale),
        bool(saturation),
        bool(phase_masses),
        c_aq,
        s_g,
        total,
        mass_g,
        mass_aq,
    )
    return (
        c_aq,
        s_g if saturation else None,
        total,
        mass_g if phase_masses else None,
        mass_aq if phase_masses else None,
    )
