"""Compiled kernel for the flash and the mass of a fluid system.

From a signal, the flash gives the relative concentration in the aqueous phase and the
gas saturation as two linear ramps (:class:`darsia.SimpleFlash`). The mass of the
injected component per pore volume then is

    m = rho_g s_g + chi_aq c_aq max(1 - s_g, 0),

with the gas density rho_g and the aqueous capacity chi_aq, which depend on pressure and
temperature and thereby on the pixel row only (:class:`darsia.FluidMassAnalysis`). The
kernel computes all fields in a single parallel pass, with the arithmetic of
`SimpleFlash.__call__` and `FluidMassAnalysis.mass_analysis`; change them together.

The kernel requires numba; without it `NUMBA_AVAILABLE` is False and callers use the
array-based implementation.

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = ["NUMBA_AVAILABLE", "row_values", "flash_and_mass"]


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


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", parallel=True)
    def _flash_and_mass(
        signal,
        min_aq,
        max_aq,
        min_g,
        max_g,
        density_gas,
        aqueous_capacity,
        use_keep_aq,
        keep_aq,
        use_keep_g,
        keep_g,
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
                x = signal[i, j]

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
                if use_keep_aq and not keep_aq[i, j]:
                    ca = 0.0
                if use_keep_g and not keep_g[i, j]:
                    sg = 0.0

                # Mass in the gas and in the aqueous phase
                mg = rho * sg
                aqueous = 1 - sg
                if aqueous < 0:
                    aqueous = 0.0
                ma = chi * ca * aqueous

                c_aq[i, j] = ca
                s_g[i, j] = sg
                mass_g[i, j] = mg
                mass_aq[i, j] = ma
                mass[i, j] = mg + ma


def flash_and_mass(
    signal: np.ndarray,
    bounds: tuple[float, float, float, float],
    density_gas: np.ndarray,
    aqueous_capacity: np.ndarray,
    keep_aq: np.ndarray | None = None,
    keep_g: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple
        Concentration in the aqueous phase, gas saturation, total mass, mass in the gas
        phase and mass in the aqueous phase, each of shape (rows, cols).

    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    fields = [np.empty(signal.shape, dtype=np.float64) for _ in range(5)]
    no_mask = np.zeros((1, 1), dtype=np.bool_)
    _flash_and_mass(
        signal,
        *[float(bound) for bound in bounds],
        np.ascontiguousarray(density_gas, dtype=np.float64),
        np.ascontiguousarray(aqueous_capacity, dtype=np.float64),
        keep_aq is not None,
        no_mask if keep_aq is None else np.ascontiguousarray(keep_aq, dtype=np.bool_),
        keep_g is not None,
        no_mask if keep_g is None else np.ascontiguousarray(keep_g, dtype=np.bool_),
        *fields,
    )
    return tuple(fields)
