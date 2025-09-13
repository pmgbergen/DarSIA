"""Module with interface to compute and visualize Wasserstein distances."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


import darsia


def wasserstein_distance(
    mass_1: darsia.Image,
    mass_2: darsia.Image,
    method: str,
    weight: Optional[darsia.Image] = None,
    **kwargs,
):
    """Unified access to Wasserstein distance computation between images with same mass.

    Args:
        mass_1 (darsia.Image): image 1, source distribution
        mass_2 (darsia.Image): image 2, destination distribution
        method (str): method to use ("newton", "bregman", or "cv2.emd")
        **kwargs: additional arguments (only for "newton" and "bregman")
            - options (dict): options for the method.

    """
    # Define method for computing 1-Wasserstein distance

    if method.lower() in ["newton", "bregman", "gprox"]:
        # Use Finite Volume Iterative Method (Newton or Bregman)

        # Extract grid - implicitly assume mass_2 to generate same grid
        grid: darsia.Grid = darsia.generate_grid(mass_1)

        # Fetch options and define Wasserstein method
        options = kwargs.get("options", {})

        # Define method
        match method.lower():
            case "newton":
                w1 = darsia.WassersteinDistanceNewton(grid, weight, options)
            case "bregman":
                w1 = darsia.WassersteinDistanceBregman(grid, weight, options)
            case "gprox":
                if weight is not None:
                    raise NotImplementedError(
                        "Weighted Gprox not implemented for anisotropic meshes"
                    )
                w1 = darsia.WassersteinDistanceGproxPGHD(grid, options)
            case _:
                raise NotImplementedError(f"Method {method} not implemented.")

    elif method.lower() == "cv2.emd":
        # Use Earth Mover's Distance from CV2
        assert weight is None, "Weighted EMD not supported by cv2."
        preprocess = kwargs.get("preprocess")
        w1 = darsia.EMD(preprocess)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    # Compute and return Wasserstein distance
    return w1(mass_1, mass_2)


def wasserstein_distance_to_vtk(
    path: Union[str, Path],
    info: dict,
) -> None:
    """Write the output of the Wasserstein distance to a VTK file.

    Args:
        path (Union[str, Path]): path to the VTK file
        info (dict): information dictionary

    NOTE: Requires pyevtk to be installed.

    """
    data = [
        (key, info[key], format)
        for key, format in [
            ("src", darsia.Format.SCALAR),
            ("dst", darsia.Format.SCALAR),
            ("mass_diff", darsia.Format.SCALAR),
            ("flux", darsia.Format.VECTOR),
            ("weighted_flux", darsia.Format.VECTOR),
            ("pressure", darsia.Format.SCALAR),
            ("transport_density", darsia.Format.SCALAR),
            ("weight", darsia.Format.TENSOR),
            ("weight_inv", darsia.Format.TENSOR),
        ]
    ]
    darsia.plotting.to_vtk(path, data)
