"""Module with interface to compute and visualize Wasserstein distances."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import darsia


def wasserstein_distance(
    mass_src: darsia.Image,
    mass_dst: darsia.Image,
    method: Literal["newton", "bregman", "gprox", "cv2.emd"] = "newton",
    weight: Optional[darsia.Image] = None,
    **kwargs,
) -> float | tuple[float, dict]:
    """Unified access to Wasserstein distance computation between images with same mass.

    Args:
        mass_src (darsia.Image): source distribution
        mass_dst (darsia.Image): destination distribution
        method (Literal["newton", "bregman", "gprox", "cv2.emd"]): method to use
        **kwargs: additional arguments (only for "newton", "bregman", "gprox")
            - options (dict): options for the method.

    Returns:
        float | tuple[float, dict: Wasserstein distance or (distance, info)

    """
    # Define method for computing 1-Wasserstein distance
    method_name = method.lower()

    if method_name in ["newton", "bregman", "gprox"]:
        # Use Finite Volume Iterative Method (Newton or Bregman)

        # Extract grid - implicitly assume mass_dst to generate same grid
        grid: darsia.Grid = darsia.generate_grid(mass_dst)

        # Fetch options and define Wasserstein method
        options = kwargs.get("options", {})

        # Define method
        match method_name:
            case "newton":
                w1 = darsia.BeckmannNewtonSolver(grid, weight, options)
            case "bregman":
                w1 = darsia.BeckmannBregmanSolver(grid, weight, options)
            case "gprox":
                w1 = darsia.BeckmannGproxPGHDSolver(grid, weight, options)

    elif method_name == "cv2.emd":
        # Use Earth Mover's Distance from CV2
        assert weight is None, "Weighted EMD not supported by cv2."
        preprocess = kwargs.get("preprocess")
        w1 = darsia.EMD(preprocess)

    else:
        raise NotImplementedError(f"Method {method_name} not implemented.")

    # Compute and return Wasserstein distance
    return w1(mass_src, mass_dst)


def wasserstein_distance_to_vtk(
    path: Path,
    info: dict,
) -> None:
    """Write the output of the Wasserstein distance to a VTK file.

    Args:
        path (Path): path to the VTK file
        info (dict): information dictionary output of darsia.wasserstein_distance

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
    darsia.plotting.to_vtk(Path(path), data)
