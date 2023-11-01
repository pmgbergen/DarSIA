"""Plot the 2d Wasserstein distance between two mass distributions.

This file mostly serves as template for the plotting of the Wasserstein distance. In
particular, grid information etc. may need to be adapted.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import darsia


def plot_2d_wasserstein_distance(
    grid: darsia.Grid,
    mass_diff: np.ndarray,
    flux: np.ndarray,
    potential: np.ndarray,
    transport_density: np.ndarray,
    **kwargs,
) -> None:
    """Plot the 2d Wasserstein distance between two mass distributions.

    The inputs are assumed to satisfy the layout of the Beckman solution.

    Args:
        grid (darsia.Grid): grid
        mass_diff (np.ndarray): difference of mass distributions
        flux (np.ndarray): fluxes
        potential (np.ndarray): potential
        transport_density (np.ndarray): transport density
        kwargs: additional keyword arguments

    """
    # Fetch options
    name = kwargs.get("name", None)
    save_plot = kwargs.get("save", False)
    if save_plot:
        folder = kwargs.get("folder", ".")
        Path(folder).mkdir(parents=True, exist_ok=True)
    show_plot = kwargs.get("show", True)

    # Control of flux arrows
    scaling = kwargs.get("scaling", 1.0)
    resolution = kwargs.get("resolution", 1)

    # Meshgrid
    Y, X = np.meshgrid(
        grid.voxel_size[0] * (0.5 + np.arange(grid.shape[0] - 1, -1, -1)),
        grid.voxel_size[1] * (0.5 + np.arange(grid.shape[1])),
        indexing="ij",
    )

    # Plot the potential
    plt.figure("Beckman solution potential")
    plt.pcolormesh(X, Y, potential, cmap="turbo")
    plt.colorbar(label="potential")
    plt.quiver(
        X[::resolution, ::resolution],
        Y[::resolution, ::resolution],
        scaling * flux[::resolution, ::resolution, 0],
        -scaling * flux[::resolution, ::resolution, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.25,
        width=0.005,
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    # plt.ylim(top=0.08)  # TODO rm?
    if save_plot:
        plt.savefig(
            folder + "/" + name + "_beckman_solution_potential.png",
            dpi=500,
            transparent=True,
        )

    # Plot the fluxes
    plt.figure("Beckman solution fluxes")
    plt.pcolormesh(X, Y, mass_diff, cmap="turbo")  # , vmin=-1, vmax=3.5)
    plt.colorbar(label="mass difference")
    plt.quiver(
        X[::resolution, ::resolution],
        Y[::resolution, ::resolution],
        scaling * flux[::resolution, ::resolution, 0],
        -scaling * flux[::resolution, ::resolution, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.25,
        width=0.005,
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    # plt.ylim(top=0.08)
    plt.text(
        0.0025,
        0.075,
        name,
        color="white",
        alpha=0.9,
        rotation=0,
        fontsize=14,
    )  # TODO rm?
    if save_plot:
        plt.savefig(
            folder + "/" + name + "_beckman_solution_fluxes.png",
            dpi=500,
            transparent=True,
        )

    # Plot the transport density
    plt.figure("L1 optimal transport density")
    plt.pcolormesh(X, Y, transport_density, cmap="turbo")
    plt.colorbar(label="flux modulus")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    # plt.ylim(top=0.08)  # TODO rm?
    if save_plot:
        plt.savefig(
            folder + "/" + name + "_beckman_solution_transport_density.png",
            dpi=500,
            transparent=True,
        )

    if show_plot:
        plt.show()
    else:
        plt.close("all")