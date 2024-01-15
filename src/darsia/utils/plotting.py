"""Plot the 2d Wasserstein distance between two mass distributions.

This file mostly serves as template for the plotting of the Wasserstein distance. In
particular, grid information etc. may need to be adapted.

"""

from pathlib import Path
from typing import Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

import darsia


def plot_2d_wasserstein_distance(
    info: dict,
    **kwargs,
) -> None:
    """Post-processing utility to plot the 2d Wasserstein distance.

    Args:
        info (dict): information about the Beckman solution, output of
            darsia.wasserstein_distance.

    """
    # Fetch fields
    grid = info["grid"]
    mass_diff = info["mass_diff"]
    flux = info["flux"]
    pressure = info["pressure"]
    transport_density = info["transport_density"]

    # Fetch options
    path = kwargs.get("path", None)
    save_plot = kwargs.get("save", False)
    if save_plot:
        Path(path).mkdir(parents=True, exist_ok=True)
        dpi = kwargs.get("dpi", 500)
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

    # Plot the pressure
    plt.figure("Beckman solution pressure")
    plt.pcolormesh(X, Y, pressure, cmap="turbo")
    plt.colorbar(label="pressure")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # Save the plot
    if save_plot:
        plt.savefig(
            Path(str(path) + "_beckman_solution_pressure.png"),
            dpi=dpi,
            transparent=True,
        )

    # Plot the fluxes
    plt.figure("Beckman solution fluxes")
    plt.pcolormesh(X, Y, mass_diff, cmap="turbo")
    plt.colorbar(label="mass difference")
    plt.quiver(
        X[::resolution, ::resolution],
        Y[::resolution, ::resolution],
        -scaling * flux[::resolution, ::resolution, 1],
        scaling * flux[::resolution, ::resolution, 0],
        angles="xy",
        alpha=0.25,
        width=0.01,
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # Save the plot
    if save_plot:
        plt.savefig(
            Path(str(path) + "_beckman_solution_fluxes.png"),
            dpi=dpi,
            transparent=True,
        )

    # Plot the transport density
    plt.figure("L1 optimal transport density")
    plt.pcolormesh(X, Y, transport_density, cmap="turbo")
    plt.colorbar(label="flux modulus")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # Save the plot
    if save_plot:
        plt.savefig(
            Path(str(path) + "_beckman_solution_transport_density.png"),
            dpi=dpi,
            transparent=True,
        )

    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close("all")


def to_vtk(
    path: Union[str, Path],
    data: list[tuple[Union[darsia.Image, np.ndarray], str]],
) -> None:
    """Write data to a VTK file.

    Args:
        path (Union[str, Path]): path to the VTK file
        data (list[tuple[Union[darsia.Image, np.ndarray], str]]): data to write, includes
            the data and the name of the data. Require at least one data point to be an
            image.

    NOTE: Requires pyevtk to be installed.

    """
    try:
        from pyevtk.hl import gridToVTK  # type: ignore [import-not-found]

        # Check whether the data contains at least one image, and pick the first one
        image = None
        for d in data:
            name, img = d
            if isinstance(img, darsia.Image):
                image = img
                break
        assert image is not None, "At least one data point must be an image."

        # Extract grid axes
        dim = image.space_dim
        indices = [
            darsia.interpret_indexing(index, "ijk"[:dim])[0] for index in "xyz"[:dim]
        ]
        revert = [
            darsia.interpret_indexing(index, "ijk"[:dim])[1] for index in "xyz"[:dim]
        ]

        def conditional_flip(x, cond):
            return np.ascontiguousarray(np.flip(x)) if cond else x

        xyz = [
            conditional_flip(
                np.linspace(
                    image.origin[i],
                    image.opposite_corner[i],
                    image.num_voxels[indices[i]] + 1,
                ),
                revert[i],
            )
            for i in range(dim)
        ]
        if image.space_dim == 1:
            x = xyz[0]
            y = np.array([0])
            z = np.array([0])
        elif image.space_dim == 2:
            x, y = xyz
            z = np.array([0])
        elif image.space_dim == 3:
            x, y, z = xyz
        target_shape = tuple(
            np.maximum(1, np.array([x.size, y.size, z.size]) - 1).tolist()
        )

        # Convert cell data to right format
        cellData = {}
        for d in data:
            name, img = d
            if isinstance(img, darsia.Image):
                img = img.img.copy()
            assert isinstance(img, np.ndarray), "Data must be of type np.ndarray."

            is_scalar = img.ndim == image.space_dim
            if not is_scalar:
                assert (
                    img.ndim == image.space_dim + 1
                ), "Data must be of dimension %d or %d." % (
                    image.space_dim,
                    image.space_dim + 1,
                )
            assert np.allclose(
                img.shape[: image.space_dim], image.img.shape[: image.space_dim]
            ), "Data must have the same shape."

            # Convert to cartesian indexing
            if is_scalar:
                img = np.asfortranarray(
                    darsia.matrixToCartesianIndexing(img, image.space_dim).reshape(
                        target_shape, order="F"
                    )
                )
            else:
                img = [
                    np.asfortranarray(
                        darsia.matrixToCartesianIndexing(
                            img[..., i], image.space_dim
                        ).reshape(target_shape, order="F")
                    )
                    for i in range(img.shape[-1])
                ]
                while len(img) < 3:
                    img.append(np.zeros(target_shape))
                assert len(img) == 3, "pyevtk only allows 3 components"
                # Flip the directions as part of the conversion
                if image.space_dim == 1:
                    img = tuple(img)
                elif image.space_dim == 2:
                    img = (-img[1], img[0], img[2])
                elif image.space_dim == 3:
                    img = (-img[1], img[2], img[0])
            cellData[name] = img

        # Make directory if necessary
        Path(path).mkdir(parents=True, exist_ok=True)

        # Write to VTK
        gridToVTK(str(Path(path)), x, y, z, cellData=cellData)

    except ImportError:
        warn("pyevtk not installed. Cannot save as vtk.")
