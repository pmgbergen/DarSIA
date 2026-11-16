"""Load data from FluidFlowerConfig experiments."""

from pathlib import Path
from typing import Literal

import darsia
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.results_folder import (
    DEFAULT_MASS_FORMATS,
    field_map_folder,
)


def mass_files(config: FluidFlowerConfig) -> list[Path]:
    """Mass maps written by the mass analysis, as `.npz` files.

    The mass analysis writes the export mode "mass" to `<results>/mass/<format>`, one
    folder per export format of `[analysis.mass] formats`; the files of the npz formats
    are returned.

    Parameters
    ----------
    config : FluidFlowerConfig
        Configuration of the experiment.

    Returns
    -------
    list[Path]
        Sorted paths of the `.npz` files.

    Raises
    ------
    FileNotFoundError
        If no mass maps in npz format are found.

    """
    formats = config.analysis.mass.formats or DEFAULT_MASS_FORMATS
    specs = ImageExportFormats(config, formats).formats
    folders = [
        field_map_folder(config.data.results, "mass") / spec.folder_name
        for spec in specs
        if spec.type == "npz"
    ]
    files = sorted(
        path for folder in folders if folder.is_dir() for path in folder.glob("*.npz")
    )
    if len(files) == 0:
        searched = ", ".join(str(folder) for folder in folders) or "none"
        raise FileNotFoundError(
            "No mass maps found. Run the mass analysis with 'mass' in "
            "[analysis.mass] export and an npz format in [analysis.mass] formats "
            f"(searched: {searched})."
        )
    return files


def load_data(
    config: FluidFlowerConfig,
    data: Literal["mass"],
    time: float,
    tol: float | None = None,
) -> darsia.Image | None:
    """Load data from a given FluidFlowerConfig.

    Parameters
    ----------
    config : FluidFlowerConfig
        Configuration of the experiment.
    data : Literal["mass"]
        Type of data to load. Currently only "mass" is supported.
    time : float
        Time point at which to load the data (in seconds).
    tol : float | None
        Tolerance for time matching (in hours). If None, exact match is
        required.

    Returns
    -------
    darsia.Image
        Loaded data image.
    """

    # Fetch available analyzed mass files
    if data == "mass":
        available_files = mass_files(config)
    else:
        raise ValueError(f"Data type {data} not recognized.")

    # Fetch experiment protocol
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # Find closest time point
    path = experiment.find_images_for_times(
        times=time,
        tol=tol * 3600 if tol is not None else None,
        data=available_files,
    )

    if path is None:
        return None
    else:
        return darsia.imread(path)
