"""Image porosity: computation, storage and reading.

The image porosity marks pixels of the baseline that show the usual (pore space)
colour of their facies (1) and pixels that do not (0). It is computed in three steps:

1. detection with the method of `[image_porosity] method` (if `active`), else ones;
2. `cutoff`: detected values below it become 0;
3. `boundary_band`: pixels within that many pixels of a label boundary become 0. Pixels
   there mix the colours of two facies and represent neither.

The boolean porosity is the image porosity above `tol` with detection, else the full
domain; `tol` is applied when reading, so it can change without a new setup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.results_folder import image_porosity_folder
from darsia.restoration.averaging import label_boundary_mask

from .methods import METHODS

logger = logging.getLogger(__name__)

IMAGE_POROSITY_FILE = "image_porosity.npz"
METADATA_FILE = "metadata.json"


def detect_image_porosity(
    baseline: darsia.Image, labels: darsia.Image, config: ImagePorosityConfig
) -> darsia.Image:
    """Step 1 only: the detected image porosity, or ones without detection.

    Parameters
    ----------
    baseline : darsia.Image
        Baseline image.
    labels : darsia.Image
        Labels image.
    config : ImagePorosityConfig
        Configuration of the image porosity.

    Returns
    -------
    darsia.Image
        Image porosity in [0, 1] on the grid of the baseline.

    """
    if not config.active:
        return darsia.ones_like(baseline, mode="voxels", dtype=np.float32)
    method = METHODS[config.method]
    with darsia.timed(f"image porosity ({config.method})"):
        return method.detect(baseline, labels, config.method_options())


def compute_image_porosity(
    baseline: darsia.Image, labels: darsia.Image, config: ImagePorosityConfig
) -> darsia.Image:
    """Image porosity: detection, cutoff and boundary band.

    Parameters
    ----------
    baseline : darsia.Image
        Baseline image.
    labels : darsia.Image
        Labels image.
    config : ImagePorosityConfig
        Configuration of the image porosity.

    Returns
    -------
    darsia.Image
        Image porosity in [0, 1] on the grid of the baseline.

    """
    porosity = detect_image_porosity(baseline, labels, config)
    porosity.img[porosity.img < config.cutoff] = 0.0
    porosity.img[label_boundary_mask(labels, config.boundary_band)] = 0.0
    return porosity


def boolean_image_porosity(
    porosity: darsia.Image, config: ImagePorosityConfig
) -> darsia.Image:
    """Boolean porosity: image porosity above `tol` with detection, else all True."""
    if config.active:
        return porosity > config.tol
    return darsia.ones_like(porosity, mode="voxels", dtype=bool)


def _method_name(config: ImagePorosityConfig) -> str | None:
    return config.method if config.active else None


def _darsia_version() -> str:
    try:
        return metadata.version("darsia")
    except metadata.PackageNotFoundError:
        return "unknown"


def save_image_porosity(
    folder: Path,
    porosity: darsia.Image,
    config: ImagePorosityConfig,
    inputs: dict[str, Path] | None = None,
) -> None:
    """Store an image porosity with its metadata and an illustration.

    Parameters
    ----------
    folder : Path
        Folder of the method (`results_folder.image_porosity_folder`).
    porosity : darsia.Image
        Image porosity.
    config : ImagePorosityConfig
        Configuration it was computed with.
    inputs : dict[str, Path], optional
        Input files (e.g. baseline, labels); their modification times are stored.

    """
    from darsia.presets.workflows.setup.illustrations import (
        save_scalar_map_illustration,
    )

    folder.mkdir(parents=True, exist_ok=True)
    porosity.save(folder / IMAGE_POROSITY_FILE, verbose=False)
    info = {
        "method": _method_name(config),
        "options": asdict(config.method_options()) if config.active else None,
        "cutoff": config.cutoff,
        "boundary_band": config.boundary_band,
        "darsia_version": _darsia_version(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            name: {
                "path": str(path),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
            for name, path in (inputs or {}).items()
            if Path(path).exists()
        },
    }
    (folder / METADATA_FILE).write_text(json.dumps(info, indent=1))

    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    save_scalar_map_illustration(
        porosity.img,
        figures / "image_porosity.jpg",
        title="Image porosity",
        colorbar_label="Image porosity",
    )
    if config.active:
        save_scalar_map_illustration(
            boolean_image_porosity(porosity, config).img.astype(float),
            figures / "boolean_porosity.jpg",
            title=f"Boolean porosity (> {config.tol})",
            colorbar_label="Porous",
        )
    logger.info(f"Image porosity stored in {folder}.")


class ImagePorosity:
    """Stored image porosity of the configured method, read on first use.

    Without detection, the boolean porosity needs no file (full domain); the image
    porosity itself is always read from the file, as it carries the boundary band.
    """

    def __init__(self, config: ImagePorosityConfig | None, results: Path) -> None:
        """
        Parameters
        ----------
        config : ImagePorosityConfig | None
            Configuration (`[image_porosity]`); None means the defaults.
        results : Path
            The results folder (`[data].results`).

        """
        self.config = config if config is not None else ImagePorosityConfig()
        self.folder = image_porosity_folder(results, _method_name(self.config))
        """Folder of the stored image porosity."""
        self._map: darsia.Image | None = None

    @property
    def file(self) -> Path:
        """File of the stored image porosity."""
        return self.folder / IMAGE_POROSITY_FILE

    @property
    def map(self) -> darsia.Image:
        """The stored image porosity (read on first use, not to be modified)."""
        if self._map is None:
            if not self.file.exists():
                method = _method_name(self.config) or "no detection"
                raise FileNotFoundError(
                    f"No image porosity ({method}) at {self.file}. Run the setup step "
                    "'Image porosity' (GUI: Setup > Image porosity; CLI: "
                    "user_interface_setup --image-porosity)."
                )
            with darsia.timed("load image porosity"):
                self._map = darsia.imread(self.file)
        return self._map

    def map_on(self, grid: darsia.Image) -> darsia.Image:
        """Image porosity on the grid of `grid` (the stored one on its own grid)."""
        return darsia.transfer(self.map, grid)

    def boolean_on(self, grid: darsia.Image) -> darsia.Image:
        """Boolean porosity on the grid of `grid`."""
        if not self.config.active:
            return darsia.ones_like(grid, mode="voxels", dtype=bool)
        return darsia.transfer(boolean_image_porosity(self.map, self.config), grid)


def load_image_porosity(
    config: ImagePorosityConfig | None, results: Path
) -> ImagePorosity:
    """Stored image porosity of the configured method (read on first use).

    Parameters
    ----------
    config : ImagePorosityConfig | None
        Configuration (`[image_porosity]`); None means the defaults.
    results : Path
        The results folder (`[data].results`).

    """
    return ImagePorosity(config, results)
