"""Image porosity: computation, storage and reading.

The image porosity marks pixels of the baseline that show the usual (pore space)
colour of their facies (1) and pixels that do not (0). It is computed in three steps:

1. detection with the method of `[image_porosity] method` (`"none"`: ones), comparing
   pixels within the regions of `[image_porosity] regions` (labels, facies, global);
2. `cutoff`: values below it become 0;
3. `boundary_band`: pixels within that many pixels of a label boundary become 0. Pixels
   there mix the colours of two facies and represent neither.

The boolean porosity is the image porosity above `tol`; `tol` is applied when
reading, so it can change without a new setup. Whether a step uses the image porosity
is decided by that step (its own options), not here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import darsia
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.results_folder import image_porosity_folder
from darsia.restoration.averaging import label_boundary_mask

from .methods import METHODS

logger = logging.getLogger(__name__)

IMAGE_POROSITY_FILE = "image_porosity.npz"
METADATA_FILE = "metadata.json"


def detect_image_porosity(
    baseline: darsia.Image, regions: darsia.Image, config: ImagePorosityConfig
) -> darsia.Image:
    """Step 1 only: the image porosity of the configured method (`"none"`: ones).

    Parameters
    ----------
    baseline : darsia.Image
        Baseline image.
    regions : darsia.Image
        Regions on the grid of the baseline; pixels are compared within each region
        (e.g. `Rig.regions(config.regions)`: labels, facies or one global region).
    config : ImagePorosityConfig
        Configuration of the image porosity.

    Returns
    -------
    darsia.Image
        Image porosity in [0, 1] on the grid of the baseline.

    """
    if regions.img.shape[:2] != baseline.img.shape[:2]:
        raise ValueError(
            f"Regions {regions.img.shape[:2]} and baseline "
            f"{baseline.img.shape[:2]} differ."
        )
    method = METHODS[config.method]
    with darsia.timed(f"image porosity ({config.method}, {config.regions})"):
        return method.detect(baseline, regions, config.method_options())


def compute_image_porosity(
    baseline: darsia.Image,
    regions: darsia.Image,
    labels: darsia.Image,
    config: ImagePorosityConfig,
) -> darsia.Image:
    """Image porosity: detection, cutoff and boundary band.

    Parameters
    ----------
    baseline : darsia.Image
        Baseline image.
    regions : darsia.Image
        Regions of the detection (see `detect_image_porosity`).
    labels : darsia.Image
        Labels image; the boundary band follows the label boundaries.
    config : ImagePorosityConfig
        Configuration of the image porosity.

    Returns
    -------
    darsia.Image
        Image porosity in [0, 1] on the grid of the baseline.

    """
    porosity = detect_image_porosity(baseline, regions, config)
    porosity.img[porosity.img < config.cutoff] = 0.0
    porosity.img[label_boundary_mask(labels, config.boundary_band)] = 0.0
    return porosity


def boolean_image_porosity(
    porosity: darsia.Image, config: ImagePorosityConfig
) -> darsia.Image:
    """Boolean porosity: image porosity above `tol`."""
    return porosity > config.tol


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
        display_step,
        save_scalar_map_illustration,
    )

    folder.mkdir(parents=True, exist_ok=True)
    porosity.save(folder / IMAGE_POROSITY_FILE, verbose=False)
    options = config.method_options()
    info = {
        "method": config.method,
        "regions": config.regions,
        "options": None if options is None else asdict(options),
        "cutoff": config.cutoff,
        "boundary_band": config.boundary_band,
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
    # Figures from every n-th pixel (display only; about 2000 pixels wide)
    step = display_step(porosity.img.shape)
    save_scalar_map_illustration(
        porosity.img[::step, ::step],
        figures / "image_porosity.jpg",
        title="Image porosity",
        colorbar_label="Image porosity",
    )
    save_scalar_map_illustration(
        (porosity.img[::step, ::step] > config.tol).astype(float),
        figures / "boolean_porosity.jpg",
        title=f"Boolean porosity (> {config.tol})",
        colorbar_label="Porous",
    )
    logger.info(f"Image porosity stored in {folder}.")


class ImagePorosity:
    """Stored image porosity of the configured method, read on first use."""

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
        self.folder = image_porosity_folder(results, self.config.method)
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
                raise FileNotFoundError(self._missing_message())
            with darsia.timed("load image porosity"):
                self._map = darsia.imread(self.file)
        return self._map

    def _missing_message(self) -> str:
        """What is missing and the two ways out: set it up, or switch the method to
        one that has been set up."""
        available = sorted(
            path.parent.name
            for path in self.folder.parent.glob(f"*/{IMAGE_POROSITY_FILE}")
        )
        message = (
            "This step needs the image porosity of [image_porosity] method = "
            f"{self.config.method!r}, but it has not been set up (expected: "
            f"{self.file}). {'Either run' if available else 'Run'} the setup step "
            "'Image porosity' with this config (GUI: Setup > Image porosity; CLI: "
            "user_interface_setup --image-porosity)"
        )
        if available:
            message += (
                ", or set [image_porosity] method to one that has been set up: "
                + ", ".join(repr(name) for name in available)
            )
        return message + "."

    def map_on(self, grid: darsia.Image) -> darsia.Image:
        """Image porosity on the grid of `grid` (the stored one on its own grid)."""
        return darsia.transfer(self.map, grid)

    def boolean_on(self, grid: darsia.Image) -> darsia.Image:
        """Boolean porosity on the grid of `grid`."""
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
