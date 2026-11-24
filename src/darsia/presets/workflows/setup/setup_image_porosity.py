"""Setup step "Image porosity": compute and store the image porosity of the rig.

Runs after the rig setup, on the stored (corrected) baseline and the labels of the rig.
The result is stored per detection method below `<results>/setup/image_porosity/`
(see `results_folder.image_porosity_folder`) and read by the restoration and the colour
calibration. See `processing/image_porosity` for the methods.
"""

import logging
import time
from pathlib import Path
from typing import Type

from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


@required_sections("data", "rig", "image_porosity")
def setup_image_porosity(
    cls: Type[Rig],
    path: Path | list[Path],
    show: bool = False,
) -> None:
    """Compute and store the image porosity configured in `[image_porosity]`.

    Parameters
    ----------
    cls : Type[Rig]
        Class of the rig.
    path : Path | list[Path]
        Path(s) to the config file(s).
    show : bool
        Whether to show the image porosity.

    """
    from darsia.presets.workflows.processing.image_porosity import (
        compute_image_porosity,
        save_image_porosity,
    )
    from darsia.presets.workflows.results_folder import image_porosity_folder

    logger.info("\033[92mSetting up image porosity...\033[0m")
    tic = time.time()

    config = FluidFlowerConfig(path, require_data=False, require_results=True)
    config.check(*list_required_sections(setup_image_porosity))
    assert config.data is not None
    assert config.rig is not None
    porosity_config = config.image_porosity
    assert porosity_config is not None

    rig = cls.load(config.rig.path)

    porosity = compute_image_porosity(
        rig.baseline,
        rig.regions(porosity_config.regions),
        rig.labels,
        porosity_config,
    )
    folder = image_porosity_folder(config.data.results, porosity_config.method)
    inputs = {
        "baseline": Path(config.rig.path) / "baseline.npz",
        "labels": Path(config.rig.path) / "labels.npz",
    }
    if porosity_config.regions == "facies":
        inputs["facies"] = Path(config.rig.path) / "facies.npz"
    save_image_porosity(folder, porosity, porosity_config, inputs=inputs)
    if show:
        porosity.show(title="Image porosity")

    logger.info(f"Image porosity setup in {time.time() - tic:.2f} s.")
