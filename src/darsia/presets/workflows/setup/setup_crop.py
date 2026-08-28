"""Interactive setup of crop correction via CropAssistant.

This routine uses an interactive matplotlib point-picker to allow the user to
select the four corners of the FluidFlower region in a baseline image, then
stores the crop configuration in the run's TOML config file.
"""

import logging
import time
from pathlib import Path

import toml

import darsia
from darsia.presets.workflows.config.corrections import CropCorrectionConfig
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@required_sections("data", "corrections", "rig")
def setup_crop_correction(path: Path | list[Path], show: bool = False) -> None:
    """Setup and store crop correction configuration interactively.

    Uses CropAssistant to let the user click 4 corner points on the baseline
    image, then stores the resulting crop configuration in the run's TOML file.
    Width and height default to the rig configuration; if not available there,
    the user is prompted for them interactively.

    Args:
        path: Path(s) to the config file(s).
        show: Whether to show intermediate results.
    """
    logger.info("\033[92mSetting up crop correction...\033[0m")

    tic = time.time()

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check(*list_required_sections(setup_crop_correction))

    assert config.data is not None
    assert config.rig is not None

    baseline_path = config.data.baseline

    if config.data.use_cache:
        assert config.data.raw_cache is not None
        baseline_relative = None
        for folder in config.data.folders:
            try:
                baseline_relative = baseline_path.resolve().relative_to(
                    folder.resolve()
                )
                break
            except ValueError:
                continue
        if baseline_relative is None:
            baseline_relative = Path(baseline_path.name)
        cache_path = config.data.raw_cache / baseline_relative.with_suffix(".npz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            baseline_path = cache_path
        else:
            original_baseline = darsia.imread(baseline_path)
            original_baseline.save(cache_path)
            baseline_path = cache_path

    baseline_image = darsia.imread(baseline_path)

    logger.info("Starting interactive crop corner selection...")
    crop_assistant = darsia.CropAssistant(
        baseline_image, width=config.rig.width, height=config.rig.height
    )
    result_dict = crop_assistant()

    logger.info("Crop corners selected successfully.")

    crop_config = CropCorrectionConfig(**result_dict["crop"])

    if show:
        cropped = crop_assistant.image.crop(crop_assistant.config)
        cropped.show()

    config_path = Path(path) if isinstance(path, (str, Path)) else Path(path[0])
    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    if "corrections" not in config_dict:
        config_dict["corrections"] = {}
    if "curvature" not in config_dict["corrections"]:
        config_dict["corrections"]["curvature"] = {}

    config_dict["corrections"]["curvature"]["crop"] = {
        "top_left": list(crop_config.top_left) if crop_config.top_left else None,
        "bottom_left": (
            list(crop_config.bottom_left) if crop_config.bottom_left else None
        ),
        "bottom_right": (
            list(crop_config.bottom_right) if crop_config.bottom_right else None
        ),
        "top_right": list(crop_config.top_right) if crop_config.top_right else None,
        "width": float(crop_config.width),
        "height": float(crop_config.height),
        "in_meters": bool(crop_config.in_meters),
    }

    if "active" not in config_dict["corrections"]["curvature"]:
        config_dict["corrections"]["curvature"]["active"] = []

    active_list = config_dict["corrections"]["curvature"]["active"]
    if "crop" not in active_list:
        active_list.append("crop")

    with open(config_path, "w") as f:
        toml.dump(config_dict, f)

    logger.info(f"Crop correction saved to {config_path}.")
    logger.info(
        f"Corners: top_left={crop_config.top_left}, "
        f"bottom_left={crop_config.bottom_left}, "
        f"bottom_right={crop_config.bottom_right}, "
        f"top_right={crop_config.top_right}"
    )

    logger.info(f"Crop correction setup in {time.time() - tic:.2f} s.")
