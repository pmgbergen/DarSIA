"""Utils for user interface of preset workflows."""

import logging
import shutil
from pathlib import Path

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)


def download_data(path: Path):
    """Download raw data for preset workflows."""

    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("data")

    # ! ---- LOAD EXPERIMENT ----
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD IMAGES ----

    image_paths = select_image_paths(
        config,
        experiment,
        all=False,
        sub_config=config.download,
        source=config.download.source,
    )

    # Define the source and destination directories
    destination_dir = config.download.folder
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Reduce to paths that do not exist yet in the destination folder if skip_existing is True
    if config.download.skip_existing:
        destination_dir = config.download.folder
        image_paths = [
            p for p in image_paths if not (destination_dir / p.name).exists()
        ]
        destination_paths = [destination_dir / p.name for p in image_paths]
        logger.info(
            f"Found {len(image_paths)} files to download after skipping existing files."
        )

    # Estimate the size of the data to be downloaded
    total_size = sum(p.stat().st_size for p in image_paths)
    total_size_MB = total_size / (1024 * 1024)
    total_size_GB = total_size / (1024 * 1024 * 1024)
    total_size_string = (
        f"{total_size_MB:.2f} MB" if total_size_MB < 1024 else f"{total_size_GB:.2f} GB"
    )

    # Ask user for confirmation if the total size is larger than 1 GB
    response = input(
        f"The total size of the data to download is {total_size_string}. Do you want to proceed? (y/n): "
    )
    if response.lower() != "y":
        print("Download cancelled.")
        return

    # Copy selected files to destination
    for src_path, dst_path in zip(image_paths, destination_paths):
        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied {src_path} to {dst_path}")
