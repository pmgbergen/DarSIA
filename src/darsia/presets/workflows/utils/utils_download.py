"""Utils for user interface of preset workflows."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadPlan:
    """Resolved selection and metadata for a download action."""

    image_paths: list[Path]
    destination_paths: list[Path]
    destination_dir: Path
    total_size_bytes: int
    total_size_string: str


def _format_size(total_size: int) -> str:
    """Format bytes as MB/GB string."""
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    return f"{total_size_mb:.2f} MB" if total_size_mb < 1024 else f"{total_size_gb:.2f} GB"


def prepare_download_data(path: Path | list[Path] | list[str] | str) -> DownloadPlan:
    """Resolve selected files, destination paths and total size for download."""

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
    if destination_dir is None:
        raise ValueError(
            "Download folder is not configured. "
            "Specify [download.folder] in the configuration or ensure a valid "
            "results path is available to determine a default download folder."
        )
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Reduce to paths that do not exist yet in the destination folder if skip_existing is True
    if config.download.skip_existing:
        image_paths = [
            p for p in image_paths if not (destination_dir / p.name).exists()
        ]
        logger.info(
            f"Found {len(image_paths)} files to download after skipping existing files."
        )

    # Define destination paths for all images to be copied
    destination_paths = [destination_dir / p.name for p in image_paths]

    # Estimate the size of the data to be downloaded
    total_size = sum(p.stat().st_size for p in image_paths)
    total_size_string = _format_size(total_size)

    return DownloadPlan(
        image_paths=image_paths,
        destination_paths=destination_paths,
        destination_dir=destination_dir,
        total_size_bytes=total_size,
        total_size_string=total_size_string,
    )


def download_data(path: Path | list[Path] | list[str] | str, require_confirmation: bool = True):
    """Download raw data for preset workflows."""

    plan = prepare_download_data(path)
    if len(plan.image_paths) == 0:
        logger.info("No files selected for download.")
        return

    # Ask user for confirmation
    if require_confirmation:
        response = input(
            f"""The total size of the data to download is {plan.total_size_string}. """
            """Do you want to proceed? (y/n): """
        )
        if response.lower() != "y":
            print("Download cancelled.")
            return

    # Copy selected files to destination
    for src_path, dst_path in zip(plan.image_paths, plan.destination_paths):
        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied {src_path} to {dst_path}")
