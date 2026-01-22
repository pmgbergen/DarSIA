"""Module to find events."""

import darsia
from pathlib import Path
from datetime import datetime


def find_images_for_datetimes(
    folder: Path,
    imaging_protocol: darsia.ImagingProtocol,
    datetimes: list[datetime],
) -> list[Path]:
    """Find images in the folder that are closest to the specified times.

    Args:
        folder (Path): Path to the folder containing images.
        imaging_protocol (darsia.ImagingProtocol): Imaging protocol with time information.
        times (list[float]): List of times (in seconds) to find corresponding images for.

    Returns:
        list[darsia.Image]: List of images corresponding to the specified times.

    """
    # Restrict df from imagign_interval to available image ids
    available_paths = list(folder.glob("*"))
    available_image_ids = {imaging_protocol.image_id(p): p for p in available_paths}
    df = imaging_protocol.df[
        imaging_protocol.df["image_id"].isin(available_image_ids.keys())
    ]

    # Collect the closest images
    closest_image_paths = []

    for dt in datetimes:
        closest_available_time = min(
            df["datetime"], key=lambda t: abs((t - dt).total_seconds())
        )
        image_path = available_image_ids[
            df[df["datetime"] == closest_available_time]["image_id"].values[0]
        ]
        closest_image_paths.append(image_path)

    return list(set(closest_image_paths))
