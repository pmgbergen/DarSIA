"""Module to find events."""

from datetime import datetime
from pathlib import Path

import darsia


def find_images_for_datetimes(
    folder: Path,
    imaging_protocol: darsia.ImagingProtocol,
    datetimes: list[datetime],
) -> list[Path]:
    """Find images in the folder that are closest to the specified times.

    Parameters
    ----------
    folder : Path
        Path to the folder containing images.
    imaging_protocol : darsia.ImagingProtocol
        Imaging protocol with time information.
    datetimes : list of datetime
        List of datetimes to find corresponding images for.

    Returns
    -------
    list[Path]
        List of image paths corresponding to the specified datetimes.
    """
    # Restrict df from imaging_protocol to available paths
    available_paths = list(folder.glob("*"))
    available_paths_by_key: dict[str, Path] = {}
    for p in available_paths:
        for key in imaging_protocol._candidate_protocol_paths(p):
            if key in imaging_protocol.datetime_by_path_key:
                available_paths_by_key[key] = p
                break
    df = imaging_protocol.df[
        imaging_protocol.df["path"].isin(available_paths_by_key.keys())
    ]

    # Collect the closest images
    closest_image_paths = []

    for dt in datetimes:
        closest_available_time = min(
            df["datetime"], key=lambda t: abs((t - dt).total_seconds())
        )
        image_path = available_paths_by_key[
            df[df["datetime"] == closest_available_time]["path"].values[0]
        ]
        closest_image_paths.append(image_path)

    return list(set(closest_image_paths))
