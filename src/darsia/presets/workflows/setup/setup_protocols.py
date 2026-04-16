"""Setup utilities for protocol CSV files."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig

logger = logging.getLogger(__name__)

_SUPPORTED_MODES = {"exif", "ctime"}


def get_modification_time(filepath: Path) -> datetime:
    """Get file modification time as datetime."""
    return datetime.fromtimestamp(filepath.stat().st_mtime)


def _extract_exif_datetime(path: Path) -> datetime | None:
    """Extract EXIF datetime from an image."""
    with Image.open(path) as img:
        exif_data = img.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ["DateTimeOriginal", "DateTime"]:
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            logger.warning("%s: No EXIF datetime found.", path.name)
        else:
            logger.warning("%s: No EXIF data found.", path.name)
    return None


def _protocol_path(
    protocol: Path | tuple[Path, str] | None,
    key: str,
) -> Path | None:
    """Resolve protocol path from config entry."""
    if protocol is None:
        return None
    if isinstance(protocol, tuple):
        logger.warning(
            "Protocol '%s' is configured with sheet tuple; only the file path is used.",
            key,
        )
        return Path(protocol[0])
    return Path(protocol)


def _imaging_protocol_paths(
    protocol: Path | tuple[Path, str] | dict[Path, Path | tuple[Path, str]] | None,
    folders: list[Path],
) -> dict[Path, Path]:
    if protocol is None:
        return {}
    if isinstance(protocol, dict):
        protocol_map = {
            Path(folder): _protocol_path(value, "imaging")
            for folder, value in protocol.items()
        }
        missing = [folder for folder in folders if folder not in protocol_map]
        extra = [folder for folder in protocol_map if folder not in folders]
        if missing:
            raise ValueError(
                "Missing imaging protocol entries for folder(s): "
                + ", ".join(str(folder) for folder in missing)
            )
        if extra:
            raise ValueError(
                "Imaging protocol configured for unknown folder(s): "
                + ", ".join(str(folder) for folder in extra)
            )
        return {
            folder: path for folder, path in protocol_map.items() if path is not None
        }

    if len(folders) > 1:
        raise ValueError(
            "Multiple [data].folders require [protocols].imaging to be a per-folder table."
        )
    single_path = _protocol_path(protocol, "imaging")
    assert single_path is not None
    return {folders[0]: single_path}


def _assert_csv(path: Path, key: str) -> None:
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Protocol '{key}' must be configured as CSV for setup generation: {path}"
        )


def _overwrite_conflicts(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def preview_protocol_setup_conflicts(path: Path | list[Path]) -> list[Path]:
    """Return protocol target files that already exist."""
    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("protocol")
    assert config.protocol is not None

    imaging_targets = _imaging_protocol_paths(
        config.protocol.imaging, config.data.folders
    )
    targets = list(imaging_targets.values())
    targets.extend(
        [
            _protocol_path(config.protocol.injection, "injection"),
            _protocol_path(
                config.protocol.pressure_temperature, "pressure_temperature"
            ),
        ]
    )
    return _overwrite_conflicts([p for p in targets if p is not None])


def _extract_imaging_protocol_dataframe(
    files: list[Path], pad: int, mode: str, root: Path
) -> pd.DataFrame:
    file_paths: list[str] = []
    file_ids: list[int] = []
    date_times: list[datetime] = []

    for i, filename in enumerate(files):
        logger.info("Processing file %s / %s", i + 1, len(files))
        image_id = (
            int(Path(filename).stem[-pad:]) if pad > 0 else int(Path(filename).stem)
        )
        if mode == "exif":
            date_time = _extract_exif_datetime(filename)
        elif mode == "ctime":
            date_time = get_modification_time(filename)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'exif' or 'ctime'.")

        if date_time is None:
            logger.warning(
                "Skipping %s because no datetime could be extracted.", filename
            )
            continue
        file_paths.append(filename.relative_to(root).as_posix())
        file_ids.append(image_id)
        date_times.append(date_time)

    if len(date_times) == 0:
        raise ValueError(
            "No datetimes could be extracted from images. "
            "Use [protocols].imaging_mode = 'ctime' or provide EXIF metadata."
        )

    return pd.DataFrame(
        {"path": file_paths, "image_id": file_ids, "datetime": date_times}
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_injection_template(path: Path, start: datetime, end: datetime) -> None:
    df = pd.DataFrame(
        {
            "id": [1],
            "location_x": [0.0],
            "location_y": [0.0],
            "start": [start],
            "end": [end],
            "rate_kg/s": [0.0],
        }
    )
    _write_csv(df, path)


def _write_pressure_temperature_template(path: Path, start: datetime) -> None:
    df = pd.DataFrame(
        {
            "datetime": [start],
            "pressure_bar": [1.013],
            "temperature_celsius": [20.0],
            "pressure_gradient_bar": [0.0],
            "temperature_gradient_celsius": [0.0],
        }
    )
    _write_csv(df, path)


def setup_imaging_protocol(
    path: Path | list[Path],
    *,
    force: bool = False,
    show: bool = False,
) -> None:
    """Generate imaging/injection/pressure-temperature protocol CSV templates."""
    logger.info("\033[92mSetting up protocol CSV templates...\033[0m")
    del show

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("data", "protocol")
    assert config.data is not None
    assert config.protocol is not None
    assert config.protocol.imaging is not None

    imaging_targets = _imaging_protocol_paths(
        config.protocol.imaging, config.data.folders
    )
    injection_path = _protocol_path(config.protocol.injection, "injection")
    pressure_temperature_path = _protocol_path(
        config.protocol.pressure_temperature, "pressure_temperature"
    )

    for imaging_path in imaging_targets.values():
        _assert_csv(imaging_path, "imaging")
    if injection_path is not None:
        _assert_csv(injection_path, "injection")
    if pressure_temperature_path is not None:
        _assert_csv(pressure_temperature_path, "pressure_temperature")

    conflicts = _overwrite_conflicts(
        [
            *imaging_targets.values(),
            *[p for p in [injection_path, pressure_temperature_path] if p is not None],
        ]
    )
    if conflicts and not force:
        conflict_text = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Protocol file(s) already exist: {conflict_text}. Use --force to overwrite."
        )

    mode = config.protocol.imaging_mode
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported [protocols].imaging_mode '{mode}'. "
            f"Supported values are: {sorted(_SUPPORTED_MODES)}."
        )

    overall_start: datetime | None = None
    overall_end: datetime | None = None
    suffix = config.data.baseline.suffix
    for folder, imaging_path in imaging_targets.items():
        files = sorted(path for path in folder.rglob(f"*{suffix}") if path.is_file())
        if len(files) == 0:
            raise FileNotFoundError(
                f"No image files with suffix {suffix} found in {folder}."
            )
        imaging_df = _extract_imaging_protocol_dataframe(
            files, config.data.pad, mode, folder
        )
        _write_csv(imaging_df, imaging_path)
        logger.info("Saved imaging protocol CSV to %s", imaging_path)

        start = pd.to_datetime(imaging_df["datetime"]).min().to_pydatetime()
        end = pd.to_datetime(imaging_df["datetime"]).max().to_pydatetime()
        overall_start = start if overall_start is None else min(overall_start, start)
        overall_end = end if overall_end is None else max(overall_end, end)

    assert overall_start is not None
    assert overall_end is not None
    if injection_path is not None:
        _write_injection_template(injection_path, overall_start, overall_end)
        logger.info("Saved injection protocol CSV template to %s", injection_path)
    if pressure_temperature_path is not None:
        _write_pressure_temperature_template(pressure_temperature_path, overall_start)
        logger.info(
            "Saved pressure-temperature protocol CSV template to %s",
            pressure_temperature_path,
        )
