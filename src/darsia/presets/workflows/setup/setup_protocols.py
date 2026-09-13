"""Setup utilities for protocol CSV files."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from natsort import natsorted, ns
from PIL import Image
from PIL.ExifTags import TAGS

from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)

logger = logging.getLogger(__name__)

_SUPPORTED_MODES = {"exif", "ctime", "interval", "Load from CSV"}


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
    protocol: dict[Path, Path | tuple[Path, str]] | None,
    folders: list[Path],
) -> dict[Path, Path]:
    if protocol is None:
        return {}
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
    return {folder: path for folder, path in protocol_map.items() if path is not None}


def _assert_csv(path: Path, key: str) -> None:
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Protocol '{key}' must be configured as CSV for setup generation: {path}"
        )


def _overwrite_conflicts(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _generation_targets(
    config: FluidFlowerConfig,
    imaging_targets: dict[Path, Path],
    injection_path: Path | None,
    pressure_temperature_path: Path | None,
) -> list[Path]:
    """Targets that generation would actually write to, given the configured modes.

    Excludes any target whose protocol type is set to 'Load from CSV' (already
    authored / left to the user): those are never overwritten when they already
    exist, so an existing file there is not an overwrite conflict. (A 'Load from
    CSV' injection target that doesn't exist yet still gets an empty template
    written, but creating a new file is never an overwrite conflict either way.)
    """
    targets = []
    if config.protocols.imaging_mode != "Load from CSV":
        targets.extend(imaging_targets.values())
    if (
        injection_path is not None
        and config.protocols.injection_mode != "Load from CSV"
    ):
        targets.append(injection_path)
    if (
        pressure_temperature_path is not None
        and config.protocols.pressure_temperature_mode != "Load from CSV"
    ):
        targets.append(pressure_temperature_path)
    return targets


def preview_protocol_setup_conflicts(path: Path | list[Path]) -> list[Path]:
    """Return protocol target files that already exist."""
    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check("protocols")
    assert config.protocols is not None

    imaging_targets = _imaging_protocol_paths(
        config.protocols.imaging, config.data.folders
    )
    injection_path = _protocol_path(config.protocols.injection, "injection")
    pressure_temperature_path = _protocol_path(
        config.protocols.pressure_temperature, "pressure_temperature"
    )
    targets = _generation_targets(
        config, imaging_targets, injection_path, pressure_temperature_path
    )
    return _overwrite_conflicts(targets)


def _extract_imaging_protocol_dataframe(
    files: list[Path], mode: str, root: Path
) -> pd.DataFrame:
    file_paths: list[str] = []
    date_times: list[datetime] = []

    for i, filename in enumerate(files):
        logger.info("Processing file %s / %s", i + 1, len(files))
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
        date_times.append(date_time)

    if len(date_times) == 0:
        raise ValueError(
            "No datetimes could be extracted from images. "
            "Use [protocols].imaging_mode = 'ctime' or provide EXIF metadata."
        )

    return pd.DataFrame({"path": file_paths, "datetime": date_times})


def _build_interval_datetimes(
    files: list[Path], start: datetime, interval_seconds: float
) -> list[datetime]:
    """Compute prescribed-cadence datetimes for a sorted file list.

    No per-file I/O: each file's datetime is ``start + index * interval_seconds``.
    """
    return [start + timedelta(seconds=i * interval_seconds) for i in range(len(files))]


def _resolve_start_reference(
    config: FluidFlowerConfig, folder: Path, files: list[Path]
) -> datetime:
    """Resolve [protocols].start_reference to a concrete datetime for `folder`.

    'first_image' resolves per folder (that folder's first sorted file). 'fixed'
    resolves to the configured ISO datetime regardless of folder/files. 'baseline'
    and 'image' resolve to the timestamp of a specific image, which must be found
    within `files` (raises FileNotFoundError otherwise, letting a caller searching
    across several folders try the next one).
    """
    choice = config.protocols.start_reference
    mode = config.protocols.imaging_mode
    extraction_mode = "ctime" if mode == "interval" else mode

    if choice == "fixed":
        fixed = config.protocols.start_reference_fixed
        if not fixed:
            raise ValueError(
                "start_reference='fixed' requires [protocols].start_reference_fixed."
            )
        return datetime.fromisoformat(fixed)

    if choice == "first_image":
        if not files:
            raise ValueError(f"No images found in {folder} to resolve start_reference.")
        target = files[0]
    elif choice == "baseline":
        target = config.data.baseline
    elif choice == "image":
        if config.protocols.start_reference_image is None:
            raise ValueError(
                "start_reference='image' requires [protocols].start_reference_image."
            )
        target = config.protocols.start_reference_image
    else:
        raise ValueError(f"Unknown start_reference: {choice}")

    if choice in {"baseline", "image"} and target not in files:
        matches = [f for f in files if f.name == target.name]
        if not matches:
            raise FileNotFoundError(
                f"Reference image {target} not found in folder {folder}."
            )
        target = matches[0]

    if extraction_mode == "exif":
        date_time = _extract_exif_datetime(target)
        if date_time is None:
            raise ValueError(f"Could not extract EXIF datetime from {target}.")
        return date_time
    return get_modification_time(target)


def _resolve_global_start_reference(
    config: FluidFlowerConfig, imaging_targets: dict[Path, Path]
) -> datetime | None:
    """Resolve the single, run-wide start_reference datetime, or None.

    Returns None for 'first_image' (resolved per folder instead, by the caller).
    For 'baseline'/'fixed'/'image', tries each folder in turn until the reference
    resolves (the reference image need not live in every folder).
    """
    choice = config.protocols.start_reference
    if choice == "first_image":
        return None

    suffix = config.data.baseline.suffix
    last_error: Exception | None = None
    for folder in imaging_targets:
        files = natsorted(
            (folder / name for name in os.listdir(folder) if name.endswith(suffix)),
            alg=ns.IGNORECASE,
        )
        try:
            return _resolve_start_reference(config, folder, files)
        except FileNotFoundError as exc:
            last_error = exc
            continue
    raise FileNotFoundError(
        f"Could not resolve start_reference={choice!r} in any configured imaging "
        "folder."
    ) from last_error


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# Volumetric rate units whose csv column needs a paired "density kg/m3" column
# for DarSIA's read side to convert to kg/s (see InjectionProtocol._load_protocol
# in experiment/protocols.py).
_VOLUMETRIC_INJECTION_RATE_UNITS = ("mL/s", "mL/min", "mL/hr")


def _write_injection_template(
    path: Path,
    start: datetime,
    end: datetime,
    rate: float,
    coordinates: tuple[float, float],
    rate_unit: str = "kg/s",
    density: float = 0.0,
) -> None:
    """Write a single-row injection protocol at a constant rate for the whole run.

    `rate_unit` selects the csv column DarSIA's read side expects for that unit;
    the raw value is written as entered, unconverted. `density` (kg/m3) is only
    written when `rate_unit` is volumetric (mL/s, mL/min, mL/hr), which need it
    to convert to kg/s on read.
    """
    rate_column = f"rate_{rate_unit}"
    columns = {
        "id": [1],
        "location_x": [coordinates[0]],
        "location_y": [coordinates[1]],
        "start": [start],
        "end": [end],
        rate_column: [rate],
    }
    if rate_unit in _VOLUMETRIC_INJECTION_RATE_UNITS:
        columns["density kg/m3"] = [density]
    df = pd.DataFrame(columns)
    _write_csv(df, path)


def _write_empty_injection_template(path: Path) -> None:
    """Write a header-only injection-protocol CSV, no guessed values.

    Used for injection_mode='Load from CSV' when no file exists yet: gives the user a
    correctly-shaped file to fill in by hand, rather than plausible-looking
    placeholder numbers (id=1, rate=0.0, etc.) that could be mistaken for real data.
    """
    df = pd.DataFrame(
        columns=["id", "location_x", "location_y", "start", "end", "rate_kg/s"]
    )
    _write_csv(df, path)


def _write_pressure_temperature_template(
    path: Path, start: datetime, pressure_bar: float, temperature_celsius: float
) -> None:
    df = pd.DataFrame(
        {
            "datetime": [start],
            "pressure_bar": [pressure_bar],
            "temperature_celsius": [temperature_celsius],
            "pressure_gradient_bar": [0.0],
            "temperature_gradient_celsius": [0.0],
        }
    )
    _write_csv(df, path)


@required_sections("data", "protocols")
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
    config.check(*list_required_sections(setup_imaging_protocol))
    assert config.data is not None
    assert config.protocols is not None
    assert config.protocols.imaging is not None

    imaging_targets = _imaging_protocol_paths(
        config.protocols.imaging, config.data.folders
    )
    injection_path = _protocol_path(config.protocols.injection, "injection")
    pressure_temperature_path = _protocol_path(
        config.protocols.pressure_temperature, "pressure_temperature"
    )

    for imaging_path in imaging_targets.values():
        _assert_csv(imaging_path, "imaging")
    if injection_path is not None:
        _assert_csv(injection_path, "injection")
    if pressure_temperature_path is not None:
        _assert_csv(pressure_temperature_path, "pressure_temperature")

    conflicts = _overwrite_conflicts(
        _generation_targets(
            config, imaging_targets, injection_path, pressure_temperature_path
        )
    )
    if conflicts and not force:
        conflict_text = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Protocol file(s) already exist: {conflict_text}. Use --force to overwrite."
        )

    mode = config.protocols.imaging_mode
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported [protocols].imaging_mode '{mode}'. "
            f"Supported values are: {sorted(_SUPPORTED_MODES)}."
        )

    global_start_reference = _resolve_global_start_reference(config, imaging_targets)

    overall_start: datetime | None = None
    overall_end: datetime | None = None
    suffix = config.data.baseline.suffix
    for folder, imaging_path in imaging_targets.items():
        files = natsorted(
            (folder / name for name in os.listdir(folder) if name.endswith(suffix)),
            alg=ns.IGNORECASE,
        )
        if len(files) == 0:
            raise FileNotFoundError(
                f"No image files with suffix {suffix} found in {folder}."
            )

        folder_start = (
            global_start_reference
            if global_start_reference is not None
            else _resolve_start_reference(config, folder, files)
        )

        if mode == "Load from CSV":
            if not imaging_path.exists():
                raise FileNotFoundError(
                    f"imaging_mode='Load from CSV' but {imaging_path} does not "
                    "exist. "
                    "Author it by hand, or choose 'exif'/'ctime'/'interval' to "
                    "generate it instead."
                )
            end = pd.to_datetime(pd.read_csv(imaging_path)["datetime"]).max()
            end = end.to_pydatetime()
        else:
            if mode == "interval":
                interval_map = config.protocols.imaging_interval_seconds or {}
                if folder not in interval_map:
                    raise ValueError(
                        f"Missing imaging interval (seconds) for folder {folder}. Set "
                        "it in [protocols.imaging_interval_seconds]."
                    )
                imaging_df = pd.DataFrame(
                    {
                        "path": [f.relative_to(folder).as_posix() for f in files],
                        "datetime": _build_interval_datetimes(
                            files, folder_start, interval_map[folder]
                        ),
                    }
                )
            else:
                imaging_df = _extract_imaging_protocol_dataframe(files, mode, folder)
            _write_csv(imaging_df, imaging_path)
            logger.info("Saved imaging protocol CSV to %s", imaging_path)
            end = pd.to_datetime(imaging_df["datetime"]).max().to_pydatetime()

        if overall_start is None:
            overall_start = folder_start
        overall_end = end if overall_end is None else max(overall_end, end)

    assert (
        overall_start is not None
    ), "No imaging data found; cannot determine overall start time."
    assert (
        overall_end is not None
    ), "No imaging data found; cannot determine overall end time."
    if injection_path is not None:
        if config.protocols.injection_mode == "constant":
            _write_injection_template(
                injection_path,
                overall_start,
                overall_end,
                config.protocols.injection_rate,
                config.protocols.injection_coordinates,
                config.protocols.injection_rate_unit,
                config.protocols.injection_density,
            )
            logger.info("Saved injection protocol CSV template to %s", injection_path)
        elif not injection_path.exists():
            _write_empty_injection_template(injection_path)
            logger.info(
                "Saved empty injection protocol CSV template to %s", injection_path
            )
    if (
        pressure_temperature_path is not None
        and config.protocols.pressure_temperature_mode == "constant"
    ):
        _write_pressure_temperature_template(
            pressure_temperature_path,
            overall_start,
            config.protocols.pressure_bar,
            config.protocols.temperature_celsius,
        )
        logger.info(
            "Saved pressure-temperature protocol CSV template to %s",
            pressure_temperature_path,
        )
