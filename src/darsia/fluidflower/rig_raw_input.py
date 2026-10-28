"""Explicit scientific RAW input for rebuilt workflow rig models.

The existing calibrated camera-U renderer supplies linear camera RGB. Rig
corrections are then built by the original setup sequence on that same grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import numpy as np


def camera_binding(value, config_path=None):
    """Validate and resolve one explicit, serializable camera profile binding."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "configuration", "sha256", "profile_id", "run_id"
    }:
        raise ValueError("data.raw_camera requires configuration, sha256, profile_id and run_id")
    if any(not isinstance(value[key], str) or not value[key].strip()
           for key in ("configuration", "profile_id")):
        raise ValueError("data.raw_camera requires an explicit profile path and id")
    digest = value["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("data.raw_camera.sha256 must be an exact lowercase SHA-256")
    run = value["run_id"]
    if not isinstance(run, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run) is None:
        raise ValueError("data.raw_camera.run_id must be one explicit safe lowercase run id")
    path = Path(value["configuration"])
    if config_path is not None:
        from darsia.presets.workflows.config.utils import _config_path

        path = _config_path(value["configuration"], config_path)
    if not path.is_absolute():
        raise ValueError("data.raw_camera.configuration must resolve to an absolute path")
    return dict(configuration=str(path), sha256=digest, profile_id=value["profile_id"], run_id=run)


def profile_parameters(binding):
    """Load the exact profile and verify its explicit run-to-profile assignment."""
    from darsia.fluidflower.native.io import read_bound_json, resolve_asset_path

    bound = camera_binding(binding)
    if bound is None:
        raise ValueError("RAW rig input requires data.raw_camera")
    from .storage_paths import resolve_storage_path
    profile = resolve_storage_path(resolve_asset_path(bound["configuration"], {}))
    document = read_bound_json(profile, bound["sha256"])
    parameters = document.get("registration", document)
    assignments = document.get("source", {}).get("profiles", {})
    selected = [value for key, value in assignments.items()
                if str(key).casefold() == bound["run_id"]]
    if (len(selected) != 1 or selected[0] != bound["profile_id"]
            or bound["profile_id"] not in parameters["common"]["profiles"]):
        raise ValueError("RAW rig profile differs from its exact run assignment")
    return parameters


def require_raw_rebuild(corrections_config):
    """Do not reuse a photometric correction built from a previous input basis."""
    relative = getattr(corrections_config, "relative_color", None)
    if relative is not None and relative.path is not None:
        raise ValueError("RAW rig setup must rebuild relative color from RAW images; "
                         "a precomputed relative-color path cannot be reused")


def read_raw_image(path, binding, *, transformations=None, dtype=float,
                   expected_sha256=None, **kwargs):
    """Decode RAW on U and use white-normalized linear values for rig setup.

    White balance and gamma are absent. The DN scaling matches the existing
    native RAW shape-chain input; all transformations receive linear values.
    The decoder verifies source bytes before and after reading the sensor.
    """
    import darsia
    from darsia.fluidflower.image_formats import image_family
    from darsia.fluidflower.native import raw_positioning
    from darsia.fluidflower.native.io import sha256_file, resolve_asset_path
    from darsia.fluidflower.storage_paths import resolve_storage_path
    from darsia.fluidflower.photograph_io import raw_capture_time
    from darsia._opencv_runtime import scientific_thread, completed_frame

    ipp_before = scientific_thread("read-raw-rig-image:before")

    path = resolve_storage_path(resolve_asset_path(path, {}))
    if image_family(path) != "raw":
        raise ValueError("A RAW rig model requires RAW optical inputs")
    bound = camera_binding(binding)
    if bound is None:
        raise ValueError("RAW rig input requires data.raw_camera")
    parameters = profile_parameters(bound)
    sensor = raw_positioning.read_sensor(
        path, expected_sha256=(sha256_file(path) if expected_sha256 is None else expected_sha256), parameters=parameters,
        profile_id=bound["profile_id"],
    )
    linear = raw_positioning.camera_u(sensor, parameters=parameters)
    if not np.isfinite(linear).all():
        raise ValueError("RAW rig camera-U grid extends outside its sensor")
    scale = np.float32(sensor.metadata["white_level"] - sensor.metadata["black_level"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("RAW rig input requires a finite positive white range")
    values = np.moveaxis(linear, 0, -1) / scale
    kwargs.setdefault("date", raw_capture_time(path))
    kwargs.setdefault("color_space", "RGB")
    kwargs["series"] = False
    image = darsia.OpticalImage(
        values, transformations=transformations, **kwargs,
    ).img_as(dtype)
    completed_frame(ipp_before, "read-raw-rig-image:after")
    return image
