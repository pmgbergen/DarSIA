"""Shared, configuration-driven U photographs for GUI and scientific callers."""

from pathlib import Path

import numpy as np

from . import raw_positioning
from .io import read_bound_json, sha256_file


def _parameters(profile_path, profile_sha256, profile_id):
    document = read_bound_json(Path(profile_path), profile_sha256)
    parameters = document.get("registration", document)
    if profile_id not in parameters["common"]["profiles"]:
        raise ValueError("Camera-U profile id is not declared in the bound profile")
    return parameters


def camera_u_size(*, profile_path, profile_sha256, profile_id):
    """Declared (width, height) of U; reading the photograph verifies its sensor."""
    parameters = _parameters(profile_path, profile_sha256, profile_id)
    return tuple(parameters["common"]["sensor"]["camera_wh"])


def load_camera_u_linear(path, *, profile_path, profile_sha256, profile_id,
                         expected_sha256=None):
    """Return CHW float32, unbalanced black-subtracted camera DN on U."""
    parameters = _parameters(profile_path, profile_sha256, profile_id)
    digest = sha256_file(path) if expected_sha256 is None else expected_sha256
    sensor = raw_positioning.read_sensor(
        path, expected_sha256=digest, parameters=parameters, profile_id=profile_id)
    return raw_positioning.camera_u(sensor, parameters=parameters)


def load_camera_u_rgb8(path, *, profile_path, profile_sha256, profile_id,
                      factor=1, expected_sha256=None, source_exception=None):
    """Return HWC uint8 display RGB on the same calibrated U lattice.

    LibRaw supplies the existing camera-WB/standard-gamma display conversion.
    The shared G lens map locates that mixed display RGB; it is never used as
    scientific DN. Profiles, dimensions and coefficients come from data.
    """
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("A positive integer display reduction is required")
    import rawpy
    from PIL import Image

    path = Path(path)
    parameters = _parameters(profile_path, profile_sha256, profile_id)
    digest = sha256_file(path) if expected_sha256 is None else expected_sha256
    display_profile_id = profile_id
    if source_exception is not None:
        from ..clearwater_source_choice import source_instrument_parameters
        parameters, display_profile_id = source_instrument_parameters(parameters,
            binding=source_exception, path=path, expected_sha256=digest,
            profile_path=profile_path, profile_sha256=profile_sha256, profile_id=profile_id)
    sensor = raw_positioning.read_sensor(
        path, expected_sha256=digest, parameters=parameters, profile_id=display_profile_id)
    before = raw_positioning._version(path)
    with rawpy.imread(str(path)) as source:
        display = source.postprocess(use_camera_wb=True, output_bps=8,
                                     half_size=False, user_flip=0)
    if (raw_positioning._version(path) != before
            or sha256_file(path) != digest):
        raise ValueError("RAW photograph changed during its display rendering")
    width, height = parameters["common"]["sensor"]["camera_wh"]
    warped = raw_positioning.sample_rgb_values(
        sensor, display, np.eye(3), shape_hw=(height, width),
        parameters=parameters, chromatic=False)
    if not np.isfinite(warped).all():
        raise ValueError("The declared camera-U display extends outside its sensor")
    rgb = np.moveaxis(np.clip(np.rint(warped), 0, 255).astype(np.uint8), 0, -1)
    if factor > 1:
        size = (max(1, width // factor), max(1, height // factor))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX))
    return rgb
