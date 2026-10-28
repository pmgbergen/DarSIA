"""RAW-only renderer port for the existing native/ring arithmetic.

The optional positional view paths support the old array-level calling
convention during integration. They are never read, hashed or required here.
The new producer must bind ``raw_acquisition`` and the shape assets instead.
"""

from datetime import datetime
from typing import Mapping

from . import raw_positioning


def render(arw_path, camera_path=None, review_path=None, *,
           expected_source_hashes: Mapping, content_box, parameters: Mapping,
           source_id: str, profile_id: str):
    """Render one RAW acquisition using explicitly bound, unchanged rig models."""
    settings = parameters["raw_positioning"]
    metadata = dict(settings["image_metadata"])
    if isinstance(metadata.get("date"), str):
        metadata["date"] = datetime.fromisoformat(metadata["date"])
    shape = tuple(parameters["common"]["sensor"]["content_wh"])[::-1]
    if tuple(content_box[2:]) != shape[::-1]:
        raise ValueError("RAW native placement differs from the existing content grid")
    sensor = raw_positioning.read_sensor(
        arw_path, expected_sha256=expected_source_hashes["arw_sha256"],
        parameters=parameters, profile_id=profile_id)
    scientific, certificate = raw_positioning.render_native(
        sensor, shape_assets=settings["shape_assets"], shape_hw=shape,
        image_metadata=metadata, parameters=parameters)
    # Numeric consumers still use these names for grids/coefficients. None is
    # an acquisition-JPEG binding; the producer certificate above is RAW-only.
    certificate.update(
        source_id=source_id, run=source_id, profile=profile_id,
        sources=dict(arw_sha256=sensor.source["sha256"]),
        white_level=sensor.metadata["white_level"],
        review_to_camera_h=certificate["native_to_camera_u_h"],
        camera_jpeg_wh=certificate["camera_u_wh"],
        raw_visible_wh=sensor.metadata["raw_visible_wh"],
        sony_distortion_table=list(sensor.distortion),
        sony_ca_red_table=list(sensor.ca_red),
        sony_ca_blue_table=list(sensor.ca_blue),
        legacy_numeric_grid_names_only=True)
    return scientific, certificate
