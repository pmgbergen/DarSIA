"""The crop section written by the crop setup holds plain integers."""

import numpy as np
import toml

from darsia.presets.workflows.config.corrections import CropCorrectionConfig
from darsia.presets.workflows.setup.setup_crop import _crop_section


def test_crop_section_is_written_with_plain_integers():
    corners = {
        "top_left": (np.int64(743), np.int64(69)),
        "bottom_left": np.array([5236, 36]),
        "bottom_right": (5289, 7828),
        "top_right": None,
    }
    config = CropCorrectionConfig(**corners, width=0.97, height=0.54, in_meters=True)
    section = _crop_section(config)

    assert section["top_left"] == [743, 69]
    assert all(type(x) is int for x in section["bottom_left"])
    assert section["top_right"] is None

    text = toml.dumps({"crop": section})
    assert "np." not in text
    assert toml.loads(text)["crop"]["top_left"] == [743, 69]
