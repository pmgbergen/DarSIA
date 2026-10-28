"""Explicit geometry schemas preserve original matrices without alias guessing."""

import json
import numpy as np
import pytest
from tests.test_n1_pipeline import tiny_config
from darsia.fluidflower import n1_pipeline as legacy
from darsia.fluidflower.mass import n1_pipeline as core
from darsia.fluidflower.mass import n1_geometry as geometry
from darsia.fluidflower.mass.production_precision import PixelMath


def test_canonical_config_requires_fields_before_output(tiny_config, tmp_path):
    """A missing explicit schema cannot silently select a historical field."""
    config, descriptor = tiny_config
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="geometry_fields"):
        core.run_n1(
            dict(run=config["run"], configuration=descriptor), output, "cpu", "float32"
        )
    assert not output.exists()


def test_explicit_alternate_names_retain_native_math(tiny_config):
    """Renamed keys yield identical numerical fields and original provenance."""
    config, descriptor = tiny_config
    expected = legacy.N1Pipeline(
        config, precision="float32", pixel_math=PixelMath("float32")
    ).load_frame(config["frames"][0])
    config = json.loads(json.dumps(config))
    fields = dict(
        source_to_native="reference_to_native",
        original_domain_to_native="domain_reference_to_native",
        carried_homography_provenance="original_transform",
    )
    config["geometry_fields"] = fields
    for spec in config["frames"]:
        for name in ("current", "baseline"):
            row = spec[name]["binding"]["geometry"]
            row[fields["source_to_native"]] = row.pop("H_ac28_to_frame_native")
        row = spec["original_domain"]
        row[fields["original_domain_to_native"]] = row.pop("H_ac28_to_frame_native")
    actual = core.N1Pipeline(
        config, precision="float32", pixel_math=PixelMath("float32")
    ).load_frame(config["frames"][0])
    for field in ("current_rgb", "baseline_rgb", "depth_m", "porosity", "phase_labels"):
        assert getattr(actual[0], field).dtype == getattr(expected[0], field).dtype
        assert (
            getattr(actual[0], field).tobytes() == getattr(expected[0], field).tobytes()
        )
    for left, right in zip(actual[1:4], expected[1:4]):
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
    proof = actual[0].provenance["original_bottom_domain"]
    assert (
        proof["original_transform"]
        == expected[0].provenance["original_bottom_domain"]["original_H_ac28_to_native"]
    )
    assert "original_H_ac28_to_native" not in proof


def test_explicit_and_legacy_mappings_cannot_disagree(tiny_config):
    """Compatibility cannot overwrite a declared competing source field map."""
    config, _ = tiny_config
    config["geometry_fields"] = dict(
        source_to_native="other",
        original_domain_to_native="other",
        carried_homography_provenance="other",
    )
    with pytest.raises(ValueError, match="differ"):
        legacy.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))


def test_metadata_supplies_transform_not_the_helper():
    """The public coordinate helper requires the declared field key."""
    fields = dict(source_to_native="transform")
    actual = geometry.frame_coordinates(
        np.eye(3),
        {"transform": np.eye(3)},
        {"phase_shape_hw": [2, 3], "native_origin_xy": [0, 0]},
        fields=fields,
    )
    np.testing.assert_array_equal(
        actual[0], np.broadcast_to(np.arange(3, dtype=np.float32), (2, 3))
    )
    np.testing.assert_array_equal(
        actual[1], np.broadcast_to(np.arange(2, dtype=np.float32)[:, None], (2, 3))
    )
    with pytest.raises(KeyError):
        geometry.frame_coordinates(
            np.eye(3),
            {"different": np.eye(3)},
            {"phase_shape_hw": [2, 3], "native_origin_xy": [0, 0]},
            fields=fields,
        )
