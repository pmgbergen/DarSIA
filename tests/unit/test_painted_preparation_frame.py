from copy import deepcopy

import pytest

from darsia.fluidflower.painted_adoption_table import _frame
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import FRAME_FROM_STATIC_REFERENCES, RigFrame


def payload(tmp_path):
    geometry = reference_frame_geometry({
        "top_left": [0., 0.], "top_right": [0., 100.],
        "bottom_right": [60., 100.], "bottom_left": [60., 0.]},
        measurements_m={"width": .9, "height": .49})
    return RigFrame(**geometry, run_id="sample", source="reviewed reference points",
        image=str(tmp_path / "photograph.png"), image_size=(101, 61)).to_dict()


def test_legacy_frame_and_explicit_reference_metadata_keep_exact_geometry(tmp_path):
    original = payload(tmp_path)
    original["reference_geometry"]["point_descriptions"] = {"top_left": "top_of_free_water"}
    expected = deepcopy(original)
    frame = _frame(original)
    assert frame.to_dict() == expected
    frame.reference_geometry["point_descriptions"]["top_left"] = "changed locally"
    assert original == expected
    del original["reference_geometry"]
    assert _frame(original).to_dict() == original


@pytest.mark.parametrize("field,value", [
    ("unexpected", {}), ("reference_geometry", []), ("reference_geometry", None),
    ("reference_geometry", {"measurement": float("nan")}),
    ("reference_geometry", {"measurement": float("inf")}),
    ("reference_geometry", {1: "not a JSON object key"}),
    ("disabled_corners", "top_left"), ("disabled_corners", ["unknown"]),
    ("disabled_corners", ["top_left", "top_left"]),
    ("disabled_corners", ["top_left"]),
])
def test_invalid_extra_geometry_is_rejected(tmp_path, field, value):
    original = payload(tmp_path)
    original[field] = value
    with pytest.raises(ValueError):
        _frame(original)


def test_measured_model_retains_disabled_corner_identity(tmp_path):
    original = payload(tmp_path)
    original.update(source=FRAME_FROM_STATIC_REFERENCES, disabled_corners=["top_left"])
    frame = _frame(original)
    assert frame.complete()
    assert frame.to_dict() == original


def test_reference_metadata_never_supplies_missing_frame_coordinates(tmp_path):
    original = payload(tmp_path)
    del original["corners"]["top_left"]
    with pytest.raises(ValueError, match="every explicit preparation frame corner"):
        _frame(original)
