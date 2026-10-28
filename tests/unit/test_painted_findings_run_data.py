"""Run descriptions from finite metadata without products or adoption authority."""

from copy import deepcopy

import pytest

from darsia.fluidflower.map_adoption import adoption_run_data


@pytest.mark.parametrize("state", ["unknown", "indicator"])
def test_adoption_run_data_keeps_six_fields_and_explicit_state(tmp_path, state):
    """Retain the actual source locator and leave caller metadata untouched."""
    labels = tmp_path / "absent-products" / "labels.npz"
    frame_row = dict(fields=dict(shape_corrected_baseline=dict(
        path="relative/source.npz", sha256="a" * 64, size_bytes=9)), unrelated="retained")
    before = deepcopy(frame_row)

    result = adoption_run_data("campaign-run", frame_row, str(labels), acquisition_state=state)

    assert result == dict(run_id="campaign-run", enabled=True, geometry_choice="run_override",
                          acquisition_state=state, labels_path=str(labels),
                          source_image_path="relative/source.npz")
    assert list(result) == ["run_id", "enabled", "geometry_choice", "acquisition_state",
                            "labels_path", "source_image_path"]
    assert frame_row == before and not labels.parent.exists()
    result["source_image_path"] = "changed returned metadata"
    assert frame_row == before


@pytest.mark.parametrize("frame_row", [{}, {"fields": {"shape_corrected_baseline": None}}])
def test_adoption_run_data_keeps_missing_photo_empty_without_synthesizing_a_binding(tmp_path, frame_row):
    """Missing source metadata supplies no descriptor, proof or inferred state."""
    labels = tmp_path / "absent-products" / "labels.npz"
    before = deepcopy(frame_row)

    result = adoption_run_data("campaign-run", frame_row, labels, acquisition_state="clearwater")

    assert result == dict(run_id="campaign-run", enabled=True, geometry_choice="run_override",
                          acquisition_state="clearwater", labels_path=str(labels), source_image_path="")
    assert frame_row == before and not labels.parent.exists()
