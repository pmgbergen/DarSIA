"""Applying saved RGB geometry has no RAW decoder or RAW Build dependency."""
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower import corrected_setup as setup
from .test_geometry_imported_rasters import imported  # noqa: F401
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_saved_rgb_maps_apply_to_all_runs_without_original_raw_images(registered_setup, monkeypatch):
    from darsia.fluidflower import rig_raw_input, rig_raw_build

    def forbidden(*args, **kwargs):
        raise AssertionError("Apply attempted a RAW decode or RAW Build")

    monkeypatch.setattr(rig_raw_input, "read_raw_image", forbidden)
    monkeypatch.setattr(rig_raw_build, "read_build_request", forbidden)
    data = registered_setup
    assert not list(data.series.rglob("*.arw"))
    assert not list(data.series.rglob("*.dng"))
    for run in ("one", "two"):
        exported = setup.export_corrected_setup(data.project, reference=SimpleNamespace(run_id=run))
        receipt = setup.apply_corrected_selected_run(data.project, exported)
        assert json.loads(receipt.read_bytes())["state"] == "complete"
        completed = setup._apply_completion(exported.destination, exported.receipt)
        assert completed["complete"] is True and completed["run_id"] == run
        assert not exported.receipt["actual_Build_run"]
