"""RAW Change through the unchanged rig RAW Change renderer (package D2e).

A synthetic RAW review whose baseline and frames bind rig classes gains a
``change`` photograph, rendered by the pinned renderer with the configured
display and Change parameters. The fixture holds a configured series' method
parameters (its provenance path omitted); the camera-U renderer is stubbed.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import phase_photographic_assets as photos
from darsia.fluidflower import raw_photographic_review as review
from darsia.fluidflower.native import display_modes, raw_photograph
from darsia.fluidflower.native.config import configure_parameters, reset_parameters
from darsia.fluidflower.phase_saved_assets import asset
from .test_raw_photographic_review import PHASE_HW, _write_json, inputs  # noqa: F401

PARAMETERS = json.loads((Path(__file__).parents[1] / "fixtures/display_change_parameters.json")
                        .read_text(encoding="utf-8"))
RENDER_DISPLAY_MODES = display_modes.render_display_modes  # Before any fixture stubs it.


@pytest.fixture
def bound(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(display_modes, "render_display_modes", RENDER_DISPLAY_MODES)
    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8",
                        lambda path, **kwargs: np.full((12, 20, 3), 90, dtype=np.uint8))
    configuration = _write_json(tmp_path / "change_configuration.json", dict(
        registration=dict(common=dict(profiles=dict(late={}))), display_modes=PARAMETERS["display_modes"]))
    token = configure_parameters(dict(
        display_modes=PARAMETERS["display_modes"], change_display=PARAMETERS["change_display"],
        saved_review_metadata=dict(display_views=dict(rig_contract=PARAMETERS["rig_contract"]))))
    domain = np.zeros(PHASE_HW, dtype=bool)
    domain[3:13, 5:21] = True
    empty = np.zeros_like(domain)
    np.savez(tmp_path / "phase_classes.npz", modelled=domain, filled_observed=empty,
             filled_unobserved=empty, rig_domain=domain)
    yield dict(inputs, configuration=configuration, classes=asset(tmp_path / "phase_classes.npz"), domain=domain)
    reset_parameters(token)


def _write(tmp_path, bound, *, reference_classes=True, modes=None, root="review"):
    frames = [dict(frame, rig_classes=bound["classes"]) for frame in bound["frames"]]
    reference = dict(bound["reference"], **({"rig_classes": bound["classes"]} if reference_classes else {}))
    return review.write_raw_review_root(
        tmp_path / root, run="ac52", native_configuration=bound["configuration"],
        raw_binding=bound["binding"], reference=reference, frames=frames, modes=modes)


def test_change_comes_from_the_pinned_renderer_on_each_records_rig_classes(tmp_path, bound):
    manifest = json.loads(_write(tmp_path, bound).read_text(encoding="utf-8"))
    assert manifest["modes"][-1] == "change" and manifest["frames"][1]["rig_classes"] == bound["classes"]
    root = tmp_path / "review"
    change = np.asarray(Image.open(photos.resolve_change_display(root, "ac52", 1)))
    assert change.shape == (*PHASE_HW, 3) and change.dtype == np.uint8
    direct = review.render_change_asset(
        review.raw_sources(json.loads(Path(bound["frames"][1]["native_record"]["path"]).read_text())["full"],
                           manifest["frames"][1]["audit"]),
        bound["classes"], baseline=_baseline(bound), run="ac52", frame_index=1,
        parameters=PARAMETERS["display_modes"])
    assert np.array_equal(change, direct)
    assert photos.change_display_available(root, "ac52", frame_count=2)
    assert np.asarray(Image.open(photos.resolve_change_display(root, "ac52", None, reference=True))).shape[:2] == PHASE_HW


def _baseline(bound):
    acquisitions = json.loads(Path(bound["reference"]["acquisition_manifest"]["path"]).read_text())
    record = acquisitions["records"][0]
    row = json.loads(Path(record["native_row"]["path"]).read_text())
    return review.raw_sources(record["full"], row["raw_producer_audit"])


def test_change_needs_the_rig_classes_of_b_and_every_frame(tmp_path, bound):
    manifest = json.loads(_write(tmp_path, bound, reference_classes=False).read_text(encoding="utf-8"))
    assert "change" not in manifest["modes"]  # B without rig classes: no Change by default.
    assert not photos.change_display_available(tmp_path / "review", "ac52", frame_count=2)
    with pytest.raises(ValueError, match="rig classes of baseline B"):
        _write(tmp_path, bound, reference_classes=False, modes=("natural", "change"), root="explicit")


def test_change_needs_the_reviews_own_bound_display_parameters(tmp_path, bound):
    with pytest.raises(ValueError, match="own bound display parameters"):
        review.render_change_asset({}, bound["classes"], baseline={}, run="ac52", frame_index=0,
                                   parameters=dict(PARAMETERS["display_modes"], modes=["natural"]))
