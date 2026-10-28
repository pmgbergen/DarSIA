"""New-preview policy keeps source provenance and displays authorized continuation."""
from copy import deepcopy
from dataclasses import replace
import threading

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import painting_photo_continuation as display
from darsia.fluidflower import photo_edge_continuation as policy
from darsia.fluidflower import painting_run_preview as core
from darsia.fluidflower import painting_preview_frame_domain as old_domain

from .test_painting_series_frame_models import context  # noqa: F401
from .test_painting_series_frame_models import _files
from ..test_photo_edge_continuation import closed_case  # noqa: F401


def _drawing(context, *, left=0.):
    view = context.view
    view.session.polygons = [lp.Polygon(2, [(0., left), (0., 159.), (119., 159.), (119., left)])]
    view.session.strokes = []
    view._render_marks()
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False
    sources = view.run_preview.sources
    sources.setCurrentIndex(next(i for i in range(sources.count()) if sources.itemData(i) == ("drawing", None)))
    return view


def _other(context, qtbot):
    preview = context.view.run_preview
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    qtbot.waitUntil(lambda: (not preview.busy() and not preview._pending and preview._waiting_key is None), timeout=30000)
    assert preview.last_result is not None, preview.note.text()
    return preview


def test_new_preview_distinguishes_photo_continuation_from_original_filled_provenance(context, qtbot, monkeypatch):
    view = _drawing(context)
    before = (_files(context.project.root, context.series), deepcopy(view.session.to_dict()), context.settings_path.read_bytes())
    overlays, threads = [], []
    original_set, original_attach = view.run_preview.canvas.set_proposal, display.attach
    def observed_set(rgba):
        overlays.append(None if rgba is None else rgba.copy())
        return original_set(rgba)
    def observed_attach(*args, **kwargs):
        threads.append(threading.get_ident())
        return original_attach(*args, **kwargs)
    monkeypatch.setattr(view.run_preview.canvas, "set_proposal", observed_set)
    monkeypatch.setattr(display, "attach", observed_attach)
    preview = _other(context, qtbot)
    result = preview.last_result
    value = result.measurements[display.KEY]
    assert value["continued_pixels"] == 600 and value["inside_gap_pixels"] == 0
    assert result.uncovered[:, :5].all(), "Original photo-origin provenance is retained"
    assert value["display_continued"][:, :5].all() and not value["display_gaps"].any()
    assert not value["display_continued"][:, 5:].any()
    assert np.all(result.layers == 2), "No alternative numerical extrapolator was used"
    assert tuple(overlays[-1][42, 0]) == (0, 220, 255, 150)
    assert "600 pixels continued" in preview.note.text() and "0 true gap pixels" in preview.note.text()
    assert old_domain.KEY in result.measurements, "The older diagnostic remains distinct"
    assert threads and all(thread != context.gui_thread for thread in threads)
    view.set_facies_visible(view.session.layer(2).facies, False)
    assert preview.last_result is result and not preview.busy()
    assert np.all(overlays[-1][value["display_continued"], 3] == 0), "Hidden facies also hide continuation marks"
    assert _files(context.project.root, context.series) == before[0]
    assert view.session.to_dict() == before[1]
    context.settings.sync()
    assert context.settings_path.read_bytes() == before[2]


def test_zero_donor_and_inside_photo_undrawn_pixels_stay_true_magenta_gaps(context, qtbot, monkeypatch):
    view = _drawing(context, left=10.)
    overlays = []
    original = view.run_preview.canvas.set_proposal
    def observed(rgba):
        overlays.append(None if rgba is None else rgba.copy())
        return original(rgba)
    monkeypatch.setattr(view.run_preview.canvas, "set_proposal", observed)
    result = _other(context, qtbot).last_result
    value = result.measurements[display.KEY]
    assert value["continued_pixels"] == 0
    assert value["inside_gap_pixels"] == 15 * 120
    assert value["display_gaps"][:, :15].all()
    assert not value["display_gaps"][:, 15:].any()
    assert np.all(overlays[-1][:, :15] == (255, 0, 255, 95))


def test_reduced_preview_masks_use_exact_existing_producer_resize_without_mutating_packet(closed_case):
    case = closed_case
    session = lp.PaintingSession("source", image_path=str(case.photos["source"]), image_size=(10, 8),
        layers=[lp.PaintedLayer(2, "Layer", "#335577", "Material"), lp.PaintedLayer(3, "Other", "#557733", "Other")],
        facies={"Material": lp.FaciesEntry("porous_medium", .3, 10.), "Other": lp.FaciesEntry("porous_medium", .3, 10.)})
    contract = policy.build_contracts(case.project, case.index, "source", ["target"], [2, 3])["target"]
    result = core.prepare_preview(session, case.source, geometry_index=case.index, target_run="target", config={},
        photo_loader=lambda run, _model: np.asarray(Image.open(case.photos[run])), display_limit=8)
    before = (result.layers.tobytes(), result.uncovered.tobytes(), result.photo.tobytes(), deepcopy(result.measurements))
    attached = display.attach(result, case.source, contract)
    value = attached.measurements[display.KEY]
    assert attached.layers is result.layers and attached.photo is result.photo and attached.uncovered is result.uncovered
    assert result.H_native_to_display == ((.5, 0., -.25), (0., .5, -.25), (0., 0., 1.))
    assert value["continued_pixels"] == 80 and value["inside_gap_pixels"] == 0
    assert value["display_continued"].shape == (5, 8)
    assert not value["display_continued"].flags.writeable
    assert (result.layers.tobytes(), result.uncovered.tobytes(), result.photo.tobytes(), result.measurements) == before


@pytest.mark.parametrize("failure", ["photo", "model", "resize", "layers", "filled"])
def test_preview_identity_or_provenance_drift_is_not_relabelled_continued(closed_case, failure):
    case = closed_case
    session = lp.PaintingSession("source", image_path=str(case.photos["source"]), image_size=(10, 8),
        layers=[lp.PaintedLayer(2, "Layer", "#335577", "Material")], facies={"Material": lp.FaciesEntry("porous_medium", .3, 10.)})
    contract = policy.build_contracts(case.project, case.index, "source", ["target"], [2])["target"]
    result = core.prepare_preview(session, case.source, geometry_index=case.index, target_run="target", config={},
        photo_loader=lambda run, _model: np.asarray(Image.open(case.photos[run])))
    if failure == "photo":
        result = replace(result, photo_path=str(case.photos["source"]))
    elif failure == "model":
        binding = deepcopy(result.geometry_binding)
        binding["models"]["target"]["model"]["sha256"] = "0" * 64
        result = replace(result, geometry_binding=binding)
    elif failure == "resize":
        result = replace(result, H_native_to_display=((1., 0., 1.), (0., 1., 0.), (0., 0., 1.)))
    elif failure == "layers":
        values = result.layers.copy()
        values[0, 0] = 0
        result = replace(result, layers=values)
    else:
        values = result.uncovered.copy()
        values[0, 0] = not values[0, 0]
        result = replace(result, uncovered=values)
    with pytest.raises(ValueError):
        display.attach(result, case.source, contract)
