"""Descriptive gaps belong to the placed drawing frame, never the whole photo.

Pure cases use tiny synthetic grids and exact homographies. The actual host
case uses private photos/models and strict private USER settings before host
construction, through the shared series-frame fixture.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import painting_preview_frame_domain as domain
from darsia.fluidflower import painting_run_preview as preview_core
from darsia.fluidflower import painting_photo_continuation as photo_continuation
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_series_frame_models import context  # noqa: F401
from .test_painting_series_frame_models import _files


def _case(tmp_path, *, source_shape=(21, 21), target_shape=None,
          corners=(2., 2., 18., 18.), matrix=None, half=False, native=None):
    target_shape = source_shape if target_shape is None else target_shape
    matrix = np.eye(3) if matrix is None else np.asarray(matrix, dtype=np.float64)
    source_path, target_path = tmp_path / "source.png", tmp_path / "target.png"
    source = placement.RunGeometry("run_a", matrix.copy(), {"path": str(source_path)},
                                   source_shape, str(tmp_path / "source-model.json"))
    target = placement.RunGeometry("run_b", matrix.copy(), {"path": str(target_path)},
                                   target_shape, str(tmp_path / "target-model.json"))
    top, left, bottom, right = corners
    frame = RigFrame(corners=dict(top_left=(top, left), top_right=(top, right),
        bottom_right=(bottom, right), bottom_left=(bottom, left)),
        width_m=(right-left)*.001, height_m=(bottom-top)*.001,
        run_id="run_a", image=str(source_path), image_size=source_shape[::-1])
    session = lp.PaintingSession("run_a", image_path=str(source_path),
        image_size=source_shape[::-1],
        layers=[lp.PaintedLayer(2, "Layer", "#336699", "Material")],
        facies={"Material": lp.FaciesEntry(porosity=.3, permeability=100.)})
    if native is None:
        native = np.full(source_shape, 2, dtype=np.uint8)
        native[10, 10] = 0
    else:
        native = native.copy()
    placed, filled = placement.place_layer_map(native, source, target, target_shape)
    shown, uncovered = (placed[::2, ::2].copy(), filled[::2, ::2].copy()) if half else (placed, filled)
    display_shape = shown.shape
    sy, sx = display_shape[0]/target_shape[0], display_shape[1]/target_shape[1]
    display_h = ((sx, 0., (sx-1.)/2.), (0., sy, (sy-1.)/2.), (0., 0., 1.))
    def bound(model):
        return dict(model={"path": model.source}, entry={"run_id": model.run_id},
            reference_image=deepcopy(model.reference_image), shape_hw=list(model.shape_hw),
            metric_to_grid=model.metric_to_grid.tolist())
    packet = preview_core.RunPreview("run_b", np.full((*display_shape, 3), 73, np.uint8),
        shown, uncovered, str(target_path), "run_a", "Private drawing",
        measurements={"preserved": {"value": 17}}, native_shape_hw=target_shape,
        display_shape_hw=display_shape, H_native_to_display=display_h,
        geometry_binding={"models": {"run_a": bound(source), "run_b": bound(target)}})
    return SimpleNamespace(packet=packet, session=session, frame=frame, native=native)


def _snapshot(case):
    return dict(native=case.native.copy(), photo=case.packet.photo.copy(),
        layers=case.packet.layers.copy(), uncovered=case.packet.uncovered.copy(),
        measurements=deepcopy(case.packet.measurements), binding=deepcopy(case.packet.geometry_binding),
        session=deepcopy(case.session.to_dict()),
        frame=None if case.frame is None else deepcopy(case.frame.to_dict()))


def _unchanged(case, before):
    for name, array in (("native", case.native), ("photo", case.packet.photo),
                        ("layers", case.packet.layers), ("uncovered", case.packet.uncovered)):
        assert array.dtype == before[name].dtype and array.shape == before[name].shape
        assert array.tobytes() == before[name].tobytes()
    assert case.packet.measurements == before["measurements"]
    assert case.packet.geometry_binding == before["binding"]
    assert case.session.to_dict() == before["session"]
    assert (None if case.frame is None else case.frame.to_dict()) == before["frame"]


def test_one_known_native_gap_has_exact_metric_area_and_far_edge_box(tmp_path):
    case = _case(tmp_path)
    before = _snapshot(case)
    attached = domain.attach_frame_domain(case.packet, case.session, case.frame, case.native)
    value = attached.measurements[domain.KEY]
    assert value["status"] == "available" and value["adopted"] is False
    assert value["inside_uncovered_pixels"] == 1
    assert value["inside_uncovered_cm2"] == pytest.approx(.01, rel=1.e-12)
    assert value["outside_pixels"] == 21*21 - 17*17
    assert value["display_inside"][2, 2] and not value["display_inside"][1, 2]
    assert value["display_gaps"][10, 10] and value["display_gaps"].sum() == 1
    assert value["beyond_edge_strip_pixels"] == 1 and value["edge_strip_mm"] == 5.
    assert value["beyond_edge_strip_regions"] == {
        "top": {"pixels": 1, "bbox_native_rc": [[10, 10], [10, 10]],
                "bbox_metric_mm": [[10., 10.], [10., 10.]]}}
    assert value["projected_frame_native_xy"] == [[2., 2.], [18., 2.], [18., 18.], [2., 18.]]
    assert not value["display_inside"].flags.writeable and not value["display_gaps"].flags.writeable
    assert attached is not case.packet and domain.KEY not in case.packet.measurements
    assert attached.photo is case.packet.photo and attached.layers is case.packet.layers
    assert attached.uncovered is case.packet.uncovered
    _unchanged(case, before)


def test_projective_pixel_footprint_uses_its_four_inverse_mapped_corners(tmp_path):
    # Both cameras use x=u/(1-.01*u), y=v/(1-.01*u) in metric coordinates.
    # Placement is identity, but one camera pixel has a nonconstant metric area.
    case = _case(tmp_path, matrix=((1., 0., 0.), (0., 1., 0.), (.01, 0., 1.)))
    before = _snapshot(case)
    value = domain.frame_domain(case.packet, case.session, case.frame, case.native)
    left, right, k = Fraction(19, 2), Fraction(21, 2), Fraction(1, 100)
    width = right/(1-k*right) - left/(1-k*left)
    # The inverse-mapped quadrilateral is an exact trapezoid. Its left and
    # right heights differ; centre-Jacobian or constant pixel area is insufficient.
    area_mm2 = width * (1/(1-k*left) + 1/(1-k*right)) / 2
    assert value["inside_uncovered_pixels"] == 1
    assert value["inside_uncovered_cm2"] == pytest.approx(float(area_mm2/100), rel=1.e-11)
    assert value["inside_uncovered_cm2"] > .013
    _unchanged(case, before)


def test_native_counts_survive_exact_half_size_pixel_centre_display(tmp_path):
    case = _case(tmp_path, source_shape=(20, 20), corners=(2., 2., 17., 17.), half=True)
    value = domain.frame_domain(case.packet, case.session, case.frame, case.native)
    assert case.packet.H_native_to_display == ((.5, 0., -.25), (0., .5, -.25), (0., 0., 1.))
    assert value["display_inside"].shape == (10, 10)
    assert value["display_inside"][5, 5] and not value["display_inside"][0, 0]
    assert value["display_gaps"][5, 5]
    assert value["inside_uncovered_pixels"] == 1
    assert value["inside_uncovered_cm2"] == pytest.approx(.01)
    assert value["outside_pixels"] == 20*20 - 16*16


def test_frame_extending_beyond_source_photo_keeps_real_target_gaps(tmp_path):
    case = _case(tmp_path, source_shape=(10, 10), target_shape=(20, 20),
        corners=(0., 0., 19., 19.), native=np.full((10, 10), 2, np.uint8))
    before = _snapshot(case)
    value = domain.frame_domain(case.packet, case.session, case.frame, case.native)
    assert value["display_inside"].all(), "A frame must be projected before source-photo clipping"
    assert value["display_gaps"][15, 15] and not value["display_gaps"][5, 5]
    assert value["outside_pixels"] == 0
    assert value["inside_uncovered_pixels"] == 20*20 - 10*10
    assert value["inside_uncovered_cm2"] == pytest.approx(3.)
    assert value["beyond_edge_strip_pixels"] > 0
    assert sum(row["pixels"] for row in value["beyond_edge_strip_regions"].values()) == value["beyond_edge_strip_pixels"]
    assert np.all(case.packet.layers == 2), "Edge replication alone does not make a gap decided"
    _unchanged(case, before)


def test_legacy_frame_without_size_still_requires_its_actual_run_photo_and_corners(tmp_path):
    case = _case(tmp_path)
    expected = domain.frame_domain(case.packet, case.session, case.frame, case.native)
    case.frame.image_size = None
    value = domain.frame_domain(case.packet, case.session, case.frame, case.native)
    assert value["projected_frame_native_xy"] == expected["projected_frame_native_xy"]
    assert np.array_equal(value["display_inside"], expected["display_inside"])
    assert value["inside_uncovered_pixels"] == expected["inside_uncovered_pixels"]


@pytest.mark.parametrize("invalid", ["collapsed", "crossed", "overflow"])
def test_invalid_frame_is_rejected_even_when_the_native_map_has_no_gaps(tmp_path, invalid):
    case = _case(tmp_path, native=np.full((21, 21), 2, np.uint8))
    assert not np.any(case.native == 0)
    if invalid == "collapsed":
        case.frame.corners["top_right"] = case.frame.corners["top_left"]
    elif invalid == "crossed":
        case.frame.corners["top_right"], case.frame.corners["bottom_right"] = (
            case.frame.corners["bottom_right"], case.frame.corners["top_right"])
    else:
        case.frame.corners = {name: (row + 1.e12, col + 1.e12)
                              for name, (row, col) in case.frame.corners.items()}
    before = _snapshot(case)
    value = domain.attach_frame_domain(case.packet, case.session, case.frame, case.native).measurements[domain.KEY]
    assert value["status"] == "unavailable" and value["reason"]
    assert "display_inside" not in value and "inside_uncovered_pixels" not in value
    _unchanged(case, before)


@pytest.mark.parametrize("invalid", ["missing_frame", "incomplete_frame", "frame_run", "frame_photo",
    "frame_grid", "source_photo", "target_photo", "native_grid", "resize", "display_grid"])
def test_unproved_frame_photo_or_resize_yields_reason_without_changing_packet(tmp_path, invalid):
    case = _case(tmp_path)
    if invalid == "missing_frame":
        case.frame = None
    elif invalid == "incomplete_frame":
        del case.frame.corners["top_right"]
    elif invalid == "frame_run":
        case.frame.run_id = "other_run"
    elif invalid == "frame_photo":
        case.frame.image = str(tmp_path / "other-frame-photo.png")
    elif invalid == "frame_grid":
        case.frame.image_size = (22, 21)
    elif invalid == "source_photo":
        case.session.image_path = str(tmp_path / "other-source.png")
    elif invalid == "target_photo":
        case.packet = replace(case.packet, photo_path=str(tmp_path / "other-target.png"))
    elif invalid == "native_grid":
        case.packet = replace(case.packet, native_shape_hw=(22, 21))
    elif invalid == "resize":
        case.packet = replace(case.packet, H_native_to_display=((1., 0., .25), (0., 1., 0.), (0., 0., 1.)))
    elif invalid == "display_grid":
        case.packet = replace(case.packet, display_shape_hw=(20, 21))
    before = _snapshot(case)
    attached = domain.attach_frame_domain(case.packet, case.session, case.frame, case.native)
    value = attached.measurements[domain.KEY]
    assert value["status"] == "unavailable" and value["adopted"] is False
    assert value["reason"] and "display_inside" not in value and "inside_uncovered_cm2" not in value
    assert attached.photo is case.packet.photo and attached.layers is case.packet.layers
    _unchanged(case, before)


def test_actual_worker_renderer_keeps_outside_transparent_and_inside_gaps_visible_when_hidden(
        context, qtbot, monkeypatch):
    view, preview = context.view, context.view.run_preview
    # A real stale classifier exercises the existing current-manual fallback;
    # neither its arrays nor the source drawing may be altered by display work.
    proposal = lp.LayerProposal(np.full((120, 160), 3, np.uint8),
        np.full((120, 160), 255, np.uint8), (160, 120),
        lp.proposal_digest(view.session, view.current_frame()))
    view._proposal_ready(proposal)
    view.add_polygon(lp.Polygon(3, [(20., 20.), (20., 40.), (40., 40.), (40., 20.)]))
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    before = (_files(context.series, context.project.root), deepcopy(view.session.to_dict()),
              proposal.classes.copy(), proposal.confidence.copy(), context.settings_path.read_bytes())
    overlays, worker_threads, native_inputs = [], [], []
    original_attach, original_overlay = domain.attach_frame_domain, preview.canvas.set_proposal
    def attached(result, session, frame, native_layers):
        worker_threads.append(threading.get_ident())
        native_inputs.append(native_layers.copy())
        return original_attach(result, session, frame, native_layers)
    def overlay(rgba):
        overlays.append(None if rgba is None else rgba.copy())
        return original_overlay(rgba)
    monkeypatch.setattr(domain, "attach_frame_domain", attached)
    monkeypatch.setattr(preview.canvas, "set_proposal", overlay)
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    qtbot.waitUntil(lambda: (not view.busy() and not preview.busy()
        and preview._waiting_key is None and not preview._pending), timeout=30000)
    result = preview.last_result
    assert result is not None, preview.note.text()
    value = result.measurements[domain.KEY]
    assert value["status"] == "available", value.get("reason")
    assert worker_threads and all(number != context.gui_thread for number in worker_threads)
    assert value["inside_uncovered_pixels"] > 0 and value["inside_uncovered_cm2"] > 0.
    assert "Target own frame" in preview.note.text() and "cm²" in preview.note.text()
    current = result.measurements[photo_continuation.KEY]
    for layer_id in (2, 3):
        view.set_facies_visible(view.session.layer(layer_id).facies, False)
    assert preview.last_result is result and not preview.busy()
    rgba = overlays[-1]
    assert np.all(rgba[~current["display_inside"], 3] == 0)
    assert np.all(rgba[current["display_gaps"]] == (255, 0, 255, 95))
    assert np.all(rgba[current["display_inside"] & ~current["display_gaps"], 3] == 0)
    assert _files(context.series, context.project.root) == before[0]
    assert view.session.to_dict() == before[1]
    assert np.array_equal(proposal.classes, before[2]) and np.array_equal(proposal.confidence, before[3])
    context.settings.sync()
    assert context.settings_path.read_bytes() == before[4]
    # Missing frame authority is descriptive failure only: retain checked
    # photo/map arrays and show its reason instead of painting a whole-photo gap.
    # An old packet has no positive continuation policy. Its original missing
    # N365 authority behavior remains independent of new-placement metadata.
    legacy = replace(result, measurements={key: item for key, item in result.measurements.items()
        if key != photo_continuation.KEY})
    unavailable = original_attach(legacy, view.session, None, native_inputs[-1])
    assert unavailable.photo is result.photo and unavailable.layers is result.layers
    preview._render_result(unavailable, view.session)
    assert "Inside-frame gaps unavailable:" in preview.note.text()
    assert preview.canvas.raw_size == (160, 120)
    assert preview.canvas.preview["path"] == result.photo_path
    assert np.all(overlays[-1][result.uncovered | (result.layers == 0), 3] == 0)
