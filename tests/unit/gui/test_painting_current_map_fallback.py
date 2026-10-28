"""The default read-only preview can show today's manual map, never fake a version.

Actual hosts use only the synthetic two-run project and strict private USER
INI constructors from the series-frame fixture. All preview computations below
are exercised only when this test file is explicitly run by its caller.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import layer_painting_polygons_only as polygons
from darsia.fluidflower import painting_run_preview as preview_core

from .test_painting_series_frame_models import context  # noqa: F401
from .test_painting_series_frame_models import _files


_FALLBACK_NAME = "Current manual map (the proposal is out of date)"


def _proposal(context, method, *, stale):
    view = context.view
    if method == "polygons":
        proposal = polygons.make_proposal(view.image, view.session, frame=view.current_frame())
    else:
        proposal = lp.LayerProposal(np.full((120, 160), 3, dtype=np.uint8),
            np.full((120, 160), 255, dtype=np.uint8), (160, 120),
            lp.proposal_digest(view.session, view.current_frame()))
    view._proposal_ready(proposal)
    assert polygons.proposal_current(proposal, view.session, view.current_frame())
    if stale:
        view.add_polygon(lp.Polygon(3, [(20., 20.), (20., 40.), (40., 40.), (40., 20.)]))
        assert not polygons.proposal_current(proposal, view.session, view.current_frame())
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    return proposal


def _preserved(context, proposal):
    context.settings.sync()
    return dict(files=_files(context.series, context.project.root),
        session=deepcopy(context.view.session.to_dict()),
        settings=context.settings_path.read_bytes(),
        classes=proposal.classes.copy(), confidence=proposal.confidence.copy(),
        digest=proposal.digest,
        source=deepcopy(getattr(proposal, "source", None)),
        inside=None if not hasattr(proposal, "inside") else proposal.inside.copy())


def _unchanged(context, proposal, before):
    assert _files(context.series, context.project.root) == before["files"]
    assert context.view.session.to_dict() == before["session"]
    context.settings.sync()
    assert context.settings_path.read_bytes() == before["settings"]
    assert np.array_equal(proposal.classes, before["classes"])
    assert np.array_equal(proposal.confidence, before["confidence"])
    assert proposal.digest == before["digest"]
    assert getattr(proposal, "source", None) == before["source"]
    if before["inside"] is not None:
        assert np.array_equal(proposal.inside, before["inside"])


def _observe_source(monkeypatch):
    calls = []
    original = preview_core.preview_layers
    def observed(session, proposal, corrections, frame, *, source="map", record=None):
        entry = dict(source=source, record=deepcopy(record), classes=None)
        calls.append(entry)
        value = original(session, proposal, corrections, frame, source=source, record=record)
        entry["classes"] = value[1].copy()
        return value
    monkeypatch.setattr(preview_core, "preview_layers", observed)
    return calls


def _other(context, qtbot):
    preview = context.view.run_preview
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    qtbot.waitUntil(lambda: (not context.view.busy() and not preview.busy()
        and preview._waiting_key is None and not preview._pending), timeout=30000)
    return preview


@pytest.mark.parametrize("method", ["classifier", "polygons"])
def test_default_current_map_uses_current_manual_polygons_for_stale_proposal(context, qtbot, monkeypatch, method):
    proposal = _proposal(context, method, stale=True)
    assert context.view.run_preview.sources.currentData() == ("map", None)
    expected = manual.manual_map(context.view.session, (160, 120),
                                 frame=context.view.current_frame()).classes
    assert expected[50, 50] == 2 and expected[30, 30] == 3
    before = _preserved(context, proposal)
    calls = _observe_source(monkeypatch)
    preview = _other(context, qtbot)
    assert preview.last_result is not None, preview.note.text()
    result = preview.last_result
    assert result.run_id == "run_b" and Path(result.photo_path) == context.photos["run_b"]
    assert len(calls) == 1 and calls[0]["source"] == "manual" and calls[0]["record"] is None
    assert np.array_equal(calls[0]["classes"], expected)
    # The synthetic target model translates the source five columns right.
    assert result.layers[50, 55] == 2 and result.layers[30, 35] == 3
    assert result.layers[90, 100] == lp.OUTSIDE
    assert result.source_name == _FALLBACK_NAME
    assert preview.note.text().count("the proposal is out of date") == 1
    assert "ValueError" not in preview.note.text() and "Preview unavailable" not in preview.note.text()
    assert not preview.progress_bar.isVisibleTo(preview.widget)
    _unchanged(context, proposal, before)


def test_reference_manual_map_selection_remains_explicit_without_stale_notice(context, qtbot, monkeypatch):
    proposal = _proposal(context, "classifier", stale=True)
    view = context.view
    view.map_shown.setCurrentIndex(view.map_shown.findData(manual.MANUAL_MAP_KIND))
    qtbot.waitUntil(lambda: (not view.busy() and view._manual_thread is None
        and not view._manual_timer.isActive() and view._manual_current()), timeout=30000)
    view.save_session()
    view._save_timer.stop()
    before = _preserved(context, proposal)
    expected = view._manual.classes.copy()
    calls = _observe_source(monkeypatch)
    preview = _other(context, qtbot)
    assert preview.last_result is not None, preview.note.text()
    assert len(calls) == 1 and calls[0]["source"] == "manual" and calls[0]["record"] is None
    assert np.array_equal(calls[0]["classes"], expected)
    assert "out of date" not in preview.note.text()
    assert preview.last_result.source_name != _FALLBACK_NAME
    _unchanged(context, proposal, before)


@pytest.mark.parametrize("failure", ["model", "photo", "stored_version"])
def test_model_photo_and_explicit_version_failures_do_not_become_a_different_map(context, qtbot, monkeypatch, failure):
    proposal = _proposal(context, "classifier", stale=failure == "stored_version")
    view = context.view
    record = None
    if failure == "model":
        # Corrupt only the private model; its existing index descriptor remains
        # authoritative, so the real measured-model reader must refuse it.
        (context.project.root / "display-models" / "run_b.json").write_bytes(b"{}")
    elif failure == "photo":
        original_loader = view._reference_photo_loader()
        def unavailable(run, model):
            if run == "run_b":
                raise ValueError("Private target photo unavailable")
            return original_loader(run, model)
        monkeypatch.setattr(view, "_reference_photo_loader", lambda: unavailable)
    else:
        drawn = manual.manual_map(view.session, (160, 120), frame=view.current_frame())
        record = manual.save_manual_version(context.project, view.session, drawn,
                                             name="Private saved map")
        view.versions = view._read_versions(view.session.run_id)
        view.run_preview.refresh()
        sources = view.run_preview.sources
        index = next(i for i in range(sources.count())
            if sources.itemData(i)[1] is not None
            and sources.itemData(i)[1].get("folder") == record["folder"])
        sources.setCurrentIndex(index)
        (Path(record["folder"]) / "manual_map.npz").write_bytes(b"private changed file")
    before = _preserved(context, proposal)
    calls = _observe_source(monkeypatch)
    preview = _other(context, qtbot)
    assert preview.last_result is None, "An invalid model/photo/version must never show a replacement map"
    assert len(calls) == 1
    assert calls[0]["source"] == ("version" if record is not None else "map")
    assert (calls[0]["record"] is not None) == (record is not None)
    message = preview.note.text()
    assert "Preview unavailable" in message and "ValueError" not in message
    assert _FALLBACK_NAME not in message
    if failure == "photo":
        assert "Private target photo unavailable" in message
    elif failure == "model":
        assert "reference model the index bound" in message
    else:
        assert "changed since it was saved" in message
    assert not preview.progress_bar.isVisibleTo(preview.widget)
    _unchanged(context, proposal, before)
