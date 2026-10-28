"""Original owned reader failures remain visible outside technical Details."""
from types import SimpleNamespace

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_camera_jpg_display as camera
from darsia.fluidflower import painting_guided_review_bindings as bindings
from darsia.fluidflower.painting_series_adoption_ui import SeriesAdoptionController
from darsia.fluidflower.painting_target_correction_ui import TargetCorrectionDialog

from ..test_painted_jpg_native_preview import (  # noqa: F401
    jpeg_case, registered_setup, imported, synthetic_payload_admission,
)
from .test_painting_guided_review_ui import GuidedBackend, _dialog, _settle, _stage


def test_original_preview_failure_replaces_loading_placeholder_without_retry_or_answer(qtbot):
    backend = GuidedBackend()
    original = backend.preview
    calls = []
    def preview(review, run, component):
        calls.append((run, component))
        if run == "ac18":
            raise ValueError("The original RGB display encoding is not declared by this preview route")
        return original(review, run, component)
    backend.preview = preview
    widget, _ = _dialog(qtbot, backend=backend)
    assert not widget.busy() and widget.packet is None
    assert widget.image_stack.currentWidget() is widget.photo_placeholder
    assert widget.photo_placeholder.text() == "This photograph could not be opened. Open Details for the cause."
    assert widget.guided_status.text() == widget.photo_placeholder.text()
    assert "original RGB display encoding" in widget.status.text()
    assert "ValueError" not in widget.guided_status.text()
    assert "is loading" not in widget.photo_placeholder.text()
    assert calls == [("ac18", "c"*64)] and not widget.technical_details.isVisible()
    assert widget.next_button.isEnabled() and not widget.recheck.isEnabled()
    def failed_catalogue():
        raise ValueError("Synthetic catalogue failure, not a photograph read")
    widget._start(("findings-catalogue", 9), failed_catalogue)
    _settle(qtbot, widget)
    assert widget.packet is None and widget.guided_stage == "deviation"
    assert "Synthetic catalogue failure" in widget.status.text()
    assert "photograph could not" not in widget.guided_status.text()
    assert "photograph could not" not in widget.photo_placeholder.text()
    for stage in ("check", "adopt"):
        _stage(qtbot, widget, stage)
        assert "photograph could not" not in widget.guided_status.text()
    _stage(qtbot, widget, "deviation")
    widget.next_button.click()
    _settle(qtbot, widget)
    assert widget.packet["run_id"] == "ac61"
    assert "could not be opened" not in widget.guided_status.text()
    assert len(calls) == 2 and backend.approvals == [] and widget.approval is None
    assert widget._candidate is None and widget._acquisition_actions == []


def test_explicit_jpeg_target_reader_keeps_legacy_default_and_loads_real_original_target(qtbot, jpeg_case):
    fixture = jpeg_case
    current = camera.read_gap_preview(fixture.review, fixture.run, fixture.key)
    owned = []
    for reader in (None, camera.read_target_paint):
        options = {} if reader is None else dict(reader=reader)
        widget = TargetCorrectionDialog(fixture.review, fixture.run, fixture.key,
            actor="Synthetic reviewer", destination=fixture.case.root / "unused-target-output",
            current_display_packet=current, **options)
        qtbot.addWidget(widget, before_close_func=lambda current: current.wait_for_workers())
        widget.show()
        qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
        owned.append(widget)
    legacy, explicit = owned
    assert legacy.reader is gaps.read_target_paint and legacy.packet is legacy.draft is None
    assert "original RGB display encoding" in legacy.status.text()
    from darsia.fluidflower.painted_target_corrections import TargetCorrectionDraft
    assert explicit.reader is camera.read_target_paint and type(explicit.draft) is TargetCorrectionDraft
    assert explicit.packet["target_correction_basis"]["science"] == fixture.row["target_correction_science"]
    assert legacy.binding is explicit.binding is None
    assert not (fixture.case.root / "unused-target-output").exists()


def test_guided_factory_explicit_target_reader_preserves_caller_injection(monkeypatch):
    from darsia.fluidflower import painting_guided_review_ui as guided
    made = []
    def constructor(*args, **options):
        made.append(options)
        return SimpleNamespace()
    monkeypatch.setattr(guided, "GuidedGapReviewDialog", constructor)
    default = bindings.guided_gap_dialog({}, actor="Synthetic")
    custom = lambda *_args: None
    supplied = bindings.guided_gap_dialog({}, actor="Synthetic", target_paint_reader=custom)
    assert default.target_paint_reader is camera.read_target_paint
    assert supplied.target_paint_reader is custom
    assert all("target_paint_reader" not in options for options in made)


def test_original_controller_forwards_only_explicit_owned_target_reader(monkeypatch, tmp_path):
    from darsia.fluidflower import layer_painting as painting
    from darsia.fluidflower import painting_target_correction_ui as target_ui
    made = []
    def constructor(*args, **options):
        made.append(options)
        signal = SimpleNamespace(connect=lambda _slot: None)
        return SimpleNamespace(saved=signal, finished=signal, show=lambda: None)
    monkeypatch.setattr(target_ui, "TargetCorrectionDialog", constructor)
    monkeypatch.setattr(painting, "painting_folder", lambda _project: tmp_path / "painting")
    selected = lambda *_args: None
    request = dict(review={}, run_id="synthetic", component_id="c"*64, point_native_xy=None)
    for explicit in (False, True):
        findings = SimpleNamespace(hide=lambda: None)
        if explicit:
            findings.target_paint_reader = selected
        owner = SimpleNamespace(_current=lambda: True, gap_dialog=findings,
            view=SimpleNamespace(_reviewer=lambda: "Synthetic", project=object()),
            _target_saved=lambda *_args: None, _target_closed=lambda *_args: None)
        SeriesAdoptionController._paint_gap(owner, request)
    # This finite constructor captures wiring only, with no target acceptance
    # or successful scientific result. Legacy callers supply no new reader.
    assert "reader" not in made[0] and made[1]["reader"] is selected
