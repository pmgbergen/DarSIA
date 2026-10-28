"""Replay only starts explicitly and retains ownership through cancellation."""

from threading import Event

from PySide6.QtCore import QCoreApplication, QThread

from darsia.fluidflower.normalisation_replay_step import NormalisationReplayStep


def saved_raw_document(reference=None):
    return dict(phase_normalisation=dict(field_sources=dict(v5_raw_photometry=dict(source={"path": "saved.json"}))),
                image_normalisation=dict(reference_run=reference),
                image_selection=dict(runs=dict(sample=dict(selected=[dict(source_path="photo.ARW")]))))


def test_prepare_and_reproduce_use_explicit_selection_in_worker(qtbot, tmp_path):
    calls = []
    def prepare(root, run, frame):
        calls.append(("prepare", root, run, frame, QThread.currentThread()))
        return dict(rows=[{}, {}], token="saved inputs")
    def execute(request, output, *, progress, check_cancelled):
        check_cancelled()
        calls.append(("execute", request, output, QThread.currentThread()))
        progress("Comparing saved bits")
        return output + "/reproduced"
    panel = NormalisationReplayStep(preparer=prepare, executor=execute)
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "FF_AC")
    panel.install_snapshot(("ac19", "ac28"), document=saved_raw_document("ac28"))
    assert not calls and panel.run.currentData() == "ac28" and not panel.start.isEnabled()
    panel.frame.setValue(1)
    panel.prepare.click()
    qtbot.waitUntil(panel.can_close)
    assert not panel.start.isEnabled()
    panel.output.setText(str(tmp_path))
    panel.start.click()
    qtbot.waitUntil(panel.can_close)
    assert calls[0][1:4] == (tmp_path, "ac28", 1)
    assert calls[1][1:3] == (dict(rows=[{}, {}], token="saved inputs"), str(tmp_path))
    assert all(call[-1] != QCoreApplication.instance().thread() for call in calls)
    assert "Exact saved lighting" in panel.status.text()
    panel.frame.setValue(2)
    assert not panel.start.isEnabled()


def test_context_switch_cancels_and_discards_old_replay(qtbot, tmp_path):
    entered, release = Event(), Event()
    def prepare(*_args):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release replay")
        return dict(rows=[{}])
    panel = NormalisationReplayStep(preparer=prepare)
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "FF_AC")
    panel.install_snapshot(("ac28",), document=saved_raw_document())
    panel.prepare.click()
    try:
        qtbot.waitUntil(entered.is_set)
        panel.set_context(tmp_path, "OTHER")
        assert not panel.prepare.isEnabled() and not panel.start.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(panel.can_close)
    assert panel._request is None and "OTHER:" in panel.status.text()
    assert "not available" in panel.status.text()


def test_replay_availability_tracks_selected_series_without_enabling_unsupported_runs(qtbot, tmp_path):
    panel = NormalisationReplayStep()
    qtbot.addWidget(panel)
    assert "Choose a series" in panel.status.text()
    panel.set_context(tmp_path, "FF_UM")
    panel.install_snapshot(("um01",))
    assert "FF_UM:" in panel.status.text() and "FF_AC" not in panel.status.text()
    assert "not available" in panel.status.text() and not panel.prepare.isEnabled()
    panel.set_context(tmp_path, "FF_AC")
    assert "not available" in panel.status.text()
    panel.install_snapshot(("ac28",), document=saved_raw_document())
    assert panel.prepare.isEnabled() and "FF_AC:" in panel.status.text()
    assert "prepare the saved inputs" in panel.status.text()
    panel.set_context(None, None)
    assert "Choose a series" in panel.status.text() and not panel.prepare.isEnabled()


def test_read_only_preview_cannot_start_replay(qtbot, tmp_path):
    panel = NormalisationReplayStep(allow_run=False)
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "FF_AC")
    panel.install_snapshot(("ac28",), document=saved_raw_document())
    assert not panel.prepare.isEnabled() and not panel.start.isEnabled()
    assert "read only" in panel.status.text()


def test_processed_setup_selection_explicitly_excludes_raw_replay(qtbot, tmp_path):
    panel = NormalisationReplayStep()
    qtbot.addWidget(panel)
    document = saved_raw_document()
    document["image_selection"]["runs"]["sample"]["selected"][0]["source_path"] = "photo.jpg"
    panel.set_context(tmp_path, "SERIES_X")
    panel.install_snapshot(("sample",), document=document)
    assert not panel.prepare.isEnabled()
    assert "processed images" in panel.status.text()
    assert "not required" in panel.status.text()


def test_cancel_retains_worker_until_it_returns(qtbot, tmp_path):
    entered, release = Event(), Event()
    def execute(_request, _output, *, progress, check_cancelled):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release replay")
        check_cancelled()
        raise AssertionError("cancelled replay must not finish")
    panel = NormalisationReplayStep(preparer=lambda *_: dict(rows=[{}]), executor=execute)
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "FF_AC")
    panel.install_snapshot(("ac28",), document=saved_raw_document())
    panel.prepare.click()
    qtbot.waitUntil(panel.can_close)
    panel.output.setText(str(tmp_path))
    panel.start.click()
    try:
        qtbot.waitUntil(entered.is_set)
        panel.cancel.click()
        assert not panel.can_close() and not panel.prepare.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(panel.can_close)
    assert "cancelled" in panel.status.text()
    assert not panel.start.isEnabled()
