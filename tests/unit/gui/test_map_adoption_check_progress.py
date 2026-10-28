"""An owned Check shows real worker stages and elapsed time until its outcome."""
import threading
from types import SimpleNamespace

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption_view as view
from darsia.fluidflower.adoption_check_progress import report_check_progress

from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401


def test_real_check_progress_stays_responsive_and_retires_without_overwriting_the_outcome(
        qtbot, registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = geometry.validate_geometry_config
    on_gui = []

    def validate(candidate):
        on_gui.append(threading.current_thread() is threading.main_thread())
        report_check_progress("Validating the complete geometry revision…")
        entered.set()
        assert release.wait(30), "The test must release the genuine validator"
        return original(candidate)

    monkeypatch.setattr(geometry, "validate_geometry_config", validate)
    clock = [10.0]
    monkeypatch.setattr(view, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    dialog = view.AdoptionDialog(data.project, prepared["folder"], actor="synthetic")
    qtbot.addWidget(dialog, before_close_func=lambda owned: owned.wait_for_workers())
    retired = []
    dialog.operation_retired.connect(retired.append)
    assert dialog.check()
    worker = dialog._worker
    try:
        qtbot.waitUntil(lambda: entered.is_set() and "complete geometry revision" in dialog.status.text(), timeout=30000)
        assert dialog.busy() and dialog.candidate is None
        assert not dialog.save_button.isEnabled() and not dialog.check_button.isEnabled()
        assert dialog._check_progress_timer.isActive()
        clock[0] += 7
        qtbot.waitUntil(lambda: "Elapsed: 7 s; current stage: 7 s." in dialog.status.text(), timeout=3000)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert on_gui == [False]
    assert dialog.candidate is not None, dialog.status.text()
    assert retired == ["check"] and dialog.save_button.isEnabled()
    assert not dialog._check_progress_timer.isActive()
    outcome = dialog.status.text()
    worker.progress.emit("Late progress from the retired Check")
    dialog._refresh_check_progress()
    assert dialog.status.text() == outcome
