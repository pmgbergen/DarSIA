"""Local close choices and original commands; no worker or process API runs."""
from types import SimpleNamespace
import threading

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLineEdit, QMainWindow, QVBoxLayout, QWidget

from darsia.fluidflower import job_control, native_jobs_close as close_ui
from darsia.fluidflower.job_launch_state import begin_launch_request
from darsia.fluidflower.jobs_panel import JobsPanel
from .test_jobs_metadata import put
from .test_native_sidebar import native_window  # noqa: F401


def associated_job(tmp_path, *, master="fixture", launch=True):
    root = tmp_path / "job"
    bound = put(root / "job.json", dict(expected_runs=["run"], precision="float32"))
    put(root / "ui" / "master.json", dict(format="darsia-job-ui-master-v1",
        machine=master, job=bound))
    if launch:
        begin_launch_request(root, machine=master, backend="cpu", workers=1, kind="local")
    return dict(root=str(root), name="Registered job", job=bound)


@pytest.fixture
def close_window(qtbot, tmp_path):
    registration = associated_job(tmp_path)
    panel = JobsPanel(local_machine="fixture")
    jobs = SimpleNamespace(panel=panel,
        registry=SimpleNamespace(known_jobs=lambda: [registration]))

    class Window(QMainWindow):
        def closeEvent(self, event):
            if not self.controller.allow_close():
                event.ignore()
            else:
                super().closeEvent(event)

    window = Window()
    controller = window.controller = close_ui.NativeJobsCloseController(window, jobs)
    panel.setParent(window)
    qtbot.addWidget(window)
    window.show()
    yield window, controller, registration
    controller.cancel()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller._approved = True
    window.close()


def test_reconnected_job_gets_embedded_cancel_keep_choice_without_mutation(close_window, qtbot):
    window, controller, registration = close_window
    assert controller.jobs.panel._launch_records == []
    assert not window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert controller.dock.isVisible() and not controller.dock.isFloating()
    assert controller.stop_button.isEnabled()
    assert "Registered job" in controller.status.text()
    assert job_control.read_control(registration["root"])["sequence"] == 0
    qtbot.keyClick(controller.cancel_button,Qt.Key.Key_Space)
    assert window.isVisible() and not controller.dock.isVisible()
    assert not window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller.keep_button.click()
    assert not window.isVisible()
    assert job_control.read_control(registration["root"])["sequence"] == 0


def test_stop_close_requests_original_stop_and_keeps_waiting_after_single_stopped_log(close_window, qtbot):
    window, controller, registration = close_window
    window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller.stop_button.click()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert job_control.read_control(registration["root"])["action"] == "stop"
    assert window.isVisible() and controller.timer.isActive()
    from pathlib import Path
    log = Path(registration["root"]) / "logs" / "fixture-cpu.log"
    log.parent.mkdir()
    log.write_text(job_control.utc() + ' {"kind":"stopped"}\n', encoding="utf-8")
    controller._check_retirement()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert window.isVisible() and "Waiting for worker retirement" in controller.status.text()
    controller.cancel_button.click()
    assert not controller.timer.isActive() and window.isVisible()
    assert job_control.read_control(registration["root"])["action"] == "stop"
    window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller.keep_button.click()
    assert not window.isVisible()


@pytest.mark.parametrize("width", [760, 5120])
def test_old_launch_close_choices_stay_together_and_available(close_window, qtbot, width):
    window, controller, registration = close_window
    window.resize(width, 900)
    window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert "no service status link" in controller.status.text()
    assert controller.keep_button.text() == "Close without waiting"
    assert controller.cancel_button.text() == "Return to DarSIA"
    for button in (controller.stop_button, controller.keep_button, controller.cancel_button):
        assert button.isVisible()
        assert button.geometry().right() < 760
    assert controller.keep_button.isEnabled() and controller.cancel_button.isEnabled()
    controller.stop_button.click()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller._check_retirement()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert window.isVisible() and controller.timer.isActive()
    assert "no service status link" in controller.status.text()
    assert controller.keep_button.isVisible() and controller.keep_button.isEnabled()
    assert controller.cancel_button.isVisible() and controller.cancel_button.isEnabled()
    controller.keep_button.click()
    assert not window.isVisible()
    assert job_control.read_control(registration["root"])["action"] == "stop"


def test_close_review_fits_720_with_a_tall_central_page_and_restores_it(close_window,qtbot):
    window,controller,registration=close_window
    page=QWidget(window)
    page.setMinimumHeight(1100)
    layout=QVBoxLayout(page)
    value=QLineEdit("Keep this unsaved selection",page)
    layout.addWidget(value)
    window.setCentralWidget(page)
    page.show()
    window.setMaximumSize(1024,720)
    window.resize(1024,720)
    window.close()
    qtbot.waitUntil(lambda:controller.thread is None)
    assert page.isHidden() and window.centralWidget() is page
    assert window.height()<=720
    for button in (controller.stop_button,controller.keep_button,controller.cancel_button):
        top=button.mapTo(window,QPoint(0,0))
        assert button.isVisible() and top.y()>=0
        assert top.y()+button.height()<=window.height()
        assert top.x()+button.width()<=window.width()
    assert window.focusWidget() is controller.cancel_button
    controller.cancel_button.click()
    qtbot.waitUntil(lambda:not controller.dock.isVisible())
    assert page.isVisible() and window.centralWidget() is page
    assert value.text()=="Keep this unsaved selection"
    assert job_control.read_control(registration["root"])["sequence"]==0
    # Finish the second close explicitly before pytest-qt disposes its window;
    # the native close review itself intentionally starts a finite reader.
    window.close()
    qtbot.waitUntil(lambda:controller.thread is None)
    controller.keep_button.click()
    assert not window.isVisible()


def test_failed_prestart_without_service_link_does_not_claim_retirement(tmp_path):
    from pathlib import Path

    registration = associated_job(tmp_path)
    log = Path(registration["root"]) / "logs" / "fixture-cpu.log"
    log.parent.mkdir()
    log.write_text(job_control.utc() + ' {"kind":"startup_failure","traceback":"Old source version"}\n', encoding="utf-8")
    review = close_ui.review_jobs([registration])
    assert not review["errors"] and len(review["jobs"]) == 1
    assert "close without waiting" in review["jobs"][0]["retirement_note"]
    assert job_control.read_control(registration["root"])["action"] != "stop"


def test_unavailable_close_review_keeps_explicit_close_and_return(close_window, qtbot, monkeypatch):
    window, controller, registration = close_window

    def unavailable():
        raise OSError("Service status unavailable")

    monkeypatch.setattr(controller, "_review_jobs", unavailable)
    window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert not controller.stop_button.isEnabled()
    assert controller.keep_button.text() == "Close without waiting"
    assert controller.keep_button.isVisible() and controller.keep_button.isEnabled()
    assert controller.cancel_button.isVisible() and controller.cancel_button.isEnabled()
    controller.keep_button.click()
    assert not window.isVisible()
    assert job_control.read_control(registration["root"])["sequence"] == 0


def test_close_without_waiting_discards_blocked_passive_retirement_read(close_window, qtbot, monkeypatch):
    window, controller, registration = close_window
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def read():
        entered.set()
        assert release.wait(3)
        returned.set()
        return dict(jobs=[], errors=[])
    monkeypatch.setattr(controller, '_review_jobs', read)
    window.close()
    qtbot.waitUntil(entered.is_set)
    controller.keep_button.click()
    assert not window.isVisible()
    assert controller.thread is None
    assert not release.is_set()
    assert job_control.read_control(registration['root'])['sequence'] == 0
    release.set()
    qtbot.waitUntil(returned.is_set)


def test_close_without_waiting_still_waits_for_an_actual_stop_write(close_window, qtbot, monkeypatch):
    window, controller, registration = close_window
    entered, release = threading.Event(), threading.Event()
    def write(*args, **kwargs):
        entered.set()
        assert release.wait(3)
    monkeypatch.setattr(close_ui, 'stop_reviewed_jobs', write)
    window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller.stop_button.click()
    qtbot.waitUntil(entered.is_set)
    controller.keep_button.click()
    assert window.isVisible()
    assert controller.thread is not None
    release.set()
    qtbot.waitUntil(lambda: not window.isVisible())


def test_stop_close_finishes_only_after_bound_retirement_reader_accepts(close_window, qtbot, monkeypatch):
    window, controller, registration = close_window
    retired = False
    original = close_ui.review_jobs
    monkeypatch.setattr(close_ui, "review_jobs", lambda rows, **kwargs:
        original(rows, retirement_reader=lambda request: retired))
    window.close()
    qtbot.waitUntil(lambda: controller.thread is None)
    controller.stop_button.click()
    qtbot.waitUntil(lambda: controller.thread is None)
    assert window.isVisible()
    retired = True
    controller._check_retirement()
    qtbot.waitUntil(lambda: not window.isVisible())
    assert not controller.timer.isActive()
    assert job_control.read_control(registration["root"])["action"] == "stop"


def test_old_claim_alone_does_not_claim_a_running_service(tmp_path):
    registration = associated_job(tmp_path, launch=False)
    from pathlib import Path
    put(Path(registration["root"]) / "claimed" / "fixture" / "run.json",
        dict(run="run", attempt_id="a" * 32))
    assert close_ui.review_jobs([registration]) == dict(jobs=[], errors=[])


def test_native_close_preserves_existing_busy_guard_then_uses_registered_job_choice(
    native_window, qtbot, tmp_path, monkeypatch,
):
    window = native_window
    jobs = window.ensure_jobs()
    registration = associated_job(tmp_path)
    jobs.panel.local_machine = "fixture"
    monkeypatch.setattr(jobs.registry, "known_jobs", lambda: [registration])
    busy = True
    monkeypatch.setattr(jobs, "can_close", lambda: not busy)
    assert not window.close()
    assert not hasattr(window, "_jobs_close_controller")
    assert job_control.read_control(registration["root"])["sequence"] == 0
    busy = False
    assert not window.close()
    controller = window._jobs_close_controller
    qtbot.waitUntil(lambda: controller.thread is None)
    controller.keep_button.click()
    assert not window.isVisible()


def test_native_close_discards_only_blocked_job_display_read(native_window, qtbot, monkeypatch):
    window = native_window
    jobs = window.ensure_jobs()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def reader(row):
        entered.set()
        assert release.wait(3)
        returned.set()
        return dict(root=row["root"])
    monkeypatch.setattr(jobs.registry, "known_jobs", lambda: [])
    jobs.panel.reader = reader
    jobs.panel.set_context(jobs_provider=lambda: [dict(root="display-only-fixture")])
    qtbot.waitUntil(entered.is_set)
    assert not jobs.can_close()
    assert window.close()
    assert not window.isVisible() and not release.is_set()
    release.set()
    qtbot.waitUntil(returned.is_set)


def test_explicit_native_close_retries_after_existing_guard_retires(native_window, qtbot, tmp_path, monkeypatch):
    window = native_window
    jobs = window.ensure_jobs()
    registration = associated_job(tmp_path)
    jobs.panel.local_machine = 'fixture'
    monkeypatch.setattr(jobs.registry, 'known_jobs', lambda: [registration])
    window.close()
    controller = window._jobs_close_controller
    qtbot.waitUntil(lambda: controller.thread is None)
    busy = True
    monkeypatch.setattr(jobs, 'can_close', lambda: not busy)
    controller.keep_button.click()
    assert window.isVisible() and controller.close_timer.isActive()
    busy = False
    qtbot.waitUntil(lambda: not window.isVisible())
    assert not controller.close_timer.isActive()


def test_native_close_stops_future_shared_service_polls_and_return_resumes(native_window, monkeypatch):
    jobs = native_window.ensure_jobs()
    shared = jobs.shared_jobs
    calls = []
    shared._service = SimpleNamespace(role='master', tick=lambda: calls.append('tick'))
    shared.timer.setInterval(60000)
    shared.timer.start()
    jobs.prepare_close()
    assert not shared.timer.isActive()
    shared._tick()
    assert calls == []
    jobs.cancel_close()
    assert shared.timer.isActive()
    shared.timer.stop()
    shared._service = None


def test_other_master_keeps_close_choice_but_disables_stop(tmp_path):
    registration = associated_job(tmp_path, master="other")
    review = close_ui.review_jobs([registration])
    with pytest.raises(ValueError, match="recorded master"):
        close_ui.stop_reviewed_jobs(review, machine="fixture")
    assert job_control.read_control(registration["root"])["sequence"] == 0


def _retirement_case(tmp_path, monkeypatch, mismatch=None):
    from pathlib import Path
    from darsia_cache import first_qualification, setup_service

    root = tmp_path / "job"
    job = put(root / "job.json", dict(format="darsia-alias-job-v1", format_version=1, expected_runs=["run"]))
    request = begin_launch_request(root, machine="fixture", backend="cpu", workers=1, kind="local")
    launch = Path(request["launch_directory"])
    launch.mkdir()
    stores = put(tmp_path / "stores.json", dict(synthetic=True))
    scope = dict(job=job)
    provider_pin = put(tmp_path / "provider.json", dict(
        format="darsia-setup-job-service-provider-v1", scope=scope))
    service_pin = put(tmp_path / "service.json", dict(format="darsia-setup-job-service-request-v1",
        job=job, provider=provider_pin, stores_source=stores))
    setup_service.record_prepared_service(["python", "--job", str(root)], launch, service_pin)
    original_launch = dict(path="synthetic-native-launch", sha256="a" * 64, size_bytes=1)
    primary = dict(pid=123, process_create_time="123.0")
    retired_pin = put(tmp_path / "retired.json", dict(
        format="darsia-native-owned-worker-retirement-v1", native_launch=original_launch,
        primary=primary, whole_family_kernel_observation=True,
        terminal=dict(accounting=dict(active_processes=1 if mismatch == "family" else 0))))
    eligibility = dict(original_retirement_verifier_accepted=mismatch != "eligibility",
        retirement=retired_pin, observer=dict(pid=321, process_create_time="321.0"))
    receipt_pin = put(tmp_path / "eligibility.json", dict(
        format="darsia-cache-service-lifecycle-v1", kind="eligibility", value=eligibility,
        actor=eligibility["observer"], binding=dict(policy=stores, profile=provider_pin,
            manifest=dict(native_launch_context=original_launch, owner=primary))))
    native = dict(request=service_pin if mismatch != "request" else stores,
        provider=provider_pin, scope=scope, stores_source=stores, complete=True,
        observer_native_handles_closed=mismatch != "handles", retirement=retired_pin,
        eligibility=dict(eligibility, receipt=receipt_pin), passed=False)
    monkeypatch.setattr(first_qualification, "_slot_read", lambda provider, slot: (native, {}))
    return request, native


@pytest.mark.parametrize("mismatch", [None, "request", "handles", "family", "eligibility"])
def test_prepared_service_link_uses_exact_native_whole_family_receipts(tmp_path, monkeypatch, mismatch):
    request, native = _retirement_case(tmp_path, monkeypatch, mismatch)
    assert close_ui._service_retired(request) is (mismatch is None)
    assert not native["passed"]  # Cooperative Stop need not complete the scientific job.


def test_service_status_reuses_provider_cache_but_rechecks_native_completion(
    tmp_path, monkeypatch,
):
    from pathlib import Path
    from darsia.fluidflower import jobs_metadata
    from darsia_cache import first_qualification

    request, native = _retirement_case(tmp_path / "retirement", monkeypatch)
    reader = jobs_metadata.JobSnapshotReader()
    original_read = jobs_metadata.read_json
    parsed, slots = [], []

    def read(path):
        parsed.append(Path(path).name)
        return original_read(path)

    monkeypatch.setattr(jobs_metadata, "read_json", read)
    monkeypatch.setattr(first_qualification, "_slot_read", lambda provider, slot:
        (slots.append(slot) or native, {}))
    native["observer_native_handles_closed"] = False
    assert close_ui._service_retired(request, reader=reader) is False
    native["observer_native_handles_closed"] = True
    assert close_ui._service_retired(request, reader=reader) is True
    assert parsed.count("provider.json") == 1
    assert slots == ["native_completion", "native_completion"]


def _ordinary_shared_request(tmp_path):
    from pathlib import Path
    root=tmp_path/"shared-job"
    put(root/"job.json",dict(format="darsia-alias-job-v1",format_version=1,expected_runs=["run"]))
    reference,master_format,_=job_control._bindings(root)
    put(root/"ui/master.json",dict(format=master_format,machine="fixture",job=reference(root/"job.json")))
    request=begin_launch_request(root,machine="fixture",backend="cpu",workers=6,kind="local")
    put(Path(request["launch_directory"])/"launch.json",dict(format="darsia-job-local-launch-v2",
        job=reference(root/"job.json"),machine="fixture",backend="cpu",workers=6,started=True,
        entrypoint="python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker",
        launcher_pid=None,requested_utc=job_control.utc()))
    return close_ui.read_launch_requests(root)[0],dict(root=str(root),job=close_ui.descriptor(root/"job.json"))


def test_current_shared_launch_is_identified_without_a_cache_service_link(tmp_path):
    request,registration=_ordinary_shared_request(tmp_path)
    review=close_ui.review_jobs([registration])
    assert not review["errors"] and len(review["jobs"])==1
    note=review["jobs"][0]["retirement_note"]
    assert "shared worker" in note and "older launch" not in note
    assert close_ui._service_retired(request,local_machine="fixture") is False


@pytest.mark.parametrize("live",[False,True])
def test_shared_close_reuses_original_worker_and_supervisor_epoch_verifier(tmp_path,monkeypatch,live):
    from pathlib import Path
    from darsia.fluidflower import job_management
    from darsia.fluidflower.pipeline import job_queue
    request,registration=_ordinary_shared_request(tmp_path)
    root=Path(registration["root"])
    put(root/"claimed/fixture-cpu/run.json",dict(run="run",attempt_id="a"*32))
    job_control.publish_control(root,"stop",machine="fixture",expected_previous=None)
    log=root/"logs/fixture-cpu.log"
    log.parent.mkdir()
    log.write_text(job_control.utc()+' {"kind":"stopped"}\n',encoding="utf-8")
    request=close_ui.read_launch_requests(root)[0]
    original_binding=object()
    monkeypatch.setattr(job_management,"_retirement_binding",lambda *args:original_binding)
    calls=[]
    def verify(actual_root,claimed,**options):
        calls.append((actual_root,claimed,options))
        assert options["local_output"] is original_binding
        if live:raise ValueError("Original worker or supervisor PID/birth is still running")
        return dict(original_retirement="accepted")
    monkeypatch.setattr(job_queue,"retired_claim_evidence",verify)
    assert close_ui._service_retired(request,local_machine="fixture") is (not live)
    assert len(calls)==1 and calls[0][2]["expected_action"]=="stop"
    assert close_ui._service_retired(request,local_machine="remote") is False
    assert len(calls)==1  # This machine never treats remote PIDs as local evidence.


def test_shared_stopped_log_without_original_claim_proof_never_means_retired(tmp_path):
    from pathlib import Path
    request,registration=_ordinary_shared_request(tmp_path)
    root=Path(registration["root"])
    job_control.publish_control(root,"stop",machine="fixture",expected_previous=None)
    log=root/"logs/fixture-cpu.log"
    log.parent.mkdir()
    log.write_text(job_control.utc()+' {"kind":"stopped"}\n',encoding="utf-8")
    request=close_ui.read_launch_requests(root)[0]
    assert close_ui._service_retired(request,local_machine="fixture") is False
