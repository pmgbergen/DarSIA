"""Real page clicks and local metadata, with injected transports only."""

import json
from pathlib import Path
import threading

import pytest

from darsia.fluidflower import job_launch_state as launches
from darsia.fluidflower import jobs_panel as jobs
from darsia.fluidflower import remote_worker_launch as remote


def _put(path, value):
    """Write a synthetic metadata file below the temporary test folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return jobs.descriptor(path)


def _job(root):
    """Create only the immutable metadata needed by launch reservations."""
    _put(root / "job.json", dict(expected_runs=["run-one"], precision="float32"))
    return root


def _snapshot(root):
    """Supply the selected job's display facts without running an analysis."""
    return dict(
        root=str(root), name="Fixture job", series="campaign", state="Waiting",
        job_facts=dict(precision="float32"), run_count=1, completed_runs=0,
        committed=0, total_frames=2, remaining=2, runs=[], workers=[], machines=[],
        registered_machines=[], logs=[], errors=[], launch_requests=[],
    )


@pytest.fixture
def action_page(tmp_path, qtbot):
    """Attach the real page and selected job, with no process-capable launcher."""
    root = _job(tmp_path / "job")
    calls = []

    def transport(root, **kwargs):
        calls.append((root, kwargs))
        return dict(job_root=root, started=True, numerical_progress_claimed=False)

    panel = jobs.JobsPanel(
        reader=lambda row: _snapshot(row["root"]), local_machine="local",
        local_launcher=transport, remote_launcher=transport,
    )
    page = jobs.NewAnalysisPage(
        panel,
        facts_reader=lambda row: dict(
            runs=["run-one"], frames=2, precision="float32", storage_mm=[None]),
    )
    panel.attach_new_analysis(page)
    qtbot.addWidget(panel, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    qtbot.addWidget(page, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    panel._show_snapshot(_snapshot(root))
    page.created_root = str(root)
    page.sync_actions()
    page.resize(1280, 800)
    page.show()
    assert page.start_button.isEnabled()
    yield panel, page, root, calls
    qtbot.waitUntil(lambda: panel.can_close() and page.can_close(), timeout=10000)


@pytest.mark.parametrize("refusal", ["pending_request", "unreadable_registered_job"])
def test_start_revalidation_refusal_shows_the_current_reason(action_page, refusal, tmp_path):
    """A stale enabled page action cannot turn a fresh refusal into success."""
    panel, page, root, calls = action_page
    if refusal == "pending_request":
        launches.begin_launch_request(root, machine="local", backend="cpu", workers=1, kind="local")
        panel.known_jobs_provider = lambda: [dict(root=str(root))]
    else:
        panel.known_jobs_provider = lambda: [dict(root=str(tmp_path / "unavailable-job"))]
    errors = []
    panel.error_reported.connect(errors.append)
    page.start_button.click()
    assert not calls and not errors
    assert page.action_label.text() == "Start was not requested. " + panel.start_button.toolTip()
    assert "Launch" in page.action_label.text()
    assert "Start requested" not in page.action_label.text()
    assert page._pending_action is None and not page.start_button.isEnabled()


@pytest.mark.parametrize("outcome", ["local_success", "remote_success", "local_failure"])
def test_start_feedback_follows_real_transport_signals(action_page, outcome, monkeypatch):
    """Only the existing launch signal confirms a request; failures keep details."""
    panel, page, root, calls = action_page
    seen = []

    def transport(root, **kwargs):
        seen.append((page.action_label.text(), page._pending_action))
        calls.append((root, kwargs))
        if outcome == "local_failure":
            raise RuntimeError("Synthetic transport refused the request")
        return dict(job_root=root, started=True, numerical_progress_claimed=False)

    panel.local_launcher = panel.remote_launcher = transport
    if outcome == "remote_success":
        monkeypatch.setattr(remote, "require_unc", lambda path: None)
        record = dict(
            machine="remote", ssh_alias="fixture", python=r"C:\fixture\python.exe",
            checkout=r"C:\fixture\checkout", ssh_executable=r"C:\fixture\ssh.exe",
            default_shell="windows-powershell", job=jobs.descriptor(root / "job.json"),
        )
        record["source"] = _put(root / "ui" / "machines" / "registered.json", record)
        panel.current["registered_machines"] = [record]
        panel._refresh_targets()
        panel.target_combo.setCurrentIndex(1)
    assert page.start_button.isEnabled()
    page.start_button.click()
    assert seen == [("Requesting start…", "start")]
    assert len(calls) == 1 and page._pending_action is None
    if outcome == "local_failure":
        assert page.action_label.text() == (
            "Start request failed. RuntimeError: Synthetic transport refused the request")
        assert "Traceback" not in page.action_label.text()
        assert "Traceback" in panel.error_text.toPlainText()
    else:
        assert page.action_label.text() == "Start requested. Follow the progress on Jobs."
    assert len(list((root / "ui" / "launches").glob("*.request.json"))) == 1
    assert panel.current["committed"] == 0


def test_start_metadata_error_replaces_request_feedback(action_page):
    """The existing metadata error path is visible beside the clicked action."""
    panel, page, root, calls = action_page
    panel.current["errors"] = ["Synthetic job metadata is unreadable"]
    page.start_button.click()
    assert not calls and not list((root / "ui" / "launches").glob("*.request.json"))
    assert page.action_label.text() == "Start request failed. Synthetic job metadata is unreadable"
    assert "metadata errors" in panel.error_text.toPlainText()
    assert page._pending_action is None


def _prepare_create(panel, page, root, qtbot):
    """Fill the existing form without creating a plan or binding real inputs."""
    panel.series_root, panel.series_id = root.parent, "campaign"
    panel.selection_provider = lambda *args: []
    page.set_plans([dict(path=str(root.parent / "plan.toml"), sha256="a" * 64)])
    page.form.job_edit.setText(str(root.parent / "created-job"))
    qtbot.waitUntil(lambda: page.can_close() and page.facts is not None, timeout=10000)
    assert page.create_button.isEnabled()


def test_async_create_failure_clears_busy_feedback_and_retry_can_succeed(action_page, qtbot):
    """A failed creator retires its page state and the next creation can finish."""
    panel, page, root, calls = action_page
    _prepare_create(panel, page, root, qtbot)
    entered, release = threading.Event(), threading.Event()

    def failing_creator(plan, job_root, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("Synthetic creator was not released")
        raise FileExistsError("The selected job folder already exists")

    panel.job_creator = failing_creator
    page.create_button.click()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert page.action_label.text() == "Creating job files… No worker is started."
        assert page._pending_action == "create"
    finally:
        release.set()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert page.action_label.text() == (
        "Job file creation failed. FileExistsError: The selected job folder already exists")
    assert "Traceback" in panel.error_text.toPlainText()
    assert page._pending_action is None and page.create_button.isEnabled()

    def creator(plan, job_root, **kwargs):
        _job(Path(job_root))
        return dict(root=job_root, name="Created job", series_label="campaign", active=True)

    panel.job_creator = creator
    page.create_button.click()
    qtbot.waitUntil(lambda: panel.can_close() and page._pending_action is None, timeout=10000)
    assert page.created_root == str(root.parent / "created-job")
    assert page.action_label.text().startswith("Job files created: ")
    assert not calls


def test_unrelated_signals_and_reset_do_not_claim_page_actions(action_page, qtbot):
    """Unrelated results and a result arriving after Reset leave feedback alone."""
    panel, page, root, calls = action_page
    before = page.action_label.text(), page.created_root
    panel.local_worker_launch_requested.emit(dict(started=True))
    panel.remote_worker_launch_requested.emit(dict(transport_started=True))
    panel.error_reported.emit("An unrelated metadata read failed")
    panel.job_created.emit(dict(root=str(root.parent / "other-job")))
    assert (page.action_label.text(), page.created_root) == before
    _prepare_create(panel, page, root, qtbot)
    entered, release = threading.Event(), threading.Event()

    def creator(plan, job_root, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("Synthetic creator was not released")
        _job(Path(job_root))
        return dict(root=job_root, name="Created job", series_label="campaign", active=True)

    panel.job_creator = creator
    page.create_button.click()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        page.reset()
        assert page._pending_action is None and page.created_root is None
    finally:
        release.set()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert page.action_label.text() == "No job files created on this page yet."
    assert page.created_root is None and not calls


def test_unrelated_worker_failure_during_creation_preserves_its_success(action_page, qtbot):
    """A separate worker's forwarded error cannot retire the active creator."""
    panel, page, root, calls = action_page
    _prepare_create(panel, page, root, qtbot)
    entered, release = threading.Event(), threading.Event()
    failures = []
    creation_failures = []
    panel.error_reported.connect(failures.append)
    panel.job_creation_failed.connect(creation_failures.append)

    def creator(plan, job_root, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Synthetic creator was not released")
        _job(Path(job_root))
        return dict(root=job_root, name="Created job", series_label="campaign", active=True)

    def unrelated_read():
        raise RuntimeError("Synthetic unrelated metadata read failed")

    panel.job_creator = creator
    page.create_button.click()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        panel._read_thread, panel._read_worker = panel._start_call(
            unrelated_read, panel._read_ready, panel._read_closed)
        qtbot.waitUntil(lambda: panel._read_thread is None, timeout=5000)
        assert len(failures) == 1 and not creation_failures
        assert "Synthetic unrelated metadata read failed" in failures[0]
        assert page._pending_action == "create"
        assert page.action_label.text() == "Creating job files… No worker is started."
    finally:
        release.set()
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert page.created_root == str(root.parent / "created-job")
    assert page.action_label.text().startswith("Job files created: ")
    assert page._pending_action is None and not calls


def test_immediate_creator_failure_emits_the_creation_specific_signal(action_page, qtbot):
    """An immediate failure is identified in the original error slot."""
    panel, page, root, calls = action_page
    _prepare_create(panel, page, root, qtbot)
    failures = []
    creation_failures = []
    panel.error_reported.connect(failures.append)
    panel.job_creation_failed.connect(creation_failures.append)

    def creator(plan, job_root, **kwargs):
        raise FileExistsError("The selected job folder already exists")

    panel.job_creator = creator
    page.create_button.click()
    assert panel._create_worker is not None
    qtbot.waitUntil(panel.can_close, timeout=10000)
    assert len(failures) == 1 and creation_failures == failures
    assert page.action_label.text() == (
        "Job file creation failed. FileExistsError: The selected job folder already exists")
    assert "Traceback" in panel.error_text.toPlainText()
    assert page._pending_action is None and not calls
