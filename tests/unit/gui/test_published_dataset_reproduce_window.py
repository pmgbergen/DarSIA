"""The launcher window and the command line give the same report (offscreen Qt)."""

import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower.published_dataset import reproduce_cli
from darsia.fluidflower.published_dataset.comparison import IDENTICAL
from tests.unit.published_dataset.test_reproduce import INSTALLED, FakeRunner, dataset  # noqa: F401


def test_command_line_and_window_give_the_same_report(dataset, tmp_path, qtbot):
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow

    arguments = reproduce_cli.parser().parse_args(["--dataset", str(dataset), "--results", str(tmp_path / "cli"),
                                                   "--workers", "1"])
    assert reproduce_cli.run(arguments, runner=FakeRunner(), machine={"cpu": "fixture"}, installed=INSTALLED) == 0
    window = ReproduceWindow(runner_factory=FakeRunner, installed=INSTALLED, fingerprint={},
                             machine={"cpu": "fixture"},
                             inventory={"cpu": {"physical_cores": 4, "logical_cores": 8},
                                        "memory": {"available_bytes": 16 * 1024**3}})
    qtbot.addWidget(window)
    window.dataset_edit.setText(str(dataset))
    window.results_edit.setText(str(tmp_path / "window"))
    window.open_dataset()
    qtbot.waitUntil(lambda: window.reproduction is not None and not window.busy(), timeout=20000)
    assert window.workers_spin.value() == 2 and "measured on the reference machine" in window.advice_label.text()
    assert window.scope_buttons["quick"].isEnabled() and window.precision_label.text().startswith("float32")
    window.workers_spin.setValue(1)
    window.start()
    qtbot.waitUntil(lambda: window.result is not None or "Stopped" in window.status_label.text(), timeout=20000)
    assert window.status_label.text() == f"Verdict: {IDENTICAL}." and window.report_button.isEnabled()
    assert window.progress_table.rowCount() == 2

    def comparable(folder):
        value = json.loads((folder / "report" / "report.json").read_text(encoding="utf-8"))
        value.pop("created_utc")
        value.pop("timing")
        return value
    assert comparable(tmp_path / "cli") == comparable(tmp_path / "window")


def test_runtime_mapping_is_an_explicit_optional_launcher_setting(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow

    window = ReproduceWindow()
    qtbot.addWidget(window)
    assert window.settings().stage_runtimes is None
    selected = tmp_path / "original-runtimes.json"
    window.stage_runtimes_edit.setText("  " + str(selected) + "  ")
    assert window.settings().stage_runtimes == str(selected)
    other = tmp_path / "selected-runtimes.json"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(other), "Runtime mapping (*.json)"))
    window._choose_stage_runtimes()
    assert window.settings().stage_runtimes == str(other)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))
    window._choose_stage_runtimes()
    assert window.settings().stage_runtimes == str(other)
    assert window.runner is None and window.tasks == []
    assert not selected.exists() and not other.exists()


def test_gui_command_line_preserves_explicit_stage_runtime_mapping(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.published_dataset import reproduce_window as module

    window = module.ReproduceWindow()
    qtbot.addWidget(window)
    opened = []
    monkeypatch.setattr(window, "open_dataset", lambda: opened.append(window.dataset_edit.text()))
    monkeypatch.setattr(module, "ReproduceWindow", lambda: window)
    # Keep the existing test Qt event loop; this checks argument transfer only.
    application = SimpleNamespace(exec=lambda: 0)
    monkeypatch.setattr(module, "QApplication", SimpleNamespace(instance=lambda: application))
    arguments = reproduce_cli.parser().parse_args(["--gui", "--dataset", str(tmp_path / "dataset"),
        "--results", str(tmp_path / "results"), "--stage-runtimes", str(tmp_path / "runtimes.json")])
    assert module.main(arguments) == 0
    assert window.settings().stage_runtimes == str(tmp_path / "runtimes.json")
    assert opened == [str(tmp_path / "dataset")]
    assert window.results_edit.text() == str(tmp_path / "results")
    assert window.tasks == []


def test_staged_dataset_does_not_offer_a_launcher_commit_override(qtbot, monkeypatch):
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow

    window = ReproduceWindow()
    qtbot.addWidget(window)
    window.override_check.setChecked(True)
    window.quick_spin.blockSignals(True)
    monkeypatch.setattr(window, "refresh_settings", lambda *args: None)
    # This is a display-state fixture, never admitted as a completed dataset.
    manifest = dict(identity=dict(title="Staged", series_id="SYN", record="R"),
        runs=[dict(run="r1", frames=[dict(case_id="one")])], plan=dict(precision="float32"),
        machine=dict(threads=1), quick_check_valid=False,
        expected=dict(layer=dict(resolution_mm=1)), staged_reproduction=dict(stages={"p1-r1": {}}))
    mismatch = "The launcher checkout is newer than the original producing job"
    check = SimpleNamespace(manifest=manifest, runs=["r1"], code=dict(same=False, note=mismatch),
                            environment=[], problems=[])
    window._opened(SimpleNamespace(check=check), check, "")
    assert window.override_check.isHidden() and not window.override_check.isEnabled()
    assert window.settings().override_commit is False
    assert not window.scope_buttons["quick"].isEnabled()
    assert "stage" in window.dataset_label.text().lower()
    assert mismatch not in window.dataset_label.text()
    assert window.tasks == [] and window.result is None


@pytest.mark.parametrize("change, message", [
    ("zero_workers", "at least one worker"),
    ("runtime_mapping", "Keep the admitted stage runtimes"),
    ("commit_override", "original commit"),
])
def test_staged_resume_refusal_is_visible_and_keeps_pause_until_valid_retry(
        qtbot, tmp_path, monkeypatch, change, message):
    from darsia.fluidflower.published_dataset import staged_runner
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow

    window = ReproduceWindow()
    qtbot.addWidget(window)
    window.show()
    window.workers_spin.setValue(1)
    window.threads_spin.setValue(2)
    window.stage_runtimes_edit.setText("explicit-runtimes.json")
    selected = staged_runner.StagedReplayRunner()
    initial = window.settings()
    selected.settings = initial
    selected.control = tmp_path / "staged-control.json"
    window.runner = selected
    window.pause()
    before = selected.control.read_bytes(), selected.control.stat().st_mtime_ns
    if change == "zero_workers":
        window.workers_spin.setValue(0)
    elif change == "runtime_mapping":
        window.stage_runtimes_edit.setText("another-runtimes.json")
    else:
        window.override_check.setChecked(True)
    monkeypatch.setattr(staged_runner, "ThreadPoolExecutor",
                        lambda *args, **kwargs: pytest.fail("This GUI check must not launch scientific workers"))

    window.resume()
    assert window.status_label.isVisible()
    assert window.status_label.text().startswith("Not resumed:\n")
    assert message in window.status_label.text()
    assert window.resume_button.isEnabled() and not window.pause_button.isEnabled()
    assert selected.settings is initial and not selected._resume.is_set()
    assert (selected.control.read_bytes(), selected.control.stat().st_mtime_ns) == before
    assert window.tasks == [] and window.result is None

    window.workers_spin.setValue(1)
    window.stage_runtimes_edit.setText(initial.stage_runtimes)
    window.override_check.setChecked(False)
    window.resume()
    assert window.status_label.text() == "Running…"
    assert not window.resume_button.isEnabled() and window.pause_button.isEnabled()
    assert selected._resume.is_set() and selected.settings == window.settings()
    assert json.loads(selected.control.read_bytes()) == {"action": "run"}
    assert window.tasks == [] and selected.evidence is None and selected.outputs == {}


def test_staged_resume_forwards_supplied_memory_inventory_and_keeps_pause_on_refusal(
        qtbot, tmp_path, monkeypatch):
    """Paused metadata state only: no real dataset admission or scientific start.

    Exercise the actual window -> ReplayRunner -> StagedReplayRunner resume
    chain and original memory guard, with explicit fixture memory observations.
    """
    from darsia.fluidflower.published_dataset import replay, resources, staged_runner
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow

    gib = 1024 ** 3
    inventory = {"memory": {"available_bytes": 4 * gib}}
    assert resources.LAUNCHER_RESERVE_BYTES == gib

    def forbidden(*args, **kwargs):
        pytest.fail("This paused metadata fixture must not probe the host or launch scientific workers")

    monkeypatch.setattr(resources, "machine_inventory", forbidden)
    monkeypatch.setattr(staged_runner, "ThreadPoolExecutor", forbidden)
    window = ReproduceWindow(inventory=inventory)
    qtbot.addWidget(window)
    window.show()
    window.workers_spin.setValue(1)
    window.threads_spin.setValue(1)
    window.stage_runtimes_edit.setText("explicit-runtimes.json")
    facade = replay.ReplayRunner(inventory=inventory, launch=forbidden)
    selected = facade._staged()
    assert selected.inventory is inventory
    initial = window.settings()
    selected.settings = initial
    selected.check = SimpleNamespace(manifest={"machine": {"peak_memory_bytes": 2 * gib}})
    selected.scope = SimpleNamespace(runs=["r1", "r2"])
    selected.control = tmp_path / "staged-control.json"
    window.runner = facade
    window.pause()
    before = selected.control.read_bytes(), selected.control.stat().st_mtime_ns
    buttons = window.resume_button.isEnabled(), window.pause_button.isEnabled()
    assert buttons == (True, False) and not selected._resume.is_set()
    assert selected.outputs == {}

    window.workers_spin.setValue(2)
    window.resume()

    assert window.status_label.isVisible()
    message = window.status_label.text()
    assert message.startswith("Not resumed:\nMemory: 2 worker(s)")
    assert "5.0 GiB" in message and "4.0 GiB" in message
    assert selected.settings is initial and not selected._resume.is_set()
    assert (selected.control.read_bytes(), selected.control.stat().st_mtime_ns) == before
    assert (window.resume_button.isEnabled(), window.pause_button.isEnabled()) == buttons
    assert facade.staged_runner is selected and selected.inventory is inventory
    assert window.tasks == [] and window.result is None
    assert facade.workers == [] and selected.evidence is None and selected.outputs == {}

    window.workers_spin.setValue(1)
    window.resume()

    assert window.status_label.text() == "Running…"
    assert not window.resume_button.isEnabled() and window.pause_button.isEnabled()
    assert selected.settings == window.settings() and selected.settings.workers == 1
    assert selected._resume.is_set()
    assert json.loads(selected.control.read_bytes()) == {"action": "run"}
    assert window.tasks == [] and window.result is None
    assert facade.workers == [] and selected.evidence is None and selected.outputs == {}
