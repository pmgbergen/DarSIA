"""Local Qt fixtures only. No actual job, SSH, hardware probe or source writes."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import threading
import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView

from darsia.fluidflower import measurement_receipts as receipts
from darsia.fluidflower import resource_advisor_panel as ui

GIB = ui.GIB


def source(name="measured.json"):
    return {"path": "fixture/" + name, "sha256": "d" * 64, "size_bytes": 100}


def job(precision="float32"):
    return {
        "job_id": "local-ui-fixture",
        "name": "Saved job",
        "precision": precision,
        "pipeline": "native-phase-n1",
        "commit": "a" * 40,
    }


def sample(precision="float32", backend="cpu"):
    return {
        "machine": "Main",
        "backend": backend,
        "precision": precision,
        "threads": 2,
        "pipeline": "native-phase-n1",
        "git_commit": "a" * 40,
        "frames": 2,
        "wall_seconds": 100,
        "compute_and_input_seconds": 60,
        "publication_and_checkpoint_seconds": 40,
        "peak_worker_rss_bytes": None,
        "source": source(),
    }


def machine(*, history=None):
    return {
        "machine": "Main",
        "backend": "cpu",
        "threads": 2,
        "device_id": "0",
        "inventory": {
            "host": "main-host",
            "recorded_utc": "2026-10-16T00:00:00+00:00",
            "cpu": {"physical_cores": 16, "logical_cores": 32},
            "memory": {"total_bytes": 128 * GIB, "available_bytes": 100 * GIB},
            "accelerators": [
                {
                    "family": "cuda",
                    "name": "Recorded GPU",
                    "device_id": "0",
                    "memory_bytes": 24 * GIB,
                }
            ],
        },
        "history": history if history is not None else [sample("float64")],
    }


def table_text(table, row):
    return [table.item(row, column).text() for column in range(table.columnCount())]


@pytest.fixture
def panel(qtbot):
    def forbidden():
        pytest.fail("No real hardware provider may run from constructor/set_context")

    value = ui.ResourceAdvisorPanel(inventory_provider=forbidden)
    qtbot.addWidget(value)
    return value


def test_open_context_and_refresh_only_read_supplied_snapshots(panel, qtbot, tmp_path):
    saved = tmp_path / "series.toml"
    saved.write_bytes(b'[method]\nactive="phase_model"\n')
    before = saved.read_bytes()
    rows = [machine()]
    original = deepcopy(rows)
    context = job()
    original_job = deepcopy(context)
    calls, choices = [], []
    panel.workers_selected.connect(choices.append)

    def provider():
        calls.append("read")
        return rows

    panel.set_context(context, series_label="<b>Series</b>", machines_provider=provider)
    panel.show()
    qtbot.waitUntil(panel.isVisible)
    assert calls == ["read"] and choices == []
    assert rows == original and context == original_job and saved.read_bytes() == before
    assert panel.context_label.textFormat() == Qt.TextFormat.PlainText
    assert "<b>Series</b>" in panel.context_label.text()
    assert (
        panel.hardware_table.editTriggers()
        == QAbstractItemView.EditTrigger.NoEditTriggers
    )
    assert panel.can_close()


def test_historical_precision_threads_and_time_scopes_are_visible(panel):
    panel.set_machines([machine()])
    panel.set_context(job(), series_label="FF_AC")
    cells = table_text(panel.history_table, 0)
    assert cells[1:5] == ["cpu", "float64", "2", "2"]
    assert cells[5] == "100.00 s total / 50.00 s per frame"
    assert cells[6] == "60.00 s total / 30.00 s per frame"
    assert cells[7] == "40.00 s total / 20.00 s per frame"
    assert cells[8:10] == [ui.UNKNOWN, ui.UNKNOWN]
    assert any(
        "not a current-job speed prediction" in label.text()
        for label in panel.findChildren(ui.QLabel)
    )
    assert panel._advice["workers"] is None
    assert "float32" in panel.context_label.text()


def test_cpu64_rss_is_not_reused_as_float32_memory(panel):
    old = sample("float64")
    old["peak_worker_rss_bytes"] = GIB
    panel.set_machines([machine(history=[old])])
    panel.set_context(job())
    assert panel._advice["workers"] is None
    assert "No peak worker RSS measurement matches" in panel.reason_label.text()
    panel.policy_budget.setChecked(True)
    assert panel._advice["workers"] == 3
    assert "not measured RSS" in panel.reason_label.text()
    assert "40 GiB" in panel.reason_label.text()


def test_matching_peak_rss_and_explicit_reserve_bound_count(panel):
    measured = sample()
    measured["peak_worker_rss_bytes"] = 12 * GIB
    panel.set_machines([machine(history=[measured])])
    panel.set_context(job())
    assert panel._advice["workers"] == 5
    assert panel._advice["core_limit"] == 8 and panel._advice["ram_limit"] == 5
    assert "Largest recorded peak worker RSS" in panel.reason_label.text()
    panel.threads_spin.setValue(3)
    assert panel._advice["workers"] is None


def test_mismatched_pipeline_commit_or_unbound_receipt_cannot_supply_measured_rss():
    for field, other in [("pipeline", "n1"), ("git_commit", "b" * 40), ("source", {})]:
        measured = sample()
        measured["peak_worker_rss_bytes"] = GIB
        measured[field] = other
        assert (
            ui.resource_advice(machine(history=[measured]), job(), "cpu", 2)["workers"]
            is None
        )


def test_gpu_total_vram_and_host_ram_are_not_available_gpu_memory(panel):
    measured = sample(backend="cuda")
    measured["peak_device_memory_bytes"] = 5 * GIB
    measured["device_id"] = "0"
    row = machine(history=[measured])
    row["backend"] = "cuda"
    panel.set_machines([row])
    panel.set_context(job())
    panel.policy_budget.setChecked(True)
    assert "24.00 GiB" in table_text(panel.hardware_table, 0)[4]
    assert panel._advice["workers"] is None and "not VRAM" in panel.reason_label.text()
    row["inventory"]["accelerators"][0]["available_bytes"] = 12 * GIB
    panel.set_machines([row])
    assert panel._advice["workers"] == 2
    assert "Separate measured device-memory limit = 2" in panel.reason_label.text()
    row["history"][0]["device_id"] = "another-device"
    panel.set_machines([row])
    assert panel._advice["workers"] is None


def test_zero_capacity_stays_zero_and_total_ram_does_not_replace_available(panel):
    row = machine()
    row["inventory"]["memory"]["available_bytes"] = 30 * GIB
    panel.set_machines([row])
    panel.set_context(job())
    panel.policy_budget.setChecked(True)
    assert panel._advice["workers"] == 0
    assert not panel.suggested_button.isEnabled()
    del row["inventory"]["memory"]["available_bytes"]
    panel.set_machines([row])
    assert panel._advice["workers"] is None
    assert "total RAM is not a free-memory measurement" in panel.reason_label.text()


def test_override_is_explicit_signal_only_and_does_not_mutate_job(panel, qtbot):
    context = job()
    original = deepcopy(context)
    choices = []
    panel.workers_selected.connect(choices.append)
    panel.set_context(context)
    panel.set_machines([machine()])
    panel.workers_spin.setValue(4)
    assert choices == []
    panel.use_button.click()
    assert len(choices) == 1
    chosen = choices[0]
    assert (
        chosen["workers"] == 4
        and chosen["threads"] == 2
        and chosen["precision"] == "float32"
    )
    assert chosen["machine"] == "Main" and chosen["backend"] == "cpu"
    assert chosen["override"] and chosen["starts_job"] is False
    assert context == original and panel._advice["workers"] is None


def test_missing_context_does_not_choose_precision_or_emit_worker_action(panel):
    choices = []
    panel.workers_selected.connect(choices.append)
    panel.set_machines([machine()])
    panel.set_context({})
    panel.use_button.click()
    assert not panel.use_button.isEnabled() and choices == []
    assert "Not recorded" in panel.context_label.text()


def test_refresh_error_clears_old_machine_facts_before_new_context(panel):
    panel.set_machines([machine()])
    panel.set_context(job())

    def failed():
        raise OSError("saved inventory unavailable verbatim")

    panel.set_context(job(), series_label="Another series", machines_provider=failed)
    assert panel.hardware_table.rowCount() == 0 and panel.history_table.rowCount() == 0
    assert panel.machine_combo.count() == 0 and not panel.use_button.isEnabled()
    assert "saved inventory unavailable verbatim" in panel.probe_label.text()


def test_invalid_saved_facts_clear_stale_context_and_show_error(panel):
    panel.set_machines([machine()])
    panel.set_context(job())
    row = machine()
    row["history"] = [None]
    panel.set_context(job(), machines_provider=lambda: [row])
    assert panel.hardware_table.rowCount() == 0 and panel.history_table.rowCount() == 0
    assert "history a list of sample objects" in panel.probe_label.text()
    assert not panel.use_button.isEnabled()


def test_explicit_fake_hardware_inspection_runs_off_ui_thread_and_retires(qtbot):
    calls, delivered, chosen = [], [], []
    main_thread = threading.get_ident()

    def provider():
        calls.append(threading.get_ident())
        return {"host": "LOCAL_FIXTURE", "cpu": {"logical_cores": 8}}

    panel = ui.ResourceAdvisorPanel(inventory_provider=provider)
    qtbot.addWidget(panel)
    panel.hardware_inspected.connect(delivered.append)
    panel.workers_selected.connect(chosen.append)
    panel.set_context(job())
    panel.set_machines([machine()])
    before = deepcopy(panel.machines)
    assert calls == []
    panel.inspect_button.click()
    qtbot.waitUntil(lambda: panel._probe_thread is None, timeout=5000)
    assert len(calls) == 1 and calls[0] != main_thread
    assert delivered[0]["host"] == "LOCAL_FIXTURE"
    assert panel.machines == before and chosen == [] and panel.can_close()
    assert "remote snapshots are not overwritten" in panel.probe_label.text()


def test_fake_hardware_failure_is_verbatim_and_no_action_is_sent(qtbot):
    def failed():
        raise RuntimeError("explicit fake hardware failure")

    panel = ui.ResourceAdvisorPanel(inventory_provider=failed)
    qtbot.addWidget(panel)
    errors, choices = [], []
    panel.hardware_inspection_failed.connect(errors.append)
    panel.workers_selected.connect(choices.append)
    panel.inspect_button.click()
    qtbot.waitUntil(lambda: panel._probe_thread is None, timeout=5000)
    assert "RuntimeError: explicit fake hardware failure" in errors[0]
    assert panel.probe_label.text() == errors[0] and choices == []


def test_active_inventory_read_is_discarded_on_close(qtbot):
    release = threading.Event()
    returned = threading.Event()
    entered = threading.Event()

    def provider():
        entered.set()
        assert release.wait(3)
        returned.set()
        return {"host": "LOCAL_FIXTURE"}

    panel = ui.ResourceAdvisorPanel(inventory_provider=provider)
    qtbot.addWidget(panel)
    panel.show()
    results = []
    panel.hardware_inspected.connect(results.append)
    panel.inspect_button.click()
    qtbot.waitUntil(entered.is_set)
    assert not panel.can_close()
    panel.close()
    assert not panel.isVisible() and panel.can_close()
    release.set()
    qtbot.waitUntil(returned.is_set, timeout=5000)
    assert results == []


def test_duplicate_machine_snapshots_rejected_and_input_copy_is_independent(panel):
    row = machine()
    with pytest.raises(ValueError, match="Duplicate"):
        panel.set_machines([row, row])
    panel.set_machines([row])
    row["inventory"]["cpu"]["physical_cores"] = 999
    assert "16 physical" in table_text(panel.hardware_table, 0)[1]
