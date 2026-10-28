"""Actual local receipt import, no historical NAS lookup, RAW or jobs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from .test_resource_advisor_panel import receipts, ui, job, table_text


def put(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(doc, ensure_ascii=False) + "\n").encode())
    return receipts._read(path)[1]


def originals(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def completion(root, *, machine="Second", precision="float32"):
    doc = dict(
        complete=True,
        worker_references_closed=True,
        whole_run=True,
        run="ac52",
        frames=2,
        records=[dict(frame_id="ac52_f1"), dict(frame_id="ac52_f9")],
        ordered_case_ids=["ac52_f1", "ac52_f9"],
        backend="cpu",
        precision=precision,
        wall_seconds=120,
        commit="a" * 40,
        actual_worker=dict(machine=machine, pid=123, process_create_time="123.5"),
        task=dict(
            run="ac52",
            threads=1,
            backend="cpu",
            precision=precision,
            commit="a" * 40,
            pipeline="native-phase-n1",
        ),
    )
    path = root / "completed.json"
    put(path, doc)
    return path, doc


def historical(root):
    code = put(root / "lock.json", dict(commit="b" * 40))
    packet = put(
        root / "worker_packet.json",
        dict(
            format="ffac-round4-main-operational-packet-v1",
            host="Main",
            code_lock=code,
            tasks=dict(
                ac52=dict(
                    run="ac52",
                    settings=dict(
                        host="Main", threads=2, precision="float64", backend="cpu"
                    ),
                )
            ),
            working_inputs={"RAW": "must never be followed"},
        ),
    )
    doc = dict(
        complete=True,
        worker_references_closed=True,
        run="ac52",
        stage="heldout",
        frames=2,
        records=[dict(frame_id="ac52_f1"), dict(frame_id="ac52_f9")],
        backend="cpu",
        precision="float64",
        wall_seconds=200,
        actual_worker=dict(pid=456, process_create_time="456.5"),
        packet=packet,
        code_lock=code,
    )
    path = root / "historical_completed.json"
    put(path, doc)
    return path, doc


def trial(root):
    path, doc = completion(root)
    bound = receipts._read(path)[1]
    summary = dict(
        complete=True,
        run="ac52",
        case_count=2,
        precision="float32",
        code_commit="a" * 40,
        actual_worker=doc["actual_worker"],
        whole_run_completion=bound,
        timings=dict(
            pipeline_wall_seconds=120,
            input_read_preprocess_and_compute_seconds=70,
            frame_publication_and_start_checkpoint_seconds=45,
        ),
    )
    target = root / "trial_summary.json"
    put(target, summary)
    return target, summary


def test_current_completion_reads_actual_machine_precision_threads_no_inferred_subtimes(
    tmp_path,
):
    path, _ = completion(tmp_path)
    before = originals(tmp_path)
    sample = receipts.load_measurement_receipt(path)
    assert (
        sample["machine"] == "Second"
        and sample["precision"] == "float32"
        and sample["threads"] == 1
    )
    assert (
        sample["frames"] == 2
        and sample["wall_seconds"] == 120
        and sample["pipeline"] == "native-phase-n1"
    )
    assert (
        sample["compute_and_input_seconds"] is None
        and sample["publication_and_checkpoint_seconds"] is None
    )
    assert (
        sample["peak_worker_rss_bytes"] is None
        and sample["peak_device_memory_bytes"] is None
    )
    assert originals(tmp_path) == before


def test_historical_cpu64_binds_packet_and_lock_no_cpu32_substitution(tmp_path):
    path, _ = historical(tmp_path)
    before = originals(tmp_path)
    sample = receipts.load_measurement_receipt(path)
    assert (
        sample["machine"] == "Main"
        and sample["precision"] == "float64"
        and sample["threads"] == 2
    )
    assert (
        sample["pipeline"] == "historical-round4-heldout"
        and sample["git_commit"] == "b" * 40
    )
    assert (
        len(sample["source_bindings"]) == 3
        and sample["compute_and_input_seconds"] is None
    )
    assert (
        originals(tmp_path) == before
        and not (tmp_path / "must never be followed").exists()
    )


def test_trial_summary_uses_only_explicit_subtimes_bound_to_same_completion(tmp_path):
    path, summary = trial(tmp_path)
    sample = receipts.load_measurement_receipt(path)
    assert sample["source"] == summary["whole_run_completion"]
    assert sample["measurement_receipt"]["path"] == str(path)
    assert (
        sample["compute_and_input_seconds"] == 70
        and sample["publication_and_checkpoint_seconds"] == 45
    )
    assert sample["wall_seconds"] == 120 and sample["peak_worker_rss_bytes"] is None


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("precision", "float64", "identity differs"),
        ("code_commit", "c" * 40, "identity differs"),
        ("case_count", 39, "identity differs"),
    ],
)
def test_trial_identity_mismatch_refuses(tmp_path, field, value, message):
    path, summary = trial(tmp_path)
    summary[field] = value
    put(path, summary)
    with pytest.raises(ValueError, match=message):
        receipts.load_measurement_receipt(path)


def test_trial_changed_bound_completion_or_wall_time_refuses(tmp_path):
    path, summary = trial(tmp_path)
    summary["timings"]["pipeline_wall_seconds"] = 121
    put(path, summary)
    with pytest.raises(ValueError, match="wall time differs"):
        receipts.load_measurement_receipt(path)
    original = Path(summary["whole_run_completion"]["path"])
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA/path/size changed"):
        receipts.load_measurement_receipt(path)


def test_unknown_schema_and_nonfinite_data_do_not_gain_fabricated_context(tmp_path):
    path = tmp_path / "unknown.json"
    put(
        path,
        dict(
            complete=True,
            worker_references_closed=True,
            frames=2,
            records=[dict(frame_id="a"), dict(frame_id="b")],
            run="ac52",
            precision="float32",
            backend="cpu",
            wall_seconds=12,
        ),
    )
    with pytest.raises(ValueError, match="Unsupported measured completion"):
        receipts.load_measurement_receipt(path)
    path.write_bytes(b'{"wall_seconds":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        receipts.load_measurement_receipt(path)


def test_duplicate_frames_wrong_task_or_invalid_memory_are_rejected(tmp_path):
    path, doc = completion(tmp_path)
    doc["records"][1]["frame_id"] = "ac52_f1"
    put(path, doc)
    with pytest.raises(ValueError, match="duplicated"):
        receipts.load_measurement_receipt(path)
    path, doc = completion(tmp_path)
    doc["task"]["precision"] = "float64"
    put(path, doc)
    with pytest.raises(ValueError, match="differs"):
        receipts.load_measurement_receipt(path)
    path, doc = completion(tmp_path)
    doc["peak_worker_rss_bytes"] = True
    put(path, doc)
    with pytest.raises(ValueError, match="peak-memory"):
        receipts.load_measurement_receipt(path)


def test_local_immutable_registry_creates_nothing_until_explicit_import_and_reloads(
    tmp_path,
):
    sources = tmp_path / "sources"
    path, _ = completion(sources)
    before = originals(sources)
    registry = receipts.MeasurementRegistry(tmp_path / "local_UI_registry")
    assert registry.samples() == [] and not registry.root.exists()
    imported = registry.import_receipt(path)
    stored = Path(imported["registry_path"]).read_bytes()
    assert registry.import_receipt(path)["source"] == imported["source"]
    assert Path(imported["registry_path"]).read_bytes() == stored
    assert originals(sources) == before
    loaded = receipts.MeasurementRegistry(registry.root).merge_machines([])
    assert loaded[0]["machine"] == "Second" and loaded[0]["inventory"] == {}
    assert loaded[0]["history"][0]["source"] == imported["source"]
    assert len(list(registry.root.glob("*.json"))) == 1


def test_bare_and_summary_import_deduplicate_same_original_sample_keep_explicit_subtimes(
    tmp_path,
):
    path, summary = trial(tmp_path / "sources")
    registry = receipts.MeasurementRegistry(tmp_path / "registry")
    registry.import_receipt(path)
    registry.import_receipt(summary["whole_run_completion"]["path"])
    rows = [dict(machine="Second", inventory=dict(cpu=dict(logical_cores=4)), history=[])]
    before = deepcopy(rows)
    merged = registry.merge_machines(rows)
    assert rows == before and len(merged[0]["history"]) == 1
    assert merged[0]["history"][0]["compute_and_input_seconds"] == 70


def test_registry_refresh_reverifies_actual_source_sha_and_refuses_changed_receipt(
    tmp_path,
):
    path, _ = completion(tmp_path / "sources")
    registry = receipts.MeasurementRegistry(tmp_path / "registry")
    registry.import_receipt(path)
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="SHA/path/size changed"):
        registry.merge_machines([])


def test_explicit_qt_receipt_import_visible_and_reloaded_without_hardware_or_jobs(
    tmp_path, qtbot
):
    path, _ = historical(tmp_path / "sources")
    registry = receipts.MeasurementRegistry(tmp_path / "registry")

    def forbidden():
        pytest.fail("historical metadata import must not inspect hardware")

    panel = ui.ResourceAdvisorPanel(
        measurement_registry=registry, inventory_provider=forbidden
    )
    qtbot.addWidget(panel)
    choices = []
    imported = []
    panel.workers_selected.connect(choices.append)
    panel.measurement_imported.connect(imported.append)
    panel.set_context(job())
    assert panel.machine_combo.count() == 0 and not registry.root.exists()
    panel.import_measurement_receipt(path)
    assert not panel.can_close()
    qtbot.waitUntil(panel.can_close, timeout=5000)
    assert len(imported) == 1 and choices == []
    cells = table_text(panel.history_table, 0)
    assert cells[0:5] == ["Main", "cpu", "float64", "2", "2"]
    assert (
        cells[5] == "200.00 s total / 100.00 s per frame"
        and cells[6:10] == [ui.UNKNOWN] * 4
    )
    assert (
        panel._advice["workers"] is None and "SHA256" in panel.measurement_label.text()
    )
    second = ui.ResourceAdvisorPanel(
        measurement_registry=receipts.MeasurementRegistry(registry.root),
        inventory_provider=forbidden,
    )
    qtbot.addWidget(second)
    second.set_context(job())
    assert second.history_table.rowCount() == 1
    assert table_text(second.history_table, 0) == cells


def test_filechooser_cancel_imports_nothing(tmp_path, qtbot, monkeypatch):
    registry = receipts.MeasurementRegistry(tmp_path / "registry")
    panel = ui.ResourceAdvisorPanel(measurement_registry=registry)
    qtbot.addWidget(panel)
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    panel.import_measurement_button.click()
    assert panel.can_close() and not registry.root.exists()


def test_import_failure_literal_and_does_not_create_registry(tmp_path, qtbot):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{broken")
    registry = receipts.MeasurementRegistry(tmp_path / "registry")
    panel = ui.ResourceAdvisorPanel(measurement_registry=registry)
    qtbot.addWidget(panel)
    errors = []
    panel.measurement_import_failed.connect(errors.append)
    panel.import_measurement_receipt(bad)
    qtbot.waitUntil(panel.can_close, timeout=5000)
    assert (
        "JSONDecodeError" in errors[0] and panel.measurement_label.text() == errors[0]
    )
    assert not registry.root.exists()
