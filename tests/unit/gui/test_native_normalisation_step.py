from dataclasses import replace
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tomllib

import pytest

from darsia.fluidflower import normalisation_step as ui


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "published"
    expected = {}
    runs = ("zz1", "zz2", "zz3", "zz4")
    rank = ["zz2", "zz1", "zz3", "zz4"]
    documents = {
        "v5_20260929/RESULTS.md": "Reference run ZZ1.\n",
        "v5_20260929/HANDOFF_20260929.md": "Saved report, 2020-04-05\n",
        "v5_20260929/protocol_rank.py": '"""Rank SERIES_X runs by saved chemistry"""\nraise RuntimeError("NEVER EXECUTE")\n',
        "v5_20260929/protocols_parsed.json": {
            run: {
                "run": run,
                "file": f"{run}/protocol.xlsx",
                "notes": "Saved protocol text",
            }
            for run in runs
        },
        "v5_20260929/proto_parse.py": "raise RuntimeError('NEVER EXECUTE')\n",
        "v5_20260929/dye_curve_distance_per_run.json": {
            run: {"median_abs": 0.01} for run in runs
        },
        "v5_20260929/protocol_chemistry_ranking.json": {
            "rank": rank,
            "rows": {
                run: {
                    "score": index,
                    "flags": [],
                    "why": ["pH z 0.1"],
                    "BTB_gL": 0.4,
                    "NaOH_gL": 0.04,
                    "ph": [8, 9, 8, 9],
                }
                for index, run in enumerate(rank)
            },
        },
        "two_step_20260928/v4_params.json": {
            "chart_stat": dict(zip(runs, [0, 0.05, 0.004, 0.006])),
            "state": dict(zip(runs, ["same", "new", "same", "same"])),
        },
        "two_step_20260928/v4.py": "raise RuntimeError('NEVER EXECUTE')\n",
        "v5_20260929/v5.py": "raise RuntimeError('NEVER EXECUTE')\n",
        "v5_20260929/scores_v5_RESULT_cdye.json": {
            "rows": [
                {"run": run, "v5": {"all": 0.012, "G": 0.011, "chroma": 0.009}}
                for run in runs
            ]
        },
    }
    for relative, document in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            document.encode()
            if isinstance(document, str)
            else json.dumps(document).encode()
        )
        path.write_bytes(raw)
        expected[relative] = hashlib.sha256(raw).hexdigest()
    provider = ui.PublishedV5Provider(expected)
    return root, runs, provider


@pytest.fixture
def series(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    original = b'\xef\xbb\xbfseries_id = "SERIES_X"\r\nname = "Unchanged"\r\n[runs]\r\nids = ["zz1", "zz2", "zz3", "zz4"]\r\n[method]\r\nactive = "colour_path"\r\n[method.phase_model]\r\nmodel_version = "inactive-model"\r\nunknown = 42\r\n[custom]\r\nvalue = [1, 2, 3]\r\nspecial = nan\r\n[phase_normalisation]\r\nunknown = "keep"\r\n'
    (root / "series.toml").write_bytes(original)
    return root, original


def test_saved_values_and_raw_vs_majority_rank_no_execution(evidence):
    root, runs, provider = evidence
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    snapshot = provider(root, "SERIES_X", runs)
    assert snapshot.report_reference == "zz1"
    assert snapshot.suggested_reference == "zz1"
    assert snapshot.snapshot_date == "2020-04-05"
    assert [row["run"] for row in snapshot.rows] == ["zz2", "zz1", "zz3", "zz4"]
    assert not snapshot.rows[0]["eligible"]
    assert "majority" in snapshot.rows[0]["exclusion"]
    assert snapshot.rows[1]["eligible"]
    assert snapshot.rows[0]["S"] == 0.05
    assert snapshot.rows[0]["residual_all"] == 0.012
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_changed_protocol_snapshot_refuses_unchanged_ranking(evidence):
    root, runs, provider = evidence
    path = root / "v5_20260929/protocols_parsed.json"
    data = json.loads(path.read_bytes())
    data["zz1"]["notes"] = "Externally changed recipe/protocol"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="origin differs"):
        provider(root, "SERIES_X", runs)


def test_missing_source_and_wrong_series_are_missing(evidence):
    root, runs, provider = evidence
    with pytest.raises(ValueError, match="another experimental series"):
        provider(root, "OTHER", runs)
    (root / "v5_20260929/v5.py").unlink()
    with pytest.raises(FileNotFoundError):
        provider(root, "SERIES_X", runs)


def test_missing_run_no_majority_and_saved_state_disagreement(evidence):
    root, runs, provider = evidence
    with pytest.raises(ValueError, match="every selected run"):
        provider(root, "SERIES_X", runs + ("zz5",))
    with pytest.raises(ValueError, match="no majority"):
        provider(root, "SERIES_X", ("zz1", "zz2"))
    path = root / "two_step_20260928/v4_params.json"
    document = json.loads(path.read_bytes())
    document["state"]["zz2"] = "same"
    raw = json.dumps(document).encode()
    path.write_bytes(raw)
    changed = dict(provider.expected_sources)
    changed["two_step_20260928/v4_params.json"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="S and light class disagree"):
        ui.PublishedV5Provider(changed)(root, "SERIES_X", runs)


def test_saved_chemistry_flags_and_existing_recipe_reasons_exclude(evidence):
    root, runs, provider = evidence
    path = root / "v5_20260929/protocol_chemistry_ranking.json"
    document = json.loads(path.read_bytes())
    document["rows"]["zz1"]["flags"] = ["buffer/acid"]
    document["rows"]["zz3"]["why"].append("BTB 120% of standard")
    raw = json.dumps(document).encode()
    path.write_bytes(raw)
    changed = dict(provider.expected_sources)
    changed["v5_20260929/protocol_chemistry_ranking.json"] = hashlib.sha256(
        raw
    ).hexdigest()
    snapshot = ui.PublishedV5Provider(changed)(root, "SERIES_X", runs)
    assert snapshot.suggested_reference == "zz4"
    assert not next(r for r in snapshot.rows if r["run"] == "zz1")["eligible"]


def test_explicit_save_preserves_inactive_unknown_unrelated_and_encoding(
    evidence, series
):
    source, runs, provider = evidence
    root, original = series
    snapshot = provider(source, "SERIES_X", runs)
    saved = ui.save_reference(
        root,
        "SERIES_X",
        snapshot,
        "zz1",
        False,
        expected_bytes=original,
        provider=provider,
    )
    parsed = tomllib.loads(saved.decode("utf-8-sig"))
    old = tomllib.loads(original.decode("utf-8-sig"))
    assert saved.startswith(b"\xef\xbb\xbf") and b"\r\n" in saved
    assert parsed["method"] == old["method"]
    assert ui._same_values(parsed["custom"], old["custom"])
    assert parsed["phase_normalisation"]["unknown"] == "keep"
    assert ui.saved_ready(parsed["phase_normalisation"], snapshot)
    assert len(parsed["phase_normalisation"]["ranking"]) == 4
    assert parsed["phase_normalisation"]["sources"] == list(snapshot.sources)


def test_override_saved_but_cannot_claim_old_corrections_use_it(evidence, series):
    source, runs, provider = evidence
    root, original = series
    snapshot = provider(source, "SERIES_X", runs)
    raw = ui.save_reference(
        root,
        "SERIES_X",
        snapshot,
        "zz4",
        True,
        expected_bytes=original,
        provider=provider,
    )
    record = tomllib.loads(raw.decode("utf-8-sig"))["phase_normalisation"]
    assert record["selection"] == "override" and record["reference_run"] == "zz4"
    assert not ui.saved_ready(record, snapshot)
    with pytest.raises(ValueError, match="automatic choice"):
        ui.record_for(snapshot, "zz4", False)


def test_stale_manifest_rejected_without_overwrite(evidence, series):
    source, runs, provider = evidence
    root, original = series
    snapshot = provider(source, "SERIES_X", runs)
    path = root / "series.toml"
    changed = original.replace(b"Unchanged", b"New external edit")
    path.write_bytes(changed)
    with pytest.raises(ui.ManifestChangedError, match="since this step"):
        ui.save_reference(
            root,
            "SERIES_X",
            snapshot,
            "zz1",
            False,
            expected_bytes=original,
            provider=provider,
        )
    assert path.read_bytes() == changed


def test_race_during_save_preserves_external_edit_and_no_temporaries(
    evidence, series, monkeypatch
):
    source, runs, provider = evidence
    root, original = series
    snapshot = provider(source, "SERIES_X", runs)
    path = root / "series.toml"
    changed = original.replace(b"Unchanged", b"Concurrent edit")
    same = ui._same_values

    def external_write_then_compare(first, second):
        path.write_bytes(changed)
        return same(first, second)

    monkeypatch.setattr(ui, "_same_values", external_write_then_compare)
    with pytest.raises(ui.ManifestChangedError, match="while saving"):
        ui.save_reference(
            root,
            "SERIES_X",
            snapshot,
            "zz1",
            False,
            expected_bytes=original,
            provider=provider,
        )
    assert path.read_bytes() == changed
    assert list(root.iterdir()) == [path]


def test_source_modified_after_preview_rejected_before_manifest_write(evidence, series):
    source, runs, provider = evidence
    root, original = series
    snapshot = provider(source, "SERIES_X", runs)
    (source / "v5_20260929/RESULTS.md").write_text("Changed report")
    with pytest.raises(ValueError, match="origin differs"):
        ui.save_reference(
            root,
            "SERIES_X",
            snapshot,
            "zz1",
            False,
            expected_bytes=original,
            provider=provider,
        )
    assert (root / "series.toml").read_bytes() == original


def test_saved_record_tamper_and_coverage_change_are_not_ready(evidence):
    source, runs, provider = evidence
    snapshot = provider(source, "SERIES_X", runs)
    record = ui.record_for(snapshot, "zz1", False)
    record["saved_utc"] = "2020-01-01T00:00:00Z"
    assert ui.saved_ready(record, snapshot)
    record["ranking"][0]["S"] = 0.003
    assert not ui.saved_ready(record, snapshot)
    fresh = ui.record_for(snapshot, "zz1", False)
    fresh["saved_utc"] = "2020-01-01T00:00:00Z"
    assert not ui.saved_ready(fresh, replace(snapshot, runs=("zz1", "zz3", "zz4")))


def test_widget_preview_never_writes_explicit_save_opens_gate(qtbot, evidence, series):
    source, _runs, provider = evidence
    root, original = series
    panel = ui.NormalisationStep(provider=provider)
    qtbot.addWidget(panel)
    signals = []
    panel.readiness_changed.connect(signals.append)
    panel.set_context(root, "SERIES_X")
    assert not panel.ready
    panel.set_source(source)
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert not panel.ready and panel.save_button.isEnabled()
    assert (root / "series.toml").read_bytes() == original
    assert panel.reference.currentData() == "zz1"
    assert panel.table.item(0, 1).text() == "ZZ2"
    panel.save_button.click()
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert panel.ready and signals[-1] is True
    assert "Done" in panel.status.text()
    panel.override.setChecked(True)
    panel.reference.setCurrentIndex(panel.reference.findData("zz4"))
    assert not panel.ready
    panel.save_button.click()
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert not panel.ready and "another reference" in panel.status.text()


def test_readonly_preview_context_switch_and_unknown_origin_close_gate(
    qtbot, evidence, series
):
    source, _runs, provider = evidence
    root, original = series
    panel = ui.NormalisationStep(provider=provider, allow_save=False)
    qtbot.addWidget(panel)
    panel.set_context(root, "SERIES_X")
    panel.set_source(source)
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert not panel.ready and not panel.save_button.isEnabled()
    panel._save()
    assert (root / "series.toml").read_bytes() == original
    panel.set_context(root, "FOREIGN")
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert panel.source_folder is None and panel.snapshot is None
    assert not panel.ready and "Missing" in panel.status.text()


def test_reloaded_valid_selection_requires_same_bound_report(qtbot, evidence, series):
    source, runs, provider = evidence
    root, original = series
    snapshot = provider(source, "SERIES_X", runs)
    ui.save_reference(
        root,
        "SERIES_X",
        snapshot,
        "zz1",
        False,
        expected_bytes=original,
        provider=provider,
    )
    panel = ui.NormalisationStep(provider=provider)
    qtbot.addWidget(panel)
    panel.set_context(root, "SERIES_X")
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert panel.ready
    (source / "v5_20260929/protocols_parsed.json").write_text("{}")
    panel.reload()
    qtbot.waitUntil(lambda: panel.can_close() and not panel.metadata_busy())
    assert not panel.ready and "origin differs" in panel.status.text()
