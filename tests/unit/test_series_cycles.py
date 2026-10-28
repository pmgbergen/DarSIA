"""Explicit cycle provenance through the ordinary series writer and reader."""

from dataclasses import replace
import tomllib

import pytest

from darsia.fluidflower.series_creation import (
    RunDraft, SeriesDraft, check_runs, create_series_folder, series_files,
)
from darsia.fluidflower.series_cycles import (
    FORMAT, CycleRun, cycle_runs_table, parse_cycle_runs,
)
from darsia.fluidflower.series_paths import config_files, series_manifest


# Explicit N59 conversion input, not an inferred runtime rule or real series data.
N59_B20_TESTONLY = (
    ("e0", "e0", 1, None),
    ("e1_c1", "e1", 1, None), ("e1_c2", "e1", 2, "e1_c1"),
    ("e1_c3", "e1", 3, "e1_c2"),
    ("e2_c1", "e2", 1, None), ("e2_c2", "e2", 2, "e2_c1"),
    ("e2_c3", "e2", 3, "e2_c2"), ("e2_c4", "e2", 4, "e2_c3"),
    ("g1_c1", "g1", 1, None), ("g1_c2", "g1", 2, "g1_c1"),
    ("g1_c3", "g1", 3, "g1_c2"), ("g1_c4", "g1", 4, "g1_c3"),
    ("g2_c1", "g2", 1, None), ("g2_c2", "g2", 2, "g2_c1"),
    ("g2_c3", "g2", 3, "g2_c2"), ("g2_c4", "g2", 4, "g2_c3"),
    ("g3_c1", "g3", 1, None), ("g3_c2", "g3", 2, "g3_c1"),
    ("g3_c3", "g3", 3, "g3_c2"), ("g3_c4", "g3", 4, "g3_c3"),
)


def _draft(tmp_path, cycles):
    runs = []
    for cycle in cycles:
        folder = tmp_path / "TESTONLY_photos" / cycle.run_id
        folder.mkdir(parents=True)
        (folder / "own_baseline.jpg").write_bytes(b"TESTONLY, never decoded")
        runs.append(RunDraft(cycle.run_id, [folder], baseline="own_baseline.jpg", cycle=cycle))
    return SeriesDraft("FF_RE", "TESTONLY cycle series", "medium", tmp_path / "DarSIA", runs)


def test_explicit_B20_roundtrip_keeps_ordinary_own_configurations(tmp_path, monkeypatch):
    cycles = tuple(CycleRun(*row) for row in N59_B20_TESTONLY)
    draft = _draft(tmp_path, cycles)
    assert check_runs(draft.runs) == []
    root = create_series_folder(draft)
    monkeypatch.setenv("DARSIA_FFRE_PROJECT_ROOT", str(root))
    manifest = series_manifest("ffre")
    assert manifest["runs"]["ids"] == [row[0] for row in N59_B20_TESTONLY]
    assert parse_cycle_runs(manifest) == cycles
    assert manifest["cycle_runs"]["format"] == FORMAT
    for run, cycle in zip(draft.runs, cycles):
        common, own = config_files("ffre", cycle.run_id)
        assert common == root / "config/common.toml"
        assert own == root / "config/runs" / (cycle.run_id + ".toml")
        value = tomllib.loads(own.read_text(encoding="utf-8"))
        assert value["data"]["baseline"] == run.baseline
        assert value["data"]["folders"] == [str(run.folders[0])]
        assert value["data"]["results"] == str(root.parent / "Results" / cycle.run_id)
        assert value["protocols"]["injection"] == str(root / "protocols" / cycle.run_id / "injection_protocol.csv")
        assert set(manifest["cycle_runs"]["runs"][cycle.run_id]) <= {"experiment", "number", "previous_run_id"}
        # This foundation neither chooses nor manufactures time zero, dose or DIC.
        assert "background_dic" not in value and "time_zero" not in value


def test_arbitrary_explicit_cycle_ids_are_not_FF_RE_runtime_policy(tmp_path):
    rows = (CycleRun("trial_a", "rig_trial", 1), CycleRun("trial_b", "rig_trial", 2, "trial_a"))
    draft = _draft(tmp_path, rows)
    draft.series_id = "CustomStudy"
    manifest = tomllib.loads(series_files(draft, created="TESTONLY")["series.toml"])
    assert parse_cycle_runs(manifest) == rows
    assert manifest["runs"]["ids"] == ["trial_a", "trial_b"]


def test_legacy_series_has_no_implicit_cycle_migration(tmp_path):
    draft = _draft(tmp_path, (CycleRun("ac52", "ac52", 1),))
    draft.series_id = "FF_AC"
    draft.runs[0].cycle = None
    files = series_files(draft, created="TESTONLY")
    manifest = tomllib.loads(files["series.toml"])
    assert "cycle_runs" not in manifest
    assert parse_cycle_runs(manifest) == ()
    assert manifest["runs"]["ids"] == ["ac52"]
    assert check_runs(draft.runs) == []


def test_cycle_creation_never_overwrites_or_migrates_existing_series(tmp_path):
    draft = _draft(tmp_path, (CycleRun("first", "trial", 1),))
    draft.darsia_folder.mkdir()
    original = b'schema_version = 1\nseries_id = "FF_RE"\n# OLD SIX RUNS\n'
    path = draft.darsia_folder / "series.toml"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="already holds a series"):
        create_series_folder(draft)
    assert path.read_bytes() == original
    assert not (draft.darsia_folder / "config").exists()


@pytest.mark.parametrize("number", [True, False, 0, -1, 1.0, "1"])
def test_cycle_ordinal_must_be_an_explicit_positive_integer(number):
    with pytest.raises(ValueError, match="positive integer"):
        CycleRun("first", "trial", number)


@pytest.mark.parametrize("run,experiment", [("../run", "trial"), ("Run", "trial"), ("run", ""), ("run", "../group")])
def test_cycle_ids_are_not_paths_or_inferred_names(run, experiment):
    with pytest.raises(ValueError, match="explicit lower-case IDs"):
        CycleRun(run, experiment, 1)


def test_cycle_previous_is_required_and_never_assumed():
    with pytest.raises(ValueError, match="first cycle"):
        CycleRun("first", "trial", 1, "another")
    with pytest.raises(ValueError, match="explicit previous"):
        CycleRun("second", "trial", 2)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reversed", "gap", "foreign_previous", "reopened_group"])
def test_cycle_table_refuses_incomplete_or_cross_experiment_links(kind):
    first = CycleRun("first", "trial", 1)
    second = CycleRun("second", "trial", 2, "first")
    rows, ids = [first, second], ["first", "second"]
    if kind == "missing": rows = [first]
    if kind == "duplicate": rows, ids = [first, first], ["first", "first"]
    if kind == "reversed": rows = [second, first]
    if kind == "gap": rows[1] = replace(second, number=3)
    if kind == "foreign_previous": rows[1] = replace(second, previous_run_id="unrelated")
    if kind == "reopened_group":
        rows = [first, CycleRun("other", "another", 1), second]
        ids = [row.run_id for row in rows]
    with pytest.raises(ValueError):
        cycle_runs_table(rows, ids)


def test_mixed_cycle_and_noncycle_drafts_are_not_silently_partial(tmp_path):
    draft = _draft(tmp_path, (CycleRun("first", "trial", 1), CycleRun("second", "trial", 2, "first")))
    draft.runs[1].cycle = None
    assert any("every run exactly" in text for text in check_runs(draft.runs))
    with pytest.raises(ValueError, match="every run exactly"):
        series_files(draft)


@pytest.mark.parametrize("kind", ["null_section", "format", "extra_run", "missing_run", "method_field", "wrong_previous"])
def test_manifest_reader_refuses_unknown_or_incomplete_cycle_metadata(tmp_path, monkeypatch, kind):
    draft = _draft(tmp_path, (CycleRun("first", "trial", 1), CycleRun("second", "trial", 2, "first")))
    manifest = tomllib.loads(series_files(draft, created="TESTONLY")["series.toml"])
    section = manifest["cycle_runs"]
    if kind == "null_section": manifest["cycle_runs"] = None
    if kind == "format": section["format"] = "UNREVIEWED"
    if kind == "extra_run": section["runs"]["extra"] = {"experiment": "extra", "number": 1}
    if kind == "missing_run": section["runs"].pop("second")
    if kind == "method_field": section["runs"]["first"]["background_dic"] = 0
    if kind == "wrong_previous": section["runs"]["second"]["previous_run_id"] = "another"
    with pytest.raises(ValueError):
        parse_cycle_runs(manifest)


def test_original_series_manifest_reader_checks_explicit_cycle_links(tmp_path, monkeypatch):
    draft = _draft(tmp_path, (CycleRun("first", "trial", 1), CycleRun("second", "trial", 2, "first")))
    root = create_series_folder(draft)
    path = root / "series.toml"
    text = path.read_text(encoding="utf-8").replace('previous_run_id = "first"', 'previous_run_id = "foreign"')
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("DARSIA_FFRE_PROJECT_ROOT", str(root))
    with pytest.raises(ValueError, match="same experiment"):
        series_manifest("ffre")
