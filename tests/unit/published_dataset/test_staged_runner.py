"""Metadata/file-only guards for exact staged input transport and selection.

No test here starts a scientific worker, substitutes a numerical function, or
claims that copied inputs constitute a completed RAW reproduction.
"""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from darsia.fluidflower.published_dataset import staged_runner as runner
from darsia.fluidflower.published_dataset.reproduce import Settings
from darsia.fluidflower.published_dataset.staged_package import FORMAT
from .synthetic import sha, write


def _row(alias, relative, data, *, camera):
    path = alias + "/" + relative
    return dict(root_alias=alias, relative_path=relative,
                path=path if camera else "staged/roots/" + path,
                sha256=sha(data), size_bytes=len(data), camera=camera)


def _archive(dataset, run, entries):
    path = dataset / ("raw_" + run + ".zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return path


def _fixture(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    raw, metadata = b"same original acquisition fixture bytes", b"original selected configuration bytes"
    first = _row("camera", "r1/one.ARW", raw, camera=True)
    second = _row("camera", "r1/two.ARW", raw, camera=True)
    config = _row("series", "runs/r1.json", metadata, camera=False)
    section = dict(roots={"camera": "staged/roots/camera", "series": "staged/roots/series"},
        input_assets=[first, second, config], transport=[],
        camera_members={"r1": [first["path"], second["path"]]})
    write(dataset / config["path"], metadata)
    _archive(dataset, "r1", [(first["path"], raw), (second["path"], raw)])
    return SimpleNamespace(dataset=dataset, section=section, first=first, second=second,
                           config=config, raw=raw, metadata=metadata)


def test_materialization_keeps_distinct_original_locators_even_when_hashes_match(tmp_path):
    fixture = _fixture(tmp_path)
    roots = runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")
    assert fixture.first["sha256"] == fixture.second["sha256"]
    for row in fixture.section["input_assets"]:
        path = roots[row["root_alias"]] / row["relative_path"]
        assert path.is_file() and sha(path.read_bytes()) == row["sha256"]
    assert roots["camera"].joinpath(fixture.first["relative_path"]) != roots["camera"].joinpath(fixture.second["relative_path"])
    assert roots["results"].is_dir()
    assert not list(roots["results"].iterdir())


@pytest.mark.parametrize("change", ["missing_archive", "missing_member", "unknown_member", "tampered_bytes", "duplicate_member"])
def test_materialization_refuses_incomplete_or_changed_selected_raw_archives(tmp_path, change):
    fixture = _fixture(tmp_path)
    entries = [(fixture.first["path"], fixture.raw), (fixture.second["path"], fixture.raw)]
    if change == "missing_archive":
        source = fixture.dataset / "raw_r1.zip"
        source.rename(source.with_suffix(".unavailable"))
    elif change == "missing_member":
        _archive(fixture.dataset, "r1", entries[:1])
    elif change == "unknown_member":
        _archive(fixture.dataset, "r1", entries + [("camera/unbound.ARW", fixture.raw)])
    elif change == "tampered_bytes":
        _archive(fixture.dataset, "r1", [(fixture.first["path"], b"X" * len(fixture.raw)), entries[1]])
    else:
        with pytest.warns(UserWarning, match="Duplicate name"):
            _archive(fixture.dataset, "r1", entries + entries[:1])
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")


def test_materialization_checks_later_archive_bytes_even_when_target_already_matches(tmp_path):
    fixture = _fixture(tmp_path)
    fixture.section["camera_members"]["r2"] = [fixture.first["path"]]
    _archive(fixture.dataset, "r2", [(fixture.first["path"], b"X" * len(fixture.raw))])
    with pytest.raises(ValueError, match="bytes differ"):
        runner.materialize_inputs(fixture.dataset, fixture.section, ["r1", "r2"], tmp_path / "inputs")
    # The first valid source remains; the second source never overwrites it.
    assert (tmp_path / "inputs/camera" / fixture.first["relative_path"]).read_bytes() == fixture.raw


def test_identical_existing_input_is_retained_but_its_new_source_is_still_verified(tmp_path):
    target = tmp_path / "original.bin"
    data = b"bound original bytes"
    target.write_bytes(data)
    before = target.stat().st_mtime_ns
    bound = dict(sha256=sha(data), size_bytes=len(data))
    runner._copy_input(io.BytesIO(data), target, bound)
    assert target.stat().st_mtime_ns == before and target.read_bytes() == data
    with pytest.raises(ValueError, match="bytes differ"):
        runner._copy_input(io.BytesIO(b"X" * len(data)), target, bound)
    assert target.stat().st_mtime_ns == before and target.read_bytes() == data


def test_materialization_refuses_conflicting_bytes_at_one_original_input_location(tmp_path):
    fixture = _fixture(tmp_path)
    conflicting = deepcopy(fixture.config)
    conflicting.update(path="staged/conflicting.json", sha256=sha(b"different"), size_bytes=len(b"different"))
    write(fixture.dataset / conflicting["path"], b"different")
    fixture.section["input_assets"].append(conflicting)
    with pytest.raises(ValueError, match="Conflicting staged input destination"):
        runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")
    assert (tmp_path / "inputs/series" / fixture.config["relative_path"]).read_bytes() == fixture.metadata


def test_materialization_never_follows_an_input_root_outside_owned_transport(tmp_path):
    fixture = _fixture(tmp_path)
    fixture.section["roots"]["../outside"] = "staged/roots/outside"
    with pytest.raises(ValueError, match="relative path"):
        runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")
    assert not (tmp_path / "outside").exists()


def test_unselected_run_archive_is_not_needed_for_selected_exact_input_scope(tmp_path):
    fixture = _fixture(tmp_path)
    unselected = _row("camera", "r2/other.ARW", b"unselected", camera=True)
    fixture.section["input_assets"].append(unselected)
    fixture.section["camera_members"]["r2"] = [unselected["path"]]
    roots = runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")
    assert not (roots["camera"] / unselected["relative_path"]).exists()
    assert (roots["camera"] / fixture.first["relative_path"]).read_bytes() == fixture.raw


def test_historical_aliases_preserve_same_hash_locators_and_explicit_consumer_location(tmp_path):
    fixture = _fixture(tmp_path)
    roots = runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")
    for index, row in enumerate((fixture.first, fixture.second)):
        logical = dict(path="/original/camera/source" + str(index) + ".ARW", sha256=row["sha256"],
                       size_bytes=row["size_bytes"])
        fixture.section["transport"].append(dict(logical=logical,
            asset={key: row[key] for key in ("path", "sha256", "size_bytes")}))
    # Historical proof payloads may have a separate archive copy, while an
    # explicitly declared input consumer must resolve to its materialized row.
    alternate = dict(path="staged/expected/original.json", sha256=fixture.config["sha256"],
                     size_bytes=fixture.config["size_bytes"])
    write(fixture.dataset / alternate["path"], fixture.metadata)
    fixture.section["transport"].append(dict(
        logical=dict(alternate, path="/original/results/configuration.json"), asset=alternate,
        input_reference=dict(root_alias="series", relative_path=fixture.config["relative_path"],
                             sha256=fixture.config["sha256"], bytes=fixture.config["size_bytes"])))
    aliases = runner.historical_aliases(fixture.dataset, fixture.section, roots)
    assert aliases[0]["logical"]["sha256"] == aliases[1]["logical"]["sha256"]
    assert aliases[0]["physical"]["path"] != aliases[1]["physical"]["path"]
    assert Path(aliases[2]["physical"]["path"]) == (roots["series"] / fixture.config["relative_path"]).resolve()
    assert all(row["logical"]["sha256"] == row["physical"]["sha256"] for row in aliases)


def test_historical_aliases_refuse_changed_exact_consumer_binding(tmp_path):
    fixture = _fixture(tmp_path)
    roots = runner.materialize_inputs(fixture.dataset, fixture.section, ["r1"], tmp_path / "inputs")
    row = fixture.config
    fixture.section["transport"] = [dict(
        logical=dict(path="/original/config.json", sha256=row["sha256"], size_bytes=row["size_bytes"]),
        asset={key: row[key] for key in ("path", "sha256", "size_bytes")},
        input_reference=dict(root_alias="series", relative_path=row["relative_path"], sha256="f" * 64,
                             bytes=row["size_bytes"]))]
    with pytest.raises(ValueError, match="exact input consumer"):
        runner.historical_aliases(fixture.dataset, fixture.section, roots)


@pytest.mark.parametrize("change, message", [
    ("override", "original commit"), ("prefix", "complete sealed frame selection"),
    ("missing_runtime", "exact Python and clean checkout"),
])
def test_preflight_refuses_runtime_overrides_partial_selection_and_missing_runtime(tmp_path, change, message):
    selected = runner.StagedReplayRunner()
    settings = Settings(workers=1, threads=1, override_commit=change == "override",
                        stage_runtimes=None if change == "missing_runtime" else "explicit-runtimes.json")
    scope = SimpleNamespace(frames={"r1": 1} if change == "prefix" else {}, runs=["r1"])
    # No fabricated successful stage manifest is needed for these early guards.
    check = SimpleNamespace(root=tmp_path, manifest={})
    errors = selected.preflight(tmp_path / "results", check=check, scope=scope, settings=settings)
    assert len(errors) == 1 and message in errors[0]
    assert selected.evidence is None and selected.outputs == {}
    assert not (tmp_path / "results").exists()


def test_raw_independent_numerical_record_cannot_claim_full_staged_reproduction(tmp_path):
    selected = runner.StagedReplayRunner()
    check = SimpleNamespace(root=tmp_path, manifest=dict(identity=dict(record="S"),
        staged_reproduction=dict(format=FORMAT)))
    errors = selected.preflight(tmp_path / "results", check=check,
        scope=SimpleNamespace(frames={}, runs=["r1"]),
        settings=Settings(workers=1, threads=1))
    assert errors and "only to an R record" in errors[0]
    assert selected.evidence is None and not (tmp_path / "results").exists()


@pytest.mark.parametrize("pause_when", ["before_setup", "during_setup"])
def test_pause_is_preserved_before_any_runtime_probe_or_materialization(tmp_path, monkeypatch, pause_when):
    from darsia.fluidflower.published_dataset import staged_package

    selected = runner.StagedReplayRunner()
    settings = Settings(workers=1, threads=1, stage_runtimes="explicit-runtimes.json")
    results = tmp_path / "results"
    check = SimpleNamespace(root=tmp_path / "dataset", manifest={})
    if pause_when == "before_setup":
        selected.pause()
        assert selected.control is None and not results.exists()

    class SetupBoundary(Exception):
        pass

    def stop_at_first_metadata_read(dataset, manifest):
        # Stop before any runtime probe, copying, or stage launch. The real
        # control file and state are the only behavior exercised here.
        assert dataset == check.root and manifest is check.manifest
        assert selected.settings is settings
        assert selected.control == results / "staged-control.json"
        assert json.loads(selected.control.read_bytes())["action"] == (
            "pause" if pause_when == "before_setup" else "run")
        if pause_when == "during_setup":
            selected.pause()
        assert not selected._resume.is_set()
        assert json.loads(selected.control.read_bytes()) == {"action": "pause"}
        selected.resume(settings)
        assert selected._resume.is_set()
        assert json.loads(selected.control.read_bytes()) == {"action": "run"}
        raise SetupBoundary()

    monkeypatch.setattr(staged_package, "readback_staged", stop_at_first_metadata_read)
    with pytest.raises(SetupBoundary):
        selected.run(check, results, SimpleNamespace(runs=["r1"]), settings)
    assert selected.evidence is None and selected.outputs == {}
    assert list(results.iterdir()) == [selected.control]


def test_resume_before_worker_setup_retains_selected_settings_without_writing_files(tmp_path, monkeypatch):
    from darsia.fluidflower.published_dataset import staged_package

    selected = runner.StagedReplayRunner()
    initial = Settings(workers=1, threads=2, stage_runtimes="explicit-runtimes.json")
    resumed = replace(initial, threads=1)
    selected.pause()
    selected.resume(resumed)
    assert selected.control is None and selected._resume.is_set()

    class SetupBoundary(Exception):
        pass

    def stop_at_first_metadata_read(dataset, manifest):
        assert selected.settings is resumed
        assert json.loads(selected.control.read_bytes()) == {"action": "run"}
        raise SetupBoundary()

    monkeypatch.setattr(staged_package, "readback_staged", stop_at_first_metadata_read)
    with pytest.raises(SetupBoundary):
        selected.run(SimpleNamespace(root=tmp_path, manifest={}), tmp_path / "results",
                     SimpleNamespace(runs=["r1"]), initial)
    assert selected.evidence is None and selected.outputs == {}


@pytest.mark.parametrize("control_exists", [False, True])
@pytest.mark.parametrize("changes, message", [
    ({"workers": 0}, "at least one worker"),
    ({"workers": True}, "at least one worker"),
    ({"workers": 1.5}, "at least one worker"),
    ({"threads": 0}, "positive thread count"),
    ({"threads": True}, "positive thread count"),
    ({"threads": 1.5}, "positive thread count"),
    ({"backend": "metal"}, "CPU or CUDA"),
    ({"backend": []}, "CPU or CUDA"),
    ({"override_commit": True}, "original commit"),
    ({"override_commit": 0}, "original commit"),
    ({"remove_scratch": "false"}, "scratch files"),
    ({"stage_runtimes": None}, "exact Python and clean checkout"),
    ({"stage_runtimes": " "}, "exact Python and clean checkout"),
    ({"stage_runtimes": "another-runtimes.json"}, "Keep the admitted stage runtimes"),
])
def test_invalid_resume_preserves_paused_settings_control_and_event(
        tmp_path, monkeypatch, control_exists, changes, message):
    from darsia.fluidflower.published_dataset import replay

    selected = runner.StagedReplayRunner()
    initial = Settings(workers=1, threads=2, stage_runtimes="explicit-runtimes.json")
    selected.settings = initial
    if control_exists:
        selected.control = tmp_path / "staged-control.json"
    selected.pause()
    before = ({path.name: (path.read_bytes(), path.stat().st_mtime_ns)
               for path in tmp_path.iterdir()})

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid Resume must not publish Run, wake the runner or start a pool")

    monkeypatch.setattr(replay, "_write_json", forbidden)
    monkeypatch.setattr(selected._resume, "set", forbidden)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", forbidden)
    with pytest.raises(ValueError, match=message):
        selected.resume(replace(initial, **changes))
    assert selected.settings is initial
    assert not selected._resume.is_set()
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in tmp_path.iterdir()} == before
    assert selected.evidence is None and selected.outputs == {}


def test_invalid_resume_before_setup_keeps_settings_unset_and_pause_requested(tmp_path):
    selected = runner.StagedReplayRunner()
    selected.pause()
    with pytest.raises(ValueError, match="at least one worker"):
        selected.resume(Settings(workers=0, threads=1, stage_runtimes="explicit-runtimes.json"))
    assert selected.settings is None and selected.control is None
    assert not selected._resume.is_set()
    assert list(tmp_path.iterdir()) == []


def test_valid_resume_updates_execution_settings_and_control_without_replacing_runtimes(tmp_path, monkeypatch):
    selected = runner.StagedReplayRunner()
    initial = Settings(workers=1, threads=2, stage_runtimes="explicit-runtimes.json")
    selected.settings = initial
    selected.control = tmp_path / "staged-control.json"
    admitted = {"p1-r1": object(), "p2-r1": object()}
    selected.runtimes = admitted
    selected.pause()
    monkeypatch.setattr(runner, "ThreadPoolExecutor",
                        lambda *args, **kwargs: pytest.fail("Resume wakes the existing runner; it does not launch a pool"))
    resumed = replace(initial, workers=2, threads=3, backend="cuda", remove_scratch=False)
    selected.resume(resumed)
    assert selected.settings is resumed and selected.runtimes is admitted
    assert selected._resume.is_set()
    assert json.loads(selected.control.read_bytes()) == {"action": "run"}
    assert selected.evidence is None and selected.outputs == {}


@pytest.mark.parametrize("changes, message", [
    ({"workers": 0}, "at least one worker"),
    ({"threads": 0}, "positive thread count"),
    ({"backend": "metal"}, "CPU or CUDA"),
    ({"remove_scratch": "false"}, "scratch files"),
])
def test_preflight_rejects_invalid_execution_settings_before_runtime_reads(tmp_path, monkeypatch, changes, message):
    from darsia.fluidflower.published_dataset import staged_runtime

    selected = runner.StagedReplayRunner()
    settings = Settings(workers=1, threads=1, stage_runtimes="explicit-runtimes.json")
    monkeypatch.setattr(staged_runtime, "load_runtimes",
                        lambda *args, **kwargs: pytest.fail("Invalid settings must fail before runtime reads"))
    errors = selected.preflight(tmp_path / "results",
        check=SimpleNamespace(root=tmp_path / "dataset", manifest={}),
        scope=SimpleNamespace(runs=["r1"], frames={}), settings=replace(settings, **changes))
    assert len(errors) == 1 and message in errors[0]
    assert selected.settings is None and selected.control is None
    assert not (tmp_path / "results").exists()
