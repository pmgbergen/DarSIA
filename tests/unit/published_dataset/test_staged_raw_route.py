"""Opt-in genuine RAW -> original P1 -> unchanged saved-calibration N1 proof.

Nothing here substitutes a sensor, renderer, inference, statistics or mass
function. The ordinary test suite skips this module's fixture when no genuine
fixture is selected; a skip is explicitly not scientific qualification.

Scheduled QA supplies DARSIA_STAGED_RAW_FIXTURE=/absolute/fixture.json and
DARSIA_STAGED_RAW_FIXTURE_SHA256=<independently pinned SHA-256>. Its immutable
JSON has format darsia-genuine-staged-raw-fixture-v1, run, series_id, roots
(input alias -> absolute existing directory), aliases (original DataReader
logical/physical byte-identical descriptors), phase1_job and phase2_job (full
job.json descriptors), phase1_software and phase2_software (full descriptors of
existing export-software JSON inventories), and environment (full descriptor
of an original content-hashed distribution fingerprint). The jobs and their
complete original publication trees must be readable at their bound paths;
there is no path guessing or legacy-origin backfill. Fixture payloads, source
geometry and frozen models must be the genuine original bytes.

This finite acceptance fixture requires the original selection to contain one
to three float32 CPU frames, with one thread, at the exact current clean code
commit for both stages. It never truncates or reselects a production run. QA
must arrange its reserved CPU affinity before opting in. All newly computed
products go under pytest's fresh temporary results directory. No queue, job
producer, source edit, model fitting or source publication is performed.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from types import SimpleNamespace
import uuid

import pytest


FORMAT = "darsia-genuine-staged-raw-fixture-v1"


@dataclass
class GenuineFixture:
    manifest: dict
    reader: object
    phase1: object
    phase2: object
    phase1_task: dict
    phase2_task: dict
    phase2_config: dict
    input_stores: object
    phase2_rows: tuple
    checkout: Path


def _document(bound):
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    return DataReader().json(bound)


def _software(path, job):
    from darsia.fluidflower.published_dataset.export_software import software_members
    request = SimpleNamespace(record="R", software_mode="references", software_inventory=Path(path),
                              software_folder=None)
    return software_members(request, SimpleNamespace(job=job), job["commit"])


def _stores(series_id, roots):
    from darsia.fluidflower.published_dataset.replay import store_map_text
    from darsia.fluidflower.store_map import parse_store_map
    return parse_store_map(store_map_text(series_id, roots).encode("utf-8"))


@pytest.fixture(scope="module")
def genuine_fixture():
    selected = os.environ.get("DARSIA_STAGED_RAW_FIXTURE")
    if not selected:
        pytest.skip("No genuine RAW/P1/N1 fixture selected; coupled scientific qualification is NOT claimed")
    expected = os.environ.get("DARSIA_STAGED_RAW_FIXTURE_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        pytest.fail("The genuine fixture requires its independently pinned manifest SHA-256")
    path = Path(selected)
    if not path.is_absolute() or not path.is_file():
        pytest.fail("The explicitly selected genuine fixture manifest is unavailable")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        pytest.fail("The genuine RAW fixture manifest differs from its independently pinned bytes")
    from darsia.fluidflower.pipeline.job_inputs import _json_bytes
    manifest = _json_bytes(raw)
    assert manifest["format"] == FORMAT
    required = {"format", "run", "series_id", "roots", "aliases", "phase1_job", "phase2_job",
                "phase1_software", "phase2_software", "environment"}
    assert set(manifest) == required, "Explicit finite original fixture fields are required"
    assert isinstance(manifest["roots"], dict) and "series" in manifest["roots"]
    assert all(Path(value).is_absolute() and Path(value).is_dir() for value in manifest["roots"].values())

    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.pipeline.reproduction_lineage import verify_phase1_origin
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog, verify_inputs
    from darsia.fluidflower.pipeline.staged_data import validate_phase2_run
    from darsia.fluidflower.published_dataset.dataset_check import installed_commit
    from darsia.fluidflower.published_dataset.environment import compare_fingerprints, distribution_fingerprint
    from darsia.fluidflower.published_dataset.job_reader import FinishedJob
    from darsia.fluidflower.series_paths import darsia_source_checkout

    job1, job2 = _document(manifest["phase1_job"]), _document(manifest["phase2_job"])
    assert all(Path(manifest[role]["path"]).name == "job.json" for role in ("phase1_job", "phase2_job"))
    phase1, phase2 = (FinishedJob(Path(manifest[role]["path"]).parent)
                      for role in ("phase1_job", "phase2_job"))
    assert phase1.job == job1 and phase2.job == job2
    run = manifest["run"]
    task1, task2 = phase1.tasks[run], phase2.tasks[run]
    assert task1["pipeline"] == "native-phase-statistics" and task2["pipeline"] == "n1"
    assert task1["ordered_case_ids"] == task2["ordered_case_ids"]
    assert 1 <= len(task1["ordered_case_ids"]) <= 3, "Choose an original finite selection; never truncate it"
    for task in (task1, task2):
        assert task["precision"] == "float32" and task["threads"] == 1
        assert task.get("backend", "any") in {"cpu", "any"} and "resume_from" not in task
    installed = installed_commit()
    assert installed["clean"] is True
    assert installed["commit"] == task1["commit"] == task2["commit"] == job1["commit"] == job2["commit"]
    checkout = darsia_source_checkout()
    assert checkout is not None
    checkout = Path(checkout).resolve()
    # Exact original input roots may contain a saved-results alias. Output
    # admission uses a separate store map, never rebinds these input locations.
    stores = _stores(manifest["series_id"], {key: Path(value) for key, value in manifest["roots"].items()})
    rows1, rows2 = resolve_catalog(task1, stores, manifest["series_id"]), resolve_catalog(task2, stores, manifest["series_id"])
    verified1, verified2 = verify_inputs(task1, rows1), verify_inputs(task2, rows2)
    config1, config2 = verified1.metadata("run.configuration"), verified2.metadata("run.configuration")
    validate_phase2_run(config2)
    assert config1["ordered_case_ids"] == task1["ordered_case_ids"]
    assert all(frame["native_task"]["kind"] == "raw_rig" for frame in config1["frames"])
    for frame in config1["frames"]:
        actual = verified1.for_use("frame." + frame["case_id"] + ".native.acquisition")
        assert actual.reference["sha256"] == frame["source"]["source_sha256"]
    source = config2["phase1"]
    assert source.get("origin", {}).get("job") == manifest["phase1_job"], "No inferred P1 producer is accepted"
    lineage = verify_phase1_origin(source["origin"], configuration=source["configuration"],
        completion=source["completion"], expected_commit=source["code_commit"],
        ordered_case_ids=task2["ordered_case_ids"])
    assert lineage["task"] == task1 and lineage["job"] == job1
    reader = DataReader(manifest["aliases"])
    assert reader.json(source["configuration"]) == config1
    for role, job in (("phase1_software", job1), ("phase2_software", job2)):
        path = DataReader().path(manifest[role])
        _software(path, job)
    fingerprint = _document(manifest["environment"])
    assert fingerprint["file_digests"] == "content", "Recorded package versions alone cannot qualify installed bits"
    actual = distribution_fingerprint(hash_files=True)
    assert compare_fingerprints(fingerprint, actual) == []
    assert fingerprint["sha256"] == actual["sha256"]
    return GenuineFixture(manifest, reader, phase1, phase2, task1, task2, config2, stores, rows2, checkout)


def _binding(fixture, original, results, stage, rows=None):
    from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
    from darsia.fluidflower.pipeline.run_io import RunIOBinding
    from darsia.fluidflower.pipeline.store_runtime import local_attempt_output, resolve_catalog, verify_inputs
    from darsia.fluidflower.published_dataset.replay import replay_task

    task = replay_task(original, attempt_id=uuid.uuid4().hex, job_id="genuine-staged-reproduction-" + stage)
    inputs = resolve_catalog(task, fixture.input_stores, fixture.manifest["series_id"]) if rows is None else rows
    verified = verify_inputs(task, inputs)
    roots = {key: Path(value) for key, value in fixture.manifest["roots"].items()}
    roots["results"] = results
    outputs = _stores(fixture.manifest["series_id"], roots)
    output, evidence = local_attempt_output(outputs, fixture.manifest["series_id"],
                                            task["job_id"], task["run"], task["attempt_id"])
    local = LocalOutputBinding(outputs.machine_id, results, output, evidence)
    binding = RunIOBinding.replay(task, verified, local, repository=fixture.checkout, admitted_commit=task["commit"])
    return task, binding


def test_genuine_raw_original_p1_couples_to_original_n1_with_bit_identical_outputs(genuine_fixture, tmp_path):
    import numpy as np
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.pipeline.phase_statistics import run_phase1
    from darsia.fluidflower.n1_pipeline import run_n1
    from darsia.fluidflower.published_dataset.staged_replay import verify_coupling

    fixture = genuine_fixture
    results = tmp_path / "fresh-results"
    results.mkdir()
    task1, binding1 = _binding(fixture, fixture.phase1_task, results, "phase1")
    events1 = []
    with binding1.activate():
        fresh = run_phase1(dict(task1, environment_lock=fixture.phase1.job["environment_lock"]),
                           binding1.output, "cpu", "float32", events1.append)
        replay_reader = DataReader(unc_only=True)
        rows2, receipt = verify_coupling(fixture.phase2_config,
            fixture.phase2_config["phase1"]["completion"], fresh["phase1_completion"],
            fixture.reader, replay_reader, fixture.phase2_rows)
    assert fresh["complete"] is True and fresh["mass_evaluation_calls"] == 0
    assert [row["case_id"] for row in events1 if row.get("state") == "phase1_frame_completed"] == task1["ordered_case_ids"]
    assert receipt["scientific_qualification"] is False and receipt["production"] is False
    assert receipt["substitutions"] and receipt["statistics"]
    replaced = {use: row for row in rows2 for use in row.uses}
    for case in task1["ordered_case_ids"]:
        for role in ("current.centre.log", "labels", "classes", "area"):
            assert replaced["frame." + case + ".mass." + role].path.resolve().is_relative_to(binding1.output.resolve())
    task2, binding2 = _binding(fixture, fixture.phase2_task, results, "phase2", rows2)
    events2 = []
    with binding2.activate():
        completed = run_n1(dict(task2, environment_lock=fixture.phase2.job["environment_lock"]),
                           binding2.output, "cpu", "float32", events2.append)
    assert completed["complete"] is True
    assert [row["case_id"] for row in events2 if row.get("state") == "frame_completed"] == task2["ordered_case_ids"]
    for case in task2["ordered_case_ids"]:
        product = fixture.phase2.product(task2["run"], "frames/" + case + "/result.json")
        expected = fixture.reader.json(dict(path=product.local_path, sha256=product.sha256, size_bytes=product.size_bytes))
        actual = json.loads((binding2.output / "frames" / case / "result.json").read_bytes())
        for field in ("total_mass_g", "aqueous_mass_g", "gas_mass_g"):
            assert type(expected[field]) is type(actual[field])
            assert struct.pack("!d", expected[field]) == struct.pack("!d", actual[field])
        layer = fixture.phase2.layer(task2["run"], case)
        assert layer is not None, "Genuine original fixture must include its stored numerical layer"
        original_layer = fixture.reader.path(dict(path=layer.local_path, sha256=layer.sha256, size_bytes=layer.size_bytes))
        with np.load(original_layer, allow_pickle=False) as old, np.load(
                binding2.output / "frames" / case / "storage_stored_frame.npz", allow_pickle=False) as new:
            assert set(old.files) == set(new.files)
            for name in old.files:
                a, b = old[name], new[name]
                assert a.dtype == b.dtype and a.shape == b.shape
                assert a.tobytes(order="C") == b.tobytes(order="C"), name
    # The fixture's own records remain historical; this is a test result only.
    (tmp_path / "coupling-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def test_genuine_origin_rejects_another_selection_before_science(genuine_fixture):
    from darsia.fluidflower.pipeline.reproduction_lineage import verify_phase1_origin
    source = genuine_fixture.phase2_config["phase1"]
    with pytest.raises(ValueError, match="selection"):
        verify_phase1_origin(source["origin"], configuration=source["configuration"], completion=source["completion"],
            expected_commit=source["code_commit"], ordered_case_ids=["not-the-original-selected-case"])


def test_genuine_origin_tampered_binding_is_not_accepted(genuine_fixture):
    from darsia.fluidflower.pipeline.reproduction_lineage import verify_phase1_origin
    source = genuine_fixture.phase2_config["phase1"]
    origin = deepcopy(source["origin"])
    origin["job"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes changed"):
        verify_phase1_origin(origin, configuration=source["configuration"], completion=source["completion"],
            expected_commit=source["code_commit"])


def test_genuine_route_has_no_missing_origin_fallback(genuine_fixture):
    from darsia.fluidflower.pipeline.reproduction_lineage import verify_phase1_origin
    source = genuine_fixture.phase2_config["phase1"]
    with pytest.raises(ValueError, match="Explicit original phase-one reproduction origin"):
        verify_phase1_origin(None, configuration=source["configuration"], completion=source["completion"],
            expected_commit=source["code_commit"])


def test_actual_raw_consumer_rejects_replaced_payload_bytes_before_science(genuine_fixture, tmp_path):
    from dataclasses import replace
    from darsia.fluidflower.pipeline.store_runtime import InputHashMismatch, resolve_catalog, verify_inputs
    fixture = genuine_fixture
    rows = resolve_catalog(fixture.phase1_task, fixture.input_stores, fixture.manifest["series_id"])
    use = "frame." + fixture.phase1_task["ordered_case_ids"][0] + ".native.acquisition"
    row, = [item for item in rows if use in item.uses]
    corrupt = tmp_path / "corrupted-input.ARW"
    with corrupt.open("wb") as stream:
        stream.truncate(row.reference["bytes"])
    with pytest.raises(InputHashMismatch, match="SHA/bytes"):
        verify_inputs(fixture.phase1_task, (replace(row, path=corrupt),))


@pytest.mark.parametrize("stage", ["phase1", "phase2"])
def test_genuine_stage_software_must_match_original_job(genuine_fixture, tmp_path, stage):
    fixture = genuine_fixture
    inventory = _document(fixture.manifest[stage + "_software"])
    inventory["code"]["commit"] = "0" * 40
    changed = tmp_path / "wrong-software.json"
    changed.write_text(json.dumps(inventory), encoding="utf-8")
    job = getattr(fixture, stage).job
    with pytest.raises(ValueError, match="Software code commit differs"):
        _software(changed, job)
