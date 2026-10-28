"""Genuine tiny C fits and synthetic verified-P1 application metadata only."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.mass import n1_refit
from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.pipeline import common_calibration as common
from darsia.fluidflower.pipeline import refit_runner, run_io
from test_n1_refit import make_rows, make_row
from test_refit_runner import write_plan, descriptor


def write(path, value):
    """Write canonical synthetic JSON and bind its exact published bytes."""
    path.write_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return descriptor(path)


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    """Fit and publish one synthetic global calibration with the original runner."""
    root = tmp_path_factory.mktemp("common-synthetic-global-fit")
    plan, configuration = write_plan(root)
    for run, entries in plan["statistics"].items():
        for row in entries:
            row["case_id"] = run + "_f" + str(row["frame_index"])
    configuration = write(Path(configuration["path"]), plan)
    cases = [row["case_id"] for run in sorted(plan["statistics"]) for row in plan["statistics"][run]]
    task = dict(pipeline=refit_runner.PIPELINE, precision="float32", run="refit", commit="c" * 40,
        configuration=configuration, backend="cpu", attempt_id="a" * 32,
        ordered_case_ids=cases, job_id="synthetic-common-fit", threads=1)
    output = root / "attempt"
    binding = SimpleNamespace(configuration_descriptor=configuration, require_output=lambda _output: None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_io, "for_task", lambda value: binding if value is task else None)
        result = refit_runner.run_refit(task, output, "cpu", "float32")
    assert result["stop_required"] is False
    result = dict(result, task=task, attempt_id=task["attempt_id"], commit=task["commit"])
    completed = write(root / "completed.json", result)
    return SimpleNamespace(root=root, plan=plan, configuration=configuration, task=task,
        output=output, result=result, completed=completed,
        binding=dict(completion=completed, plan=configuration))


def application(tmp_path, run="held"):
    """Write one independent held-out application with its own P1 sources."""
    rows = make_rows([run])
    specs, records = [], []
    for row in rows:
        case = run + "_f" + str(row["frame_index"])
        path = tmp_path / (case + ".json.gz")
        path.write_bytes(gzip.compress(json.dumps(row).encode()))
        specs.append(dict(case_id=case, frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
                          mass_spec=dict(selected_target_h=row["selected_target_h"]),
                          source=dict(source_sha256=row["source_binding"]["source_sha256"])))
        records.append(dict(record=dict(case_id=case, statistics=descriptor(path))))
    config = dict(run=run, frames=specs)
    configured = write(tmp_path / "p1-config.json", config)
    completed = write(tmp_path / "p1-closed.json", dict(synthetic=True, run=run))
    return dict(configuration=config, records=records, configuration_descriptor=configured,
        completion_descriptor=completed, payload_bytes_verified=True, scientific_arrays_decoded=False)


def join(genuine, verified, anchor, **changes):
    """Verify the actual global calibration against the application source anchor."""
    arguments = dict(binding=genuine.binding, verified_phase1=verified,
        fitted_calibration=genuine.result["frozen_calibration"], fitted_anchor=anchor,
        expected_commit="c" * 40, reader=DataReader())
    arguments.update(changes)
    return common.verify_frozen_refit(**arguments)


def test_catalogue_statistics_decode_only_exact_named_bytes(tmp_path):
    """Catalogue statistics decode only exact named bytes."""
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    path = tmp_path / "statistics.json.gz"
    path.write_bytes(gzip.compress(b'{"run":"sample","affected_pore_volume_m3":1.0}'))
    bound = descriptor(path)
    catalogue = InputCatalog(dict(series=str(tmp_path)))
    assert catalogue.statistics_document(bound, "calibration", "sample.statistics")["run"] == "sample"
    path.write_bytes(gzip.compress(b'{"run":"sample","affected_pore_volume_m3":2.0}'))
    with pytest.raises(ValueError, match="SHA/size"):
        catalogue.statistics_document(bound, "calibration", "sample.statistics")


def test_heldout_application_uses_unchanged_own_anchor_and_fit_content_is_byte_identical(genuine, tmp_path):
    """Heldout application uses unchanged own anchor and fit content is byte identical."""
    verified = application(tmp_path)
    saved = [genuine.configuration, *genuine.result["files"].values()]
    before = {row["path"]: Path(row["path"]).read_bytes() for row in saved}
    plan_before = deepcopy(genuine.plan)
    reader = DataReader()
    proof = common.heldout_application_anchor(verified, genuine.plan, reader=reader)
    rules = refit_runner.validate_plan(genuine.plan)
    rows = [refit_runner.read_statistics(reader, row["record"]["statistics"]) for row in verified["records"]]
    expected = n1_refit.anchors_by_run(rows, rules)["held"]
    assert proof["anchor"] == {key: expected[key] for key in ("elapsed_h", "affected_pore_volume_m3")}
    assert proof["usage"] == "application" and proof["role"] == "held out"
    assert all(proof[key] is False for key in (
        "entered_fit_objective", "entered_training_rows", "changed_coefficients", "entered_group_value"))
    actual = join(genuine, verified, proof["anchor"])
    assert actual["application"] == proof and actual["actual_frozen_refit_verified"] is True
    assert genuine.plan == plan_before and "held" not in genuine.plan["statistics"]
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())
    # A genuine refit of the same original fitting rows is unchanged too. The
    # application rows are never an argument to that original numerical fit.
    fit_rows = refit_runner.load_rows(genuine.plan, DataReader())
    fitted_again = n1_refit.fit_candidate(fit_rows, rules)
    saved_fit = json.loads(before[genuine.result["files"]["fit.json"]["path"]])
    assert fitted_again == saved_fit
    assert "held" not in fitted_again["calibration_runs"]
    assert "held" not in fitted_again["anchors_by_run"]


@pytest.mark.parametrize("change", ["foreign_run", "excluded", "fit_role", "missing_role", "changed_own_stats", "wrong_anchor"])
def test_heldout_application_refuses_role_and_own_source_changes(genuine, tmp_path, change):
    """Heldout application refuses role and own source changes."""
    verified = application(tmp_path)
    plan = deepcopy(genuine.plan)
    if change == "foreign_run":
        verified["records"][0]["record"]["statistics"] = genuine.plan["statistics"]["r1"][0]["statistics"]
    elif change == "excluded":
        plan["runs"]["excluded"] = ["held"]
    elif change == "fit_role":
        verified["configuration"]["run"] = "r1"
    elif change == "missing_role":
        plan["runs"]["heldout"] = []
    elif change == "changed_own_stats":
        Path(verified["records"][0]["record"]["statistics"]["path"]).write_bytes(b"changed")
    else:
        proof = common.heldout_application_anchor(verified, plan, reader=DataReader())
        anchor = dict(proof["anchor"], affected_pore_volume_m3=proof["anchor"]["affected_pore_volume_m3"] * 2)
        with pytest.raises(ValueError, match="anchor"):
            join(genuine, verified, anchor)
        return
    with pytest.raises(ValueError):
        common.heldout_application_anchor(verified, plan, reader=DataReader())


@pytest.mark.parametrize("field", ["stop_required", "commit", "parameters", "task", "frozen_calibration"])
def test_frozen_refit_join_requires_actual_global_result_not_a_calibration_hash(genuine, tmp_path, field):
    """Frozen refit join requires actual global result not a calibration hash."""
    verified = application(tmp_path)
    anchor = common.heldout_application_anchor(verified, genuine.plan, reader=DataReader())["anchor"]
    result = deepcopy(genuine.result)
    if field == "stop_required":
        result[field] = True
    elif field == "commit":
        result[field] = "d" * 40
    elif field == "parameters":
        result[field]["gain"] *= 2
    elif field == "task":
        result[field]["ordered_case_ids"] = list(reversed(result[field]["ordered_case_ids"]))
    else:
        result[field] = genuine.result["files"]["fit.json"]
    altered = write(tmp_path / "changed-completed.json", result)
    with pytest.raises(ValueError):
        join(genuine, verified, anchor, binding=dict(genuine.binding, completion=altered))


def test_closed_refit_product_extractor_uses_actual_fit_and_frozen_files(genuine):
    """Closed refit product extractor uses actual fit and frozen files."""
    claim = SimpleNamespace(task=genuine.task, token=genuine.task["attempt_id"], output=genuine.output)
    rows, metadata, accessible = common.extract_refit_products(genuine.result, claim, "Synthetic")
    assert {row["product_id"] for row in rows} == {"fit.json", "frozen_calibration.json"}
    assert set(metadata) == set(accessible) == {"fit.json", "frozen_calibration.json"}
    for row in rows:
        raw = Path(accessible[row["product_id"]]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"] and len(raw) == row["bytes"]


@pytest.mark.parametrize("change", ["extra", "missing", "foreign", "open", "wrong_attempt"])
def test_refit_extractor_refuses_added_missing_or_foreign_outputs(genuine, change):
    """Refit extractor refuses added missing or foreign outputs."""
    result = deepcopy(genuine.result)
    claim = SimpleNamespace(task=genuine.task, token=genuine.task["attempt_id"], output=genuine.output)
    if change == "extra":
        result["files"]["other.json"] = result["files"]["fit.json"]
    elif change == "missing":
        result["files"].pop("fit.json")
    elif change == "foreign":
        result["files"]["fit.json"]["path"] = str(genuine.root / "foreign.json")
    elif change == "open":
        result["worker_references_closed"] = False
    else:
        result["attempt_id"] = "b" * 32
    with pytest.raises(ValueError):
        common.extract_refit_products(result, claim, "Synthetic")


def test_positive_sealer_binds_genuine_c_and_preserves_original_template_provenance(genuine, tmp_path, monkeypatch):
    """Positive sealer binds genuine c and preserves original template provenance."""
    from tests.unit import test_staged_data as fixture
    from darsia.fluidflower.pipeline import staged_data
    data = fixture.phase1_data.__wrapped__()
    data["run"] = "held"
    spec = data["frames"][0]
    spec["source"]["run"] = "held"
    spec["elapsed_h"] = spec["source"]["elapsed_h"] = spec["mass_spec"]["selected_target_h"] = 2.25
    row = make_row("held", spec["frame_index"], 2.25)
    row["source_binding"] = deepcopy(spec["source"]["source_binding"])
    original_writer = fixture._write
    def actual_statistics(path, value=None, raw=None):
        if Path(path).name == "native_statistics.json.gz":
            raw = gzip.compress(json.dumps(row).encode())
        if isinstance(value, dict) and value.get("format") == staged_data.COMPLETION_FORMAT:
            value = dict(value, code_commit="c" * 40)
        return original_writer(path, value=value, raw=raw)
    monkeypatch.setattr(fixture, "_write", actual_statistics)
    closure = fixture._closure(tmp_path, data)
    reader = DataReader()
    verified = staged_data.verify_phase1_completion(closure["completion"], closure["configuration"],
        expected_commit="c" * 40, reader=reader)
    anchor = common.heldout_application_anchor(verified, genuine.plan, reader=reader)["anchor"]
    target = tmp_path / "sealed-with-C.json"
    sealed = staged_data.seal_phase2_configuration(target, closure["completion"], closure["configuration"],
        expected_phase1_commit="c" * 40, fitted_calibration=genuine.result["frozen_calibration"],
        fitted_anchor=anchor, reader=reader, frozen_refit=genuine.binding,
        storage_profile=dict(profile="lean", colour_intermediates="keep"))
    config = reader.json(sealed["configuration"])
    proof = config["provenance"].pop("common_frozen_refit")
    assert proof["application"]["role"] == "held out" and proof["run"] == "held"
    assert config["provenance"] == data["statistics"].get("provenance", {})
    assert proof["phase1_completion"] == closure["completion"]
    config["provenance"]["common_frozen_refit"] = proof
    class Catalogue:
        def __init__(self): self.uses = {}
        def doc(self, pin, role, use):
            self.uses[use] = pin
            return reader.json(pin)
        def add(self, pin, role, use):
            reader.path(pin)
            self.uses[use] = pin
        def statistics_document(self, pin, role, use):
            self.add(pin, role, use)
            return refit_runner.read_statistics(reader, pin)
    catalog = Catalogue()
    assert common.catalogue_refit(catalog, config) == proof
    assert "mass.common_refit.application.case-one" in catalog.uses
    assert len([key for key in catalog.uses if ".fit." in key]) == len(genuine.task["ordered_case_ids"])
    forged = deepcopy(config)
    forged["anchor"]["affected_pore_volume_m3"] *= 1.1
    altered = forged["provenance"]["common_frozen_refit"]
    altered["anchor"] = deepcopy(forged["anchor"])
    altered["application"]["anchor"] = deepcopy(forged["anchor"])
    altered["application"]["selected_anchor"]["affected_pore_volume_m3"] = forged["anchor"]["affected_pore_volume_m3"]
    with pytest.raises(ValueError, match="original own-run selection"):
        common.catalogue_refit(Catalogue(), forged)


@pytest.mark.parametrize("change_shared_copy", (False, True))
def test_genuine_c_uses_original_queue_publication_and_metadata_terminal_without_opening_producer(
    tmp_path, monkeypatch, change_shared_copy
):
    """Actual tiny numerical C + ordinary queue; synthetic I/O admission only."""
    import test_refit_jobs as fixture
    from tests.unit.test_phase1_worker_protocol import SyntheticServerClock
    from darsia.fluidflower import job_filesystem, published_jobs_snapshot
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    from darsia.fluidflower.pipeline import job_queue, publication_runtime
    from darsia.fluidflower.pipeline.store_runtime import ResolvedInput
    data = fixture.series.__wrapped__(tmp_path, monkeypatch)
    prepared, root = fixture.prepare(data, loro=False)
    job = job_queue.read_json(root / "job.json")
    task = job_queue.read_json(Path(prepared["tasks"][0]["task"]["path"]))
    # These are protocol doubles, not producer-input/physical qualification.
    unit_asset = Path(task["input_catalog"]["inputs"][0]["reference"].get("path", data["baseline"]["path"]))
    def availability(value):
        return tuple(ResolvedInput(deepcopy(row["reference"]), unit_asset, tuple(row["uses"]))
                     for row in value["input_catalog"]["inputs"])
    def output(value, token):
        local = tmp_path / "published-local"
        attempt = local / value["job_id"] / value["run"] / ".attempts" / token
        admission = job_filesystem.LocalFileAdmission(True, "synthetic-local", "unit protocol",
            "No physical filesystem qualification", str(tmp_path))
        return job_queue.LocalOutputBinding("synthetic", local, attempt, admission)
    def no_foreign(_attempt):
        raise AssertionError("Fresh C must not probe a foreign producer")
    queue = job_queue.JobQueue(root, machine_id="synthetic", server_clock=SyntheticServerClock())
    publisher = publication_runtime.make_publication_factory(root, job, "synthetic", producer_origin_for_attempt=no_foreign)
    owner = dict(machine="synthetic", pid=123, process_create_time="synthetic-C-epoch")
    claim = queue.claim("synthetic", "cpu", owner, availability=availability,
        output_factory=output, publication_factory=publisher)
    configuration = descriptor(Path(data["root"]) / "calibration/refit/job1/refit_plan.json")
    binding = SimpleNamespace(configuration_descriptor=configuration, require_output=lambda _output: None)
    with monkeypatch.context() as patch:
        patch.setattr(run_io, "for_task", lambda value: binding if value is claim.task else None)
        result = refit_runner.run_refit(claim.task, claim.output, "cpu", "float32")
    result = dict(result, task=claim.task, attempt_id=claim.token, commit=task["commit"],
        calibration_sha256=task["calibration_sha256"], actual_worker=owner)
    queue.complete(claim, result)
    from darsia.fluidflower.pipeline import refit_jobs
    # The copied completion exists before the alias done marker. It must not
    # enable a frozen result in the real GUI reader during that transition.
    done_path = root / "done/refit.json"
    held_done = tmp_path / "held-original-refit-done.json"
    done_path.rename(held_done)
    pending = refit_jobs.refit_status(root)
    assert pending["tasks"]["refit"]["state"] == "running"
    assert pending["frozen_calibration"] is None
    held_done.rename(done_path)
    done = job_queue.read_json(root / "done/refit.json")
    original_open = Path.open
    def shared_only(path, *args, **kwargs):
        assert not path.is_relative_to(claim.output), "Terminal readers cannot open producer-local products"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", shared_only)
    original_metadata = published_jobs_snapshot._original_metadata
    contexts = []
    mutated = []

    def read_original(index, products, asset, attempt, machine, *, metadata, decode_json=True):
        """Delegate the actual strict API and optionally drift one consumed copy."""
        value = original_metadata(index, products, asset, attempt, machine,
                                  metadata=metadata, decode_json=decode_json)
        contexts.append(id(metadata))
        if change_shared_copy and not mutated:
            key, _row = metadata.matches(published_jobs_snapshot._original_path(asset["path"], attempt))[0]
            bound, raw = metadata.read(key)
            shared_path = Path(bound["path"])
            shared_path.write_bytes(b" " * len(raw))
            assert shared_path.stat().st_size == len(raw)
            mutated.append(shared_path)
        return value

    monkeypatch.setattr(published_jobs_snapshot, "_original_metadata", read_original)
    if change_shared_copy:
        with pytest.raises(ValueError, match="Metadata bytes differ|metadata identity changed"):
            published_jobs_snapshot._terminal(root, root / "results/refit",
                claim.task, job, done, JobSnapshotReader().read)
        assert mutated and len(set(contexts)) == 1
        with pytest.raises(ValueError):
            refit_jobs.refit_status(root)
        return
    completed, machines, lineage = published_jobs_snapshot._terminal(root, root / "results/refit",
        claim.task, job, done, JobSnapshotReader().read)
    assert len(contexts) == 2 and len(set(contexts)) == 1
    assert completed == result and machines == {} and lineage == []
    assert completed["frozen_calibration"] is not None and completed["stop_required"] is False
    status = refit_jobs.refit_status(root)
    assert status["complete"] and status["parameters"] == result["parameters"]
    assert status["frozen_calibration_origin"] == result["frozen_calibration"]
    document, shared_frozen = refit_jobs.read_frozen_calibration(root)
    assert document["parameters"] == result["parameters"]
    assert shared_frozen == status["frozen_calibration"]
    assert not Path(shared_frozen["path"]).is_relative_to(claim.output)
    assert shared_frozen["sha256"] == result["frozen_calibration"]["sha256"]
