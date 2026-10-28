"""Original metadata/catalogue contracts; no synthetic scientific success.

These tests are authored for later scheduling. The small fixtures contain
opaque authored payloads and test metadata transport only, never RAW decoding.
The optional genuine fixture below exercises package preparation/readback only.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.published_dataset import staged_package as package
from darsia.fluidflower.published_dataset.scan import ScanHit
from darsia.fluidflower.pipeline import reproduction_lineage as lineage
from darsia.fluidflower.pipeline.job_queue import descriptor
from ..gui.test_run_io_binding import fixture as original_catalogue_fixture
from ..test_phase1_enqueue_alias_master import owned_publication


def _report(hits=()):
    value = SimpleNamespace(hits=list(hits), warnings=[], refusals=[])
    value.refuse = value.refusals.append
    return value


def _bound(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def test_original_named_catalogue_reopens_at_explicit_root_without_rewriting_metadata(tmp_path):
    task, config, bound, host, _binding = original_catalogue_fixture(tmp_path,
        declared_alias=r"outputs\original-fit\calibration.json")
    before = Path(bound["path"]).read_bytes()
    result = package._verified_catalogue(task, {"series": host}, "Synthetic")
    assert result.config == config and Path(bound["path"]).read_bytes() == before
    assert [result.catalogue.entries[key] for key in sorted(result.catalogue.entries)] == task["input_catalog"]["inputs"]
    pins = {package._key(pin["path"]): row for pin, row in result.catalogue.transport.values()}
    assert pins[package._key(config["calibration"]["path"])].path == host / "calibration.json"
    assert pins[package._key(r"outputs\original-fit\calibration.json")].path == host / "calibration.json"
    assert all(row.path.is_relative_to(host) for row in result.rows)


def test_staged_catalogue_refuses_changed_payload_before_named_consumer_walk(tmp_path):
    task, _config, _bound_config, host, _binding = original_catalogue_fixture(tmp_path)
    (host / "log.npy").write_bytes(b"changed opaque input")
    with pytest.raises(ValueError):
        package._verified_catalogue(task, {"series": host}, "Synthetic")


def test_original_proof_tree_packaging_reopens_through_original_terminal(tmp_path):
    source = owned_publication(tmp_path / "source", complete=True)
    args = dict(configuration=source.result["configuration"], completion=source.result["phase1_completion"],
                expected_commit=source.job["commit"])
    origin = lineage.bind_phase1_origin(descriptor(source.root / "job.json"), source.task["run"], **args)
    verified = lineage.verify_phase1_origin(origin, **args)
    files = package._Files()
    stage = package._publication_files(files, "p1-run-one",
        {key: origin[key] for key in package._PROOF_ROLES}, verified)
    portable = tmp_path / "portable"
    for member in files.members.values():
        target = portable / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for chunk in member.chunks():
                stream.write(chunk)
    relocated = dict(origin, **{key: dict(origin[key], path=str(portable / stage[key]["path"]))
                               for key in package._PROOF_ROLES})
    actual = lineage.verify_phase1_origin(relocated, **args)
    assert actual["task"] == verified["task"] and actual["completed"] == verified["completed"]
    assert {row["relative_path"] for row in actual["proof_files"]} == {
        row["relative_path"] for row in verified["proof_files"]}
    assert set(files.assets) == files.sealed
    indexed = next(row for row in stage["proof_files"] if row["relative_path"].endswith("native_statistics.json.gz"))
    path = portable / indexed["asset"]["path"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        lineage.verify_phase1_origin(relocated, **args)


def test_named_p2_catalogue_transport_wins_over_identical_shared_copy(tmp_path):
    source = _bound(tmp_path / "source.json", b"immutable metadata")
    files = package._Files()
    proof = files.add("staged/publications/p1-one/source.json", source["path"], source["sha256"], source["size_bytes"])
    input1 = dict(proof, path="staged/roots/old/source.json")
    input2 = dict(proof, path="staged/roots/results/source.json")
    ref = dict(root_alias="old", relative_path="source.json", sha256=source["sha256"], bytes=source["size_bytes"])
    files.alias(source, input1, input_reference=ref, priority=1)
    ref2 = dict(ref, root_alias="results")
    files.alias(source, input2, input_reference=ref2, priority=2)
    files.alias(source, proof)
    files.alias(source, input1, input_reference=ref, priority=1)
    row, = files.transport.values()
    assert row == dict(logical=source, asset=input2, input_reference=ref2)
    with pytest.raises(ValueError, match="conflicting stage catalogue"):
        files.alias(source, input1, input_reference=ref, priority=2)


@pytest.mark.parametrize("role,relative,phase,expected", (
    ("rig_model", "shape.png", "phase1", False),
    ("model", "shape.png", "phase1", False),
    ("image", "review.jpg", "phase1", True),
    ("image", "source.ARW", "phase1", True),
    ("image", "source.ARW", "phase2", False)))
def test_camera_inventory_excludes_nonimage_native_shape_and_model_assets(role, relative, phase, expected):
    reference = dict(root_alias="camera", relative_path=relative, role=role)
    assert package._camera_input(reference, ("frame.one.native.shape.0",), phase) is expected


def test_exact_frame_acquisition_does_not_select_a_different_equal_byte_locator():
    first = dict(root_alias="camera", relative_path="a.ARW", sha256="a" * 64, bytes=12, role="image")
    second = dict(first, relative_path="b.ARW")
    task = dict(input_catalog=dict(inputs=[
        dict(reference=first, uses=["frame.first.native.acquisition"]),
        dict(reference=second, uses=["frame.second.native.acquisition"])]))
    assert package._frame_acquisition(task, "second") == second
    with pytest.raises(ValueError, match="exact original named RAW acquisition"):
        package._frame_acquisition(task, "missing")
    task["input_catalog"]["inputs"][1]["reference"]["role"] = "rig_model"
    with pytest.raises(ValueError, match="exact original named RAW acquisition"):
        package._frame_acquisition(task, "second")


def test_transport_refuses_two_byte_versions_at_one_original_datareader_locator():
    files = package._Files()
    source = dict(path="C:/producer/original.json", sha256="a" * 64, size_bytes=1)
    files.alias(source, dict(source, path="staged/roots/series/original.json"))
    changed = dict(source, sha256="b" * 64)
    with pytest.raises(ValueError, match="conflicting byte identities"):
        files.alias(changed, dict(changed, path="staged/roots/series/another.json"))


@pytest.mark.parametrize("changed", ("missing-origin", "published-revision", "plan-revision"))
def test_staged_export_does_not_guess_legacy_origin_or_accept_revision_override(changed):
    job = SimpleNamespace(job=dict(commit="a" * 40), runs=["one"], tasks={"one": dict(pipeline="n1")},
                          plan_reference=lambda: dict(sha256="b" * 64))
    published = SimpleNamespace(commit="a" * 40, plan_sha256="b" * 64, proof=None,
                                configs=[dict(run="one")])
    if changed == "published-revision":
        published.commit = "c" * 40
    elif changed == "plan-revision":
        published.plan_sha256 = "c" * 64
    with pytest.raises(ValueError, match="legacy ancestry|exact original producing revision"):
        package.prepare_staged(SimpleNamespace(record="R"), job, published, {}, _report())


def test_producer_metadata_choice_is_explicit_and_does_not_allow_other_content():
    metadata = "staged/publications/p1-one/job.json"
    staged = SimpleNamespace(section=dict(sealed_metadata=[metadata]))
    def hit(file, where="key /producer"):
        return ScanHit(file, where, "local path", "recorded producer", "C:/producer/private")
    hits = [hit("series.zip:" + metadata), hit(package.MANIFEST, "key /staged_reproduction/stages/p1-one/origin"),
            hit("series.zip:expected/result.json"), hit(package.MANIFEST, "key /identity/title"),
            hit("series.zip", "member name")]
    report = _report(hits)
    package.provenance_findings(SimpleNamespace(retain_producer_metadata=False), staged, report)
    assert report.hits == hits and report.refusals and not report.warnings
    report = _report(hits)
    package.provenance_findings(SimpleNamespace(retain_producer_metadata=True), staged, report)
    assert [item.allowed for item in report.hits] == [True, True, False, False, False]
    assert len(report.warnings) == 2 and all("C:/producer/private" in text for text in report.warnings)
    assert all(old.excerpt == new.excerpt for old, new in zip(hits, report.hits))


def _schema_manifest(tmp_path):
    """Structure-only data; never a valid published/scientific fixture."""
    assets = {}
    def asset(path, token):
        value = dict(path=path, sha256=hashlib.sha256(token.encode()).hexdigest(), size_bytes=1)
        assets[path] = value
        return deepcopy(value)
    stages, origins = {}, {}
    for prefix, commit in (("p1", "a" * 40), ("p2", "b" * 40)):
        stage_id = prefix + "-one"
        root = "staged/publications/" + stage_id
        original_root = tmp_path / (prefix + "-producer")
        publication, proof, roles = {}, [], {}
        for role, relative in (("job", "job.json"), ("task", "results/one/task.json"),
                               ("done", "done/one.json"), ("products_index", "results/one/products.json")):
            value = asset(root + "/" + relative, stage_id + role)
            original = dict(value, path=str(original_root / relative))
            publication[role], roles[role] = original, value
            proof.append(dict(relative_path=relative, source=original, asset=value))
        if prefix == "p1":
            origins = dict(format=lineage.FORMAT, run="one", **deepcopy(publication))
        software_root = "software/stages/" + stage_id + "/"
        lock = asset(software_root + "requirements-worker.lock", "lock" + prefix)
        software = dict(mode="references", python="3.11.1", code=dict(commit=commit, url="https://example.org/source"),
            environment_lock=dict(path=lock["path"], sha256=lock["sha256"]),
            libraries=[dict(name="numpy", version="1.2.3")],
            inventory=asset(software_root + "versions.json", "versions" + prefix),
            source_inventory=asset(software_root + "input-inventory.json", "source" + prefix))
        stages[stage_id] = dict(pipeline="native-phase-statistics" if prefix == "p1" else "n1", run="one",
            commit=commit, precision="float32", output_bytes=4, ordered_case_ids=["one-case"], publication_root=root,
            publication=publication, proof_files=proof, environment_lock=dict(path="requirements.lock", sha256=lock["sha256"]),
            software=software, **roles)
    stages["p1-one"]["origin"] = deepcopy(origins)
    stages["p2-one"]["phase1_origin"] = deepcopy(origins)
    payload = asset("staged/roots/series/run.json", "config")
    raw = dict(path="camera/input.ARW", sha256="c" * 64, size_bytes=12)
    section = dict(format=package.FORMAT, require_bit_identical=True, stages=stages,
        runs={"one": dict(phase1="p1-one", phase2="p2-one")},
        roots=dict(series="staged/roots/series", camera="staged/roots/camera"),
        assets=list(assets.values()),
        input_assets=[dict(payload, root_alias="series", relative_path="run.json", camera=False),
                      dict(raw, root_alias="camera", relative_path="input.ARW", camera=True)],
        camera_members={"one": [raw["path"]]}, transport=[dict(logical=dict(payload, path=str(tmp_path / "original.json")),
            asset=payload, input_reference=dict(root_alias="series", relative_path="run.json", sha256=payload["sha256"], bytes=1))],
        sealed_metadata=[], provenance_policy=dict(retain_original_producer_metadata=True))
    return dict(identity=dict(record="R", series_id="Synthetic"), code=dict(published_commit="b" * 40),
        runs=[dict(run="one", frames=[dict(case_id="one-case")])], staged_reproduction=section)


def test_staged_structure_preserves_separate_original_stage_commits(tmp_path):
    manifest = _schema_manifest(tmp_path)
    assert package.validate_staged(manifest) is manifest["staged_reproduction"]
    assert manifest["staged_reproduction"]["stages"]["p1-one"]["commit"] != manifest["code"]["published_commit"]


@pytest.mark.parametrize("mutation", ("selection", "extra-stage", "missing-software", "wrong-lock",
    "missing-asset", "transport-location", "raw-members", "weak-comparison", "unacknowledged", "output-bytes"))
def test_staged_structure_refuses_incomplete_or_redirected_contract(tmp_path, mutation):
    manifest = _schema_manifest(tmp_path)
    section = manifest["staged_reproduction"]
    if mutation == "selection":
        section["stages"]["p1-one"]["ordered_case_ids"] = ["other"]
    elif mutation == "extra-stage":
        section["stages"]["p1-other"] = deepcopy(section["stages"]["p1-one"])
    elif mutation == "missing-software":
        section["stages"]["p1-one"].pop("software")
    elif mutation == "wrong-lock":
        section["stages"]["p1-one"]["environment_lock"]["sha256"] = "d" * 64
    elif mutation == "missing-asset":
        section["assets"].pop(0)
    elif mutation == "transport-location":
        section["transport"][0]["input_reference"]["root_alias"] = "other"
    elif mutation == "raw-members":
        section["camera_members"]["one"].append("camera/other.ARW")
    elif mutation == "weak-comparison":
        section["require_bit_identical"] = False
    elif mutation == "output-bytes":
        section["stages"]["p1-one"]["output_bytes"] = True
    else:
        section["provenance_policy"]["retain_original_producer_metadata"] = False
    with pytest.raises(ValueError):
        package.validate_staged(manifest)


@pytest.mark.parametrize("mode", ("references", "include"))
def test_each_stage_software_readback_rechecks_original_inventory_and_complete_assets(tmp_path, mode):
    from .test_staged_software import _case
    from darsia.fluidflower.published_dataset.staged_software import staged_software_members
    case = _case(tmp_path / "source", stages=("p1-one", "p2-one"), mode=mode)
    members, sections = staged_software_members(case.request, case.jobs)
    staged = SimpleNamespace(section=dict(assets=[], sealed_metadata=[], stages={
        key: dict(commit=job["commit"], environment_lock=deepcopy(job["environment_lock"]))
        for key, job in case.jobs.items()}))
    package.attach_staged_software(staged, sections, members)
    portable = tmp_path / "portable"
    for member in members:
        path = portable / member.name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            for chunk in member.chunks():
                stream.write(chunk)
    assets = {row["path"]: row for row in staged.section["assets"]}
    assert set(assets) == {member.name for member in members}
    assert set(staged.section["sealed_metadata"]) == {row["source_inventory"]["path"] for row in sections.values()}
    for stage in staged.section["stages"].values():
        package._readback_software(portable, stage, assets)
    first = staged.section["stages"]["p1-one"]
    path = portable / first["software"]["source_inventory"]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA/size"):
        package._readback_software(portable, first, assets)
