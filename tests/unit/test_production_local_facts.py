"""Default Production Facts/Create admission for exact local plan aliases.

The original phase-two sealer prepares metadata from the existing explicitly
synthetic completion fixture. Production then uses its real plan discriminator,
series reader, phase-two validator and rooted InputCatalog; none is replaced.
No scientific products are decoded. Create tests disclose inert named-job,
series-lock and final producer seams and do not claim resource qualification.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower import analysis_plan, production_selection as production
from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.pipeline import job_inputs, staged_data
from darsia.fluidflower.pipeline.job_queue import descriptor
from .gui.test_jobs_metadata import bytes_of, put
from .test_analysis_plan import _job
from .test_staged_data import _closure, phase1_data  # noqa: F401


@pytest.fixture
def local_production_plan(tmp_path, phase1_data):
    """One genuine rooted metadata read, with no injected Facts/admitter.

    The completion's tiny authored bytes are synthetic unit evidence, not a
    worker run. DataReader's local option is used only for fixture sealing;
    the production reader must admit local files through its original roots.
    """
    root = tmp_path / "registered-series"
    root.mkdir()
    manifest = root / "series.toml"
    manifest.write_text(
        'schema_version=1\nseries_id="unit"\n[roots]\nseries="."\n', encoding="utf-8")
    closure = _closure(root / "synthetic-completion", phase1_data)
    sealed = staged_data.seal_phase2_configuration(
        root / "phase2.json", closure["completion"], closure["configuration"],
        expected_phase1_commit="1" * 40, fitted_calibration=closure["calibration"],
        fitted_anchor=dict(elapsed_h=1., affected_pore_volume_m3=.001),
        reader=DataReader(unc_only=False))
    configuration = sealed["configuration"]
    config = json.loads(Path(configuration["path"]).read_text(encoding="utf-8"))
    selection = put(root / "selection.json", dict(complete=True, runs=[dict(
        run=config["run"], ordered_case_ids=[frame["case_id"] for frame in config["frames"]])]))
    unavailable = tmp_path / "unavailable-producer"
    aliases = [dict(logical=dict(pin, path=str(unavailable / filename)), physical=deepcopy(pin))
               for pin, filename in ((selection, "selection.json"), (configuration, "phase2.json"))]
    value = dict(format="darsia-series-v1", adopted_for_execution=True, pipeline="n1",
        job_id="local-production", selection=deepcopy(aliases[0]["logical"]),
        job=dict(_job(), calibration_sha256=config["calibration"]["sha256"]),
        runs=[dict(run=config["run"], configuration=deepcopy(aliases[1]["logical"]),
                   resources=dict(backend="cpu", threads=1))], data_aliases=aliases)
    state = dict(root=root, series_id="unit", manifest=manifest, config=config,
        configuration=configuration, selection=selection, aliases=aliases, value=value,
        plan_path=root / "execution.toml",
        choice=dict(name="Explicit named calibration", chemistry_model="titration",
                    chemistry_label="Titration", job_root=str(root / "named-job"),
                    calibration=deepcopy(config["calibration"]), series_id="unit"))

    def rewrite_plan():
        # Preserve the original plan writer's schema, including absence of a
        # top-level series_id. Add only explicit aliases and adoption state.
        text = analysis_plan._plan_text(pipeline=value["pipeline"], job_id=value["job_id"],
            job=value["job"], selection=value["selection"], runs=value["runs"])
        if value["adopted_for_execution"]:
            text = text.replace("adopted_for_execution = false", "adopted_for_execution = true")
        if "series_id" in value:
            text = "series_id = " + json.dumps(value["series_id"]) + "\n" + text
        for alias in value["data_aliases"]:
            text += "\n[[data_aliases]]\n"
            for side in ("logical", "physical"):
                text += "[data_aliases." + side + "]\n"
                text += "\n".join(key + " = " + json.dumps(alias[side][key])
                                  for key in ("path", "sha256", "size_bytes")) + "\n"
        state["plan_path"].write_text(text, encoding="utf-8")
        state["plan"] = descriptor(state["plan_path"])

    state["rewrite_plan"] = rewrite_plan
    rewrite_plan()
    return state


def _facts(state, **context):
    return production.read_production_facts(state["plan"], **dict(
        dict(series_root=state["root"], series_id=state["series_id"]), **context))


def _record_metadata_reads(monkeypatch):
    original, opened = job_inputs.InputCatalog._read_metadata, []

    def read(path):
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(job_inputs.InputCatalog, "_read_metadata", staticmethod(read))
    return opened


def test_default_facts_reads_exact_local_selection_and_configuration_without_mutation(
        local_production_plan, monkeypatch):
    p = local_production_plan
    opened = _record_metadata_reads(monkeypatch)
    before, supplied = bytes_of(p["root"]), deepcopy(p["plan"])
    result = _facts(p)
    assert "series_id" not in p["value"]  # The original writer does not require it.
    assert result == dict(runs=["run-one"], frames=1, precision="float32", storage_mm=[None],
        pipeline="n1", running_pressure_runs=[], gas_probability=False, route_blocker="",
        calibration=p["config"]["calibration"])
    assert opened == [Path(p["selection"]["path"]), Path(p["configuration"]["path"])]
    assert all(not Path(alias["logical"]["path"]).exists() for alias in p["aliases"])
    assert bytes_of(p["root"]) == before and p["plan"] == supplied


@pytest.mark.parametrize("context", [dict(series_root=None), dict(series_id=None),
    dict(series_id=""), dict(series_id=1), dict(series_id="other")])
def test_positive_n1_requires_the_captured_registered_context(local_production_plan, context):
    p = local_production_plan
    before = bytes_of(p["root"])
    with pytest.raises(ValueError, match="series|Series"):
        _facts(p, **context)
    assert bytes_of(p["root"]) == before


@pytest.mark.parametrize("field", ["sha256", "size_bytes"])
def test_plan_descriptor_mismatch_is_refused_before_metadata_open(local_production_plan, monkeypatch, field):
    p = local_production_plan
    p["plan"][field] = "f" * 64 if field == "sha256" else p["plan"][field] + 1
    opened = _record_metadata_reads(monkeypatch)
    with pytest.raises(ValueError, match="plan.*changed"):
        _facts(p)
    assert opened == []


@pytest.mark.parametrize("manifest", [
    'schema_version=2\nseries_id="unit"\n[roots]\nseries="."\n',
    'schema_version=1\nseries_id="unit"\n',
    'schema_version=1\nseries_id="other"\n[roots]\nseries="."\n',
])
def test_original_manifest_schema_roots_and_id_are_required(local_production_plan, monkeypatch, manifest):
    p = local_production_plan
    p["manifest"].write_text(manifest, encoding="utf-8")
    opened = _record_metadata_reads(monkeypatch)
    with pytest.raises(ValueError, match="schema|roots|series"):
        _facts(p)
    assert opened == []


def test_explicit_plan_series_id_must_match_both_manifest_and_capture(local_production_plan):
    p = local_production_plan
    p["value"]["series_id"] = p["series_id"]
    p["rewrite_plan"]()
    assert _facts(p)["frames"] == 1
    p["value"]["series_id"] = "another-series"
    p["rewrite_plan"]()
    with pytest.raises(ValueError, match="another registered series"):
        _facts(p)


@pytest.mark.parametrize("item", ["plan", "selection", "configuration"])
def test_same_bytes_outside_current_registered_roots_are_refused(local_production_plan, item):
    p = local_production_plan
    pin = p[item]
    outside = p["root"].parent / ("outside-" + Path(pin["path"]).name)
    outside.write_bytes(Path(pin["path"]).read_bytes())
    if item == "plan":
        p["plan"] = descriptor(outside)
    else:
        p["aliases"][0 if item == "selection" else 1]["physical"] = descriptor(outside)
        p["rewrite_plan"]()
    with pytest.raises(ValueError, match="No declared root"):
        _facts(p)


@pytest.mark.parametrize("item", ["selection", "configuration"])
@pytest.mark.parametrize("problem,match", [
    ("alias_sha", "alias changes source SHA"),
    ("alias_size", "alias changes source size"),
    ("physical_sha", "metadata SHA/size changed"),
    ("physical_size", "input size changed"),
    ("conflicting_alias", "Conflicting source input aliases"),
    ("missing_alias", "No declared root"),
])
def test_original_rooted_catalogue_enforces_alias_and_byte_identity(
        local_production_plan, item, problem, match):
    p = local_production_plan
    index = 0 if item == "selection" else 1
    alias = p["aliases"][index]
    physical = Path(alias["physical"]["path"])
    if problem == "alias_sha":
        alias["physical"]["sha256"] = "f" * 64
    elif problem == "alias_size":
        alias["physical"]["size_bytes"] += 1
    elif problem == "physical_sha":
        raw = physical.read_bytes()
        changed = raw.replace(b"case-one", b"case-two")
        assert changed != raw and len(changed) == len(raw)
        physical.write_bytes(changed)
    elif problem == "physical_size":
        physical.write_bytes(physical.read_bytes() + b" ")
    elif problem == "conflicting_alias":
        alternative = p["root"] / "same-bytes-another-location.json"
        alternative.write_bytes(physical.read_bytes())
        p["aliases"].append(dict(logical=deepcopy(alias["logical"]), physical=descriptor(alternative)))
    else:
        p["aliases"].pop(index)
    p["rewrite_plan"]()
    before = bytes_of(p["root"])
    with pytest.raises(ValueError, match=match):
        _facts(p)
    assert bytes_of(p["root"]) == before


def test_byte_valid_selection_still_requires_original_case_order(local_production_plan):
    p = local_production_plan
    changed = put(Path(p["selection"]["path"]), dict(complete=True,
        runs=[dict(run="run-one", ordered_case_ids=["different-case"])]))
    alias = p["aliases"][0]
    alias["physical"] = changed
    alias["logical"].update(sha256=changed["sha256"], size_bytes=changed["size_bytes"])
    p["value"]["selection"] = deepcopy(alias["logical"])
    p["rewrite_plan"]()
    with pytest.raises(ValueError, match="fixed selected case order"):
        _facts(p)


@pytest.mark.parametrize("problem,match", [
    ("inert", "inert until explicitly adopted"),
    ("precision", "different precision"),
    ("calibration", "cannot mix frozen calibrations"),
])
def test_rooted_route_keeps_original_adoption_precision_and_calibration_guards(
        local_production_plan, problem, match):
    p = local_production_plan
    if problem == "inert":
        p["value"]["adopted_for_execution"] = False
    elif problem == "precision":
        p["value"]["job"]["precision"] = "float64"
    else:
        p["value"]["job"]["calibration_sha256"] = "f" * 64
    p["rewrite_plan"]()
    with pytest.raises(ValueError, match=match):
        _facts(p)


@pytest.mark.parametrize("changed", ["plan", "manifest"])
def test_facts_recheck_plan_and_manifest_after_original_metadata_read(
        local_production_plan, monkeypatch, changed):
    p = local_production_plan
    original, events = job_inputs.InputCatalog.doc, []

    def read(catalog, pin, role, use):
        result = original(catalog, pin, role, use)
        if pin == p["value"]["runs"][0]["configuration"]:
            target = p["plan_path"] if changed == "plan" else p["manifest"]
            target.write_bytes(target.read_bytes() + b"\n# concurrent source edit\n")
            events.append(changed)
        return result

    monkeypatch.setattr(job_inputs.InputCatalog, "doc", read)
    with pytest.raises(ValueError, match="changed while facts were read"):
        _facts(p)
    assert events == [changed]


@pytest.fixture
def local_create_seams(local_production_plan, monkeypatch):
    """Only terminal choice/lock/producer seams; Facts and its reader stay real."""
    from darsia.fluidflower import production_resources, series_calibration_approval
    p = local_production_plan
    record = dict(locked=True, binding=dict(chemistry_model="titration",
                  calibration=deepcopy(p["choice"]["calibration"])))
    events = []

    def named(path):
        assert path == p["choice"]["job_root"]
        events.append("named calibration terminal seam")
        return deepcopy(p["choice"])

    def approved(root, *, verify):
        assert root == p["root"] and verify is True
        events.append("A2 lock terminal seam")
        return deepcopy(record)

    calls = []

    def producer(plan, target, **kwargs):
        assert not Path(target).exists()
        calls.append((deepcopy(plan), Path(target), dict(kwargs)))
        events.append("inert final producer")
        return dict(worker_started=False, numerical_calls_started=False, inert_producer=True)

    monkeypatch.setattr(production, "read_named_calibration", named)
    monkeypatch.setattr(series_calibration_approval, "read_approval", approved)
    monkeypatch.setattr(production_resources, "create_profiled_analysis_job", producer)
    return dict(state=p, record=record, calls=calls, events=events)


def _create(seams):
    p = seams["state"]
    supplied = dict(p["plan"], production_calibration=deepcopy(p["choice"]))
    target = p["root"].parent / "never-started-production"
    result = production.create_production_job(supplied, target,
        series_root=p["root"], series_id=p["series_id"], storage_profile="full",
        display_name="Local production", series_label="Unit")
    assert supplied == dict(p["plan"], production_calibration=p["choice"])
    return result


def test_create_reaches_default_rooted_facts_before_original_guarded_producer(
        local_create_seams, monkeypatch):
    seams, p = local_create_seams, local_create_seams["state"]
    opened = _record_metadata_reads(monkeypatch)
    before = bytes_of(p["root"])
    assert _create(seams) == dict(worker_started=False, numerical_calls_started=False, inert_producer=True)
    assert opened == [Path(p["selection"]["path"]), Path(p["configuration"]["path"])]
    assert seams["events"] == ["named calibration terminal seam", "A2 lock terminal seam", "inert final producer"]
    assert seams["calls"] == [(p["plan"], p["root"].parent / "never-started-production",
        dict(series_root=p["root"], series_id="unit", storage_profile="full",
             display_name="Local production", series_label="Unit"))]
    assert bytes_of(p["root"]) == before
    assert not seams["calls"][0][1].exists()


@pytest.mark.parametrize("problem,match", [
    ("plan_descriptor", "plan.*changed"),
    ("configuration_bytes", "input size changed"),
    ("captured_id", "captured series ID"),
    ("outside_root", "No declared root"),
    ("alias_identity", "alias changes source SHA"),
    ("unlocked", "Approve and lock"),
    ("locked_calibration", "locked series choice"),
])
def test_create_refuses_source_or_series_lock_changes_before_producer(local_create_seams, problem, match):
    seams, p = local_create_seams, local_create_seams["state"]
    if problem == "plan_descriptor":
        p["plan"]["size_bytes"] += 1
    elif problem == "configuration_bytes":
        path = Path(p["configuration"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
    elif problem == "captured_id":
        p["manifest"].write_text('schema_version=1\nseries_id="other"\n[roots]\nseries="."\n', encoding="utf-8")
    elif problem == "outside_root":
        outside = p["root"].parent / "outside-configuration.json"
        outside.write_bytes(Path(p["configuration"]["path"]).read_bytes())
        p["aliases"][1]["physical"] = descriptor(outside)
        p["rewrite_plan"]()
    elif problem == "alias_identity":
        p["aliases"][1]["physical"]["sha256"] = "f" * 64
        p["rewrite_plan"]()
    elif problem == "unlocked":
        seams["record"]["locked"] = False
    else:
        seams["record"]["binding"]["calibration"]["sha256"] = "f" * 64
    before = bytes_of(p["root"])
    with pytest.raises(ValueError, match=match):
        _create(seams)
    assert seams["calls"] == [] and bytes_of(p["root"]) == before
    assert not (p["root"].parent / "never-started-production").exists()


def test_create_rechecks_manifest_after_facts_and_lock_read(local_create_seams, monkeypatch):
    from darsia.fluidflower import series_calibration_approval
    seams, p = local_create_seams, local_create_seams["state"]
    original = series_calibration_approval.read_approval

    def change_after_facts(root, *, verify):
        result = original(root, verify=verify)
        p["manifest"].write_bytes(p["manifest"].read_bytes() + b"\n# changed after facts\n")
        return result

    monkeypatch.setattr(series_calibration_approval, "read_approval", change_after_facts)
    with pytest.raises(ValueError, match="changed before job creation"):
        _create(seams)
    assert seams["events"] == ["named calibration terminal seam", "A2 lock terminal seam"]
    assert seams["calls"] == []
