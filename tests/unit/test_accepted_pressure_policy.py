"""Recorded pressure DATA with neutral paths; finite geometry/config DATA.

Pressure/index/manifest fixtures are historical copies whose user folders are
neutral; the decision record file is a neutral specimen. The test series names
the decision, who decided and the pinned files in its configuration. The
original atmosphere consumer, rooted metadata catalogue and all three
provisional-data guards remain active.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import accepted_pressure_policy as policy
from darsia.fluidflower import analysis_plan as plans
from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower import staged_jobs
from darsia.fluidflower.mass.n1_pipeline import _atmosphere
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from darsia.fluidflower.pipeline.job_queue import descriptor

from .test_analysis_plan import _job
from .test_analysis_source_inputs import _save, saved_products  # noqa: F401
from .test_staged_data import phase1_data  # noqa: F401

REASON = "Use the existing short-gap station-pressure means without changing measured values or quality flags."
AUTHORITY_PIN = dict(sha256="9fba1bad185d7e4acad2e5491631a029609e156ad6d667cf2a3fc3980718f45f", size_bytes=447)
INDEX_PIN = dict(sha256="64941b1596755b278a2b8e2228f45228fd0743693749a7767ffbe175b1519840", size_bytes=28327)
MANIFEST_PIN = dict(sha256="c3b8040c2ae6090b4c8f33eb7bb507abef3ed3170b5055836d0f39ad02cc0ad2", size_bytes=540870)
BUILDER_PIN = dict(sha256="a947fb612447fd6c8aa141340fe60dbaf100a0b5b8395235fe6eaae25d947382", size_bytes=10484)
DECISIONS = dict(accepted_pressure=dict(record_id="decision-0345-pressure-short-gaps",
    record_file=AUTHORITY_PIN, actor="Test decider", confirmed_utc="2026-10-21T10:51:53Z",
    reason=REASON, runs=["ac24", "ac32", "ac41", "ac45", "ac51"],
    source_index=INDEX_PIN, source_manifest=MANIFEST_PIN, data_builder=BUILDER_PIN))
GAP_AUTHORITY = "Test decider: " + REASON


def _pin(value):
    return "{ sha256 = \"" + value["sha256"] + "\", size_bytes = " + str(value["size_bytes"]) + " }"


SERIES_TOML = ('schema_version = 1\nseries_id = "FF_AC"\n[roots]\nseries = "."\n[runs]\nids = ["ac32"]\n'
               "[decisions.accepted_pressure]\n"
               'record_id = "decision-0345-pressure-short-gaps"\n'
               "record_file = " + _pin(AUTHORITY_PIN) + "\n"
               'actor = "Test decider"\nconfirmed_utc = "2026-10-21T10:51:53Z"\n'
               'reason = "' + REASON + '"\n'
               'runs = ["ac24", "ac32", "ac41", "ac45", "ac51"]\n'
               "source_index = " + _pin(INDEX_PIN) + "\n"
               "source_manifest = " + _pin(MANIFEST_PIN) + "\n"
               "data_builder = " + _pin(BUILDER_PIN) + "\n")


def _read(bound):
    return json.loads(Path(bound["path"]).read_bytes())


def _authorization():
    return policy.human_authorization(policy.pressure_settings(DECISIONS), "FF_AC")


@pytest.fixture
def recorded_pressure(tmp_path, saved_products):
    case = saved_products
    root = case.root
    (root / "series.toml").write_text(SERIES_TOML, encoding="utf-8")
    fixture_root = Path(__file__).with_name("fixtures") / "accepted_pressure_policy"
    authority_path = root / "authority.json"
    authority_path.write_bytes((fixture_root / "N345_N375_AUTHORITY.json").read_bytes())
    authority = descriptor(authority_path)
    assert authority["sha256"] == AUTHORITY_PIN["sha256"]
    for name in ("N274_INDEX.json", "AC32_ATMOSPHERE.json", "GFI_SOURCE_MANIFEST.json"):
        (root / name).write_bytes((fixture_root / name).read_bytes())
    original_index = descriptor(root / "N274_INDEX.json")
    index = _read(original_index)
    original_atmosphere = descriptor(root / "AC32_ATMOSPHERE.json")
    atmosphere = _read(original_atmosphere)
    source_manifest = index["source_manifest"]
    source_aliases = [dict(logical=index["runs"]["FF_AC"]["ac32"]["atmosphere_series"],
                           physical=original_atmosphere),
                      dict(logical=source_manifest, physical=descriptor(root / "GFI_SOURCE_MANIFEST.json"))]
    assert original_index["sha256"] == INDEX_PIN["sha256"]
    assert atmosphere["complete"] and atmosphere["provisional_gap_default"]

    # Existing genuine finite native/geometry metadata is rebound before source
    # collection. Nothing released is edited or made scientifically authoritative.
    docs = case.documents["run-one"]
    original = deepcopy(docs.original)
    original["run"] = original["frames"][0]["source"]["run"] = "ac32"
    configuration = _save(root / "ac32-configuration.json", original)
    measured = deepcopy(docs.measured)
    measured["run"] = "ac32"
    measured_bound = _save(root / "ac32-frame.json", measured)
    frames = deepcopy(docs.frames)
    frames["run"] = "ac32"
    frames["frames"]["case-one"]["frame_model"] = measured_bound
    frame_index = _save(root / "ac32-frames.json", frames)
    geometry = _save(root / "ac32-geometry.json", dict(
        format="darsia-measured-run-geometry-v1", reference_models_complete=True,
        runs={"ac32": dict(frame_data=frame_index)}))
    return SimpleNamespace(series_root=root, original_index=original_index,
        authority=authority, original_atmosphere=original_atmosphere, atmosphere=atmosphere,
        index=index, source_manifest=source_manifest, source_aliases=source_aliases,
        configurations=[configuration], geometry=geometry, original=original)


def _publish(case, name="accepted.json"):
    return policy.prepare_accepted_pressure_index(case.original_index,
        case.series_root / name, authority=case.authority, series_root=case.series_root,
        source_aliases=case.source_aliases)


def _accepted(result):
    index = _read(result["pressure_index"])
    bound = index["runs"]["FF_AC"]["ac32"]["atmosphere_series"]
    return bound, _read(bound)


def _collect(case, pressure, path):
    return preparation.collect_phase1_source_inputs(case.configurations, path,
        series_root=case.series_root, geometry_index=case.geometry,
        pressure_index=pressure, gas_probability=False)


def test_recorded_policy_preserves_all_original_data_and_admits_exact_named_evidence(recorded_pressure):
    case = recorded_pressure
    before = {path: path.read_bytes() for path in case.series_root.rglob("*") if path.is_file()}
    result = _publish(case)
    assert result["runs"] == result["covered_runs"] == ["ac32"]
    assert result["plan_prepared"] is result["worker_started"] is False
    bound, accepted = _accepted(result)
    expected = deepcopy(case.atmosphere)
    expected["provisional_gap_default"] = False
    expected["gap_default_authority"] = GAP_AUTHORITY
    assert policy._same({key: value for key, value in accepted.items() if key != policy.FIELD}, expected)
    assert accepted["rows"] == case.atmosphere["rows"]
    assert accepted["gaps"] == case.atmosphere["gaps"]
    assert accepted["gap_default_authority"] == GAP_AUTHORITY
    assert case.atmosphere["gap_default_authority"] == "pending; needs a recorded choice"
    assert accepted["adopted"] is False
    catalog = InputCatalog({"series": str(case.series_root)})
    policy.verify_accepted_pressure(accepted, catalog=catalog)
    assert {use for row in catalog.entries.values() for use in row["uses"]} == {
        "pressure_policy." + field for field in policy._SOURCES}
    for key, source in accepted[policy.FIELD].items():
        if key == "authority":
            assert _read(source) == _authorization()
            assert _read(source)["actor"] == "Test decider" and _read(source)["runs"] == DECISIONS[
                "accepted_pressure"]["runs"]
            text = Path(source["path"]).read_text(encoding="utf-8")
            assert not any(word in text for word in ("decision-0345", "decision-0375", "decided_by"))
            assert Path(source["path"]).name == "authorization.json"
        elif isinstance(source, dict):
            actual = InputCatalog({}, case.source_aliases).physical(source["original"])
            assert Path(source["copy"]["path"]).read_bytes() == Path(actual["path"]).read_bytes()
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert descriptor(bound["path"]) == bound
    assert not list(case.series_root.rglob("execution.toml"))
    assert not list(case.series_root.rglob("job.json"))


@pytest.mark.parametrize("fault", ["coordination-record", "actor", "reason"])
def test_pressure_reread_requires_plain_exact_human_authorization(recorded_pressure, fault):
    case = recorded_pressure
    result = _publish(case)
    _, accepted = _accepted(result)
    authority = accepted[policy.FIELD]["authority"]
    assert _read(authority) == _authorization()
    assert "original" not in authority and "copy" not in authority
    assert _read(result["pressure_index"])[policy.FIELD]["authority"] == authority
    if fault == "coordination-record":
        changed_authority = case.authority  # Exact private decision, forbidden as product evidence.
    else:
        changed = _read(authority)
        if fault == "actor":
            changed["actor"] = "somebody else"
        else:
            changed["reason"] = "Recorded somewhere else"
        changed_authority = _save(case.series_root / "changed-authorization.json", changed)
    accepted[policy.FIELD]["authority"] = changed_authority
    # A changed actor or reason no longer matches the accepted gap authority text.
    with pytest.raises(ValueError, match="pressure authorization changed|decision provenance"):
        policy.verify_accepted_pressure(accepted,
            catalog=InputCatalog({"series": str(case.series_root)}))
    assert _read(case.original_atmosphere)["rows"] == case.atmosphere["rows"]
    assert not list(case.series_root.rglob("job.json"))


def test_original_prepare_refuses_provisional_data_and_verified_policy_uses_same_atmosphere_abi(recorded_pressure):
    case = recorded_pressure
    # Only this untreated legacy negative uses a local finite index locator.
    # The actual table bytes remain exact; no old campaign locator is opened.
    # The publisher below still receives the unchanged recorded N274 index.
    legacy = deepcopy(case.index)
    legacy["runs"]["FF_AC"]["ac32"]["atmosphere_series"] = case.original_atmosphere
    legacy_index = _save(case.series_root / "local-untreated-legacy-index.json", legacy)
    with pytest.raises(ValueError, match="Provisional pressure gaps"):
        _collect(case, legacy_index, case.series_root / "refused-inputs.json")
    assert not (case.series_root / "refused-inputs.json").exists()
    result = _publish(case)
    _, accepted = _accepted(result)
    inputs = _collect(case, result["pressure_index"], case.series_root / "inputs.json")
    prepared = preparation.prepare_phase1_inputs(inputs["inputs"], case.series_root / "plan",
        series_root=case.series_root, geometry_index=case.geometry,
        pressure_index=result["pressure_index"], gas_probability=False,
        job_id="testonly", job=_job())
    config = _read(prepared["runs"][0]["configuration"])
    assert policy._same(config["statistics"]["atmosphere_series"], accepted)
    original_abi = _atmosphere(dict(atmosphere_series=case.atmosphere), {"elapsed_h": 1.})
    assert config["frames"][0]["mass_spec"]["atmosphere"] == original_abi
    assert config["frames"][0]["native_task"] == case.original["frames"][0]["native_task"]
    assert prepared["adopted_for_execution"] is prepared["worker_started"] is False
    assert _read(case.original_atmosphere)["provisional_gap_default"] is True
    assert not list(case.series_root.rglob("job.json"))


@pytest.mark.parametrize("boundary", ["prepare", "plan", "queue"])
def test_changed_accepted_numeric_type_fails_before_each_original_boundary(recorded_pressure, boundary):
    case = recorded_pressure
    result = _publish(case)
    accepted_bound, accepted = _accepted(result)
    config = plans.phase1_configuration(case.original, precision="float32", atmosphere=accepted)
    # Equal numerical value, different JSON type: provenance comparison must
    # not silently admit bool/int/float coercion through Python equality.
    changed = deepcopy(accepted)
    assert changed["rows"][0]["temperature_c"] == 23.0
    changed["rows"][0]["temperature_c"] = 23
    assert changed["rows"] == accepted["rows"] and not policy._same(changed, accepted)
    resources = {"ac32": dict(backend="cpu", threads=1)}
    if boundary == "prepare":
        changed_bound = _save(case.series_root / "changed-atmosphere.json", changed)
        index = _read(result["pressure_index"])
        index["runs"]["FF_AC"]["ac32"]["atmosphere_series"] = changed_bound
        selected = _save(case.series_root / "changed-index.json", index)
        with pytest.raises(ValueError, match="changes original DATA"):
            _collect(case, selected, case.series_root / "refused-inputs.json")
        assert not (case.series_root / "refused-inputs.json").exists()
    elif boundary == "plan":
        config["statistics"]["atmosphere_series"] = changed
        with pytest.raises(ValueError, match="changes original DATA"):
            plans.write_phase1_plan(case.series_root / "refused-plan", [config],
                job_id="testonly", job=_job(), resources=resources)
        assert not (case.series_root / "refused-plan").exists()
    else:
        prepared = plans.write_phase1_plan(case.series_root / "plan", [config],
            job_id="testonly", job=_job(), resources=resources)
        value, configs, _ = plans.read_phase1_plan(prepared["plan"]["path"],
            expected_sha256=prepared["plan"]["sha256"], require_adopted=False)
        configs[0]["statistics"]["atmosphere_series"] = changed
        selected = _save(case.series_root / "changed-run.json", configs[0])
        rows = deepcopy(value["runs"])
        rows[0]["configuration"] = selected
        execution = case.series_root / "changed-execution.toml"
        execution.write_bytes(plans._plan_text(pipeline=plans.PHASE1, job_id=value["job_id"],
            job=value["job"], selection=value["selection"], runs=rows).encode("utf-8"))
        # Original metadata-only test adoption supplies the same explicit
        # precondition as existing staged-jobs tests. No real job is created.
        approved = plans.adopt_prepared_plan(execution, case.series_root / "test-approved.toml",
            expected_sha256=descriptor(execution)["sha256"], actor="synthetic test reviewer",
            reason="Prove the actual queue boundary rejects changed accepted-pressure DATA")
        with pytest.raises(ValueError, match="changes original DATA"):
            staged_jobs.phase1_inputs(approved["plan"],
                series_manifest=descriptor(case.series_root / "series.toml"))
    assert descriptor(accepted_bound["path"]) == accepted_bound
    assert not list(case.series_root.rglob("job.json"))


@pytest.mark.parametrize("fault", ["authority", "source_copy", "recipe", "scope", "observation_source", "alias_sha"])
def test_changed_authority_source_recipe_or_scope_cannot_publish_or_verify(recorded_pressure, fault):
    case = recorded_pressure
    if fault == "source_copy":
        result = _publish(case)
        _, accepted = _accepted(result)
        copied = accepted[policy.FIELD]["source_atmosphere"]["copy"]
        path = Path(copied["path"])
        raw = path.read_bytes()
        changed = raw.replace(b'"temperature_c": 23.0', b'"temperature_c": 24.0', 1)
        assert changed != raw and len(changed) == len(raw)
        path.write_bytes(changed)
        with pytest.raises(ValueError, match="source SHA/size changed"):
            policy.verify_accepted_pressure(accepted)
        return
    if fault == "authority":
        path = Path(case.authority["path"])
        raw = path.read_bytes()
        changed = raw.replace(b'"decided_by": "reviewer"', b'"decided_by": "fake"', 1)
        assert changed != raw
        path.write_bytes(changed)
        case.authority = descriptor(path)
    elif fault == "alias_sha":
        case.source_aliases[0]["physical"]["sha256"] = "0" * 64
    else:
        atmosphere = deepcopy(case.atmosphere)
        if fault == "recipe":
            filled = next(row for row in atmosphere["rows"] if row["quality"].startswith("provisional"))
            filled["station_pressure_hpa"] += 0.001
        elif fault == "scope":
            atmosphere["run"] = "ac17"  # Original policy covers the five named gap runs only.
        else:
            atmosphere["rows"][0]["observations"][0]["source"]["sha256"] = "0" * 64
        bound = _save(case.original_atmosphere["path"], atmosphere)
        case.index["runs"]["FF_AC"]["ac32"]["atmosphere_series"] = bound
        case.original_index = _save(case.original_index["path"], case.index)
    destination = case.series_root / "refused.json"
    with pytest.raises(ValueError, match="unchanged recorded pressure decision|unchanged recorded pressure index|alias changes source SHA"):
        policy.prepare_accepted_pressure_index(case.original_index, destination,
            authority=case.authority, series_root=case.series_root, source_aliases=case.source_aliases)
    assert not destination.exists() and not destination.with_name("refused_policy_sources").exists()


def test_output_copies_must_be_admitted_by_original_registered_roots_before_publication(
        recorded_pressure, tmp_path):
    case = recorded_pressure
    destination = tmp_path / "outside-registered-root.json"
    with pytest.raises(ValueError, match="No declared root contains input path"):
        policy.prepare_accepted_pressure_index(case.original_index, destination,
            authority=case.authority, series_root=case.series_root, source_aliases=case.source_aliases)
    assert not destination.exists() and not policy.pressure_policy_sources_path(destination).exists()
    assert not list(tmp_path.rglob("job.json"))


def test_publication_never_overwrites_an_original_index_or_an_existing_namespace(recorded_pressure):
    case = recorded_pressure
    before = Path(case.original_index["path"]).read_bytes()
    with pytest.raises(FileExistsError, match="fresh accepted-pressure"):
        policy.prepare_accepted_pressure_index(case.original_index, case.original_index["path"],
            authority=case.authority, series_root=case.series_root)
    assert Path(case.original_index["path"]).read_bytes() == before
    result = _publish(case)
    published = Path(result["pressure_index"]["path"]).read_bytes()
    with pytest.raises(FileExistsError, match="fresh accepted-pressure"):
        _publish(case)
    assert Path(result["pressure_index"]["path"]).read_bytes() == published
    assert not list(case.series_root.rglob("job.json"))


def test_series_without_its_recorded_pressure_decision_cannot_publish(recorded_pressure):
    case = recorded_pressure
    (case.series_root / "series.toml").write_text(SERIES_TOML.split("[decisions.")[0], encoding="utf-8")
    destination = case.series_root / "refused.json"
    with pytest.raises(ValueError, match="decisions.accepted_pressure"):
        policy.prepare_accepted_pressure_index(case.original_index, destination,
            authority=case.authority, series_root=case.series_root, source_aliases=case.source_aliases)
    assert not destination.exists() and not policy.pressure_policy_sources_path(destination).exists()


def test_pressure_settings_name_the_decision_and_the_gap_authority_names_who_decided():
    settings = policy.pressure_settings(DECISIONS)
    assert settings["actor"] == "Test decider" and settings["source_index"] == INDEX_PIN
    assert policy.gap_authority(policy.human_authorization(settings, "FF_AC")) == GAP_AUTHORITY
    for broken in (dict(DECISIONS["accepted_pressure"], actor=" "),
                   dict(DECISIONS["accepted_pressure"], runs=["ac32", "ac32"]),
                   dict(DECISIONS["accepted_pressure"], confirmed_utc="2026-10-21 10:51"),
                   {key: value for key, value in DECISIONS["accepted_pressure"].items() if key != "data_builder"}):
        with pytest.raises(ValueError):
            policy.pressure_settings(dict(accepted_pressure=broken))
