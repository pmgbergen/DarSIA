"""Production identity admission; no fitting, queue or worker starts."""
from copy import deepcopy
import json

import pytest

from darsia.fluidflower import production_selection as production
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_staged_data import phase1_data  # noqa: F401


def bound(path="frozen.json", sha="a", size=123):
    return dict(path=path, sha256=sha * 64, size_bytes=size)


def choice():
    return dict(name="Morning", chemistry_model="titration", chemistry_label="Titration",
                job_root="named-job", calibration=bound(), series_id="series-one")


def facts(**changes):
    return dict(dict(pipeline="n1", precision="float32", calibration=bound()), **changes)


def test_equal_names_do_not_admit_a_different_frozen_calibration():
    selected = choice()
    assert production.production_choice_blocker(facts(), selected) == ""
    assert "different mass calibration" in production.production_choice_blocker(
        facts(calibration=bound(sha="b")), selected)
    assert "different mass calibration" in production.production_choice_blocker(
        facts(calibration=bound(size=124)), selected)
    # Relocation is allowed only for identical bytes, not a common display name.
    assert production.same_calibration(bound(), bound(path="published/frozen.json"))


@pytest.mark.parametrize("chemistry", [None, "simple_flash", "both"])
def test_missing_or_unavailable_chemistry_is_never_labelled_titration(chemistry):
    selected = dict(choice(), chemistry_model=chemistry)
    assert "explicitly recorded Titration" in production.production_choice_blocker(facts(), selected)


def test_phase_one_remains_available_but_production_precision_is_not_promoted():
    assert production.production_choice_blocker(dict(pipeline="native-phase-statistics"), None) == ""
    assert "float32" in production.production_choice_blocker(facts(precision="float64"), choice())


def test_named_calibration_uses_verified_terminal_task_name(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import common_calibration, job_queue, refit_jobs
    root = tmp_path / "renamed-folder-is-not-the-name"
    (root / "results/refit").mkdir(parents=True)
    (root / "done").mkdir()
    job = dict(pipeline="n1-calibration-refit", expected_runs=["refit"],
               commit="c" * 40, precision="float32", calibration_sha256="b" * 64, series_id="series-one")
    task = dict(job, run="refit", job_id="Morning", configuration=bound(sha="d"))
    for path, value in ((root / "job.json", job), (root / "results/refit/task.json", task),
                        (root / "done/refit.json", {"done": True})):
        path.write_text(json.dumps(value), encoding="utf-8")
    frozen = dict(format="darsia-n1-frozen-calibration-v1", frozen=True,
                  code_commit=job["commit"], configuration=task["configuration"], chemistry_model="titration")
    calls = []
    monkeypatch.setattr(job_queue, "validate_task", lambda value: calls.append(("task", value)))
    monkeypatch.setattr(common_calibration, "refit_terminal",
                        lambda *args: (dict(task=task, frozen_calibration=bound()), {}, set()))
    monkeypatch.setattr(refit_jobs, "read_frozen_calibration", lambda path: (frozen, bound()))
    result = production.read_named_calibration(root)
    assert result["name"] == "Morning"
    assert result["chemistry_label"] == "Titration"
    assert result["job"] == descriptor(root / "job.json")
    assert calls == [("task", task)]
    monkeypatch.setattr(refit_jobs, "read_frozen_calibration", lambda path: (frozen, bound(sha="b")))
    with pytest.raises(ValueError, match="different source identities"):
        production.read_named_calibration(root)
    monkeypatch.setattr(refit_jobs, "read_frozen_calibration", lambda path: (frozen, bound()))
    # A successful file read cannot bypass the existing terminal lineage guard.
    def reject(*args):
        raise ValueError("Original closed refit/index/task/source identity differs")
    monkeypatch.setattr(common_calibration, "refit_terminal", reject)
    with pytest.raises(ValueError, match="identity differs"):
        production.read_named_calibration(root)


def test_plan_facts_reject_mixed_precision_and_calibration(monkeypatch):
    plan = bound("plan.toml", "c")
    config = dict(run="r1", frames=[], precision="float32", calibration=bound())
    value = dict(pipeline="n1", job=dict(precision="float32", calibration_sha256="a" * 64))
    configs = [config, dict(config, run="r2")]
    monkeypatch.setattr(production, "_read_production_plan", lambda *args, **kwargs: (value, configs, plan))
    assert production.read_production_facts(plan)["calibration"] == bound()
    configs[1] = dict(config, run="r2", precision="float64")
    with pytest.raises(ValueError, match="different precision"):
        production.read_production_facts(plan)
    configs[1] = dict(config, run="r2", calibration=bound(sha="b"))
    with pytest.raises(ValueError, match="mix frozen"):
        production.read_production_facts(plan)


def test_original_native_plan_inherits_job_precision_without_a_new_config_field(phase1_data, monkeypatch):
    from .test_analysis_plan import _original
    config = _original(phase1_data)
    assert "precision" not in config
    config["mass"]["calibration"] = bound()
    plan = bound("plan.toml", "c")
    value = dict(job=dict(precision="float32", calibration_sha256="a" * 64))
    monkeypatch.setattr(production, "_read_production_plan", lambda *a, **k: (value, [config], plan))
    assert production.read_production_facts(plan)["precision"] == "float32"
    config["precision"] = "float64"
    with pytest.raises(ValueError, match="different precision"):
        production.read_production_facts(plan)


def test_create_rechecks_choice_and_preserves_exact_original_creator_arguments(monkeypatch):
    from darsia.fluidflower import production_resources, staged_jobs
    plan = bound("plan.toml", "c")
    selected = choice()
    supplied = dict(plan, production_calibration=selected, production_stores_path="stores.json")
    snapshot = deepcopy(supplied)
    calls = []
    monkeypatch.setattr(production, "descriptor", lambda path: bound(str(path), "e"))
    monkeypatch.setattr(staged_jobs, "plan_pipeline", lambda row: "n1")
    monkeypatch.setattr(production, "validate_production_choice",
                        lambda *args, **kwargs: calls.append(("validate", args, kwargs)))
    monkeypatch.setattr(production_resources, "create_profiled_analysis_job",
                        lambda *args, **kwargs: calls.append(("create", args, kwargs)) or {"worker_started": False})
    resolver = object()
    kwargs = dict(series_root="registered", series_id="series-one", selection_provider=resolver,
                  storage_profile="full", display_name="Production", series_label="Series")
    assert production.create_production_job(supplied, "fresh", **kwargs) == {"worker_started": False}
    assert calls[0] == ("validate", (plan, selected), dict(series_root="registered", series_id="series-one"))
    assert calls[1] == ("create", (plan, "fresh"), dict(kwargs, stores_path="stores.json"))
    assert supplied == snapshot


def test_failed_identity_guard_never_calls_the_job_producer(monkeypatch):
    from darsia.fluidflower import production_resources, staged_jobs
    monkeypatch.setattr(staged_jobs, "plan_pipeline", lambda row: "n1")
    monkeypatch.setattr(production, "descriptor", lambda path: bound(str(path), "e"))
    monkeypatch.setattr(production_resources, "create_profiled_analysis_job", lambda *a, **k: pytest.fail("Producer must not run"))
    with pytest.raises(ValueError, match="Choose a named"):
        production.create_production_job(bound(), "fresh", series_root="series")
    def reject(*args, **kwargs):
        raise ValueError("Calibration changed")
    monkeypatch.setattr(production, "validate_production_choice", reject)
    with pytest.raises(ValueError, match="Calibration changed"):
        production.create_production_job(dict(bound(), production_calibration=choice()), "fresh", series_root="series")


def test_changed_approval_during_creation_preserves_files_without_registration(monkeypatch):
    from darsia.fluidflower import production_resources, staged_jobs
    reads = iter([bound(sha="a"), bound(sha="a"), bound(sha="b")])
    monkeypatch.setattr(production, "descriptor", lambda path: next(reads))
    monkeypatch.setattr(staged_jobs, "plan_pipeline", lambda row: "n1")
    monkeypatch.setattr(production, "validate_production_choice", lambda *a, **k: facts())
    calls = []
    monkeypatch.setattr(production_resources, "create_profiled_analysis_job", lambda *a, **k: calls.append(a) or {"worker_started": False})
    with pytest.raises(ValueError, match="Job files preserved"):
        production.create_production_job(dict(bound(), production_calibration=choice()), "fresh", series_root="series")
    assert len(calls) == 1


def test_locked_series_choice_is_checked_again_before_create(monkeypatch):
    # A2 supplies this reader; this test is intentionally part of the joint
    # integration run after that exact dependency is present.
    from darsia.fluidflower import series_calibration_approval as approval
    selected = choice()
    monkeypatch.setattr(production, "read_named_calibration", lambda root: deepcopy(selected))
    monkeypatch.setattr(production, "read_production_facts", lambda plan, **kwargs: facts())
    record = dict(locked=True, binding=dict(chemistry_model="titration", calibration=bound()))
    reads = []
    monkeypatch.setattr(approval, "read_approval", lambda root, **kwargs: reads.append((root, kwargs)) or record)
    assert production.validate_production_choice(bound(), selected, series_root="series", series_id="series-one") == facts()
    assert reads == [("series", {"verify": True})]
    record["locked"] = False
    with pytest.raises(ValueError, match="Approve and lock"):
        production.validate_production_choice(bound(), selected, series_root="series", series_id="series-one")
    record["locked"] = True
    record["binding"]["calibration"] = bound(sha="b")
    with pytest.raises(ValueError, match="locked series choice"):
        production.validate_production_choice(bound(), selected, series_root="series", series_id="series-one")
    selected["name"] = "Changed after display"
    with pytest.raises(ValueError, match="changed since"):
        production.validate_production_choice(bound(), choice(), series_root="series", series_id="series-one")


def test_local_overflow_calibration_is_read_by_its_own_job_reader(tmp_path, monkeypatch):
    from darsia.fluidflower import series_paths
    from darsia.fluidflower.pipeline import local_refit
    from darsia.fluidflower.pipeline.local_overflow_refit import FROZEN_FORMAT
    from darsia.fluidflower.mass.n1_pipeline import _write_json
    series_root = tmp_path / "series"
    series_root.mkdir()
    (series_root / "series.toml").write_text('series_id = "Finite"\n', encoding="utf-8")
    root = tmp_path / "local-calibration"
    root.mkdir()
    _write_json(root / "job.json", dict(format=production.LOCAL_JOB_FORMAT, job_id="local-calibration",
        series_manifest=descriptor(series_root / "series.toml")))
    frozen = dict(format=FROZEN_FORMAT, frozen=True, variant_f=True, chemistry_model="titration",
        parameters=dict(candidate="N1", gain=1.5, beta=.4), calibration_runs=["fit"], heldout_runs=["held"],
        descriptive_runs=[], excluded_runs=[], anchors_by_run=dict(fit=dict(elapsed_h=2.25,
        affected_pore_volume_m3=.001)), rules=dict(anchor_target_h=2.25, anchor_tolerance_h=.13))
    pin = bound("frozen.json")
    state = dict(frozen=(frozen, pin), frozen_error="")
    monkeypatch.setattr(local_refit, "read_local_refit_job", lambda job_root: state)
    monkeypatch.setattr(series_paths, "configured_series", lambda: (("finite", "Finite", series_root),))
    result = production.read_named_calibration(root)
    assert result == dict(name="local-calibration", chemistry_model="titration", chemistry_label="Titration",
        calibration=pin, job_root=str(root), job=descriptor(root / "job.json"), series_id="finite")
    for change, reason in ((dict(chemistry_model=None), "explicitly records Titration"),
                           (dict(variant_f=False), "overflow \(variant F\)"),
                           (dict(anchors_by_run={}), "anchor for every fit run")):
        state["frozen"] = (dict(frozen, **change), pin)
        with pytest.raises(ValueError, match=reason):
            production.read_named_calibration(root)
    state.update(frozen=(frozen, pin), calibration_scope="individual")
    with pytest.raises(ValueError, match="shared mass calibration"):
        production.read_named_calibration(root)
