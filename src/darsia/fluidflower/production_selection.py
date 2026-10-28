"""Production choices bound to existing plans and named calibration jobs.

This UI admission layer never rewrites a plan, calibration, seal or task. The
original plan reader, series approval and queue producer remain authoritative.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .pipeline.job_queue import descriptor, name


CHOICE_KEY = "production_calibration"
STORES_KEY = "production_stores_path"


def _bound(value):
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def same_calibration(first, second):
    """Names and locations are not identity; exact frozen bytes are."""
    return (isinstance(first, dict) and isinstance(second, dict)
            and bool(first.get("sha256"))
            and first.get("sha256") == second.get("sha256")
            and type(first.get("size_bytes")) is int
            and first["size_bytes"] == second.get("size_bytes"))


LOCAL_JOB_FORMAT = "darsia-local-refit-job-v1"


def production_calibration_fields(frozen):
    """Check the frozen fields that phase two reads; return the chemistry model.

    Phase two uses the N1 coefficients (``parameters``), each fit run's
    anchor, and for every other run the anchor rules with the run roles. A
    missing field refuses with a plain reason; nothing is filled in, and the
    chemistry model is only the one the frozen document itself records.
    """
    import math
    from .mass.n1_arithmetic import validate_parameters
    if frozen.get("frozen") is not True:
        raise ValueError("The mass calibration is not frozen.")
    if not isinstance(frozen.get("parameters"), dict):
        raise ValueError("The frozen calibration records no N1 gain and beta.")
    validate_parameters(frozen["parameters"])
    for key, label in (("calibration_runs", "fit"), ("heldout_runs", "held-out"),
                       ("descriptive_runs", "descriptive"), ("excluded_runs", "excluded")):
        if not isinstance(frozen.get(key), list):
            raise ValueError(f"The frozen calibration records no list of {label} runs.")
    anchors = frozen.get("anchors_by_run")
    if not isinstance(anchors, dict) or any(
            not isinstance(anchors.get(run), dict)
            or not all(type(anchors[run].get(key)) in (int, float) and math.isfinite(anchors[run][key])
                       for key in ("elapsed_h", "affected_pore_volume_m3"))
            for run in frozen["calibration_runs"]):
        raise ValueError("The frozen calibration records no anchor for every fit run.")
    rules = frozen.get("rules")
    if not isinstance(rules, dict) or not {"anchor_target_h", "anchor_tolerance_h"} <= set(rules):
        raise ValueError("The frozen calibration records no anchor rules.")
    return frozen.get("chemistry_model")


def _registered_series_id(series_root):
    """The registered id of the series kept in ``series_root``, else a plain refusal."""
    from .series_paths import configured_series
    from .storage_paths import same_storage_path
    for key, _name, root in configured_series():
        if same_storage_path(root, series_root):
            return key
    raise ValueError("The calibration job's series folder is not a registered series: " + str(series_root))


def _read_local_calibration(root, job_bound, job):
    """A completed local overflow (variant F) calibration, checked field by field."""
    from .pipeline.local_overflow_refit import FROZEN_FORMAT
    from .pipeline.local_refit import read_local_refit_job
    state = read_local_refit_job(root)
    if state["frozen_error"]:
        raise ValueError(state["frozen_error"])
    if state["frozen"] is None:
        raise ValueError("The local mass-calibration job has no frozen calibration yet.")
    if state.get("calibration_scope") == "individual":
        raise ValueError("Choose a shared mass calibration; per-run calibrations are not used in production.")
    frozen, bound = state["frozen"]
    if frozen.get("format") != FROZEN_FORMAT or frozen.get("variant_f") is not True:
        raise ValueError("Choose a completed overflow (variant F) or global mass-calibration job.")
    chemistry = production_calibration_fields(frozen)
    if chemistry != "titration":
        raise ValueError("Choose a calibration that explicitly records Titration. "
                         "Other chemistry models are not available on this production route yet.")
    if descriptor(root / "job.json") != job_bound:
        raise ValueError("The calibration job changed while it was read.")
    name(job["job_id"])
    return dict(name=job["job_id"], chemistry_model=chemistry, chemistry_label="Titration",
                calibration=_bound(bound), job_root=str(root), job=job_bound,
                series_id=_registered_series_id(Path(job["series_manifest"]["path"]).parent))


def read_named_calibration(job_root):
    """Read a named completed calibration through its original result readers.

    The name is the immutable task's job_id, not a folder-name guess. Opening a
    job here neither indexes a library nor approves its result for a series.
    A local overflow (variant F) calibration job is read by its own job reader.
    """
    from .jobs_metadata import JobSnapshotReader
    from .pipeline.common_calibration import refit_terminal
    from .pipeline.job_queue import validate_task
    from .pipeline.refit_jobs import read_frozen_calibration
    from .pipeline.refit_runner import FREEZE_FORMAT, PIPELINE
    from .pipeline.staged_data import _same_json

    root = Path(job_root)
    reader = JobSnapshotReader()
    job_bound = descriptor(root / "job.json")
    job = reader.read(Path(job_bound["path"]), job_bound)
    if job.get("format") == LOCAL_JOB_FORMAT:
        return _read_local_calibration(root, job_bound, job)
    task_bound = descriptor(root / "results" / "refit" / "task.json")
    task = reader.read(Path(task_bound["path"]), task_bound)
    validate_task(task)
    if (job.get("pipeline") != PIPELINE or task.get("pipeline") != PIPELINE
            or task.get("run") != "refit" or "refit" not in job.get("expected_runs", ())
            or any(task.get(key) != job.get(key)
                   for key in ("commit", "precision", "calibration_sha256"))):
        raise ValueError("Choose the completed global mass-calibration job.")
    # Positive original publication/claim lineage, not existence of a result.
    completed, _machines, _lineage = refit_terminal(
        root, task, job, reader.read(root / "done" / "refit.json"), reader.read)
    frozen, bound = read_frozen_calibration(root)
    if (frozen.get("format") != FREEZE_FORMAT or frozen.get("frozen") is not True
            or frozen.get("code_commit") != task["commit"]
            or not _same_json(frozen.get("configuration"), completed["task"]["configuration"])
            or not same_calibration(bound, completed.get("frozen_calibration"))):
        raise ValueError("The named job and frozen calibration have different source identities.")
    chemistry = frozen.get("chemistry_model")
    if chemistry != "titration":
        raise ValueError("Choose a calibration that explicitly records Titration. "
                         "Other chemistry models are not available on this production route yet.")
    if descriptor(root / "job.json") != job_bound or descriptor(Path(task_bound["path"])) != task_bound:
        raise ValueError("The calibration job changed while it was read.")
    recorded_name = completed["task"]["job_id"]
    name(recorded_name)  # Validate without changing the recorded display text.
    return dict(name=recorded_name, chemistry_model=chemistry,
                chemistry_label="Titration", calibration=_bound(bound),
                job_root=str(root), job=job_bound, task=task_bound,
                series_id=job["series_id"])


def _read_production_plan(plan, *, series_root, series_id):
    """Read positive N1 metadata through the current registered series roots."""
    from .staged_jobs import plan_pipeline, read_execution_plan

    expected = _bound(plan)
    if plan_pipeline(expected) != "n1":
        return read_execution_plan(Path(expected["path"]), expected_sha256=expected["sha256"])
    if not series_root or not isinstance(series_id, str) or not series_id:
        raise ValueError("Production facts require the current registered series root and ID.")
    from .pipeline.alias_job import _manifest, read_alias_series

    manifest_path = Path(series_root) / "series.toml"
    manifest = descriptor(manifest_path)
    _, registered, _roots = _manifest(manifest)
    if registered.get("series_id") != series_id:
        raise ValueError("The registered series manifest differs from the captured series ID.")
    value, configurations, bound = read_alias_series(
        expected["path"], expected_sha256=expected["sha256"], series_manifest=manifest)
    if bound != expected or value.get("pipeline") != "n1":
        raise ValueError("The selected production execution-plan descriptor changed.")
    # Original prepared plans need not repeat a series ID: their current
    # series membership is established by the bound manifest and rooted read.
    if value.get("series_id", series_id) != series_id:
        raise ValueError("The production plan belongs to another registered series.")
    if descriptor(Path(expected["path"])) != expected or descriptor(manifest_path) != manifest:
        raise ValueError("The production plan or registered series manifest changed while facts were read.")
    return value, configurations, bound


def read_production_facts(plan, *, series_root=None, series_id=None):
    """Read exact plan facts, with rooted admission for explicit N1 metadata.

    The caller captures the current series context before scheduling this read.
    Local aliases remain subject to the original InputCatalog's root and byte
    checks. Other pipelines keep their original execution-plan reader.
    """
    from .jobs_panel import read_plan_facts

    value, configurations, bound = _read_production_plan(
        plan, series_root=series_root, series_id=series_id)
    if bound != _bound(plan):
        raise ValueError("The selected execution-plan descriptor changed.")
    facts = read_plan_facts(plan, reader=lambda *args, **kwargs: (value, configurations, bound))
    pipeline = value.get("pipeline", "native-phase-n1")
    facts["pipeline"] = pipeline
    if any((pipeline in {"n1", "native-phase-statistics"} or "precision" in config)
           and config.get("precision") != value["job"].get("precision") for config in configurations):
        raise ValueError("The job and run configurations have different precision.")
    if pipeline != "native-phase-statistics":
        calibrations = [config["calibration"] if pipeline == "n1"
                        else config["mass"]["calibration"] for config in configurations]
        if not calibrations or any(not same_calibration(calibrations[0], item) for item in calibrations):
            raise ValueError("Production cannot mix frozen calibration identities.")
        if calibrations[0]["sha256"] != value["job"]["calibration_sha256"]:
            raise ValueError("The plan and its run calibrations differ.")
        facts["calibration"] = _bound(calibrations[0])
    return facts


def production_choice_blocker(facts, choice):
    """Explain a choice mismatch; never silently select or replace a plan."""
    if not facts:
        return "Read an adopted execution plan first."
    if facts.get("pipeline") == "native-phase-statistics":
        return ""  # Phase one precedes mass calibration and remains available.
    if facts.get("precision") != "float32":
        return ("Production requires float32 inputs. A higher precision requires "
                "new eligible phase maps and a separately admitted plan.")
    if not choice:
        return "Choose the named mass-calibration job for this production."
    if choice.get("chemistry_model") != "titration":
        return "This production route requires an explicitly recorded Titration calibration."
    if not same_calibration(facts.get("calibration"), choice.get("calibration")):
        return ("This plan uses a different mass calibration. Choose a matching plan "
                "or prepare and review a new plan; the current plan is never changed.")
    return ""


def validate_production_choice(plan, choice, *, series_root, series_id):
    """Recheck the selected job and existing series lock immediately on Create."""
    from .series_calibration_approval import read_approval

    current = read_named_calibration(choice["job_root"])
    if current != choice:
        raise ValueError("The chosen mass-calibration job changed since it was shown.")
    if current["series_id"] != series_id:
        raise ValueError("The calibration job belongs to another series.")
    facts = read_production_facts(plan, series_root=series_root, series_id=series_id)
    reason = production_choice_blocker(facts, current)
    if reason:
        raise ValueError(reason)
    approval = read_approval(series_root, verify=True)
    if approval is None or approval.get("locked") is not True:
        raise ValueError("Approve and lock this calibration in Calibration > Use for this series first.")
    if (approval["binding"].get("chemistry_model") != current["chemistry_model"]
            or not same_calibration(approval["binding"].get("calibration"), current["calibration"])):
        raise ValueError("This calibration differs from the locked series choice. "
                         "Use the explicit unlock and review in Calibration before changing it.")
    return facts


def create_production_job(plan, job_root, **kwargs):
    """Guard the selected calibration before profile admission and Jobs creation.

    Jobs supplies its captured series, saved-selection resolver and stores path.
    Both local and shared workers subsequently use that one created queue.
    """
    from .staged_jobs import plan_pipeline
    from .production_resources import create_profiled_analysis_job

    captured = deepcopy(plan)
    choice = captured.pop(CHOICE_KEY, None)
    if STORES_KEY in captured:
        kwargs["stores_path"] = captured.pop(STORES_KEY)
    bound = _bound(captured)
    manifest_path = Path(kwargs["series_root"]) / "series.toml"
    manifest = descriptor(manifest_path)
    if plan_pipeline(bound) != "native-phase-statistics":
        if choice is None:
            raise ValueError("Choose a named mass calibration before creating a production job.")
        validate_production_choice(bound, choice, series_root=kwargs.get("series_root"),
                                   series_id=kwargs.get("series_id"))
        import tomllib
        from .production_image_grid import KEY, resolve_production_selection
        if manifest_path.is_file() and KEY in tomllib.loads(manifest_path.read_text(encoding="utf-8-sig")):
            from .series_image_selection import load_run_protocol_metadata
            root = kwargs["series_root"]
            kwargs["selection_provider"] = lambda runs: [resolve_production_selection(
                root, run, load_run_protocol_metadata(root, run), check_files=False) for run in runs]
    if descriptor(manifest_path) != manifest:
        raise ValueError("The series approval or saved selection changed before job creation.")
    # Never change profile, precision, source selection, calibration or seals.
    result = create_profiled_analysis_job(bound, job_root, **kwargs)
    if descriptor(manifest_path) != manifest:
        raise ValueError("Job files preserved, but the series approval or saved selection changed "
                         "during creation. No worker was started; review the job before using it.")
    return result
