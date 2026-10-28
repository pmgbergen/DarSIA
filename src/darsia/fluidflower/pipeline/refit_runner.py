"""The N1 refit as a worker pipeline: ``n1-calibration-refit``.

A refit job has one task per fit: run ``refit`` fits the gain and beta on every
fit run and freezes the result, and run ``loro-<fit run>`` repeats the fit
without that run and predicts its frames. Each task's bound configuration is a
``darsia-n1-refit-plan-v1`` plan naming the per-frame statistics files of the
fit runs (written by phase one), the baseline calibration and the rules
(``mass.n1_refit.RefitRules``). The worker's RunIOBinding verifies every file
before the fit reads it; the fit itself is ``mass.n1_refit``.

Nothing here changes a pixel product, a map or the phase maps. Variant F is a
plan switch that stays off until its decisions exist.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import socket
import time

PIPELINE = "n1-calibration-refit"
PLAN_FORMAT = "darsia-n1-refit-plan-v1"
RESULT_FORMAT = "darsia-n1-refit-result-v1"
FREEZE_FORMAT = "darsia-n1-frozen-calibration-v1"
GLOBAL_RUN = "refit"
FOLD_PREFIX = "loro-"
INDIVIDUAL_PREFIX = "individual-"


def fold_of(run):
    """The fit run a task leaves out, or None for the global fit."""
    if run == GLOBAL_RUN or isinstance(run, str) and run.startswith(INDIVIDUAL_PREFIX) and len(run) > len(INDIVIDUAL_PREFIX):
        return None
    if isinstance(run, str) and run.startswith(FOLD_PREFIX) and len(run) > len(FOLD_PREFIX):
        return run[len(FOLD_PREFIX):]
    raise ValueError("A refit task is 'refit' or 'loro-<fit run>'")


def validate_plan(plan):
    """Admit a refit plan's identity, roles, statistics, baseline and fold."""
    if not isinstance(plan, dict) or plan.get("format") != PLAN_FORMAT:
        raise ValueError("A darsia-n1-refit-plan-v1 configuration is required")
    if "chemistry_model" in plan and plan["chemistry_model"] != "titration":
        raise ValueError("This mass-calibration chemistry model is not available yet")
    from ..mass.n1_refit import RefitRules
    rules = RefitRules.from_plan(plan)
    individual = plan.get("calibration_scope") == "individual"
    if individual != str(plan["run"]).startswith(INDIVIDUAL_PREFIX):
        raise ValueError("Individual calibration task and scope differ")
    if individual and (list(rules.fit_runs) != [plan.get("calibrated_run")]
            or plan["run"] != INDIVIDUAL_PREFIX + plan["calibrated_run"]):
        raise ValueError("Individual calibration needs exactly its named fit run")
    fold = fold_of(plan["run"])
    if fold != plan.get("fold") or (fold is not None and fold not in rules.fit_runs):
        raise ValueError("The plan's run and fold differ")
    statistics = plan["statistics"]
    if set(statistics) != set(rules.fit_runs):
        raise ValueError("The plan needs the statistics of exactly its fit runs")
    for run, frames in statistics.items():
        identities = [frame["frame_index"] for frame in frames]
        if not frames or len(set(identities)) != len(identities):
            raise ValueError("Each fit run needs distinct frames: " + run)
        for frame in frames:
            if not {"path", "sha256", "size_bytes"} <= set(frame["statistics"]):
                raise ValueError("Each statistics file needs a full descriptor")
    baseline = plan["baseline"]
    if not {"path", "sha256", "size_bytes"} <= set(baseline["calibration"]):
        raise ValueError("The baseline calibration needs a full descriptor")
    return rules


def read_statistics(reader, descriptor):
    """One verified per-frame statistics row (JSON, optionally gzip)."""
    from .refit_capacity import read_statistics_bound
    path = reader.path(descriptor)
    row, _decoded_bytes = read_statistics_bound(path, descriptor)
    reader.path(descriptor)
    return row


def load_rows(plan, reader, on_progress=None):
    """Every fit run's statistics rows, with the plan's selected target times."""
    from ..mass.n1_statistics import CONTRACT
    rows = []
    for run in sorted(plan["statistics"]):
        for frame in plan["statistics"][run]:
            row = read_statistics(reader, frame["statistics"])
            if (row.get("contract") != CONTRACT or row.get("run") != run
                    or row.get("frame_index") != frame["frame_index"]):
                raise ValueError(f"Statistics row differs from the plan: {run} frame {frame['frame_index']}")
            if row.get("selected_target_h") is None:
                row["selected_target_h"] = frame["selected_target_h"]
            elif row["selected_target_h"] != frame["selected_target_h"]:
                raise ValueError(f"Selected target differs from the plan: {run} frame {frame['frame_index']}")
            rows.append(row)
        if on_progress:
            on_progress(step="statistics_read", fit_run=run, rows=len(rows))
    return rows


def frozen_calibration(plan, fit, *, task, files):
    """The frozen calibration document of a global fit that requires no stop."""
    roles = plan["runs"]
    return dict(format=FREEZE_FORMAT, frozen=True, frozen_utc=datetime.now(timezone.utc).isoformat(),
                parameters=fit["parameters"], calibration_runs=fit["calibration_runs"],
                heldout_runs=list(roles.get("heldout", ())), descriptive_runs=list(roles.get("descriptive", ())),
                excluded_runs=list(roles.get("excluded", ())), anchors_by_run=fit["anchors_by_run"],
                rules=plan["rules"], baseline=plan["baseline"], configuration=task["configuration"],
                code_commit=task["commit"], calibration_files=files,
                colour_paths_refitted=False, colour_paths="as in the phase-one statistics",
                variant_f=False, gas_branch_fitted=False, map_mass_rescaling=False, overflow_in_objective=False,
                **({"chemistry_model": plan["chemistry_model"]} if "chemistry_model" in plan else {}),
                **({"calibration_scope": "individual", "calibrated_run": plan["calibrated_run"],
                    "source_roles": plan["source_roles"]} if plan.get("calibration_scope") == "individual" else {}))


def run_refit(task, output, backend, precision, on_progress=None):
    """Execute one refit task under the worker's admitted RunIOBinding."""
    from ..mass.n1_pipeline import DataReader
    from .run_io import for_task
    if task.get("pipeline") != PIPELINE or backend != "cpu" or precision != task.get("precision"):
        raise ValueError("A refit task runs on the CPU at its task precision")
    binding = for_task(task)
    if binding is None:
        raise ValueError("A refit task requires its admitted per-run I/O binding")
    binding.require_output(output)
    reader = DataReader(unc_only=False)
    return _execute_refit(task, output, backend, precision, reader,
                          binding.configuration_descriptor, on_progress)


def run_local_refit(task, output, on_progress=None):
    """Run the original fit directly on bound files in one local CPU process.

    The GUI adapter owns launch exclusivity, source identity and process thread
    limits. This entry does not claim worker admission or publish queue results.
    """
    from ..mass.n1_pipeline import DataReader, _is_unc
    if (task.get("pipeline") != PIPELINE or task.get("backend") != "cpu"
            or task.get("precision") != "float32"
            or type(task.get("threads")) is not int or task["threads"] != 1):
        raise ValueError("A local refit requires CPU/float32 and one thread")
    if not isinstance(task.get("commit"), str) or not task["commit"]:
        raise ValueError("A local refit requires its source commit")
    configuration = task.get("configuration")
    if (not isinstance(configuration, dict)
            or not {"path", "sha256", "size_bytes"} <= set(configuration)):
        raise ValueError("A local refit requires a full configuration descriptor")
    output = Path(output)
    if not output.is_absolute() or _is_unc(output):
        raise ValueError("A local refit requires an absolute local output path")
    reader = DataReader(unc_only=False)
    if reader.io_binding is not None:
        raise ValueError("A local refit cannot use a worker I/O binding")
    return _execute_refit(task, output, "cpu", "float32", reader,
                          configuration, on_progress)


def _execute_refit(task, output, backend, precision, reader, configuration,
                   on_progress=None):
    """Shared original plan validation, fitting and output sequence."""
    from ..mass import n1_refit
    from ..mass.n1_pipeline import _write_json
    from .job_queue import retry_transient
    plan = reader.json(configuration)
    rules = validate_plan(plan)
    if plan["run"] != task["run"]:
        raise ValueError("Task and plan runs differ")
    rules.require_variant_f_off()
    baseline = reader.json(plan["baseline"]["calibration"])
    if baseline.get("parameters") != plan["rules"]["baseline"]:
        raise ValueError("The plan's baseline parameters differ from its baseline calibration")
    output = Path(output)
    retry_transient(lambda: output.mkdir(parents=True, exist_ok=True))
    if retry_transient(lambda: any(output.iterdir())):
        raise FileExistsError("Each attempt must be fresh")

    def progress(**event):
        if on_progress:
            on_progress(dict(state="refit_progress", run=task["run"], **event))

    clock = time.perf_counter()
    rows = load_rows(plan, reader, progress)
    fold = fold_of(task["run"])
    fit = n1_refit.fit_candidate(rows, rules, excluded_run=fold, progress=progress)
    files = {"fit.json": _write_json(output / "fit.json", fit)}
    frozen = None
    if fold is not None and not fit["stop_required"]:
        files["predictions.json"] = _write_json(output / "predictions.json", dict(
            format="darsia-n1-refit-predictions-v1", heldout_fit_run=fold, parameters=fit["parameters"],
            predictions=n1_refit.predict_run(rows, fit, fold)))
    if fold is None and not fit["stop_required"]:
        progress(step="freeze")
        frozen = _write_json(output / "frozen_calibration.json",
                             frozen_calibration(plan, fit, task=task, files=dict(files)))
        files["frozen_calibration.json"] = frozen
    progress(step="complete", stop_required=fit["stop_required"])
    return dict(format=RESULT_FORMAT, complete=True, pipeline=PIPELINE, run=task["run"], fold=fold,
                backend=backend, precision=precision, host=socket.gethostname(), frames=len(rows),
                parameters=fit["parameters"], stop_required=fit["stop_required"], reason=fit.get("reason"),
                files=files, frozen_calibration=frozen, wall_seconds=time.perf_counter() - clock,
                validation=dict(state="not_scored", reason="Checks 1-3 score the frozen calibration separately"),
                worker_references_closed=True)
