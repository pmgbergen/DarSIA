"""Explicit local overflow-target preparation and gain/beta fitting.

Only the selected completed fit runs are consumed here. Held-out run levels
are deliberately pending until their own P1 products exist; they never enter
the pooled estimate. Nothing adopts a calibration or starts phase two.
"""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from ..mass import n1_refit, overflow_levels, overflow_method, overflow_statistics, overflow_refit
from ..mass.gas_contribution_review import save_gas_contributions
from ..mass.n1_pipeline import DataReader, asset, _write_json, _jsonable
from .current_inputs import current_input_reader

REQUEST_FORMAT = "darsia-local-overflow-refit-inputs-v1"
PREPARATION_FORMAT = "darsia-local-overflow-preparation-v1"
PLAN_FORMAT = "darsia-local-overflow-refit-plan-v1"
FROZEN_FORMAT = "darsia-local-overflow-frozen-calibration-v1"


def read_request(bound, *, roles, reader=None, cohort=None, policy=None):
    reader = current_input_reader(reader or DataReader(unc_only=False))
    value = overflow_method.checked_metadata(reader, bound, limit=1024 * 1024)
    from ..mass.overflow_phase_split import POLICY
    if policy == POLICY:
        if (type(value) is not dict or value.get("format") != REQUEST_FORMAT
                or not isinstance(value.get("scale_reviews"), dict) or not value["scale_reviews"]):
            raise ValueError("Choose the reviewed overflow sources for the phase split")
        analysis = roles["fit"] + roles["heldout"] + roles["descriptive"]
        overflow_method.validate_roles(roles, analysis, cohort=cohort)
        if not set(value["scale_reviews"]) <= set(analysis):
            raise ValueError("An overflow source names a run outside the selected series")
        for source in value["scale_reviews"].values():
            overflow_method.checked_metadata(reader, source, limit=16 * 1024 * 1024)
        return value
    if policy is not None:
        raise ValueError("The selected overflow phase-split policy is unsupported")
    if (type(value) is not dict or set(value) != {"format", "method", "scale_reviews"}
            or value["format"] != REQUEST_FORMAT
            or set(value["scale_reviews"]) != set(overflow_method.USABLE_RUNS)):
        raise ValueError("Choose explicit recorded overflow method and all reviewed scale sources")
    analysis = roles["fit"] + roles["heldout"] + roles["descriptive"]
    overflow_method.validate_roles(roles, analysis, cohort=cohort)
    overflow_method.checked_method(reader, value["method"])
    for source in value["scale_reviews"].values():
        overflow_method.checked_metadata(reader, source, limit=16 * 1024 * 1024)
    return value


def preparation_budget(job, plan, reference_budget, *, reader):
    """Original F serial-replay workspace formula on this local P1 metadata."""
    from ..phase.probability_storage import parse_request
    largest_pixels = count = 0
    for run in plan["runs"]["fit"]:
        result = reader.json(job["phase_sources"][run]["completion"])
        if (result.get("execution_mode") != "local" or result.get("complete") is not True
                or result.get("run") != run or not {"configuration", "phase1_completion"} <= result.keys()):
            raise ValueError("The overflow target needs this route's completed local phase inputs; prepare the phase inputs again")
        config = reader.json(result["configuration"])
        core = reader.json(result["phase1_completion"])
        if (config.get("run") != run or parse_request(config) is None
                or core.get("ordered_case_ids") != config["ordered_case_ids"]
                or len(core["frames"]) != len(config["frames"])):
            raise ValueError("The overflow target needs completed phase inputs with lossless gas probabilities; prepare the phase inputs again")
        for pin in core["frames"]:
            record = reader.json(pin)
            shape = record["native"]["full"]["phase_shape_hw"]
            if (record.get("run") != run or len(shape) != 2
                    or any(type(n) is not int or n < 1 for n in shape) or "probabilities" not in record):
                raise ValueError("F replay needs actual original grid and probability products")
            largest_pixels = max(largest_pixels, shape[0] * shape[1])
            count += 1
    decoded = reference_budget["decoded_statistics_bytes"]
    products = 4 * decoded + count * 8 * 1024**2 + 32 * 1024**2
    return dict(products_estimated_bytes=products,
        transient_estimated_bytes=2 * products,
        workspace_estimated_bytes=2048 * largest_pixels + 64 * decoded + 512 * 1024**2,
        model="Estimate from original F serial-replay formula; no measured RAM cap",
        largest_native_pixels=largest_pixels, raw_decoded=False)


def prepare(job, plans, root, *, on_progress=None, reader=None):
    """Derive current run targets, then adjusted rows without repeating RAW/P1."""
    from .local_overflow_sources import LocalOverflowRun
    reader = current_input_reader(reader or DataReader(unc_only=False))
    cohort = job.get("calibration_cohort")
    if cohort is not None:
        if cohort.get("series_manifest") != job["series_manifest"]:
            raise ValueError("The overflow calibration cohort belongs to another series input")
        reader.path(cohort["series_manifest"])
    policy = job.get("overflow_phase_split_policy")
    request = read_request(job["overflow"], roles=plans["refit"]["runs"], reader=reader, cohort=cohort, policy=policy)
    if policy is not None:
        from .local_overflow_phase_split import prepare as prepare_phase_split
        return prepare_phase_split(job, plans, root, request, reader=reader, on_progress=on_progress)
    roles = plans["refit"]["runs"]
    volumes = {run: [] for run in overflow_method.USABLE_RUNS}
    bindings = {}
    def progress(run, step, **event):
        if on_progress:
            on_progress(dict(run=run, step=step, **event))
    for run in roles["fit"]:
        if run not in overflow_method.USABLE_RUNS:
            continue
        progress(run, "overflow_current_volume")
        saved = LocalOverflowRun(job["phase_job"], run, reader=reader)
        for spec, frame in saved.frames(window=(1., 4.)):
            volumes[run].append(dict(case_id=spec["case_id"], elapsed_h=spec["elapsed_h"],
                baseline_gas_volume_m3=overflow_statistics.baseline_volume(frame)))
            del frame
        saved.finish()
        bindings[run] = deepcopy(saved.bindings)
        del saved
    levels = overflow_levels.read_measurements(reader, request["method"], request["scale_reviews"],
        volumes, roles=roles, analysis_runs=roles["fit"] + roles["heldout"] + roles["descriptive"], cohort=cohort)
    # No missing held-out P1 is claimed to establish an own or production level.
    pending = roles["heldout"] + roles["descriptive"]
    levels["runs"] = {run: levels["runs"][run] for run in roles["fit"]}
    levels["pending_nonfit_runs"] = list(pending)
    levels["analysis_run_order"] = list(roles["fit"])
    entries, reports = {}, []
    folder = Path(root) / "overflow_preparation"
    folder.mkdir(exist_ok=False)
    for run in roles["fit"]:
        progress(run, "overflow_derive")
        saved = LocalOverflowRun(job["phase_job"], run, reader=reader)
        original = {item["frame_index"]: item for item in plans["refit"]["statistics"][run]}
        destination = folder / run
        destination.mkdir()
        for spec, frame in saved.frames(frame_indices=set(original)):
            entry = original[spec["frame_index"]]
            if frame.statistics_descriptor != entry["statistics"]:
                raise ValueError("Overflow derivation identifies another original fit statistics row")
            level = levels["runs"][run]
            result = overflow_statistics.derive_statistics(frame, level=level["level"],
                uncapped_level=level["uncapped_level"])
            stage = destination / spec["case_id"]
            stage.mkdir()
            path = stage / "overflow_statistics.json.gz"
            with path.open("xb") as stream:
                stream.write(gzip.compress(json.dumps(_jsonable(result["statistics"]), allow_nan=False,
                    separators=(",", ":")).encode(), mtime=0))
            distribution = {key: value for key, value in result["distribution"].items()
                            if key != "saturation_free_gas"}
            reports.append(dict(run=run, case_id=spec["case_id"], report=result["report"],
                frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
                gas_contribution=save_gas_contributions(stage, frame, result),
                distribution=distribution, original_statistics=frame.statistics_descriptor,
                probability_roles=frame.probability_roles))
            del frame, result
        saved.finish()
        bindings[run] = deepcopy(saved.bindings)
        entries[run] = statistics_entries_f(destination)
        if [(row["frame_index"], row["selected_target_h"]) for row in entries[run]] != [
                (row["frame_index"], row["selected_target_h"]) for row in plans["refit"]["statistics"][run]]:
            raise ValueError("Overflow preparation lost the original selected acquisition order")
        del saved
    from ..mass.gas_contribution_review import contribution_scales
    preparation = _write_json(folder / "preparation.json", dict(format=PREPARATION_FORMAT,
        gas_contribution_review=contribution_scales(reports),
        inputs=job["overflow"], phase_sources=bindings, levels=levels, frames=reports, statistics=entries,
        raw_decoded=False, phase_inference_run=False, calibration_adopted=False,
        production_admitted=False, pending_nonfit_runs=pending))
    result = {}
    for run, plan in plans.items():
        result[run] = dict(deepcopy(plan), format=PLAN_FORMAT,
            statistics={key: deepcopy(entries[key]) for key in plan["statistics"]},
            variant_f={"enabled": True}, overflow_preparation=preparation)
    return result


def statistics_entries_f(folder):
    from .refit_runner import read_statistics
    reader = DataReader(unc_only=False)
    entries = []
    for path in Path(folder).glob("*/overflow_statistics.json.gz"):
        bound = asset(path)
        row = read_statistics(reader, bound)
        entries.append(dict(case_id=path.parent.name, frame_index=row["frame_index"],
            selected_target_h=row["selected_target_h"], statistics=bound))
    return sorted(entries, key=lambda row: row["frame_index"])


def run_fit(task, plan, output, *, on_progress=None):
    """Fit the adjusted gas/water statistics, preserving the ordinary OFF gate."""
    from .refit_runner import read_statistics, fold_of
    reader = DataReader(unc_only=False)
    if plan.get("format") != PLAN_FORMAT or plan.get("variant_f") != {"enabled": True}:
        raise ValueError("An explicitly prepared local overflow fit plan is required")
    if reader.json(task["configuration"]) != plan:
        raise ValueError("The overflow fit plan differs from its exact authored bytes")
    preparation = reader.json(plan["overflow_preparation"])
    if (preparation.get("format") != PREPARATION_FORMAT or preparation.get("production_admitted") is not False
            or {key: value for key, value in preparation.get("statistics", {}).items()
                if key in plan["statistics"]} != plan["statistics"]):
        raise ValueError("The exact local overflow preparation is required")
    rows = []
    for run in sorted(plan["statistics"]):
        for entry in plan["statistics"][run]:
            row = read_statistics(reader, entry["statistics"])
            if row["run"] != run or row["frame_index"] != entry["frame_index"]:
                raise ValueError("Overflow statistics differ from their bound run/frame")
            rows.append(row)
    rules = n1_refit.RefitRules.from_plan(plan)
    fold = fold_of(task["run"])
    def progress(**event):
        if on_progress:
            on_progress(dict(run=task["run"], **event))
    result = overflow_refit.fit_adjusted_rows(rows, rules, excluded_run=fold, progress=progress)
    fit = result["numeric_fit"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {"fit.json": _write_json(output / "fit.json", result)}
    frozen = None
    if fold is None and not fit["stop_required"]:
        document = dict(format=FROZEN_FORMAT, frozen=True, frozen_utc=datetime.now(timezone.utc).isoformat(),
            parameters=fit["parameters"], calibration_runs=fit["calibration_runs"],
            heldout_runs=plan["runs"]["heldout"], descriptive_runs=plan["runs"]["descriptive"],
            excluded_runs=plan["runs"]["excluded"], anchors_by_run=fit["anchors_by_run"],
            rules=plan["rules"], baseline=plan["baseline"], code_commit=task["commit"],
            configuration=task["configuration"], overflow_preparation=plan["overflow_preparation"],
            variant_f=True, free_gas_target=("reviewed own measurements; pressure-matched estimates are labelled"
                if plan.get("overflow_phase_split_policy") else "reviewed overflow with fit-only pooled shrinkage"),
            phase_labels_modified=False, gas_branch_fitted=False, map_mass_rescaling=False,
            calibration_adopted=False, production_admitted=False,
            # The chemistry the plan was fitted with, as the global refit records it.
            **({"chemistry_model": plan["chemistry_model"]} if "chemistry_model" in plan else {}),
            pending_nonfit_runs=preparation["pending_nonfit_runs"])
        if plan.get("overflow_phase_split_policy"):
            document["overflow_phase_split_policy"] = plan["overflow_phase_split_policy"]
            status = {}
            for row in preparation["frames"]:
                target = row.get("phase_split_target") or {}
                kind = target.get("kind", "model_phase_split")
                key = (row["run"], kind)
                item = status.setdefault(key, dict(run=row["run"], kind=kind, image_count=0,
                    donor_runs=[], pressure_differences_bar=[]))
                item["image_count"] += 1
                item["donor_runs"] = sorted(set(item["donor_runs"]) | {
                    donor["run"] for donor in target.get("donors", [])})
                if target.get("pressure_difference_bar") is not None:
                    item["pressure_differences_bar"].append(target["pressure_difference_bar"])
            document["overflow_phase_split_status"] = []
            for item in status.values():
                gaps = item.pop("pressure_differences_bar")
                item["pressure_difference_bar_range"] = [min(gaps), max(gaps)] if gaps else None
                document["overflow_phase_split_status"].append(item)
        if plan.get("calibration_scope") == "individual":
            document.update(calibration_scope="individual", calibrated_run=plan["calibrated_run"],
                            source_roles=deepcopy(plan["source_roles"]))
        if task.get("reference_without_overflow") is not None:
            reader.json(task["reference_without_overflow"])
            document["reference_without_overflow"] = deepcopy(task["reference_without_overflow"])
        frozen = _write_json(output / "frozen_calibration.json", document)
        files["frozen_calibration.json"] = frozen
    elif fold is not None and not fit["stop_required"]:
        files["predictions.json"] = _write_json(output / "predictions.json", dict(
            heldout_fit_run=fold, parameters=fit["parameters"], predictions=result["heldout_predictions"]
            if "heldout_predictions" in result else n1_refit.predict_run(rows, fit, fold)))
    return dict(format="darsia-local-overflow-refit-result-v1", complete=True, run=task["run"],
        pipeline="local-overflow-calibration-refit", parameters=fit["parameters"],
        stop_required=fit["stop_required"], reason=fit.get("reason"), files=files,
        frozen_calibration=frozen, calibration_adopted=False, production_admitted=False)
