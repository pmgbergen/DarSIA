"""Apply per-image overflow phase references to existing completed P1 products."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

from ..mass import overflow_measurements, overflow_statistics
from ..mass.overflow_phase_split import (
    POLICY, TargetUnavailable, own_target, phase_split_state, pressure_matched_target,
)
from ..mass.n1_pipeline import _jsonable, _write_json


def _state(saved, frame):
    value = frame.prepared["frame"]
    adopted = saved.physical.context.provenance["adopted_revision"]
    return phase_split_state(run=value.run_id, elapsed_h=value.elapsed_h, pressure_bar=value.pressure_bar,
        temperature_c=value.temperature_c, injected_mass_g=value.nominal_injected_mass_g,
        injection_stages=saved.config["statistics"]["dose"]["injection_stages"],
        adopted_revision_sha256=adopted["sha256"], recipe=value.recipe,
        math=frame.pipeline.math, tables=frame.prepared["response"]["physics"])


def _model_result(frame, reason):
    from ..mass.gas_contribution_review import capture_gas_contributions
    row = deepcopy(frame.prepared["statistics"])
    row["contract"] = overflow_statistics.STATISTICS_FORMAT
    row["overflow_steering"] = dict(format="darsia-overflow-statistics-derivation-v1",
        policy=POLICY, case_id=frame.case_id, original_statistics=deepcopy(frame.statistics_descriptor),
        kind="model_phase_split", estimated=False, overflow_reference_available=False, reason=str(reason),
        labels_modified=False, physical_geometry_modified=False, excluded_estimates_modified=False)
    gas = row["main_gas_mass_g"]
    rho = frame.pipeline.math.pixels(frame.prepared["response"]["physics"].density_gaseous_co2)
    density = rho*frame.population.baseline_saturation
    return dict(statistics=row, distribution={},
        gas_contribution_maps=capture_gas_contributions(frame, density, density), report=dict(
        baseline_gas_mass_g=gas, gas_mass_g=gas, gas_change_g=0.,
        gas_change_share=None if gas == 0 else 0., overflow_reference_available=False,
        status="Phase model — no overflow reference available at this time", reason=str(reason)))


def prepare(job, plans, root, request, *, reader, on_progress=None):
    from .local_overflow_sources import LocalOverflowRun
    from .local_overflow_refit import PREPARATION_FORMAT, PLAN_FORMAT, statistics_entries_f
    roles = plans["refit"]["runs"]
    curves, unavailable = {}, {}
    for run in roles["fit"]:
        source = request["scale_reviews"].get(run)
        if source is None:
            unavailable[run] = "This run has no reviewed overflow measurement."
            continue
        document = reader.json(source)
        if document.get("metadata", {}).get("run") != run:
            raise ValueError("The reviewed overflow source belongs to another run")
        checked = overflow_measurements.reviewed_curve(document, run=run)
        if checked["usable"]:
            curves[run] = dict(curve=checked["curve"], source=dict(
                scale_review=deepcopy(source), feed="reviewed feed C", qa=deepcopy(checked["qa"])))
        else:
            unavailable[run] = "; ".join(checked["reasons"])
    folder = Path(root) / "overflow_preparation"
    folder.mkdir(exist_ok=False)
    entries, reports, bindings, donors = {}, [], {}, []
    deferred = {}

    def save(run, spec, frame, result, target=None):
        from ..mass.gas_contribution_review import save_gas_contributions
        destination = folder / run / spec["case_id"]
        destination.mkdir(parents=True)
        with (destination / "overflow_statistics.json.gz").open("xb") as stream:
            stream.write(gzip.compress(json.dumps(_jsonable(result["statistics"]), allow_nan=False,
                separators=(",", ":")).encode(), mtime=0))
        distribution = {key: value for key, value in result["distribution"].items()
                        if key != "saturation_free_gas"}
        reports.append(dict(run=run, case_id=spec["case_id"], report=result["report"],
            frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
            gas_contribution=save_gas_contributions(destination, frame, result, target),
            phase_split_target=deepcopy(target), distribution=distribution,
            original_statistics=deepcopy(frame.statistics_descriptor), probability_roles=deepcopy(frame.probability_roles)))

    # Measured runs are processed first. Their gas-amount fractions can then
    # support explicit estimates without reading RAW or re-running inference.
    for run in roles["fit"]:
        if run not in curves:
            deferred[run] = None
            continue
        if on_progress:
            on_progress(dict(run=run, step="overflow_own_measurement"))
        saved = LocalOverflowRun(job["phase_job"], run, reader=reader)
        selected = {item["frame_index"]: item for item in plans["refit"]["statistics"][run]}
        deferred[run] = set()
        for spec, frame in saved.frames(frame_indices=set(selected)):
            if frame.statistics_descriptor != selected[spec["frame_index"]]["statistics"]:
                raise ValueError("Overflow phase split identifies another original fit statistics row")
            try:
                target = own_target(curves[run]["curve"], run=run, elapsed_h=spec["elapsed_h"], source=curves[run]["source"])
            except TargetUnavailable:
                deferred[run].add(spec["frame_index"])
                continue
            result = overflow_statistics.derive_phase_split_statistics(frame, target)
            state = _state(saved, frame)
            if state["injected_mass_g"] is not None and state["injected_mass_g"] > 0:
                donors.append(dict(state, **{key: deepcopy(target[key]) for key in
                    ("kind", "estimated", "source", "target_volume_m3")}))
            save(run, spec, frame, result, target)
        saved.finish()
        bindings[run] = deepcopy(saved.bindings)

    for run, pending in deferred.items():
        if pending == set():
            continue
        if on_progress:
            on_progress(dict(run=run, step="overflow_pressure_estimate"))
        saved = LocalOverflowRun(job["phase_job"], run, reader=reader)
        selected = {item["frame_index"]: item for item in plans["refit"]["statistics"][run]}
        for spec, frame in saved.frames(frame_indices=set(selected) if pending is None else pending):
            if frame.statistics_descriptor != selected[spec["frame_index"]]["statistics"]:
                raise ValueError("Overflow phase split identifies another original fit statistics row")
            try:
                target = pressure_matched_target(_state(saved, frame), donors, fit_runs=roles["fit"])
            except (TargetUnavailable, ValueError) as error:
                # Outside all measurement coverage is not a zero observation.
                # Keep the original model split, retaining the missing reference.
                save(run, spec, frame, _model_result(frame, error))
            else:
                save(run, spec, frame, overflow_statistics.derive_phase_split_statistics(frame, target), target)
        saved.finish()
        bindings[run] = deepcopy(saved.bindings)

    for run in roles["fit"]:
        entries[run] = statistics_entries_f(folder / run)
        if [(row["frame_index"], row["selected_target_h"]) for row in entries[run]] != [
                (row["frame_index"], row["selected_target_h"]) for row in plans["refit"]["statistics"][run]]:
            raise ValueError("Overflow phase preparation lost the original selected acquisition order")
    from ..mass.gas_contribution_review import contribution_scales
    preparation = _write_json(folder / "preparation.json", dict(format=PREPARATION_FORMAT,
        gas_contribution_review=contribution_scales(reports),
        policy=POLICY, inputs=job["overflow"], phase_sources=bindings, frames=reports, statistics=entries,
        source_status={run: "own reviewed measurement" if run in curves else "estimated when comparable measurements exist"
                       for run in roles["fit"]}, unavailable_own_sources=unavailable,
        raw_decoded=False, phase_inference_run=False, calibration_adopted=False,
        production_admitted=False, pending_nonfit_runs=roles["heldout"]+roles["descriptive"]))
    return {run: dict(deepcopy(plan), format=PLAN_FORMAT,
        statistics={key: deepcopy(entries[key]) for key in plan["statistics"]},
        variant_f={"enabled": True}, overflow_preparation=preparation, overflow_phase_split_policy=POLICY)
        for run, plan in plans.items()}
