"""Bind the active saved RAW selection to its retained calibration ancestry.

This reads selection and calibration metadata only. Map adoption may change
series.toml without changing a single acquisition; the new record binds that
actual current manifest and never rewrites the historical selection record.
"""
from copy import deepcopy
from pathlib import Path
import ntpath
import tomllib

from ..mass.n1_pipeline import DataReader, asset, _write_json

FORMAT = "ffac-local-p1-selection-data-v1"
HISTORY_POLICY = "baseline plus all selected acquisitions in original chronological order"


def _saved_target(row, saved):
    """Use the camera selection's recorded target match, never invent one."""
    import math
    actual, requested = saved.get("target_hours"), saved.get("requested_target_hours")
    tolerance = saved.get("tolerance_seconds")
    if (not isinstance(actual, list) or not isinstance(requested, list) or len(actual) != len(requested)
            or type(tolerance) not in {int, float} or not math.isfinite(tolerance) or tolerance < 0):
        return None
    matches = [nominal for time, nominal in zip(actual, requested)
               if type(time) in {int, float} and type(nominal) in {int, float}
               and math.isfinite(time) and math.isfinite(nominal)
               and abs(time - row["elapsed_h"]) <= 1e-9
               and abs(time - nominal) * 3600 <= tolerance + 1e-6 and nominal >= 0]
    return float(matches[0]) if len(matches) == 1 else None


def same_preparation_selection(previous, current):
    """Mass-role edits reuse identical physical inputs and eligible statistics."""
    def basis(value):
        value = deepcopy(value)
        value.pop("mass_fit_runs", None)
        value.pop("decision", None)
        authority = value.get("input_authority", {})
        authority.pop("current_series", None)
        for row in value.get("runs", []):
            row.pop("calibration_role", None)
        return {key: value.get(key) for key in ("input_authority", "runs", "counts", "optical_selection")}
    return basis(previous) == basis(current)


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _located(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def method_ancestry(series_root, reader):
    """The active method's retained calibration ancestry and its included runs.

    The calibration and the production selection prepare phase one with this
    one method: its global fit, retained phase plan and the registered runs
    that the calibration does not exclude. ``reader`` is the current input
    reader. Returns the bound series manifest and its document, the
    calibration, fit and plan documents with the fit and plan bindings, and
    the included runs in registered order.
    """
    root = Path(series_root)
    current = asset(root / "series.toml")
    manifest = tomllib.loads(reader.path(current).read_text(encoding="utf-8-sig"))
    method = manifest["method"][manifest["method"]["active"]]
    from ..series_method import calibration_ancestry_reference
    reference = calibration_ancestry_reference(method)
    if reference is None:
        raise ValueError("The saved calibration image selection has no retained calibration ancestry.")
    calibration = reader.json(asset(reader.resolve_path(_located(root, reference))))
    fit_binding = calibration["calibration_files"]["global"]
    fit = reader.json(fit_binding)
    plan_binding = fit["selected_phase_gate"]["phase_plan"]
    plan = reader.json(plan_binding)
    excluded = set(calibration.get("excluded_runs", ()))
    runs = [run for run in manifest["runs"]["ids"] if run not in excluded]
    return dict(current=current, manifest=manifest, calibration=calibration, fit_binding=fit_binding,
                fit=fit, plan_binding=plan_binding, plan=plan, runs=runs)


def current_calibration_selection(series_root, *, actor, reason, reader=None):
    """Prepare all included runs; retain the separate mass-calibration cohort."""
    if not actor or not reason.strip():
        raise ValueError("The current selection requires its recorded actor and reason")
    root = Path(series_root)
    from .current_inputs import current_input_reader
    reader = current_input_reader(reader)
    from ..storage_paths import storage_path_remapper
    remap = storage_path_remapper()
    path_key = lambda value: _key(remap(value))
    ancestry = method_ancestry(root, reader)
    current, manifest, calibration = ancestry["current"], ancestry["manifest"], ancestry["calibration"]
    fit_binding, fit = ancestry["fit_binding"], ancestry["fit"]
    plan_binding, plan, runs = ancestry["plan_binding"], ancestry["plan"], ancestry["runs"]
    wanted = fit["calibration_runs"]
    if len(set(wanted)) != len(wanted) or not set(wanted) <= set(runs):
        raise ValueError("The retained calibration cohort differs from the registered runs")
    from ..shared_calibration_cohort import explicit_mass_cohort
    chosen_roles = explicit_mass_cohort(manifest)
    if chosen_roles is not None:
        wanted = chosen_roles["fit"]
    decisions = {(row["run"], path_key(row["source_binding"]["raw_path"]), row["elapsed_h"]): row
                 for row in fit["frame_decisions"]}
    if len(decisions) != len(fit["frame_decisions"]):
        raise ValueError("The retained calibration repeats an acquisition decision")
    records = []
    protocol_sources = {}
    totals = dict(unique_acquisitions=0, selected_frames=0, baselines=0, fit_candidates=0, earlier_history=0)
    for run in runs:
        source = plan["runs"][run]
        inventory = reader.json(source["source_inventory"])
        original_bound = fit["selected_phase_gate"]["run_manifests"][run]
        original = reader.json(original_bound)
        saved = manifest["image_selection"]["runs"][run]
        selection_bound = dict(saved["selection_manifest"])
        selection_bound["path"] = str(_located(root, selection_bound["path"]))
        selection = reader.json(selection_bound)
        chosen = saved["selected"]
        identity = lambda row: (path_key(row["source_path"]), row["elapsed_h"])
        if [identity(row) for row in chosen] != [
                (path_key(row["chosen_image"]["source_path"]), row["data_time_h"])
                for row in selection["frames"]]:
            raise ValueError("The saved RAW choices differ from their selection manifest: " + run)
        sources = {row["source_sha256"]: row for row in inventory["sources"]}
        by_raw = {(path_key(row.get("v5_raw_path") or row.get("raw_path")), row["elapsed_h"]): row
                  for row in sources.values() if row.get("selected") or row.get("baseline")}
        if set(by_raw) != {identity(row) for row in chosen}:
            raise ValueError("The saved choices changed the original acquisition/history cohort: " + run)
        acquisitions = []
        for position, (row, camera) in enumerate(zip(chosen, selection["frames"])):
            original_source = by_raw[identity(row)]
            prior = sources.get(original_source["previous_source_sha256"])
            raw_sha = original_source.get("v5_raw_sha256") or original_source.get("raw_sha256")
            previous_sha = (prior.get("v5_raw_sha256") or prior.get("raw_sha256")) if prior else None
            # Eligibility is retained for later individual fits and holdout
            # changes. Only the refit plan's fit roles enter its objective.
            decision = decisions.get((run, *identity(row)))
            target = decision["selected_target_h"] if decision else _saved_target(row, saved)
            if decision is not None and decision["source_binding"]["raw_sha256"] != raw_sha:
                raise ValueError("The retained fit and current RAW identity differ: " + run)
            acquisitions.append(dict(current_selection_position=position,
                current_frame_index=row["frame_index"], image_id=row["image_id"],
                raw_path=row["source_path"], raw_sha256=raw_sha,
                recorded_raw_size_bytes=(original_source.get("quality") or {}).get("raw_size_bytes"),
                elapsed_h=row["elapsed_h"], camera_datetime=camera.get("camera_time"),
                baseline=bool(original_source.get("baseline")), previous_raw_sha256=previous_sha,
                source_identity_provenance=dict(jpeg_source_sha256=original_source["source_sha256"],
                    previous_jpeg_source_sha256=original_source["previous_source_sha256"],
                    inventory=deepcopy(source["source_inventory"])),
                fit_candidate=target is not None, fit_stage=decision["stage"] if decision else None,
                nominal_fit_target_h=target,
                eligibility_source=("retained_mass_frame_decision" if decision else
                    "saved_camera_target_match" if target is not None else "no_recorded_eligible_target"),
                previous_fit_included=decision["included"] if decision else None))
        baselines = [row for row in acquisitions if row["baseline"]]
        frames = [row for row in acquisitions if not row["baseline"]]
        if len(baselines) != 1 or not frames:
            raise ValueError("One baseline and the complete selected history are required: " + run)
        baseline = baselines[0]
        previous = baseline["raw_sha256"]
        if baseline["previous_raw_sha256"] is not None:
            raise ValueError("The initial baseline has a predecessor: " + run)
        for frame in frames:
            if frame["previous_raw_sha256"] != previous:
                raise ValueError("The original RAW predecessor chain changed: " + run)
            previous = frame["raw_sha256"]
        if [(path_key(row["raw_path"]), row["elapsed_h"]) for row in frames] != [
                (path_key(row["raw_path"]), row["elapsed_h"]) for row in original["frame_sources"]]:
            raise ValueError("The original selected order changed: " + run)
        counts = dict(unique_acquisitions=len(acquisitions), selected_frames=len(frames), baselines=1,
            fit_candidates=sum(row["fit_candidate"] for row in frames),
            earlier_history=sum(not row["fit_candidate"] for row in frames))
        for key, count in counts.items():
            totals[key] += count
        role = ("fit" if run in wanted else "heldout" if run in (chosen_roles["heldout"] if chosen_roles else calibration.get("heldout_runs", ()))
                else "descriptive" if run in calibration.get("descriptive_runs", ()) else "control")
        record = dict(run=run, calibration_role=role, counts=counts, current_selection_manifest=selection_bound,
            original_selection_manifest=deepcopy(original_bound), original_inventory=deepcopy(source["source_inventory"]),
            baseline=baseline, initial_previous_raw_sha256=baseline["raw_sha256"], selected_frames=frames)
        from .current_run_protocols import freeze_run_protocols
        record["current_protocols"] = freeze_run_protocols(root, manifest, record,
            reader=reader, source_cache=protocol_sources)
        records.append(record)
    unavailable = [row["run"] for row in records if row["run"] in wanted and not row["counts"]["fit_candidates"]]
    if unavailable:
        raise ValueError("Selected mass-fit runs have no recorded eligible calibration images: "
                         + ", ".join(unavailable) + ". Choose their calibration image targets first.")
    reader.path(current)
    return dict(format=FORMAT, decision=dict(actor=actor, reason=reason),
        history_policy=HISTORY_POLICY,
        input_authority=dict(global_fit=deepcopy(fit_binding), phase_plan=deepcopy(plan_binding), current_series=current),
        counts=totals, mass_fit_runs=[run for run in runs if run in wanted], runs=records,
        optical_selection=deepcopy(manifest.get("image_selection", {}).get("color_paths", {})))


def save_current_calibration_selection(series_root, output, *, actor, reason, reader=None, model_binding=None):
    """Save a new current binding; no RAW file is opened or historical record edited."""
    record = current_calibration_selection(series_root, actor=actor, reason=reason, reader=reader)
    if model_binding is not None:
        record["input_authority"]["selected_model"] = deepcopy(model_binding)
    return _write_json(Path(output), record)
