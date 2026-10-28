"""Bind the saved production image grid to the active method's ancestry.

The production grid (Setup > Image selection) names, for every run, the
photographs matched to one shared clock of target times. This record has the
form of the current calibration selection, so the existing current-build route
prepares phase one for those photographs with the same retained method: its
phase plan, statistics recipe and the current protocols. No production frame
enters a mass fit.

Each run's reference is the photograph its completed Build was made from, as
in calibration, because a run's phase history starts at the Build's baseline.
A different grid baseline is recorded in the run's record and left out.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ntpath
from pathlib import Path

from ..mass.n1_pipeline import asset

PURPOSE = "production"


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def raw_identities(paths, *, threads=4, on_progress=None):
    """Each RAW photograph's SHA-256 and size, read once with parallel reads.

    Returns a mapping from each path, as given, to its descriptor.
    ``on_progress`` receives the count of photographs read so far.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError("Choose a positive number of reading threads")
    paths = list(dict.fromkeys(str(path) for path in paths))
    result = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for path, bound in zip(paths, pool.map(asset, paths)):
            result[path] = bound
            if on_progress:
                on_progress(dict(state="raw_identity", completed=len(result), total=len(paths)))
    return result


def _acquisition(position, row, raw, *, baseline, previous, target):
    """One acquisition row in the calibration selection's form, outside any fit."""
    return dict(current_selection_position=position, current_frame_index=row["frame_index"],
        image_id=row["image_id"], raw_path=raw["path"], raw_sha256=raw["sha256"],
        recorded_raw_size_bytes=raw.get("size_bytes"), elapsed_h=row["elapsed_h"],
        camera_datetime=row["camera_time"], baseline=baseline, previous_raw_sha256=previous,
        fit_candidate=False, fit_stage=None, nominal_fit_target_h=target,
        eligibility_source="completed_build_baseline" if baseline else "production_grid_target",
        previous_fit_included=None)


def _run_frames(run, metadata, resolved, baseline, remap):
    """The Build baseline's protocol row, the grid's target rows and a left-out grid baseline."""
    key = lambda path: _key(remap(str(path)))
    found = [row for row in metadata.frames if key(row["source_path"]) == key(baseline["path"])]
    if len(found) != 1:
        raise ValueError(run + ": the completed Build's baseline photograph is not one image of this run's "
                         "imaging protocol")
    reference = found[0]
    targets = {key(match["source_path"]): float(match["target_hour"])
               for match in resolved["target_matches"] if match.get("selected")}
    frames, left_out = [], None
    for row in resolved["selected"]:
        if key(row["source_path"]) == key(baseline["path"]):
            continue
        if key(row["source_path"]) in targets:
            if not (row["elapsed_h"] > reference["elapsed_h"]
                    and row["frame_index"] > reference["frame_index"]):
                raise ValueError(run + ": a production photograph precedes the completed Build's baseline")
            frames.append((row, targets[key(row["source_path"])]))
        elif key(row["source_path"]) == key(resolved["baseline_path"]):
            left_out = dict(raw_path=row["source_path"], image_id=row["image_id"],
                            elapsed_h=row["elapsed_h"], reason="another photograph than the Build's baseline")
        else:
            raise ValueError(run + ": the saved production grid holds a photograph without a target time")
    if not frames:
        raise ValueError(run + ": the saved production grid has no photograph after the Build's baseline")
    return reference, frames, left_out


def current_production_selection(series_root, baselines, *, actor, reason, reader=None,
                                  threads=4, on_progress=None):
    """Prepare every included run's production photographs after its Build baseline.

    ``baselines`` maps each included run to the RAW descriptor its completed
    Build was made from (the Build request's baseline). Each other photograph's
    SHA-256 is read once here. Returns the selection record that current-build
    preparation consumes; nothing is written.
    """
    if not actor or not reason.strip():
        raise ValueError("The current selection requires its recorded actor and reason")
    from .current_calibration_selection import FORMAT, HISTORY_POLICY, method_ancestry
    from .current_inputs import current_input_reader
    from .current_run_protocols import freeze_run_protocols
    from ..production_image_grid import production_grid_targets, resolve_production_selection
    from ..series_image_selection import load_run_protocol_metadata
    from ..storage_paths import storage_path_remapper
    root = Path(series_root)
    reader = current_input_reader(reader)
    remap = storage_path_remapper()
    ancestry = method_ancestry(root, reader)
    missing = [run for run in ancestry["runs"] if run not in baselines]
    if missing:
        raise ValueError("The completed Build has no baseline for " + ", ".join(missing))
    plans = {}
    for run in ancestry["runs"]:
        metadata = load_run_protocol_metadata(root, run)
        resolved = resolve_production_selection(root, run, metadata)
        resolved = dict(resolved, **production_grid_targets(root, resolved))
        plans[run] = (resolved, *_run_frames(run, metadata, resolved, baselines[run], remap))
    identities = raw_identities([row["source_path"] for _resolved, _reference, frames, _left in plans.values()
                                 for row, _target in frames], threads=threads, on_progress=on_progress)
    records, protocol_sources = [], {}
    totals = dict(unique_acquisitions=0, selected_frames=0, baselines=0, fit_candidates=0, earlier_history=0)
    for run, (resolved, reference, frames, left_out) in plans.items():
        baseline = _acquisition(0, reference, dict(baselines[run]), baseline=True, previous=None, target=None)
        previous, acquisitions = baseline["raw_sha256"], []
        for position, (row, target) in enumerate(frames, start=1):
            raw = identities[str(row["source_path"])]
            acquisitions.append(_acquisition(position, row, raw, baseline=False, previous=previous, target=target))
            previous = raw["sha256"]
        counts = dict(unique_acquisitions=len(acquisitions) + 1, selected_frames=len(acquisitions),
                      baselines=1, fit_candidates=0, earlier_history=len(acquisitions))
        for name, count in counts.items():
            totals[name] += count
        record = dict(run=run, counts=counts, current_selection_manifest=deepcopy(resolved["production_manifest"]),
            baseline=baseline, initial_previous_raw_sha256=baseline["raw_sha256"], selected_frames=acquisitions,
            production_grid_baseline=left_out)
        record["current_protocols"] = freeze_run_protocols(root, ancestry["manifest"], record,
            reader=reader, source_cache=protocol_sources)
        records.append(record)
    reader.path(ancestry["current"])
    return dict(format=FORMAT, purpose=PURPOSE, decision=dict(actor=actor, reason=reason),
        history_policy=HISTORY_POLICY,
        input_authority=dict(global_fit=deepcopy(ancestry["fit_binding"]),
            phase_plan=deepcopy(ancestry["plan_binding"]), current_series=ancestry["current"]),
        counts=totals, mass_fit_runs=[], runs=records,
        optical_selection=deepcopy(ancestry["manifest"].get("image_selection", {}).get("color_paths", {})))
