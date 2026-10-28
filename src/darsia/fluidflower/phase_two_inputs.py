"""Collect phase two's source inputs from finished phase-one jobs.

Mass from finished phase maps (phase two) seals each run from its genuine
phase-one completion, the fitted mass calibration and the run's dilution
anchor. This module writes the source manifest that phase-two preparation
reads (``darsia-analysis-phase2-inputs-v1``): one row per registered run, in
the registered order, binding the run's published phase-one result and anchor.

Each completion is verified by the original phase-one verifier against its
job's commit. The anchor follows the calibration's own rule: a fit run keeps
the anchor its frozen calibration was fitted with; a held-out or descriptive
run takes the frame nearest the calibration's anchor time in its own
phase-one statistics. Nothing is sealed, adopted or started here; phase-two
preparation repeats every check.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

FORMAT = "darsia-analysis-phase2-inputs-v1"
PHASE_ONE = "native-phase-statistics"


def registered_runs(series_root):
    """The bound series manifest and its registered run order."""
    from .pipeline.alias_job import _manifest
    from .pipeline.job_queue import descriptor
    manifest = descriptor(Path(series_root) / "series.toml")
    _, registered, _ = _manifest(manifest)
    runs = (registered.get("runs") or {}).get("ids")
    if (not isinstance(runs, list) or not runs or len(set(runs)) != len(runs)
            or any(not isinstance(run, str) or not run for run in runs)):
        raise ValueError("series.toml [runs] ids must list the distinct registered runs in order")
    return manifest, runs


def published_phase_one(job_roots, runs, *, reader):
    """Each run's published phase-one result in the chosen finished jobs.

    Both the queue and the local Build route publish a run's result at
    ``<job>/results/<run>/completed.json``. Returns ``(found, missing)``:
    ``found`` maps a run to its job and result descriptors, the job's commit
    and the result; ``missing`` lists runs that no chosen job has published.
    """
    from .pipeline.job_queue import descriptor
    jobs = []
    for root in dict.fromkeys(Path(value) for value in job_roots):
        bound = descriptor(root / "job.json")
        jobs.append((root, bound, reader.json(bound)))
    if not jobs:
        raise ValueError("Choose the finished phase-one job")
    found, missing = {}, []
    for run in runs:
        places = [(root, bound, job) for root, bound, job in jobs
                  if (root / "results" / run / "completed.json").is_file()]
        if not places:
            missing.append(run)
            continue
        if len(places) > 1:
            raise ValueError(run + ": two chosen phase-one jobs published this run; choose one")
        root, bound, job = places[0]
        result = descriptor(root / "results" / run / "completed.json")
        value = reader.json(result)
        if (value.get("complete") is not True or value.get("run") != run
                or value.get("pipeline") != PHASE_ONE
                or not {"phase1_completion", "configuration"} <= set(value)):
            raise ValueError(run + ": the published result is not a complete phase-one run")
        found[run] = dict(job=bound, commit=job.get("commit"), result=result, value=value)
    return found, missing


def run_anchor(verified, frozen, *, reader):
    """The run's dilution anchor under the frozen calibration's own rule.

    ``verified`` is the original verifier's result for the run's phase one;
    ``frozen`` is the frozen calibration document. A fit run keeps the anchor
    the calibration was fitted with. A held-out or descriptive run takes its
    frame nearest the anchor time, within the calibration's tolerance, from
    its own verified statistics (as for a held-out application).
    """
    from .mass.n1_refit import RefitRules, anchors_by_run
    from .mass.n1_statistics import CONTRACT
    from .pipeline.common_calibration import phase1_statistics
    from .pipeline.refit_runner import read_statistics
    run = verified["configuration"]["run"]
    if run in frozen["calibration_runs"]:
        anchor = (frozen.get("anchors_by_run") or {}).get(run)
        if not isinstance(anchor, dict):
            raise ValueError(run + ": the frozen calibration has no anchor for this fit run")
    else:
        roles = [role for role in ("heldout_runs", "descriptive_runs") if run in frozen.get(role, ())]
        if run in frozen.get("excluded_runs", ()) or len(roles) != 1:
            raise ValueError(run + ": the calibration gives this run no held-out or descriptive role")
        rules = RefitRules.from_plan(dict(rules=frozen["rules"], runs=dict(
            fit=frozen["calibration_runs"], heldout=frozen.get("heldout_runs", []),
            excluded=frozen.get("excluded_runs", []))))
        rows = []
        for entry, spec in zip(phase1_statistics(verified), verified["configuration"]["frames"]):
            row = read_statistics(reader, entry["statistics"])
            if (row.get("contract") != CONTRACT or row.get("run") != run
                    or row.get("frame_index") != entry["frame_index"]
                    or row.get("elapsed_h") != spec["elapsed_h"]
                    or row.get("source_binding", {}).get("source_sha256") != spec["source"]["source_sha256"]):
                raise ValueError(run + ": phase-one statistics differ from their verified acquisition")
            rows.append(row)
        anchor = anchors_by_run(rows, rules)[run]
    return {key: anchor[key] for key in ("elapsed_h", "affected_pore_volume_m3")}


def overflow_targets(verified, frozen, *, reader):
    """Each production image's overflow target under the calibration's own policy.

    ``verified`` maps each run, in run order, to its verified phase one.
    ``frozen`` is a calibration fitted with each run's own overflow phase
    split. An image covered by its run's own usable reviewed measurement
    takes it; every other image takes the pressure-matched estimate from the
    calibration's fit runs with their own measurement. An image with no
    comparable donor keeps the model's split and its reason ("none").
    """
    from .mass import overflow_measurements
    from .mass.n1_pipeline import _atmosphere, _dose
    from .mass.native_optical_source import load_build_physical_context
    from .mass.overflow_phase_split import (POLICY, TargetUnavailable, frozen_tables, own_target,
                                            phase_split_state, pressure_matched_target)
    from .mass.production_precision import PixelMath
    from .pipeline.current_inputs import current_input_reader
    from .pipeline.setup_image_preparation import image_basis_reader, retained_series_aliases
    current = current_input_reader(reader)
    preparation = current.json(frozen["overflow_preparation"])
    if preparation.get("policy") != POLICY:
        raise ValueError("The chosen calibration was not fitted with each run's own overflow phase split")
    reviews = current.json(preparation["inputs"]).get("scale_reviews") or {}
    math, tables = PixelMath("float32"), frozen_tables()
    cases, targets, donors, pending = {}, {}, [], []
    for run, item in verified.items():
        statistics = item["configuration"]["statistics"]
        operation = statistics["physical_correction"]
        context = load_build_physical_context(operation, reader=image_basis_reader(current, dict(
            source_aliases=retained_series_aliases({"current": operation}, current))))
        adopted = context.provenance["adopted_revision"]["sha256"]
        curve = None
        if run in reviews:
            checked = overflow_measurements.reviewed_curve(current.json(reviews[run]), run=run)
            if checked["usable"]:
                curve = dict(curve=checked["curve"], source=dict(scale_review=deepcopy(reviews[run]),
                             feed="reviewed feed C", qa=deepcopy(checked["qa"])))
        cases[run], targets[run] = [], {}
        for spec in item["configuration"]["frames"]:
            mass_spec = dict(spec["mass_spec"], elapsed_h=spec["elapsed_h"])
            atmosphere = _atmosphere(statistics, mass_spec)
            state = phase_split_state(run=run, elapsed_h=spec["elapsed_h"],
                pressure_bar=atmosphere["pressure_bar"], temperature_c=atmosphere["temperature_c"],
                injected_mass_g=_dose(statistics, mass_spec),
                injection_stages=statistics["dose"]["injection_stages"],
                adopted_revision_sha256=adopted, recipe=statistics["recipe"], math=math, tables=tables)
            cases[run].append(spec["case_id"])
            try:
                if curve is None:
                    raise TargetUnavailable("This run has no usable reviewed overflow measurement")
                target = own_target(curve["curve"], run=run, elapsed_h=state["elapsed_h"], source=curve["source"])
            except TargetUnavailable:
                pending.append((run, spec["case_id"], state))
                continue
            targets[run][spec["case_id"]] = target
            if state["injected_mass_g"] > 0:
                donors.append(dict(state, **{key: deepcopy(target[key]) for key in
                                             ("kind", "estimated", "source", "target_volume_m3")}))
    for run, case, state in pending:
        try:
            targets[run][case] = pressure_matched_target(state, donors, fit_runs=frozen["calibration_runs"])
        except (TargetUnavailable, ValueError) as error:
            targets[run][case] = dict(kind="none", reason=str(error))
    return {run: dict(policy=POLICY, targets={case: targets[run][case] for case in order})
            for run, order in cases.items()}


def collect_phase2_source_inputs(job_roots, calibration, destination, *, series_root,
                                 overflow_phase_split=False):
    """Write the phase-two source manifest for every registered run.

    ``job_roots`` are the finished phase-one jobs; ``calibration`` binds the
    frozen calibration chosen for phase two (path, SHA-256 and size), and is
    recorded in the manifest because the anchors follow it. ``destination``
    must be a fresh file. Missing or unverifiable runs refuse before writing.
    With ``overflow_phase_split`` (default off, which writes today's manifest)
    each run also carries its images' overflow targets (overflow_targets).
    """
    from .mass.n1_pipeline import DataReader
    from .pipeline.job_inputs import MAX_METADATA_BYTES
    from .pipeline.job_queue import _immutable_bytes, descriptor, write_immutable
    from .pipeline.staged_data import verify_phase1_completion
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a fresh phase-two input manifest file")
    manifest, runs = registered_runs(series_root)
    reader = DataReader()
    frozen = reader.json(calibration)
    if frozen.get("frozen") is not True or not isinstance(frozen.get("calibration_runs"), list):
        raise ValueError("Choose a frozen mass calibration")
    if type(overflow_phase_split) is not bool:
        raise ValueError("Choose explicitly whether each run's own overflow governs its phase split")
    found, missing = published_phase_one(job_roots, runs, reader=reader)
    errors = [run + ": no chosen phase-one job has published this run" for run in missing]
    rows, verified_runs = [], {}
    for run in runs:
        if run not in found:
            continue
        item = found[run]
        try:
            verified = verify_phase1_completion(item["value"]["phase1_completion"],
                item["value"]["configuration"], expected_commit=item["commit"], reader=reader)
            if verified["configuration"]["run"] != run:
                raise ValueError("the phase-one completion belongs to another run")
            rows.append(dict(run=run, completion_bound=dict(item["value"]["phase1_completion"]),
                configuration_bound=dict(item["value"]["configuration"]),
                expected_phase1_commit=item["commit"],
                fitted_anchor=run_anchor(verified, frozen, reader=reader)))
            verified_runs[run] = verified
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{run}: {error}")
    if errors:
        raise ValueError("Phase-two inputs are incomplete; nothing was written:\n" + "\n".join(errors))
    document = dict(format=FORMAT, series_manifest=manifest, runs=rows,
        anchor_calibration=dict(calibration),
        phase_one_results={run: found[run]["result"] for run in runs})
    if overflow_phase_split:
        splits = overflow_targets(verified_runs, frozen, reader=reader)
        for row in rows:
            row["overflow_phase_split"] = splits[row["run"]]
        document["overflow_phase_split"] = True
    if len(_immutable_bytes(document)) > MAX_METADATA_BYTES:
        raise ValueError("Phase-two input manifests must remain bounded metadata")
    if descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("The registered series changed while phase-two inputs were collected")
    write_immutable(destination, document)
    return dict(inputs=descriptor(destination), runs=[row["run"] for row in rows], plan_prepared=False)


def require_anchor_calibration(document, calibration):
    """Refuse phase-two inputs whose anchors follow another mass calibration.

    Identity is the calibration file's exact bytes (SHA-256 and size), not
    its name or location.
    """
    recorded = document.get("anchor_calibration")
    if (not isinstance(recorded, dict)
            or any(recorded.get(key) != calibration.get(key) for key in ("sha256", "size_bytes"))):
        made_for = Path(recorded["path"]).name if isinstance(recorded, dict) and "path" in recorded else "none"
        raise ValueError("The saved phase-two inputs were made for another mass calibration (" + made_for
                         + "), and their run anchors follow that calibration. This plan uses "
                         + Path(calibration["path"]).name + ".")


def bind_phase_two_inputs(job_roots, calibration, destination, *, series_root, overflow_phase_split=False):
    """The phase-two inputs for the calibration chosen at Prepare.

    Writes the manifest when ``destination`` does not exist yet. An existing
    manifest is used only when it was made for this same calibration, from
    the same published phase-one results and with the same overflow
    phase-split choice; otherwise this refuses with a plain reason. A saved
    manifest is never replaced.
    """
    from .mass.n1_pipeline import DataReader
    from .pipeline.job_queue import descriptor
    destination = Path(destination)
    if not destination.exists():
        return collect_phase2_source_inputs(job_roots, calibration, destination, series_root=series_root,
                                            overflow_phase_split=overflow_phase_split)
    bound = descriptor(destination)
    reader = DataReader()
    document = reader.json(bound)
    if document.get("format") != FORMAT:
        raise ValueError("The saved phase-two input file is not a phase-two input manifest")
    require_anchor_calibration(document, calibration)
    if document.get("overflow_phase_split", False) is not overflow_phase_split:
        raise ValueError("The saved phase-two inputs were made with the overflow phase split "
                         + ("on" if document.get("overflow_phase_split") else "off")
                         + "; this plan asks for it " + ("on." if overflow_phase_split else "off."))
    _manifest, runs = registered_runs(series_root)
    found, missing = published_phase_one(job_roots, runs, reader=reader)
    if missing or document.get("phase_one_results") != {run: found[run]["result"] for run in runs}:
        raise ValueError("The saved phase-two inputs were made from other phase-one results than the "
                         "chosen phase-one jobs")
    return dict(inputs=bound, runs=[row["run"] for row in document["runs"]], plan_prepared=False)


def prepare_phase_two_plan(job_roots, calibration, inputs_destination, folder, *, series_root, job_id, job,
                           overflow_phase_split=False, **options):
    """Prepare's phase-two entry: inputs bound to the chosen calibration, then the plan.

    The inputs come from bind_phase_two_inputs; the original phase-two
    preparation then seals the plan with the same calibration. The
    ``overflow_phase_split`` switch (default off) lets each run's own overflow
    govern its images' free gas. ``options`` pass on to that preparation
    (threads, storage profile, stored resolution and original phase-one jobs).
    """
    from . import analysis_preparation
    from .mass.n1_pipeline import DataReader
    bound = bind_phase_two_inputs(job_roots, calibration, inputs_destination, series_root=series_root,
                                  overflow_phase_split=overflow_phase_split)["inputs"]
    require_anchor_calibration(DataReader().json(bound), calibration)
    return analysis_preparation.prepare_phase2_inputs(bound, folder, series_root=series_root,
        fitted_calibration=calibration, job_id=job_id, job=job, **options)
