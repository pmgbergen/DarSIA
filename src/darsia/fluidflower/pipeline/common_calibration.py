"""Exact, additive P1 -> genuine global refit -> P2 provenance joins.

No fit, phase inference or queue claim is computed here.
The common qualification must retain the actual fitting roles. In particular,
a held-out run is never added to the fit. Its separate application anchor uses
its own verified P1 rows and the unchanged original anchor-selection function.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

JOIN_FORMAT = "darsia-common-phase-calibration-join-v1"
APPLICATION_FORMAT = "darsia-frozen-calibration-heldout-application-v1"
PHASE1_INPUTS_FORMAT = "darsia-common-phase1-closure-inputs-v1"


def bind_common_phase1(plan, *, configuration, completion, code_commit, reader):
    """Add explicit closure inputs without changing any fitting roles or values."""
    from .staged_data import _bound
    if (not isinstance(plan, dict) or plan.get("format") != "darsia-n1-refit-plan-v1"
            or plan.get("run") != "refit" or "common_phase1" in plan
            or re.fullmatch(r"[0-9a-f]{40}", str(code_commit)) is None):
        raise ValueError("An explicit global plan and original P1 commit are required")
    result = deepcopy(plan)
    result["common_phase1"] = dict(format=PHASE1_INPUTS_FORMAT,
        configuration=deepcopy(_bound(configuration, where="common phase-one configuration")),
        completion=deepcopy(_bound(completion, where="common phase-one completion")),
        code_commit=code_commit)
    configured = reader.json(configuration)
    if configured.get("format") != "darsia-native-phase-statistics-run-v1":
        raise ValueError("The common closure needs its actual phase-one configuration")
    from .job_inputs import InputCatalog, _key
    aliases = deepcopy(plan.get("data_aliases", []))
    existing = {_key(row["logical"]["path"]): row for row in aliases}
    for row in configured.get("data_aliases", ()):
        key = _key(row["logical"]["path"])
        if key in existing and existing[key] != row:
            raise ValueError("The C plan conflicts with an actual P1 closure alias")
        if key not in existing:
            aliases.append(deepcopy(row)); existing[key] = row
    InputCatalog({}, aliases)  # Same SHA/size/conflict checks; no resources are opened.
    if aliases:
        result["data_aliases"] = aliases
    return result


def catalogue_common_phase1(catalogue, binding, *, use):
    """Declare the original verifier's named dependencies using metadata only.

    This walk grants no scientific verification. The unchanged original verifier
    runs later in the admitted worker with its complete positive RunIO binding.
    """
    if (not isinstance(binding, dict) or set(binding) != {"format", "configuration", "completion", "code_commit"}
            or binding["format"] != PHASE1_INPUTS_FORMAT
            or re.fullmatch(r"[0-9a-f]{40}", str(binding["code_commit"])) is None):
        raise ValueError("Explicit common phase-one closure descriptors required")
    configured = catalogue.doc(binding["configuration"], "configuration", use + ".configuration")
    for row in configured.get("data_aliases", ()):
        logical, physical = row["logical"], row["physical"]
        if (not _same_bytes(logical, physical) or catalogue.physical(logical) != physical):
            raise ValueError("The common plan must explicitly carry the original P1 closure aliases")
    closed = catalogue.doc(binding["completion"], "configuration", use + ".completion")
    frames = configured.get("frames")
    if (configured.get("format") != "darsia-native-phase-statistics-run-v1"
            or not isinstance(frames, list) or not frames
            or closed.get("format") != "darsia-phase-statistics-completion-v1"
            or closed.get("complete") is not True or closed.get("history_complete") is not True
            or closed.get("configuration") != binding["configuration"]
            or closed.get("run") != configured.get("run") or closed.get("precision") != configured.get("precision")
            or closed.get("code_commit") != binding["code_commit"]
            or closed.get("ordered_case_ids") != [frame["case_id"] for frame in frames]
            or closed.get("ordered_case_ids") != configured.get("ordered_case_ids")
            or not isinstance(closed.get("frames"), list) or len(closed["frames"]) != len(frames)):
        raise ValueError("Common phase-one closure dependency identities differ")
    for spec, commit in zip(frames, closed["frames"]):
        tag = use + ".frame." + spec["case_id"]
        record = catalogue.doc(commit, "configuration", tag + ".commit")
        if (record.get("format") != "darsia-phase-statistics-frame-v1" or record.get("complete") is not True
                or record.get("case_id") != spec["case_id"] or record.get("source") != spec["source"]
                or record.get("frame_index") != spec["frame_index"] or record.get("elapsed_h") != spec["elapsed_h"]
                or record.get("configuration") != binding["configuration"]
                or record.get("precision") != configured["precision"]
                or not isinstance(record.get("files"), dict) or not record["files"]):
            raise ValueError("Common phase-one frame dependencies differ from their selected source")
        for leaf, artifact in record["files"].items():
            catalogue.add(artifact, "calibration", tag + ".file." + leaf)
        catalogue.doc(record["native"]["audit"], "configuration", tag + ".native_audit")
        catalogue.doc(record["native_receipt"], "configuration", tag + ".native_receipt")
        history = catalogue.doc(record["phase"]["history_input"], "configuration", tag + ".history")
        for field in ("temporal_log", "temporal_valid"):
            catalogue.add(history[field], "calibration", tag + ".history." + field)
        catalogue.doc(record["before_rules"]["receipt"], "configuration", tag + ".before_receipt")
        if "photograph" in record:
            catalogue.doc(record["photograph"]["receipt"], "configuration", tag + ".photograph_receipt")
        if "probabilities" in record:
            catalogue.doc(record["probabilities"]["receipt"], "configuration", tag + ".probability_receipt")
            catalogue.doc(configured["phase"]["configuration"], "configuration", use + ".phase_configuration")


from ..published_jobs_snapshot import _same_bytes


def phase1_statistics(verified):
    """Use the original verifier's exact ordered records, never a folder glob."""
    if (not isinstance(verified, dict) or verified.get("payload_bytes_verified") is not True
            or verified.get("scientific_arrays_decoded") is not False):
        raise ValueError("Original content-verified phase-one records are required")
    config, records = verified["configuration"], verified["records"]
    if len(config["frames"]) != len(records):
        raise ValueError("Phase-one statistics omit an original selected frame")
    result = []
    for spec, item in zip(config["frames"], records):
        record = item["record"]
        if record["case_id"] != spec["case_id"]:
            raise ValueError("Phase-one statistics differ from the selected case order")
        result.append(dict(case_id=spec["case_id"], frame_index=spec["frame_index"],
            selected_target_h=spec["mass_spec"]["selected_target_h"],
            statistics=deepcopy(record["statistics"])))
    return result




def heldout_application_anchor(verified_phase1, plan, *, reader):
    """Select only application data; never append a row to a fit or group.

    The same branch applies to every explicitly non-fit production run.
    The original anchor rule operates on that run's own verified P1 statistics.
    The plan, its fitting rows, coefficients and frozen fit are never mutated.
    """
    from .refit_runner import validate_plan, read_statistics
    from ..mass.n1_refit import anchors_by_run
    from ..mass.n1_statistics import CONTRACT
    rules = validate_plan(plan)
    run = verified_phase1["configuration"]["run"]
    roles = plan["runs"]
    if run in rules.fit_runs or run in plan["statistics"]:
        raise ValueError("A fit run must use its actual frozen fit anchor, not the held-out branch")
    if run in roles.get("excluded", ()):
        raise ValueError("An excluded run is not an admitted held-out application")
    accepted = [role for role in ("heldout", "descriptive") if run in roles.get(role, ())]
    if len(accepted) != 1:
        raise ValueError("A held-out application needs its explicit unchanged non-fit production role")
    entries = phase1_statistics(verified_phase1)
    specs = verified_phase1["configuration"]["frames"]
    rows = []
    for entry, spec in zip(entries, specs):
        row = read_statistics(reader, entry["statistics"])
        if (row.get("contract") != CONTRACT or row.get("run") != run
                or row.get("frame_index") != entry["frame_index"]
                or row.get("elapsed_h") != spec["elapsed_h"]
                or row.get("source_binding", {}).get("source_sha256") != spec["source"]["source_sha256"]):
            raise ValueError("Held-out application statistics differ from their own verified P1 acquisition")
        if row.get("selected_target_h") not in (None, entry["selected_target_h"]):
            raise ValueError("Held-out application selected target differs from its verified P1")
        rows.append(row)
    chosen = anchors_by_run(rows, rules)[run]
    anchor = {key: chosen[key] for key in ("elapsed_h", "affected_pore_volume_m3")}
    return dict(format=APPLICATION_FORMAT, usage="application", role="held out",
        accepted_plan_role=accepted[0], run=run, anchor=anchor,
        selected_anchor=deepcopy(chosen), statistics=deepcopy(entries),
        phase1_configuration=deepcopy(verified_phase1["configuration_descriptor"]),
        phase1_completion=deepcopy(verified_phase1["completion_descriptor"]),
        entered_fit_objective=False, entered_training_rows=False,
        changed_coefficients=False, entered_group_value=False)


def validate_refit_result(result, *, task, expected_commit, context,
                          configuration=None, plan=None, fit=None, frozen=None):
    """Validate one original refit result and its available closed documents.

    Publication ownership and DATA reads remain with the caller. ``context`` is
    explicit: producer/terminal allow the original stopped and fold results;
    global_application requires the actual stop-free global plan and freeze.
    """
    from . import refit_runner as refit
    if context not in {"producer", "terminal", "global_application"}:
        raise ValueError("An explicit original refit validation context is required")
    if (not isinstance(task, dict) or result.get("task") != task
            or task.get("pipeline") != refit.PIPELINE
            or result.get("format") != refit.RESULT_FORMAT
            or result.get("pipeline") != refit.PIPELINE or result.get("complete") is not True
            or result.get("worker_references_closed") is not True
            or result.get("run") != task.get("run")
            or result.get("attempt_id") != task.get("attempt_id")
            or result.get("commit") != expected_commit or task.get("commit") != expected_commit
            or result.get("backend") != task.get("backend")
            or result.get("precision") != task.get("precision")
            or type(result.get("stop_required")) is not bool):
        raise ValueError("The actual closed refit task, source or result identity differs")
    fold = refit.fold_of(task["run"])
    if result.get("fold") != fold or (context != "global_application" and result.get("backend") != "cpu"):
        raise ValueError("The actual refit fold or CPU result differs")
    expected = {"fit.json"}
    if not result["stop_required"]:
        expected.add("frozen_calibration.json" if fold is None else "predictions.json")
    if not isinstance(result.get("files"), dict) or set(result["files"]) != expected:
        raise ValueError("The actual refit result omits or adds an output")
    if "frozen_calibration.json" in expected:
        if result.get("frozen_calibration") != result["files"]["frozen_calibration.json"]:
            raise ValueError("The actual refit frozen output differs from its file binding")
    elif result.get("frozen_calibration") is not None:
        raise ValueError("A stopped or fold refit cannot claim a frozen global calibration")
    if fit is not None and (fit.get("parameters") != result.get("parameters")
            or fit.get("stop_required") is not result["stop_required"]
            or fit.get("heldout_fit_run") != fold):
        raise ValueError("The saved fit differs from the actual refit result")
    if frozen is not None and (frozen.get("format") != refit.FREEZE_FORMAT
            or frozen.get("frozen") is not True or frozen.get("configuration") != task["configuration"]
            or frozen.get("code_commit") != expected_commit
            or frozen.get("parameters") != result.get("parameters")
            or frozen.get("calibration_files") != {"fit.json": result["files"]["fit.json"]}):
        raise ValueError("The frozen calibration differs from its actual refit plan/fit/source")
    rules = None
    if context == "global_application":
        if (plan is None or configuration is None or fold is not None
                or result["stop_required"] is not False or result["run"] != refit.GLOBAL_RUN
                or plan.get("run") != refit.GLOBAL_RUN or plan.get("fold") is not None
                or not _same_bytes(task.get("configuration", {}), configuration)):
            raise ValueError("An actual closed stop-free global refit on the pinned source is required")
        rules = refit.validate_plan(plan)
        if fit is not None and frozen is not None:
            if (frozen.get("anchors_by_run") != fit.get("anchors_by_run")
                    or frozen.get("calibration_runs") != fit.get("calibration_runs")
                    or list(rules.fit_runs) != frozen.get("calibration_runs")
                    or frozen.get("rules") != plan["rules"] or frozen.get("baseline") != plan["baseline"]):
                raise ValueError("The actual fit and frozen plan/parameters/anchors disagree")
            for role, key in (("heldout", "heldout_runs"), ("descriptive", "descriptive_runs"),
                              ("excluded", "excluded_runs")):
                if frozen.get(key) != list(plan["runs"].get(role, ())):
                    raise ValueError("The frozen refit changes an accepted run role")
        expected_cases = [row["case_id"] for run in sorted(plan["statistics"])
                          for row in plan["statistics"][run]]
        if task.get("ordered_case_ids") != expected_cases or result.get("frames") != len(expected_cases):
            raise ValueError("The actual C task differs from its complete plan's statistics order")
    return expected, rules


def verify_frozen_refit(binding, *, verified_phase1, fitted_calibration,
                        fitted_anchor, expected_commit, reader):
    """Join a genuine closed global C result to its exact P1 inputs and P2 data.

    The frozen document's plan/task binding, fit, parameters and anchors are
    checked independently. A calibration hash with plausible parameters alone
    cannot establish that this C plan ran. Queue/native retirement is verified
    separately by the common provider's stage evidence.
    """
    from .staged_data import _bound, _keys, _same_json
    _keys(binding, {"completion", "plan"}, where="genuine global refit binding")
    for field in ("completion", "plan"):
        _bound(binding[field], where="global refit " + field)
    completed = reader.json(binding["completion"])
    plan = reader.json(binding["plan"])
    task = completed.get("task")
    _, rules = validate_refit_result(completed, task=task, expected_commit=expected_commit,
        context="global_application", configuration=binding["plan"], plan=plan)
    fit_pin = completed["files"]["fit.json"]
    frozen_pin = completed["files"]["frozen_calibration.json"]
    for value in (fit_pin, frozen_pin):
        _bound(value, where="actual refit output")
    if (completed.get("frozen_calibration") != frozen_pin
            or not _same_bytes(frozen_pin, fitted_calibration)):
        raise ValueError("Phase two must use this actual global refit's frozen calibration")
    fit, frozen = reader.json(fit_pin), reader.json(frozen_pin)
    validate_refit_result(completed, task=task, expected_commit=expected_commit,
        context="global_application", configuration=binding["plan"], plan=plan, fit=fit, frozen=frozen)
    run = verified_phase1["configuration"]["run"]
    expected_rows = phase1_statistics(verified_phase1)
    application = None
    if run in rules.fit_runs:
        if run not in plan["statistics"] or not _same_json(plan["statistics"][run], expected_rows):
            raise ValueError(run + ": the genuine C plan does not consume this exact P1 statistics order")
    else:
        application = heldout_application_anchor(verified_phase1, plan, reader=reader)
    for rows in plan["statistics"].values():
        for row in rows:
            reader.path(row["statistics"])
    anchors = frozen.get("anchors_by_run")
    if not isinstance(anchors, dict) or (application is None and run not in anchors):
        raise ValueError(run + ": no applicable own anchor in the genuine frozen C result")
    if application is not None and run in anchors:
        raise ValueError("Held-out application data must not appear in the frozen fit anchors")
    anchor = anchors[run] if application is None else application["anchor"]
    wanted = {key: anchor[key] for key in ("elapsed_h", "affected_pore_volume_m3")}
    if not _same_json(wanted, fitted_anchor):
        raise ValueError("The P2 anchor differs from its genuine own-run fit/application anchor")
    return dict(format=JOIN_FORMAT, binding=deepcopy(binding),
        calibration=deepcopy(frozen_pin), fit=deepcopy(fit_pin),
        phase1_configuration=deepcopy(verified_phase1["configuration_descriptor"]),
        phase1_completion=deepcopy(verified_phase1["completion_descriptor"]),
        run=run, ordered_case_ids=[row["case_id"] for row in expected_rows],
        parameters=deepcopy(frozen["parameters"]), anchor=deepcopy(wanted),
        code_commit=expected_commit, application=deepcopy(application), actual_frozen_refit_verified=True,
        queue_ownership_or_retirement_verified=False)


def extract_refit_products(result, claim, machine_id):
    """Publish only closed original C files through the existing byte publisher."""
    from .products_publication import product, _bound, _verified_bytes, _json, MAX_METADATA_BYTES
    from .job_queue import name
    name(machine_id)
    task = claim.task
    if result.get("attempt_id") != claim.token:
        raise ValueError("An original closed refit result from this actual claim is required")
    expected, _ = validate_refit_result(result, task=task, expected_commit=task["commit"], context="producer")
    root = Path(claim.output).resolve()
    rows, metadata, accessible, documents = [], [], {}, {}
    for leaf in sorted(expected):
        bound = _bound(result["files"][leaf])
        path = root / leaf
        if Path(bound["path"]).resolve() != path or path.is_symlink():
            raise ValueError("A refit product is outside its actual owned attempt")
        documents[leaf] = _json(_verified_bytes(path, bound, limit=MAX_METADATA_BYTES))
        row = product(leaf, machine_id=machine_id, local_path=bound["path"],
                      sha256=bound["sha256"], bytes=bound["bytes"], kind="operational_receipt")
        rows.append(row)
        metadata.append(leaf)
        accessible[leaf] = path
    validate_refit_result(result, task=task, expected_commit=task["commit"], context="producer",
        fit=documents["fit.json"], frozen=documents.get("frozen_calibration.json"))
    return rows, metadata, accessible


def refit_terminal(root, base_task, job, done, reader):
    """Read the original C publication through its indexed metadata copies.

    No producer-local path is opened by this reader. The common supervisor
    separately verifies its own declared output bytes after natural retirement.
    """
    from . import refit_runner as refit
    from ..published_jobs_snapshot import (
        _read_reference, _same_bytes as same, _task_identity, _lease_chain,
        _owner_machine, _original_metadata, _pure, _SnapshotMetadata,
    )
    from .products_publication import _read_index, published_metadata_descriptor
    from .job_queue import _own_asset, descriptor, name
    root = Path(root)
    if (base_task.get("pipeline") != refit.PIPELINE
            or done.get("format") != "darsia-alias-done-v1" or done.get("run") != base_task["run"]
            or done["completed"]["relative_path"] != "results/" + base_task["run"] + "/completed.json"):
        raise ValueError("Actual alias refit done identity required")
    completed = _read_reference(root, done["completed"], reader)
    index_path = _own_asset(root, done["products_index"])
    index = descriptor(index_path)
    _folder, document, products = _read_index(index)
    shared_metadata = _SnapshotMetadata(index, _folder, products)
    task = completed.get("task") or {}
    if (document["run"] != base_task["run"] or document["attempt_id"] != done["attempt_id"]
            or not same(document["job"], descriptor(root / "job.json"))
            or not same(published_metadata_descriptor(index), done["completed"])
            or not _task_identity(base_task, task) or task.get("attempt_id") != done["attempt_id"]
            or completed.get("attempt_id") != done["attempt_id"]
            or completed.get("calibration_sha256") != job["calibration_sha256"]
            or completed.get("precision") != job["precision"]):
        raise ValueError("Original closed refit/index/task/source identity differs")
    expected, _ = validate_refit_result(completed, task=task,
        expected_commit=job["commit"], context="terminal")
    owner = reader(index_path.parent.parent / "owner.json")
    if (owner.get("task") != task or owner.get("token") != done["attempt_id"]
            or owner.get("actual_owner") != completed.get("actual_worker")):
        raise ValueError("The refit publication owner differs from the actual claim")
    machine = _owner_machine(owner, base_task, done["attempt_id"])
    original = products["completed.json"]
    attempt = _pure(original["local_path"]).parent
    if (not same(original, done["completed"]) or name(original["machine_id"]) != name(machine)
            or _pure(original["local_path"]).name != "candidate_completed.json"
            or attempt.name != done["attempt_id"] or attempt.parent.name != ".attempts"
            or name(attempt.parent.parent.name) != name(task["run"])
            or name(attempt.parent.parent.parent.name) != name(task["job_id"])):
        raise ValueError("The refit producer is outside its actual owned job/run/attempt")
    if set(products) != expected | {"completed.json"}:
        raise ValueError("The original refit publication omits or adds an actual output")
    documents = {}
    for leaf in expected:
        bound, product = completed["files"][leaf], products[leaf]
        if (not same(bound, product) or _pure(product["local_path"]) != attempt / leaf
                or _pure(bound["path"]) != attempt / leaf or name(product["machine_id"]) != name(machine)):
            raise ValueError("The indexed refit output differs from its original descriptor")
        documents[leaf] = _original_metadata(index, products, bound, attempt, machine, metadata=shared_metadata)
    validate_refit_result(completed, task=task, expected_commit=job["commit"], context="terminal",
        fit=documents["fit.json"], frozen=documents.get("frozen_calibration.json"))
    lineage = _lease_chain(root, base_task["run"], done["attempt_id"], base_task, reader)
    shared_metadata.finish()
    return completed, {}, lineage


def catalogue_refit(catalogue, config):
    """Positive named C/application inputs in the original runtime catalogue.

    Replay recomputes the original application anchor from SHA-verified own
    statistics as well as checking all actual C file identities.
    No calibration row is added or fit recomputed by catalogue construction.
    """
    join = config["provenance"]["common_frozen_refit"]
    if (not isinstance(join, dict) or join.get("format") != JOIN_FORMAT
            or join.get("actual_frozen_refit_verified") is not True
            or join.get("run") != config["run"] or join.get("anchor") != config["anchor"]
            or not _same_bytes(join.get("calibration", {}), config["calibration"])
            or join.get("phase1_configuration") != config["phase1"]["configuration"]
            or join.get("phase1_completion") != config["phase1"]["completion"]
            or join.get("ordered_case_ids") != [frame["case_id"] for frame in config["frames"]]):
        raise ValueError("The typed common C join differs from the original P2/P1/calibration/anchor")
    catalogue_common_phase1(catalogue, dict(format=PHASE1_INPUTS_FORMAT,
        configuration=config["phase1"]["configuration"], completion=config["phase1"]["completion"],
        code_commit=config["phase1"]["code_commit"]), use="mass.common_phase1")
    use = "mass.common_refit"
    completed = catalogue.doc(join["binding"]["completion"], "calibration", use + ".completion")
    plan = catalogue.doc(join["binding"]["plan"], "calibration", use + ".plan")
    fit = catalogue.doc(join["fit"], "calibration", use + ".fit")
    frozen = catalogue.doc(join["calibration"], "calibration", use + ".frozen")
    task = completed.get("task") or {}
    if (completed.get("files") != {"fit.json": join["fit"], "frozen_calibration.json": join["calibration"]}
            or completed.get("frozen_calibration") != join["calibration"]
            or join["parameters"] != fit.get("parameters")):
        raise ValueError("The catalogued C result is not its actual stop-free global fit/plan/frozen content")
    _, rules = validate_refit_result(completed, task=task, expected_commit=join["code_commit"],
        context="global_application", configuration=join["binding"]["plan"], plan=plan,
        fit=fit, frozen=frozen)
    catalogue.doc(plan["baseline"]["calibration"], "calibration", use + ".baseline")
    for run, rows in plan["statistics"].items():
        for row in rows:
            catalogue.add(row["statistics"], "calibration", use + ".fit." + run + "." + str(row["frame_index"]))
    application = join.get("application")
    run = config["run"]
    if run in rules.fit_runs:
        if application is not None:
            raise ValueError("A fit run cannot use held-out application data")
        anchor = frozen["anchors_by_run"][run]
        if config["anchor"] != {key: anchor[key] for key in ("elapsed_h", "affected_pore_volume_m3")}:
            raise ValueError("P2 changes its actual frozen fit anchor")
        rows = plan["statistics"][run]
        if [row["case_id"] for row in rows] != join["ordered_case_ids"]:
            raise ValueError("The genuine C plan does not consume this original P1 order")
        for entry, spec in zip(rows, config["frames"]):
            if (entry["statistics"] != config["phase1"]["frames"][spec["case_id"]]["statistics"]
                    or entry["frame_index"] != spec["frame_index"]):
                raise ValueError("The actual fit uses other phase-one statistics")
    else:
        roles = [role for role in ("heldout", "descriptive") if run in plan["runs"].get(role, ())]
        if (run in plan["statistics"] or run in frozen["anchors_by_run"] or run in plan["runs"].get("excluded", ())
                or len(roles) != 1 or not isinstance(application, dict)
                or application.get("format") != APPLICATION_FORMAT or application.get("usage") != "application"
                or application.get("role") != "held out" or application.get("accepted_plan_role") != roles[0]
                or application.get("run") != run or application.get("anchor") != config["anchor"]
                or any(application.get(field) is not False for field in (
                    "entered_fit_objective", "entered_training_rows", "changed_coefficients", "entered_group_value"))
                or application.get("phase1_configuration") != config["phase1"]["configuration"]
                or application.get("phase1_completion") != config["phase1"]["completion"]):
            raise ValueError("The non-fit application changes an accepted role, fit or own-P1 anchor binding")
        entries = application.get("statistics")
        if not isinstance(entries, list) or [row["case_id"] for row in entries] != join["ordered_case_ids"]:
            raise ValueError("The application omits or reorders its own original P1 statistics")
        own_rows = []
        from ..mass.n1_refit import anchors_by_run
        from ..mass.n1_statistics import CONTRACT
        for entry, spec in zip(entries, config["frames"]):
            bound = config["phase1"]["frames"][spec["case_id"]]["statistics"]
            if entry["statistics"] != bound or entry["frame_index"] != spec["frame_index"]:
                raise ValueError("Application statistics differ from the original P1 case")
            row = catalogue.statistics_document(bound, "calibration", use + ".application." + spec["case_id"])
            if (row.get("contract") != CONTRACT or row.get("run") != run
                    or row.get("frame_index") != spec["frame_index"]
                    or row.get("elapsed_h") != spec["elapsed_h"]
                    or row.get("source_binding", {}).get("source_sha256")
                        != spec["source_binding"]["source_sha256"]
                    or row.get("selected_target_h") not in (None, spec["selected_target_h"])):
                raise ValueError("Application statistics differ from their own P1 acquisition")
            own_rows.append(row)
        chosen = anchors_by_run(own_rows, rules)[run]
        wanted = {key: chosen[key] for key in ("elapsed_h", "affected_pore_volume_m3")}
        if wanted != config["anchor"] or chosen != application.get("selected_anchor"):
            raise ValueError("The application anchor does not reproduce the original own-run selection")
    return join
