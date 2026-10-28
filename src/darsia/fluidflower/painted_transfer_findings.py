"""Original input findings from a replayed Transfer, without adoption authority.

This module produces review DATA only.  It never creates an index, registration,
table contract, geometry candidate, approval or Store revision.  The normal
preparation, Check and adoption paths retain their original source admission.
"""

from copy import deepcopy
import json

import numpy as np


FORMAT = "darsia-fluidflower-transfer-findings-data-v1"
_ACTION_KEYS = {"analysis_exclusions", "source_choices", "display_preferences"}


def _json_snapshot(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _request_bindings(review_binding, target_bindings, explicit_actions):
    if not isinstance(target_bindings, list):
        raise ValueError("Transfer findings require the exact ordered target-binding cohort")
    actions = {} if explicit_actions is None else _json_snapshot(explicit_actions)
    if not isinstance(actions, dict) or set(actions) - _ACTION_KEYS:
        raise ValueError("Transfer findings require the exact explicit GUI action fields")
    choices = actions.get("source_choices", [])
    preferences = actions.get("display_preferences", {})
    if not isinstance(choices, list) or not isinstance(preferences, dict):
        raise ValueError("Transfer findings require literal source actions and display preferences")
    return _json_snapshot(dict(review=review_binding,
        target_bindings=target_bindings, analysis_exclusions=actions.get("analysis_exclusions"),
        source_choices=choices, display_preferences=preferences))


def _source_state(choices):
    from .geometry_facies import RUN_GEOMETRY_ACQUISITION_STATES
    from .painted_analysis_exclusions import _action

    for choice in choices:
        _action(choice, state=True)
        if choice["acquisition_state"] not in RUN_GEOMETRY_ACQUISITION_STATES:
            raise ValueError("Choose an explicit supported facies-source water state")
    return choices[-1]["acquisition_state"] if choices else "unknown"


def _display_findings(review, bindings, display_attachments):
    from .painted_findings_catalogue import review_findings_data
    from .painted_review_comparison import make_display_attachments

    preferences = bindings["display_preferences"]
    if preferences:
        if set(preferences) != {"position_policy", "water_policy", "water_actions"}:
            raise ValueError("Display preferences require the original position and water fields")
        expected = make_display_attachments(review, **preferences)
        if display_attachments is not None and expected != display_attachments:
            raise ValueError("Display attachments differ from the current explicit GUI choices")
        display_attachments = expected
    return review_findings_data(review, display_attachments=display_attachments)


def _unresolved_domains(replay, exclusions, unmapped_label_ids):
    """Keep exact current residual components and unproved zero roles visible."""
    from .painted_adoption_table import _same

    result = []
    reviewed = {row["run_id"]: {component["component_id"]: component for component in row["components"]}
                for row in replay["review"]["runs"]}
    for row in replay["runs"]:
        mask = exclusions.get(row["run_id"])
        for component, component_mask in row["components"]:
            if mask is not None and np.all(mask[component_mask]):
                continue
            bound = reviewed[row["run_id"]].get(component["component_id"])
            if bound is None or not _same(component,
                    {key: value for key, value in bound.items() if key != "review_row_sha256"}):
                raise ValueError("The current residual component needs its genuine bound review row")
            result.append(dict(kind="analysis_gap", run_id=row["run_id"], **deepcopy(bound)))
    if unmapped_label_ids:
        result.append(dict(kind="unmapped_label_role", label_ids=list(unmapped_label_ids)))
    return result


def _assemble_findings(current, replay, bindings, *, display_attachments=None, outside_context=None):
    """Consume only original source/product replay DATA and diagnostic helpers."""
    from .geometry_facies import _GeometryDiagnosticInputs, _geometry_input_report
    from .map_adoption import adoption_run_data
    from .painted_adoption_table import _checked
    from .painted_analysis_exclusions import (
        DisplayFindingsChanged, verify_analysis_exclusions,
    )
    from .painted_transfer_table_data import transfer_table_data

    review = replay["review"]
    config = deepcopy(current["configuration"])
    frame_binding = config.get("frame_binding") or {}
    frame_rows = frame_binding.get("runs") or {}
    display = frame_binding.get("display_reference_run")
    native = {row["run_id"]: row for row in replay["runs"]}
    if len(native) != len(replay["runs"]) or not native or display not in native:
        raise ValueError("The exact display reference run needs its original replayed native map")
    if set(native) - set(frame_rows):
        raise ValueError("The native review contains a run outside the current workspace frame")

    issues = []
    exclusions = {}
    approval = bindings["analysis_exclusions"]
    if approval is not None:
        try:
            # The original reader owns literal legacy/new approval dispatch and
            # replays actual source, target and component science before masks.
            exclusions = verify_analysis_exclusions(replay["preparation"], replay["folder"], approval)
        except (ValueError, OSError, KeyError, TypeError) as error:
            issues.append(dict(severity="error", code="painted_analysis_exclusions_changed", message=str(error)))

    table_data = None
    try:
        table_data = transfer_table_data(current,
            painting_binding=replay["painting_binding"], source_analysis=replay["approved_analysis"],
            effective_labels={run: row["effective_arrays"]["array"] for run, row in native.items()},
            source_fixture_drawn_ids=replay["fixture_drawn_ids"],
            native_fixture_drawn_ids={value for row in native.values() for value in row["native_drawn_ids"]},
            outside_context=outside_context)
        config["regions"] = deepcopy(table_data["regions"])
        config["permeability_unit"] = table_data["permeability_unit"]
    except (ValueError, OSError, KeyError, TypeError) as error:
        # A failed table derivation is a real source error, never the old table
        # masquerading as a successful new painted-map derivation.
        config["regions"] = []
        issues.append(dict(severity="error", code="invalid_geometry_source_mode", message=str(error)))

    state = _source_state(bindings["source_choices"])
    run_data = [adoption_run_data(run, frame_rows[run], row["base_product"]["path"],
                    acquisition_state=state) for run, row in native.items()]
    config["run_geometry"] = dict(config.get("run_geometry") or {}, runs=run_data)
    config["labels_path"] = native[display]["base_product"]["path"]
    # These are prior published authorities, not authorities for these input
    # DATA.  Their original replacements are created only by real preparation.
    for key in ("painted_adoption_table", "analysis_exclusions", "target_corrections"):
        config.pop(key, None)
    try:
        display_findings = _display_findings(review, bindings, display_attachments)
    except DisplayFindingsChanged as error:
        display_findings = error
    inputs = _GeometryDiagnosticInputs(
        fields={"labels_path": native[display]["effective_arrays"]["array"]},
        run_fields={run: {"labels": row["effective_arrays"]["array"]} for run, row in native.items()},
        analysis_exclusion=exclusions.get(display), review_findings=display_findings)
    diagnostic = _geometry_input_report(config, inputs=inputs)
    issues.extend(diagnostic["issues"])
    unmapped = table_data["unmapped_label_ids"] if table_data is not None else []
    source_bindings = deepcopy(replay["source_bindings"])
    guards = [] if table_data is None else table_data["guarded_sources"]
    for descriptor in guards:
        _checked(descriptor)
    source_bindings = dict(replay=source_bindings, table=deepcopy(guards))
    if outside_context is not None:
        source_bindings["outside_frame_proof"] = deepcopy(outside_context["outside_frame_proof"])
        source_bindings["outside_proof_bindings"] = deepcopy(outside_context["outside_proof_bindings"])
    summary = deepcopy(diagnostic["summary"])
    summary.update(error_count=sum(issue["severity"] == "error" for issue in issues),
        warning_count=sum(issue["severity"] == "warning" for issue in issues))
    result = dict(format=FORMAT, input_bindings=deepcopy(bindings),
        source_bindings=source_bindings, issues=issues, summary=summary,
        run_data=deepcopy(run_data),
        table_data=(None if table_data is None else dict(spec=deepcopy(table_data["spec"]),
            regions=deepcopy(table_data["regions"]), permeability_unit=table_data["permeability_unit"],
            parent_source=deepcopy(table_data["parent_source"]), unmapped_label_ids=list(unmapped))),
        unresolved_domains=_unresolved_domains(replay, exclusions, unmapped))
    for key in ("display_attachments", "display_findings_sha256"):
        if key in diagnostic:
            result[key] = deepcopy(diagnostic[key])
    return result


def collect_transfer_findings(current_revision_binding, review_binding, *, explicit_actions=None,
        display_attachments=None, source_context=None, target_correction_bindings=(),
        residual_destination=None):
    """Replay the current exact Transfer and collect its original input checks.

    ``source_context`` is the explicit original producer recipe, including any
    fixture source; it is never inferred from filenames.  A changed target
    cohort needs an explicitly owned, exclusive ``residual_destination`` for
    the genuine derivative review.  All filesystem work belongs in the owned
    finite metadata worker, never a GUI readiness predicate.
    """
    from .painted_adoption_table import _checked, _same, _source
    from .painted_transfer_replay import (
        replay_transfer_review, build_residual_transfer_review, materialize_transfer_findings,
    )

    current_revision_binding = _json_snapshot(current_revision_binding)
    review_binding = _json_snapshot(review_binding)
    target_bindings = _json_snapshot(target_correction_bindings)
    source_context = None if source_context is None else _json_snapshot(source_context)
    display_attachments = None if display_attachments is None else _json_snapshot(display_attachments)
    bindings = _request_bindings(review_binding, target_bindings, explicit_actions)
    current = _checked(current_revision_binding, metadata=True)
    source = _source(current)
    if not _same(source["revision"], current_revision_binding):
        raise ValueError("The collector requires the exact immutable current workspace revision")
    replay = replay_transfer_review(review_binding, source_context=source_context,
        target_bindings=target_bindings, retain_source_arrays=False)
    effective_targets = replay["target_bindings"]
    # The original target reader groups a global cohort by run while retaining
    # each run's exact subsequence.  Cross-run interleaving is not a new target
    # version and must not cause repeated residual review publication.
    selected = {}
    for binding in effective_targets:
        run = _checked(binding["version"], metadata=True)["basis"]["run_id"]
        selected.setdefault(run, []).append(binding)
    reviewed_targets = {row["run_id"]: row.get("target_correction_bindings", [])
                        for row in replay["review"]["runs"] if row.get("target_correction_bindings")}
    if not _same(reviewed_targets, selected):
        if residual_destination is None:
            raise ValueError("Changed target corrections require a genuine bound residual review")
        review = build_residual_transfer_review(replay, residual_destination, target_bindings=effective_targets)
        replay = replay_transfer_review(review, source_context=review["preparation_context"],
            target_bindings=effective_targets, retain_source_arrays=False)
        # The original writer refuses nesting a new recipe under its retained
        # source.  Keep this second exclusive output beside the new review.
        from pathlib import Path
        destination = Path(residual_destination)
        residual_destination = destination.with_name(destination.name + "_findings_inputs")
    outside_context = None
    has_residual = any(row["residual_mask"].any() for row in replay["runs"])
    approval_complete = not has_residual
    if has_residual and bindings["analysis_exclusions"] is not None:
        from .painted_analysis_exclusions import verify_analysis_exclusions
        try:
            verify_analysis_exclusions(replay["preparation"], replay["folder"], bindings["analysis_exclusions"])
        except (ValueError, OSError, KeyError, TypeError):
            # The original error is retained by _assemble_findings below.  An
            # incomplete action cannot become a successful outside proof.
            pass
        else:
            approval_complete = True
    if residual_destination is not None and approval_complete:
        outside_context = materialize_transfer_findings(replay, residual_destination,
            analysis_exclusions=bindings["analysis_exclusions"])
        # The original materializer has already replayed these exact copies.
        # Keep the unchanged scientific review binding; use only the actual
        # materialized recipe for its exclusion and outside-proof consumers.
        replay = dict(replay, preparation=outside_context["preparation"],
            folder=outside_context["folder"])
    report = _assemble_findings(current, replay, bindings, display_attachments=display_attachments,
        outside_context=outside_context)
    # Compact presentation DATA only. Reuse masks authenticated by this exact
    # replay; matching never creates a class, human action or resolved row.
    from .painted_gap_review_groups import deviation_groups_from_replay
    report["deviation_groups"] = deviation_groups_from_replay(replay)
    # Diagnostics can outlive the initial replay.  Recheck every exact input
    # consumed there, including target and renderer files, before publishing
    # the finite worker result to the GUI.
    for descriptor in replay["source_bindings"]["files"]:
        _checked(descriptor)
    if outside_context is not None:
        for descriptor in outside_context["outside_proof_bindings"]:
            _checked(descriptor)
        from .painted_analysis_exclusions import exclusion_sources
        exclusion_bindings = exclusion_sources(outside_context["preparation"])
        for descriptor in exclusion_bindings:
            _checked(descriptor)
        report["source_bindings"]["analysis_exclusions"] = deepcopy(exclusion_bindings)
    _checked(current_revision_binding)
    report["source_bindings"] = dict(report["source_bindings"],
        current_revision=current_revision_binding, review=deepcopy(replay["review"]["review_binding"]))
    return dict(review=deepcopy(replay["review"]), report=report)
