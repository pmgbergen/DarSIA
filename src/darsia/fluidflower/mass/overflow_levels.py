"""Source-bound measurements and one global fit-only overflow level table.

This does not publish/admit a preparation or run a fit. The owned preparation
must bind the returned actual source identities to its verified P1 closure and
replay these calculations before a derived row may enter the separate F fit.
"""
from copy import deepcopy

from . import overflow_method as method

NO_OVERFLOW = "no_usable_overflow"
ANCHOR_UNPROVED = "post_sampling_anchor_unproved"
NO_CURRENT_VOLUME = "no_qualifying_current_volume"


def observations_to_levels(observations, fallback_reasons, *, roles, analysis_runs, cohort=None):
    """Pure role/coverage checks around the unchanged shrinkage calculation.

    Actual current volumes and original scale-source admission belong to the
    reader below, not to an arbitrary dictionary passed to this pure kernel.
    A declared candidate count is never substituted for measured membership.
    """
    from .overflow_steering import shrink_run_levels
    method.validate_roles(roles, analysis_runs, cohort=cohort)
    if (type(observations) is not dict or type(fallback_reasons) is not dict
            or not set(observations) <= set(method.USABLE_RUNS)-set(method.CONDITIONAL_ANCHOR_RUNS)
            or set(fallback_reasons) != set(analysis_runs)-set(observations)):
        raise ValueError("Each original run needs exactly its own admitted measurement or explicit fallback")
    for run, value in observations.items():
        cases = value.get("selected_cases") if type(value) is dict else None
        if (type(value) is not dict or set(value) != {"ratio", "sigma", "selected_cases"}
                or type(cases) is not list or not cases or not all(type(case) is str and case for case in cases)
                or len(set(cases)) != len(cases)):
            raise ValueError("A run measurement needs its actual distinct selected current cases")
    for run, value in fallback_reasons.items():
        expected = (NO_OVERFLOW if run in method.NO_USABLE_OVERFLOW else
                    ANCHOR_UNPROVED if run in method.CONDITIONAL_ANCHOR_RUNS else NO_CURRENT_VOLUME)
        if (type(value) is not dict or set(value) != {"code", "reason"}
                or value["code"] != expected or type(value["reason"]) is not str or not value["reason"]):
            raise ValueError("The explicit fallback differs from that run's actual admitted route")
    numeric = {run:{key:value[key] for key in ("ratio", "sigma")}
               for run,value in observations.items()}
    result = shrink_run_levels(numeric, fit_runs=roles["fit"], runs=analysis_runs)
    result.update(format="darsia-overflow-run-levels-v1", roles=deepcopy(roles),
        analysis_run_order=list(analysis_runs), measurements=deepcopy(observations),
        fallback_reasons=deepcopy(fallback_reasons),
        historical_group_candidates=list(method.GROUP_CANDIDATES),
        historical_group_candidate_count=len(method.GROUP_CANDIDATES),
        actual_group_fit_count=len(result["group_fit_runs"]),
        group_fit_exclusions=[dict(run=run, **deepcopy(fallback_reasons[run]))
            for run in method.GROUP_CANDIDATES if run not in observations],
        group_recomputed_per_fold=False, calibration_admitted=False, production_admitted=False)
    return result


def _review_rows(reader, bound):
    # The review is named by the recorded method (or its caller) with its exact bytes.
    bound = method._source(bound, "overflow review")
    review = method.checked_metadata(reader, bound, limit=bound["size_bytes"])
    if type(review) is not dict or type(review.get("rows")) is not list or type(review.get("skipped")) is not dict:
        raise ValueError("The exact original overflow review table is required")
    rows = {row["run"]:row for row in review["rows"]}
    if len(rows) != len(review["rows"]):
        raise ValueError("The original overflow review contains duplicate run rows")
    return review, rows


def _run_measurement(reader, scale_review, current_volumes, *, original_row, run):
    from .overflow_measurements import reviewed_curve, measured_ratio, CORRECTIONS
    from ..pipeline.staged_data import _same_json
    if type(current_volumes) is not list:
        raise ValueError("Current selected volume observations must be an explicit frame list")
    document = method.checked_metadata(reader, scale_review, limit=16*1024*1024)
    metadata = document.get("metadata") if type(document) is dict else None
    if (type(metadata) is not dict or metadata.get("run") != run
            or type(metadata.get("injection_start")) is not str or not metadata["injection_start"]):
        raise ValueError("The reviewed scale source belongs to another run or lacks its original injection origin")
    checked = reviewed_curve(document, run=run)
    qa_fields = ("t_end", "late_sd", "drift", "vA_pre", "q_pre")
    if (not checked["usable"] or original_row.get("run") != run
            or not _same_json(checked["qa"], {key:original_row[key] for key in qa_fields})):
        raise ValueError("The reviewed scale source no longer reproduces its original accepted feed QA")
    pre = [row for row in document["records"]
        if row.get("status") in ("Verified", "Corrected") and row.get("weight_g") is not None
        and row["time_from_injection_h"] < 0]
    anchor = max(pre, key=lambda row:row["time_from_injection_h"])
    source = dict(scale_review=deepcopy(scale_review), injection_start=metadata["injection_start"],
        original_feed_qa=deepcopy(checked["qa"]),
        original_feed_anchor=dict(image_time=anchor.get("image_time"),
            time_from_injection_h=anchor["time_from_injection_h"], original_weight_g=anchor["weight_g"],
            effective_weight_g=CORRECTIONS.get((run, anchor.get("image_time")), anchor["weight_g"]),
            status=anchor["status"]), post_sampling_evidence=None)
    if run in method.CONDITIONAL_ANCHOR_RUNS:
        return dict(measurement=None, source=source, fallback=dict(code=ANCHOR_UNPROVED,
            reason="No authenticated source evidence places this actual original feed anchor after pH sampling."))
    # Empty original selection is explicit no-current-volume evidence. Keep
    # measured_ratio's stricter nonempty/malformed input guards unchanged.
    measured = None if not current_volumes else measured_ratio(checked["curve"], current_volumes)
    fallback = None if measured is not None else dict(code=NO_CURRENT_VOLUME,
        reason="No selected current 1--4 h frame has at least 3 mL of model gas within the reviewed measurement interval.")
    return dict(measurement=measured, source=source, fallback=fallback)


def read_run_measurement(reader, scale_review, current_volumes, *, review, run):
    """One actual original weight source/QA join; no P1 or F admission.

    The same implementation is used by the complete table reader. Current
    volumes must still come from the owned original P1 source boundary; this
    lower source/measurement function grants no proof of their origin.
    """
    if run not in method.USABLE_RUNS:
        raise ValueError("This run has no admitted original usable overflow source")
    _document, rows = _review_rows(reader, review)
    return _run_measurement(reader, scale_review, current_volumes, original_row=rows[run], run=run)


def read_measurements(reader, method_bound, scale_reviews, current_volumes, *,
                      roles, analysis_runs, sampling_anchor_evidence=None, cohort=None):
    """Reproduce original feed/QA, then compare with current own P1 gas volumes.

    Sources are named explicitly; no source directory is searched. Old model
    volume/ratio columns in the review are never used. Historical scale files
    were not byte-bound by that old review: the returned descriptors therefore
    state the actual current reviewed source bytes, without a false historical
    byte-identity claim. The immutable preparation must retain all of them.

    No authenticated post-sampling annotation has been bound for AC34/AC41 in
    the admitted current source route. Their already-recorded conditional group
    fallback applies. A supplied but unsupported claimed proof refuses rather
    than granting own-ratio use or inventing an annotation/time.
    """
    method.validate_roles(roles, analysis_runs, cohort=cohort)
    if sampling_anchor_evidence is not None and (type(sampling_anchor_evidence) is not dict
                                                or sampling_anchor_evidence):
        raise ValueError("No supported original post-sampling anchor evidence is bound to this preparation")
    if (type(scale_reviews) is not dict or set(scale_reviews) != set(method.USABLE_RUNS)
            or type(current_volumes) is not dict or set(current_volumes) != set(method.USABLE_RUNS)):
        raise ValueError("Every declared usable run needs its explicit reviewed source and current volume selection")
    selected_method = method.checked_method(reader, method_bound)
    review, original_rows = _review_rows(reader, selected_method["overflow_review"])
    observations, sources = {}, {}
    fallback = {}
    for run in method.NO_USABLE_OVERFLOW:
        fallback[run] = dict(code=NO_OVERFLOW,
            reason=review["skipped"].get(run, "The original reviewed scale data do not pass the recorded measurement QA."))
    for run in method.USABLE_RUNS:
        measured = _run_measurement(reader, scale_reviews[run], current_volumes[run],
            original_row=original_rows[run], run=run)
        sources[run] = measured["source"]
        if measured["measurement"] is None:
            fallback[run] = measured["fallback"]
        else:
            observations[run] = measured["measurement"]
    result = observations_to_levels(observations, fallback, roles=roles, analysis_runs=analysis_runs, cohort=cohort)
    result.update(method=deepcopy(method_bound), overflow_review=deepcopy(selected_method["overflow_review"]),
        reviewed_sources=sources, reviewed_weight_metadata_replayed=True,
        original_p1_ownership_or_payload_admission_verified=False)
    return result
