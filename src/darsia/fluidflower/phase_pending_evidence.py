"""Read-only accounting of saved labels against a selected model's evidence.

Counts describe human labels eligible for capture, not extracted pixels or
guaranteed distinct training samples. Image/geometry checks remain the trainer's
responsibility. Missing receipts or ledgers are unknown, never zero evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path

from . import phase_review_retractions as withdrawals
from .phase_contributions import _records, regional_inventory
from .phase_evidence_audit import _verified_pre_geometry_hash, superseded_evidence_ids
from .phase_training.logistic import CONFIDENCE_WEIGHTS, canonical_phase_class
from .review_storage import resolve_review_path

FORMAT = "phase-pending-evidence-v1"
_GEOMETRY = {"coordinate_space", "shape", "points", "x", "y", "width", "height", "frame_index", "elapsed_h"}


def _read(path, checksum=None):
    raw = resolve_review_path(path).read_bytes()
    if checksum and hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError(f"Training evidence checksum mismatch: {path}")
    return json.loads(raw.decode("utf-8-sig"))


def _signature(run, source, payload):
    field = "annotations" if source == "manual" else "phase_quiz_answers"
    rows = list(_records({"decisions": {run: {field: [payload]}}}))
    return rows[0][3] if rows else None


def _record(run, source, payload, series):
    run = str(run).casefold()
    identifier = payload.get("id") or payload.get("proposal_id")
    if not identifier:
        # This is also the existing trainer's fallback identity.
        from .phase_training.trainer import _geometry_key
        identifier = _geometry_key(run, payload)[:16]
    region_id = f"{run}:{source}:{identifier}"
    normalized_payload = dict(payload)
    if not normalized_payload.get("id") and not normalized_payload.get("proposal_id"):
        normalized_payload["id"] = identifier
    target = withdrawals._target(series, run, source, normalized_payload)
    logical_id = (target["series"], target["run"], target["kind"], target["identifier"])
    category = canonical_phase_class(payload, source)
    signatures = {_signature(run, source, payload)} - {None}
    if _verified_pre_geometry_hash(payload) is not None:
        geometry = payload["geometry_migration"]
        original = {key: value for key, value in payload.items()
                    if key not in _GEOMETRY and key != "geometry_migration"}
        original.update(geometry["source_geometry"])
        bridge = geometry.get("supersession_rebinding")
        if bridge:
            original["evidence_audit"] = bridge["original_evidence_audit"]
        signatures.add(_signature(run, source, original))
        signatures.discard(None)
    return dict(region_id=region_id, logical_id=logical_id, target=target,
        run=run, source=source, kind="blind_labels" if target["kind"] == "quiz" else "visible_regions",
        payload=payload, category=category, signatures=signatures,
        eligible=bool(category and category != "ignored"
                      and CONFIDENCE_WEIGHTS.get(str(payload.get("confidence") or "low").lower(), 0) > 0))


def _document_records(document, series):
    for run, decision in (document.get("decisions") or {}).items():
        for field, source in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
            for payload in decision.get(field) or []:
                yield _record(run, source, payload, series)


def _timestamp(value):
    if not value:
        return 0.
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.


def summarize_pending_evidence(consumed_regions, session_documents, *, series,
                               withdrawal_snapshot, applied_snapshot=None, historical_documents=(),
                               captured_regions=(), unused_reasons=None):
    """Pure accounting over verified payloads; neither reads nor writes files.

    ``consumed_regions`` are actual positively weighted model regions with
    ``run``, ``source`` and ``payload``. ``session_documents`` contain session
    metadata, ``document``, and ``eligible_for_training`` set by the caller.
    Ineligible independent holdouts remain countable in their separate column.
    """
    current_policy = withdrawals.validate_snapshot(withdrawal_snapshot)
    if current_policy["series"] not in {withdrawals._series(series), "all"}:
        raise ValueError("Evidence ledger belongs to another series")
    applied = withdrawals.validate_snapshot(applied_snapshot or withdrawals.empty_snapshot(series))
    withdrawals._assert_policy_extends(current_policy, applied)
    consumed = {}
    evidence_records = []
    for region in consumed_regions:
        record = _record(region["run"], region["source"], region["payload"], series)
        if not record["signatures"]:
            raise ValueError("Consumed region has no verifiable classification fingerprint")
        if record["logical_id"] in consumed and consumed[record["logical_id"]]["signatures"] != record["signatures"]:
            raise ValueError("Consumed model contains conflicting revisions of one label")
        consumed[record["logical_id"]] = record
        evidence_records.append((record["region_id"], record["payload"]))
    for document in historical_documents:
        evidence_records.extend((row["region_id"], row["payload"]) for row in _document_records(document, series))
    captured = dict(consumed)
    for region in captured_regions:
        record = _record(region["run"], region["source"], region["payload"], series)
        captured[record["logical_id"]] = record
        evidence_records.append((record["region_id"], record["payload"]))
    current, holdouts, excluded = {}, {}, Counter()
    eligible_session_candidates, session_pending = {}, defaultdict(Counter)
    for session_index, session in enumerate(session_documents):
        is_eligible = bool(session.get("eligible_for_training"))
        is_holdout = bool(session.get("independent_holdout")
                          or session.get("evidence_role") == "blind_transfer_validation")
        if is_eligible and not is_holdout:
            eligible_session_candidates[session_index] = {key: value for key, value in session.items() if key != "document"}
        for record in _document_records(session["document"], series):
            payload = record["payload"]
            explicit_time = max(_timestamp(payload.get("updated_utc")), _timestamp(payload.get("reclassified_utc")))
            rank = (explicit_time, float(session.get("modified_timestamp") or 0))
            record["rank"] = rank
            record["session_index"] = session_index
            if is_holdout:
                if record["logical_id"] not in holdouts or rank > holdouts[record["logical_id"]]["rank"]:
                    holdouts[record["logical_id"]] = record
                continue
            if not is_eligible:
                excluded["not_enrolled_or_locked"] += 1
                continue
            evidence_records.append((record["region_id"], record["payload"]))
            incumbent = current.get(record["logical_id"])
            if incumbent is None or rank > incumbent["rank"]:
                current[record["logical_id"]] = record
            elif rank == incumbent["rank"] and record["signatures"] != incumbent["signatures"]:
                raise ValueError("Conflicting equally recent saved revisions: " + record["region_id"])
    replacement_targets = {}
    for event in withdrawals.active_replacements(current_policy):
        if event["target"]["series"] != withdrawals._series(series):
            continue
        replacement = event["replacement"]
        record = _record(replacement["run"], "manual", replacement["annotation"], series)
        record["rank"] = (float("inf"), float("inf"))
        current[record["logical_id"]] = record
        replacement_targets[record["logical_id"]] = event["target"]
        evidence_records.append((record["region_id"], record["payload"]))
    superseded = superseded_evidence_ids(evidence_records)
    withdrawn_keys = {tuple(target[key] for key in ("series", "run", "kind", "identifier"))
                      for target in current_policy["targets"] if target["kind"] in {"manual", "quiz"}
                      and target["series"] == withdrawals._series(series)}
    by_run = defaultdict(lambda: Counter())
    pending_categories = defaultdict(Counter)
    pending, added, changed, matched = Counter(), Counter(), Counter(), Counter()
    pending_ids = []
    for key, record in current.items():
        if key in withdrawn_keys:
            excluded["withdrawn"] += 1
            continue
        if record["region_id"] in superseded:
            excluded["superseded"] += 1
            continue
        if not record["eligible"]:
            excluded["ignored_or_zero_weight"] += 1
            continue
        previous = consumed.get(key)
        kind = record["kind"]
        if previous and record["signatures"] & previous["signatures"]:
            matched[kind] += 1
            by_run[record["run"]]["consumed_" + kind] += 1
            continue
        previous_capture = captured.get(key)
        if previous_capture and record["signatures"] & previous_capture["signatures"]:
            # An unchanged annotation with conflicting/empty extraction does
            # not become new data merely because it had zero fitted weight.
            excluded["already_captured_without_contribution"] += 1
            continue
        original_id = (record["payload"].get("evidence_audit") or {}).get("original_region_id")
        replaces_used = any(row["region_id"] == original_id for row in consumed.values()) if original_id else False
        replacement = replacement_targets.get(key)
        if replacement:
            replaces_used |= tuple(replacement[name] for name in ("series", "run", "kind", "identifier")) in consumed
        change = changed if previous or previous_capture or replaces_used else added
        change[kind] += 1
        pending[kind] += 1
        by_run[record["run"]][kind] += 1
        pending_categories[record["run"]][record["category"]] += 1
        pending_ids.append(record["region_id"])
        if "session_index" in record:
            session_pending[record["session_index"]][kind] += 1
    consumed_counts = Counter(record["kind"] for record in consumed.values())
    unused = [row for key, row in captured.items() if key not in consumed
              and row["eligible"] and key not in withdrawn_keys and row["region_id"] not in superseded]
    unused_counts = Counter(row["kind"] for row in unused)
    unused_reason_counts = Counter((unused_reasons or {}).get(row["region_id"], "not_in_fitted_inventory") for row in unused)
    def counts(counter):
        values = {key: int(counter.get(key, 0)) for key in ("visible_regions", "blind_labels")}
        return dict(values, total=sum(values.values()))
    applied_events = {row["event_id"] for row in applied["events"]}
    policy_changes = [row for row in current_policy["events"] if row["event_id"] not in applied_events
                      and row["target"]["series"] == withdrawals._series(series)]
    pending_removals = {key for key, row in consumed.items()
        if key in withdrawn_keys or row["region_id"] in superseded
        or (key in current and not current[key]["eligible"])}
    eligible_sessions = [dict(eligible_session_candidates[index],
        pending_visible_regions=int(count.get("visible_regions", 0)),
        pending_blind_labels=int(count.get("blind_labels", 0)), pending_total=int(sum(count.values())))
        for index, count in session_pending.items() if sum(count.values())]
    return dict(format=FORMAT, known=True, pending=counts(pending), added=counts(added), changed=counts(changed),
        consumed=counts(consumed_counts), current_matching_consumed=counts(matched),
        captured_unused=counts(unused_counts), captured_unused_reasons=dict(unused_reason_counts),
        captured_unused_region_ids=sorted(row["region_id"] for row in unused),
        independent_holdout_labels=sum(row["eligible"] and key not in withdrawn_keys for key, row in holdouts.items()),
        excluded=dict(excluded),
        withdrawn=len(withdrawn_keys), pending_policy_events=len(policy_changes),
        pending_removed_regions=len(pending_removals),
        has_changes=bool(sum(pending.values()) or policy_changes or pending_removals),
        pending_region_ids=sorted(pending_ids), runs={run: dict(row) for run, row in by_run.items()},
        pending_categories_by_run={run: dict(row) for run, row in pending_categories.items()},
        eligible_sessions=eligible_sessions, errors=[],
        description="Unique saved training-eligible labels not represented by this model; extraction and geometry checks still apply.")


def _model_context(root):
    family, seen, todo = set(), set(), [root]
    selected_request = None
    while todo:
        folder = resolve_review_path(todo.pop()).resolve()
        if folder in seen:
            continue
        if len(seen) >= 64:
            raise ValueError("Training lineage is unexpectedly large")
        seen.add(folder)
        request = _read(folder / "training_request.json")
        if selected_request is None:
            selected_request = request
        family.add(str(request.get("model_id") or folder.name))
        settings = request.get("settings") or {}
        todo.extend(settings[key] for key in ("parent_model", "evidence_parent_model", "physics_parent_model")
                    if settings.get(key))
    return family, selected_request


def _has_training_inventory(root):
    """Check the accounting contract before opening historical lineage paths.

    An inference revision may contain frozen heads and maps without the fitted
    region receipts. File presence only permits the existing inventory reader
    to proceed; it does not replace its checksum or coverage validation.
    """
    if (root / "region_inventory.csv").is_file():
        return True
    return all((root / name).is_file() for name in (
        "regional_model.json", "evidence.json", "extraction.json",
        "regional_training_protocol.json"))


def _series_document(document, manifest, series):
    """Use the same manifest dataset binding as the trainer, never name heuristics."""
    datasets = defaultdict(set)
    for job in manifest.get("jobs") or []:
        if job.get("run") and job.get("dataset"):
            datasets[str(job["run"]).casefold()].add(withdrawals._series(job["dataset"]))
    selected = {}
    for run, decision in (document.get("decisions") or {}).items():
        assigned = datasets[str(run).casefold()]
        if len(assigned) != 1:
            raise ValueError("Missing or conflicting review dataset binding: " + str(run))
        if withdrawals._series(series) in assigned:
            selected[run] = decision
    return dict(document, decisions=selected)


def _frozen_documents(root, request, series):
    result = []
    for bundle in request.get("bundles") or []:
        values = {}
        for role in ("decisions", "combined_manifest"):
            record = bundle.get(role) or {}
            path = resolve_review_path(record.get("snapshot_path") or "")
            if not path.is_file():
                path = root / "input_snapshot" / path.name
            if not record.get("sha256"):
                raise ValueError("Frozen training evidence lacks a checksum: " + role)
            values[role] = _read(path, record["sha256"])
        result.append(_series_document(values["decisions"], values["combined_manifest"], series))
    return result


def _consumed_regions(root, documents, series):
    inventory = regional_inventory(root)
    if inventory is not None:
        model = _read(root / "regional_model.json")
        evidence = _read(root / "evidence.json", model["evidence_sha256"])
        used = {row["region_id"] for row in inventory}
        return [row for row in evidence["regions"] if row["region_id"] in used
                and withdrawals._series(row["dataset"]) == withdrawals._series(series)]
    path = root / "region_inventory.csv"
    if not path.is_file():
        raise ValueError("Selected model has no verifiable training-region inventory")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        used = {row["region_id"]: row for row in csv.DictReader(stream)
                if float(row.get("supervision_weight") or 0) > 0
                and (not row.get("dataset") or withdrawals._series(row["dataset"]) == withdrawals._series(series))}
    selected_runs = {str(run).casefold() for doc in documents for run in doc.get("decisions", {})}
    used = {key: row for key, row in used.items() if str(row.get("run")).casefold() in selected_runs}
    result = {}
    for document in documents:
        for run, decision in (document.get("decisions") or {}).items():
            for field, source in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
                for payload in decision.get(field) or []:
                    records = list(_records({"decisions": {run: {field: [payload]}}}))
                    if not records or records[0][0] not in used:
                        continue
                    key, _, category, signature = records[0]
                    if category != used[key]["phase_class"]:
                        continue
                    if key in result and result[key]["signature"] != signature:
                        raise ValueError("Legacy inventory cannot distinguish consumed revisions: " + key)
                    result[key] = dict(run=str(run).casefold(), source=source, payload=payload, signature=signature)
    if set(result) != set(used):
        raise ValueError("Frozen decisions do not explain every used training region")
    return list(result.values())


def _registered_promotions(family, series):
    """Verify saved enrollment metadata; full native-image checks stay in capture."""
    from . import phase_saved_promotions as promotions
    result = []
    for path in sorted(promotions.promotion_root(series).glob("*.json")):
        stored = _read(path)
        if withdrawals._series(stored.get("dataset")) != withdrawals._series(series) or stored.get("source_model_id") not in family:
            continue
        if stored.get("format") != promotions.FORMAT or promotions._key(stored) != path.stem:
            raise ValueError("Training enrollment receipt changed: " + str(path))
        checked = promotions.registered_promotion(stored["evaluation_root"], stored["source_model_root"],
                                                  stored["reviewer"], stored["save_root"])
        if checked is None:
            raise ValueError("Training enrollment can no longer be verified: " + str(path))
        result.append(checked)
    return result


def _verify_promoted_session(session, enrollment):
    if (session.get("target_model_id") != enrollment["evaluation_model_id"]
            or str(session.get("reviewer") or "").casefold() != enrollment["reviewer"].casefold()
            or resolve_review_path(session.get("combined_root") or "") != resolve_review_path(enrollment["combined_root"])
            or session.get("review_surface") != "open_correction"):
        raise ValueError("Enrolled review session changed identity or image source")
    context = session.get("session_path")
    decisions = resolve_review_path(session.get("decision_path") or "", context=context)
    store = resolve_review_path(enrollment["save_root"], context=context)
    if not decisions.is_relative_to(store):
        raise ValueError("Enrolled decisions are outside their selected review store")


def pending_training_evidence(model_root, review_sessions, *, series, withdrawal_snapshot=None,
                              promoted_sessions=()):
    """Worker-side receipt/ledger read for the Studio front-page indicator.

    Saved explicit enrollments are checked through the existing metadata-only
    promotion API. Optional ``promoted_sessions`` must also be receipt-backed
    enrollments verified by the caller. Independent blind-transfer validation
    remains excluded regardless.
    No writes, model loads, image reads or training are performed.
    """
    try:
        root = resolve_review_path(model_root).resolve()
        if not _has_training_inventory(root):
            raise ValueError(
                "Selected model has no verifiable training-region inventory. "
                "Saved-label training accounting is unavailable; saved phase "
                "maps can still be reviewed.")
        family, request = _model_context(root)
        documents = _frozen_documents(root, request, series)
        consumed = _consumed_regions(root, documents, series)
        captured, unused_reasons = consumed, {}
        if (root / "regional_model.json").is_file():
            model = _read(root / "regional_model.json")
            evidence = _read(root / "evidence.json", model["evidence_sha256"])
            captured = [row for row in evidence["regions"] if withdrawals._series(row["dataset"]) == withdrawals._series(series)]
            protocol = _read(root / "regional_training_protocol.json", model["protocol_sha256"])
            extraction = _read(root / "extraction.json", protocol.get("extraction_sha256"))
            unused_reasons = {row["region_id"]: row.get("status") or "not_in_fitted_inventory"
                              for row in extraction.get("coverage", [])}
        policy = withdrawal_snapshot if withdrawal_snapshot is not None else withdrawals.load_retractions(series=series)
        applied = withdrawals.model_retractions(root, series=series)
        enrollments = [*_registered_promotions(family, series), *promoted_sessions]
        promoted = {str(row["review_session_id"]): row for row in enrollments if row.get("review_session_id")}
        review_sessions = list(review_sessions)
        if promoted:
            from .phase_workflow import discover_phase_review_sessions
            stores = {resolve_review_path(row["save_root"]) for row in promoted.values()}
            review_sessions.extend(discover_phase_review_sessions(stores))
        sessions, empty_sessions, excluded_sessions = [], [], []
        for raw in review_sessions:
            session = dict(raw)
            enrolled = str(session.get("review_session_id")) in promoted
            if enrolled:
                _verify_promoted_session(session, promoted[str(session["review_session_id"])])
            belongs = session.get("target_model_id") in family or (
                enrolled and promoted[str(session["review_session_id"])]["source_model_id"] in family)
            # Independent evaluations can be bound through source_model_id,
            # while their evaluation ID intentionally is outside the lineage.
            holdout = bool(session.get("independent_holdout") or session.get("evidence_role") == "blind_transfer_validation")
            if not belongs and not (holdout and session.get("source_model_id") in family):
                continue
            session_series = session.get("dataset") or session.get("series")
            if session_series and isinstance(session_series, str) and withdrawals._series(session_series) != withdrawals._series(series):
                continue
            if not holdout and (session.get("editing_locked") or (session.get("include_in_training") is False and not enrolled)):
                excluded_sessions.append(session.get("review_session_id"))
                continue
            path = resolve_review_path(session.get("decision_path") or "", context=session.get("session_path"))
            if not path.is_file():
                if any(int(session.get(key) or 0) for key in ("manual_regions", "blind_region_labels", "frame_verdicts")):
                    raise ValueError("Saved review evidence is unavailable: " + str(path))
                # Merely opening a fresh session creates metadata before any
                # decision file. Capture skips precisely these empty sessions.
                empty_sessions.append(session.get("review_session_id"))
                continue
            manifest_path = resolve_review_path(session.get("combined_root") or "", context=session.get("session_path")) / "postflight_manifest.json"
            session["document"] = _series_document(_read(path), _read(manifest_path), series)
            session["eligible_for_training"] = (not holdout and not session.get("editing_locked")
                and (session.get("include_in_training") is not False or enrolled))
            sessions.append(session)
        result = summarize_pending_evidence(consumed, sessions, series=series, withdrawal_snapshot=policy,
                                            applied_snapshot=applied, historical_documents=documents,
                                            captured_regions=captured, unused_reasons=unused_reasons)
        result["empty_sessions"] = empty_sessions
        result["excluded_sessions"] = excluded_sessions
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(format=FORMAT, known=False, pending=None, added=None, changed=None, consumed=None,
                    has_changes=None, eligible_sessions=[], runs={}, errors=[str(error)])
