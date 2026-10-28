"""One native review lifecycle over the original, unmodified evidence stores.

The desktop's ``review_session.json`` and training locks remain authoritative
for editing and evidence admission. This small registry reserves one session
per series/reviewer before the original review process starts. It never parses,
copies, combines or deletes annotations, validation boxes or image assets.
Attach the manifest before enabling the original review window. Discard and
training completion require the original review process to have retired.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from .review_storage import resolve_review_path
from .storage_paths import _is_absolute, machine_stored_text
from .reviewer_preferences import reviewer_identifier
from .phase_training.scope import series_id

FORMAT = "native-phase-review-lifecycle-v1"
SESSION_FORMAT = "fluidflower-local-review-session-v1"
SCRATCH_SOURCE_FORMAT = "native-phase-scratch-sources-v1"
_LIMIT = 4 * 1024 * 1024
_OPEN = {"active", "discarding"}
_editing_leases = threading.local()


def _now():
    return datetime.now(UTC).isoformat()


def _identity(series, reviewer):
    person = str(reviewer or "").strip()
    if not person or reviewer_identifier(person) == "anonymous":
        raise ValueError("Set a named human reviewer before opening a review session")
    return series_id(series), reviewer_identifier(person), person


def canonical_source_identity(binding, *, series=None):
    """Capture the verified photo selection without importing any model identity.

    File/manifest hashes are checked by the source preparation boundary. This
    identity preserves its exact roots, run filters and candidate selection for
    all later lifecycle actions. ``None`` retains an original all-runs filter;
    the explicit, nonempty candidate map still bounds that selection.
    """
    if not isinstance(binding, Mapping):
        raise ValueError("Scratch review requires a verified source binding")
    selected = series if series is not None else binding.get("series_id")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("Scratch source identity requires an experimental series")
    selected = series_id(selected)
    if binding.get("series_id") is not None and series_id(binding["series_id"]) != selected:
        raise ValueError("Scratch source identity belongs to another series")

    def runs(name):
        values = binding.get(name)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Scratch source run filters must contain named runs")
        result = [value.strip() for value in values]
        if len(set(result)) != len(result):
            raise ValueError("Scratch source run filters contain repeated runs")
        return sorted(result)

    first, second = runs("ffac_runs"), runs("ffum_runs")
    candidates = binding.get("selected_candidates")
    if not isinstance(candidates, Mapping) or not candidates or any(
            not isinstance(run, str) or not run.strip()
            or not isinstance(candidate, str) or not candidate.strip()
            for run, candidate in candidates.items()):
        raise ValueError("Scratch review requires explicit named source candidates")
    candidates = {run.strip(): candidate.strip() for run, candidate in candidates.items()}
    if len(candidates) != len(binding["selected_candidates"]):
        raise ValueError("Scratch source candidates contain repeated runs")
    explicit = set(first or ()) | set(second or ())
    if set(first or ()) & set(second or ()) or not explicit.issubset(candidates):
        raise ValueError("Scratch source runs and candidates disagree")
    if first is not None and second is not None and explicit != set(candidates):
        raise ValueError("Scratch source run filters must cover the selected candidates")

    roots = {}
    for name, selected_runs in (("combined_root", None), ("ffac_root", first), ("ffum_root", second)):
        raw = binding.get(name)
        if raw in (None, "") and selected_runs == []:
            roots[name] = ""
        elif not isinstance(raw, (str, os.PathLike)) or not str(raw).strip() or not Path(raw).is_absolute():
            raise ValueError("Scratch source roots must be absolute paths")
        else:
            roots[name] = str(Path(raw))
    return dict(format=SCRATCH_SOURCE_FORMAT, series_id=selected, **roots,
                selected_candidates=dict(sorted(candidates.items())), ffac_runs=first, ffum_runs=second)


def _review_binding(value, selected):
    basis = value.get("review_basis", "model")
    model = value.get("target_model_id")
    if basis == "scratch":
        if model != "":
            raise ValueError("A scratch review must not have a target model ID")
        source = canonical_source_identity(value.get("source_identity"), series=selected)
        if value.get("source_identity") != source:
            raise ValueError("Scratch review source identity is not canonical")
        return basis, source
    if basis != "model" or not isinstance(model, str) or not model.strip():
        raise ValueError("Review session has no valid model or scratch source binding")
    return basis, None


def _root(save_root):
    if not str(save_root or "").strip() or not Path(save_root).is_absolute():
        raise ValueError("An absolute original review save root is required")
    return resolve_review_path(save_root).resolve()


def review_session_registry_path(save_root, *, series, reviewer):
    selected, owner, _person = _identity(series, reviewer)
    return _root(save_root) / ".phase_workflow" / selected / (owner + ".json")


def _read(path):
    if path.stat().st_size > _LIMIT:
        raise ValueError("Review session metadata exceeds its size limit")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Review session metadata must be an object")
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _locked(path, *, busy_message="Another review session action is in progress; refresh and retry"):
    """Nonblocking OS lock; process exit releases it without deleting a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(busy_message) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def review_editing_lease(save_root, *, series, reviewer):
    """Hold across child startup/event loop, or parent preflight/freeze/launch.

    Another process or thread fails before it can change any session metadata.
    Reentrancy on the same thread allows an attachment or completion action
    inside the caller's existing lease. Lock files persist; process exit
    releases ownership automatically, without timers or stale-lock deletion.
    """
    path = review_session_registry_path(save_root, series=series, reviewer=reviewer)
    editor_path = path.with_name(path.stem + ".editor.json")
    key = os.path.normcase(str(editor_path))
    held = getattr(_editing_leases, "held", None)
    if held is None:
        held = _editing_leases.held = set()
    if key in held:
        yield
        return
    with _locked(editor_path,
            busy_message="Close the open model correction window before changing or training this review"):
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)


def _registry(path, selected, owner):
    if not path.exists():
        return dict(format=FORMAT, series=selected, reviewer_id=owner, sessions=[])
    value = _read(path)
    if (value.get("format") != FORMAT or value.get("series") != selected
            or value.get("reviewer_id") != owner or not isinstance(value.get("sessions"), list)):
        raise ValueError("Review session registry has another owner or an unreadable format")
    seen = set()
    for row in value["sessions"]:
        if (not isinstance(row, dict) or not isinstance(row.get("review_session_id"), str)
                or not row["review_session_id"] or row["review_session_id"] in seen
                or row.get("status") not in _OPEN | {"discarded", "submitted"}
                or not isinstance(row.get("manifest_paths"), list)
                or not isinstance(row.get("history"), list)
                or not isinstance(row.get("target_model_id"), str)):
            raise ValueError("Review session registry contains an invalid lifecycle")
        if any(not isinstance(p, str) or not _is_absolute(p) for p in row["manifest_paths"]):
            raise ValueError("Review session registry contains an invalid original manifest path")
        _review_binding(row, selected)
        seen.add(row["review_session_id"])
    if sum(row["status"] in _OPEN for row in value["sessions"]) > 1:
        raise ValueError("Several active review sessions require explicit resolution")
    return value


def read_review_session_registry(save_root, *, series, reviewer):
    selected, owner, _person = _identity(series, reviewer)
    path = review_session_registry_path(save_root, series=selected, reviewer=reviewer)
    return deepcopy(_registry(path, selected, owner))


def active_review_session(save_root, *, series, reviewer):
    value = read_review_session_registry(save_root, series=series, reviewer=reviewer)
    return next((row for row in value["sessions"] if row["status"] in _OPEN), None)


def read_current_review_session(save_root, *, series, reviewer):
    """Read the current logical review; ``None`` means none is reserved."""
    return active_review_session(save_root, series=series, reviewer=reviewer)


def _held_out(value):
    return bool(value.get("independent_holdout") or value.get("evidence_role") in {
        "blind_transfer_validation", "independent_blind_gas_validation"})


def _owned(value, selected, owner):
    person = value.get("reviewer") or value.get("reviewer_id") or ""
    if reviewer_identifier(person) != owner:
        return False
    assigned = value.get("series") or value.get("dataset")
    if assigned:
        return series_id(assigned) == selected
    # Old cross-series workspaces have no series field. Do not infer their
    # ownership from a model ID or a file name: callers must pass scoped rows
    # from the original discovery/binding reader.
    return True


def _session(path, save_root, selected, owner, *, session_id=None, target_model_id=None,
             review_basis=None, source_identity=None):
    path = resolve_review_path(path).resolve()
    if path.name != "review_session.json" or not path.is_relative_to(_root(save_root)):
        raise ValueError("Review session must be an original manifest in the selected review store")
    value = _read(path)
    if value.get("format") != SESSION_FORMAT or not _owned(value, selected, owner):
        raise ValueError("Review session has another owner or an unreadable format")
    decisions = resolve_review_path(value.get("decision_dir") or path.parent, context=path).resolve()
    if decisions != path.parent:
        raise ValueError("Review session points outside its original decision directory")
    if session_id is not None and value.get("review_session_id") != session_id:
        raise ValueError("Original review session identity differs from the reserved session")
    if target_model_id is not None and value.get("target_model_id") != target_model_id:
        raise ValueError("Original review session belongs to another model")
    if review_basis is not None:
        basis, source = _review_binding(value, selected)
        if basis != review_basis or source != source_identity:
            raise ValueError("Original review session has another review basis or source identity")
    elif value.get("review_basis") == "scratch":
        _review_binding(value, selected)
    return path, value


def _action(action, person, reason):
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("Give a reason for this review session action")
    return dict(action=action, person=person, reason=reason, utc=_now())


def _find(value, session_id):
    row = next((row for row in value["sessions"] if row["review_session_id"] == session_id), None)
    if row is None:
        raise ValueError("Review session is not registered for this series and reviewer")
    return row


def reserve_review_session(save_root, *, series, reviewer, target_model_id, existing_sessions=(),
                           review_basis="model", source_identity=None):
    """Reserve an ID before launch, or reuse this owner's identically bound ID.

    ``existing_sessions`` must be the freshly discovered original sessions for
    this review store and series. Editable legacy sessions are reported; they
    are never silently merged or selected by recency. Closed registry entries
    remain in history. Changing the model, basis or scratch sources requires
    submit or discard first. Scratch sessions explicitly have no target model.
    """
    selected, owner, person = _identity(series, reviewer)
    model = str(target_model_id or "").strip()
    if review_basis == "scratch":
        if model:
            raise ValueError("A scratch review must not have a target model ID")
        source = canonical_source_identity(source_identity, series=selected)
    elif review_basis == "model" and source_identity is None:
        source = None
    else:
        raise ValueError("Choose either a model review or a source-bound scratch review")
    if review_basis == "model" and not model:
        raise ValueError("Choose a model before opening a review session")
    path = review_session_registry_path(save_root, series=selected, reviewer=reviewer)
    with review_editing_lease(save_root, series=selected, reviewer=reviewer), _locked(path):
        value = _registry(path, selected, owner)
        active = next((row for row in value["sessions"] if row["status"] in _OPEN), None)
        conflicts = []
        for session in existing_sessions:
            if not isinstance(session, Mapping):
                raise ValueError("Review discovery returned invalid session metadata")
            discovered_path = session.get("session_path")
            if not discovered_path:
                raise ValueError("Refresh saved review sessions before opening a review")
            original = resolve_review_path(discovered_path).resolve()
            if not original.is_relative_to(_root(save_root)):
                continue
            # Discovery is asynchronous. Re-read its small policy document so
            # an old cached exclusion or lock cannot conceal an editable store.
            session = dict(_read(original), session_path=str(original))
            if session.get("format") != SESSION_FORMAT:
                raise ValueError("A discovered review has an unreadable session format")
            if (not _owned(session, selected, owner) or session.get("editing_locked")
                    or session.get("include_in_training") is False or _held_out(session)):
                continue
            if active and session.get("review_session_id") == active["review_session_id"]:
                continue
            conflicts.append(str(session.get("session_path") or session.get("review_session_id") or "unknown"))
        if conflicts:
            raise ValueError("Existing editable reviews must be submitted or explicitly discarded before a new review: "
                             + "; ".join(conflicts))
        if active:
            if active["status"] != "active":
                raise ValueError("Complete the pending discard before opening another review")
            active_basis, active_source = _review_binding(active, selected)
            if active["target_model_id"] != model:
                raise ValueError("Submit or discard the active review before choosing another model")
            if (active_basis, active_source) != (review_basis, source):
                raise ValueError("Submit or discard the active review before choosing another review basis or source")
            return deepcopy(active)
        now = _now()
        row = dict(review_session_id="review-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                   + "-" + uuid.uuid4().hex[:8], target_model_id=model, review_basis=review_basis, reviewer=person,
                   status="active", created_utc=now, updated_utc=now, manifest_paths=[], history=[])
        if source is not None:
            row["source_identity"] = source
        value["sessions"].append(row)
        _write(path, value)
        return deepcopy(row)


def attach_review_manifest(save_root, *, series, reviewer, review_session_id, session_path):
    """Attach an original store created by the original companion process."""
    selected, owner, _person = _identity(series, reviewer)
    path = review_session_registry_path(save_root, series=selected, reviewer=reviewer)
    with review_editing_lease(save_root, series=selected, reviewer=reviewer), _locked(path):
        value = _registry(path, selected, owner)
        row = _find(value, review_session_id)
        if row["status"] != "active":
            raise ValueError("A closed review session cannot receive another store")
        original, manifest = _session(session_path, save_root, selected, owner,
            session_id=review_session_id, target_model_id=row["target_model_id"],
            review_basis=row.get("review_basis", "model"), source_identity=row.get("source_identity"))
        if manifest.get("editing_locked") or manifest.get("include_in_training") is False or _held_out(manifest):
            raise ValueError("Locked or independent held-out evidence cannot become an active training review")
        stored = machine_stored_text(original)
        if stored not in row["manifest_paths"]:
            row["manifest_paths"].append(stored)
            row["updated_utc"] = _now()
            _write(path, value)
        return deepcopy(row)


def _discard_manifest(path, manifest, action):
    # The ordinary pending-evidence reader already excludes this policy. The
    # original review store checks editing_locked on every attempted save.
    history = list(manifest.get("review_lifecycle_history") or [])
    if action not in history:
        history.append(action)
    manifest.update(include_in_training=False, editing_locked=True,
        review_lifecycle_status="discarded", review_lifecycle_history=history,
        updated_utc=_now())
    _write(path, manifest)


def discard_review_session(save_root, *, series, reviewer, review_session_id, reason):
    """Close one logical review and exclude all its stores, retaining every box.

    A durable ``discarding`` intent blocks new reviews if a metadata write
    fails. Retrying the same action completes that intent; no source decision
    files are altered. A frozen training session cannot be discarded.
    """
    selected, owner, person = _identity(series, reviewer)
    action = _action("discarded review session", person, reason)
    path = review_session_registry_path(save_root, series=selected, reviewer=reviewer)
    with review_editing_lease(save_root, series=selected, reviewer=reviewer), _locked(path):
        value = _registry(path, selected, owner)
        row = _find(value, review_session_id)
        if row["status"] == "discarded":
            return deepcopy(row)
        if row["status"] == "submitted":
            raise ValueError("Evidence already submitted to training cannot be discarded")
        originals = [_session(p, save_root, selected, owner, session_id=review_session_id,
            target_model_id=row["target_model_id"], review_basis=row.get("review_basis", "model"),
            source_identity=row.get("source_identity")) for p in row["manifest_paths"]]
        if any(manifest.get("training_locks") or (manifest.get("editing_locked")
                and manifest.get("review_lifecycle_status") != "discarded")
                for _p, manifest in originals):
            raise ValueError("Evidence already frozen for training cannot be discarded")
        if row["status"] == "discarding":
            action = row["discard_action"]
        else:
            row.update(status="discarding", discard_action=action, updated_utc=_now())
            _write(path, value)
        for original, manifest in originals:
            _discard_manifest(original, manifest, action)
        row.update(status="discarded", updated_utc=_now())
        row["history"].append(action)
        _write(path, value)
        return deepcopy(row)


def discard_legacy_review_sessions(save_root, *, series, reviewer, session_paths, reason):
    """Explicitly close separate legacy stores without combining their files."""
    selected, owner, person = _identity(series, reviewer)
    action = _action("discarded review session", person, reason)
    path = review_session_registry_path(save_root, series=selected, reviewer=reviewer)
    with review_editing_lease(save_root, series=selected, reviewer=reviewer), _locked(path):
        value = _registry(path, selected, owner)
        registered = {p for row in value["sessions"] for p in row["manifest_paths"]}
        originals = [_session(p, save_root, selected, owner) for p in session_paths]
        if not originals:
            raise ValueError("Select the original review sessions to discard")
        for original, manifest in originals:
            if str(original) in registered or machine_stored_text(original) in registered:
                raise ValueError("Use the logical review session action for a registered store")
            if (manifest.get("training_locks") or _held_out(manifest)
                    or (manifest.get("editing_locked") and manifest.get("review_lifecycle_status") != "discarded")):
                raise ValueError("Training-frozen or independent held-out reviews cannot be discarded here")
        for original, manifest in originals:
            _discard_manifest(original, manifest, action)
        return [str(original) for original, _manifest in originals]


def require_training_session_admission(bundles):
    """Reject excluded original sessions before native request preparation.

    Wire this at the original request boundary, before snapshots/directories
    are written, as well as native readiness. Historical locked training data
    remain eligible; independent validation and discarded sessions do not.
    This is an additional guard, never a replacement for existing admission.
    """
    for decisions, _combined_manifest in bundles:
        decisions = resolve_review_path(decisions)
        if decisions.name != "pairwise_review_decisions.json":
            continue
        path = decisions.parent / "review_session.json"
        if not path.exists():
            continue
        manifest = _read(path)
        if manifest.get("format") != SESSION_FORMAT:
            raise ValueError("Selected review has an unreadable session format")
        if manifest.get("include_in_training") is False or manifest.get("review_lifecycle_status") == "discarded":
            raise ValueError("Selected review session is excluded from training: " + str(path))
        if _held_out(manifest):
            raise ValueError("Independent held-out validation cannot be used for training: " + str(path))


def _require_frozen_revision(decisions, record, lock):
    """Verify original bytes and the derived training view without altering either."""
    def file_sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    frozen_sha = str(record.get("sha256") or "")
    snapshot = record.get("snapshot_path")
    if (not snapshot or len(frozen_sha) != 64
            or lock.get("frozen_decisions_sha256") != frozen_sha
            or file_sha(resolve_review_path(snapshot)) != frozen_sha):
        raise ValueError("The original training snapshot no longer matches its review lock")
    current_sha = file_sha(decisions)
    if lock.get("source_decisions_sha256") != current_sha:
        raise ValueError("The live review changed after its original training lock")
    if lock.get("source_matches_frozen_revision") is True and current_sha == frozen_sha:
        return
    # Original preparation can filter withdrawn/reviewer evidence and freeze
    # sampling dependencies. Its training view then intentionally differs from
    # the complete, separately captured original. Both hashes must still agree
    # with that request and the lock's observed live source.
    original_snapshot = record.get("original_snapshot_path")
    if (record.get("derived_training_view") is not True or not original_snapshot
            or record.get("source_sha256") != current_sha
            or record.get("original_snapshot_sha256") != current_sha
            or file_sha(resolve_review_path(original_snapshot)) != current_sha):
        raise ValueError("The live review does not match the original frozen evidence revision")


def complete_review_session_training(save_root, *, series, reviewer, review_session_id, request, reason):
    """Close only after the original request writer has frozen and locked inputs.

    This method does not mint locks or assert scientific acceptance. Every
    registered original store must be present in the request and carry the
    original lock for that exact model and unchanged frozen source revision;
    partial selections remain active. Derived training views retain the original
    preparer's separately verified source snapshot.
    """
    selected, owner, person = _identity(series, reviewer)
    action = _action("submitted review session to training", person, reason)
    model = str(request.get("model_id") or "").strip()
    if not model:
        raise ValueError("Training request has no model ID")
    sources = {}
    for bundle in request.get("bundles") or []:
        if not isinstance(bundle, Mapping) or not isinstance(bundle.get("decisions"), Mapping):
            continue
        record = bundle["decisions"]
        if record.get("source_path"):
            source = str(resolve_review_path(record["source_path"]).resolve())
            if source in sources and sources[source] != record:
                raise ValueError("One live review has conflicting frozen training revisions")
            sources[source] = record
    path = review_session_registry_path(save_root, series=selected, reviewer=reviewer)
    with review_editing_lease(save_root, series=selected, reviewer=reviewer), _locked(path):
        value = _registry(path, selected, owner)
        row = _find(value, review_session_id)
        if row["status"] == "submitted" and row.get("submitted_model_id") == model:
            return deepcopy(row)
        if row["status"] != "active" or not row["manifest_paths"]:
            raise ValueError("An active review with saved evidence is required for submission")
        for original_path in row["manifest_paths"]:
            original, manifest = _session(original_path, save_root, selected, owner,
                session_id=review_session_id, target_model_id=row["target_model_id"],
                review_basis=row.get("review_basis", "model"), source_identity=row.get("source_identity"))
            if manifest.get("include_in_training") is False or _held_out(manifest):
                raise ValueError("Excluded or independent evidence cannot close a training review")
            decisions = original.parent / "pairwise_review_decisions.json"
            if str(decisions) not in sources:
                raise ValueError("All stores in this review must be frozen before submission")
            locks = [lock for lock in manifest.get("training_locks") or []
                     if isinstance(lock, Mapping) and lock.get("model_id") == model]
            if not manifest.get("editing_locked") or len(locks) != 1:
                raise ValueError("The original training request has not locked this review")
            _require_frozen_revision(decisions, sources[str(decisions)], locks[0])
        row.update(status="submitted", submitted_model_id=model,
                   training_request=str(request.get("request_path") or ""), updated_utc=_now())
        row["history"].append(action)
        _write(path, value)
        return deepcopy(row)
