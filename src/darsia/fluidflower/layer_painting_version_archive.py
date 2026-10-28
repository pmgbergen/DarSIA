"""Drawing versions and reversible version actions for the painting view.

Version content stays immutable. Names, notes and reversible archive actions
are append-only events; an event never constitutes a map approval. Only the
new GUI adapter uses this module. Existing session and version readers stay
unchanged, and no file is deleted by these operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from . import layer_painting as lp
from . import layer_painting_manual as manual
from .project import utc_now

DRAWING_KIND = "drawing"
_LIMIT = 16 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}")


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Repeated version metadata key")
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError("Nonfinite version metadata")
    with Path(path).open("rb") as stream:
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT:
        raise ValueError("Version metadata exceeds the bounded read")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Version metadata must be an object")
    return value


def _safe(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Version actions need absolute paths without parent traversal")
    for candidate in reversed((path, *path.parents)):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if candidate.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Version actions refuse symlink or reparse paths")
    return path


def _roots(project, run_id):
    project_root = _safe(Path(project.root)).resolve(strict=True)
    painting = _safe(lp.painting_folder(project)).resolve(strict=False)
    versions = _safe(lp.versions_folder(project, run_id)).resolve(strict=False)
    deleted = _safe(painting / "deleted" / lp._file_stem(run_id)).resolve(strict=False)
    if not painting.is_relative_to(project_root) or not versions.is_relative_to(painting) or not deleted.is_relative_to(painting):
        raise ValueError("Version roots escape the intended geometry package")
    return versions, deleted


def _child(path, root, *, exists):
    path = _safe(path).resolve(strict=exists)
    root = _safe(root).resolve(strict=False)
    if path.parent != root or path == root:
        raise ValueError("The version is outside its intended direct-child root")
    if exists and not path.is_dir():
        raise ValueError("The selected version folder is unavailable")
    return path


def _write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf8")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _events(project, run_id):
    # Preserve the strict original parser: a damaged line never means unapproved.
    path = _safe(lp.approvals_path(project, run_id))
    if path.exists() and (not path.is_file() or path.stat().st_size > _LIMIT):
        raise ValueError("The approvals log is unavailable or exceeds the metadata limit")
    return lp.read_map_approvals(project, run_id)


def _event(project, record, action, **extra):
    # A metadata event must never append to or conceal damaged approval
    # history. Reading immutable version bytes remains possible separately.
    _events(project, record["run_id"])
    row = dict(format=lp.LAYER_APPROVAL_FORMAT, event=action,
        event_id=uuid.uuid4().hex, recorded_utc=utc_now(), run_id=record["run_id"],
        version=Path(record["folder"]).name, version_sha256=record["sha256"],
        version_record_sha256=lp._file_sha256(Path(record["folder"]) / "version.json"),
        **extra)
    # No top-level sha256: map_approval() must never treat an
    # archive, rename, note or restore event as an approval.
    line = (json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf8")
    path = _safe(lp.approvals_path(project, record["run_id"]))
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return row


@contextmanager
def _action_lock(project, run_id):
    from .device_choice import _try_lock, _unlock
    versions, _deleted = _roots(project, run_id)
    if not versions.is_dir():
        raise ValueError("The run has no saved version directory")
    path = _safe(versions / ".version_actions.lock")
    with path.open("a+b") as stream:
        if not _try_lock(stream):
            raise ValueError("Another version action is in progress")
        try:
            yield
        finally:
            _unlock(stream)


def _drawing_payload(session):
    value = session.to_dict()
    # New drawing versions preserve the in-memory raw coordinates; they do
    # not pass through the old polygon/stroke display rounding.
    value["polygons"] = [[int(row.layer_id), [[float(r), float(c)] for r, c in row.points]]
                         for row in session.polygons]
    value["strokes"] = [[int(row.layer_id), float(row.radius_px),
                         [[float(r), float(c)] for r, c in row.points]] for row in session.strokes]
    return value


def _drawing_digest(value):
    value = deepcopy(value)
    value.pop("updated_utc", None)
    value.pop("updated_by", None)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def _proposal_loader(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["meta"]))
    if metadata.get("format") == lp.LAYER_PROPOSAL_FORMAT:
        return lp.LayerProposal.load(path)
    if metadata.get("format") == "darsia-fluidflower-polygons-only-proposal-v1":
        from .layer_painting_polygons_only import load_proposal
        return load_proposal(path)
    raise ValueError("The map version contains an unsupported proposal format")


def _load_map_payload(record, proposal_loader=None):
    folder = Path(record["folder"])
    session = lp.PaintingSession.from_dict(_json(folder / "session.json"))
    proposal = (proposal_loader or _proposal_loader)(folder / "proposal.npz")
    payload = _json(folder / "corrections.json")
    corrections = [lp.Correction.from_dict(row) for row in payload.get("corrections", [])]
    return session, proposal, corrections


def _saved_map_digest(folder, proposal, corrections):
    payload = _json(Path(folder) / "session.json")
    # The old session loader supplies missing facies defaults. Hash the
    # original saved DATA through the unchanged digest function, so that
    # those load-time defaults cannot invalidate a legitimate older version.
    class SavedSession:
        def to_dict(self):
            return deepcopy(payload)
    return lp.map_digest(SavedSession(), proposal, corrections)


def _checked(project, run_id, selected, *, deleted=False, content=True, proposal_loader=None):
    versions, archived = _roots(project, run_id)
    folder = _child(Path(str(selected["folder"])), archived if deleted else versions, exists=True)
    record = _json(folder / "version.json")
    if (record.get("format") != lp.LAYER_VERSION_FORMAT or record.get("run_id") != run_id
            or record.get("kind") not in {DRAWING_KIND, "proposal", "edited map", manual.MANUAL_MAP_KIND}
            or not isinstance(record.get("sha256"), str) or _SHA.fullmatch(record["sha256"]) is None):
        raise ValueError("The actual saved version has another run, kind or SHA-256")
    for key in ("run_id", "sha256", "kind", "name"):
        if key in selected and selected[key] != record.get(key):
            raise ValueError("The selected version changed; refresh its metadata before continuing")
    expected = ({"session.json"} if record["kind"] == DRAWING_KIND else
                set(manual.MANUAL_VERSION_FILES) if manual.is_manual_version(record) else set(lp.VERSION_FILES))
    if not isinstance(record.get("files"), dict) or set(record["files"]) != expected:
        raise ValueError("The saved version does not list its exact original files")
    children = list(folder.iterdir())
    if {path.name for path in children} != expected | {"version.json"}:
        raise ValueError("The saved version has unlisted files or folders")
    for path in children:
        _safe(path)
        if not path.is_file():
            raise ValueError("The saved version must contain only its ordinary files")
    for leaf, digest in record["files"].items():
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None or lp._file_sha256(folder / leaf) != digest:
            raise ValueError("A saved version file differs from its SHA-256")
    record = dict(record, folder=str(folder))
    if content:
        if record["kind"] == DRAWING_KIND:
            payload = _json(folder / "session.json")
            session = lp.PaintingSession.from_dict(payload)
            digest = _drawing_digest(payload)
        elif manual.is_manual_version(record):
            session, map_value = manual.load_manual_version(record)
            digest = manual.manual_map_digest(session, map_value)
        else:
            session, proposal, corrections = _load_map_payload(record, proposal_loader)
            digest = _saved_map_digest(folder, proposal, corrections)
        if session.run_id != run_id or digest != record["sha256"]:
            raise ValueError("The version's content does not match its actual run or map SHA-256")
    return record


def _cites(value, digest):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"map_version", "approval", "version"} and isinstance(item, dict) and item.get("sha256") == digest:
                return True
            if key in {"map_version_sha256", "map_sha256"} and item == digest:
                return True
            if _cites(item, digest):
                return True
    elif isinstance(value, list):
        return any(_cites(item, digest) for item in value)
    return False


def _reference_metadata(project, reference_files):
    base = Path(project.root)
    for name, label in (("map_placement", "placed"), ("map_adoption", "prepared for adoption")):
        root = base / "annotations" / name
        _safe(root)
        if root.is_dir():
            children = list(root.iterdir())
            if len(children) > 4096:
                raise ValueError("Version protection metadata exceeds the bounded folder count")
            for child in children:
                _safe(child)
                index = child / "index.json"
                if child.is_dir() and index.exists():
                    yield label, _json(_safe(index)), str(index)
    root = base / "overrides" / "geometry_facies"
    for leaf in ("current.json", "revisions.jsonl"):
        path = _safe(root / leaf)
        if not path.exists():
            continue
        if leaf.endswith(".jsonl"):
            with path.open("rb") as stream:
                raw = stream.read(_LIMIT + 1)
            if len(raw) > _LIMIT:
                raise ValueError("Geometry revision history exceeds the bounded read")
            for line in raw.decode("utf8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Geometry revision history is malformed")
                    yield "adopted", value, str(path)
        else:
            yield "adopted", _json(path), str(path)
    root = _safe(base / "results" / "geometry")
    if root.is_dir():
        children = list(root.iterdir())
        if len(children) > 4096:
            raise ValueError("Geometry history exceeds the bounded folder count")
        for child in children:
            _safe(child)
            path = child / "revision.json"
            if child.is_dir() and path.exists():
                yield "adopted", _json(_safe(path)), str(path)
    for path in reference_files:
        yield "referenced by placement or adoption", _json(_safe(path)), str(path)



def _exact_cites(value, record):
    """Match a literal saved folder/version, independently of same-map aliases."""
    if isinstance(value, dict):
        if value.get("sha256") == record["sha256"] and (
                value.get("folder") == record["folder"]
                or value.get("version") == Path(record["folder"]).name):
            return True
        return any(_exact_cites(item, record) for item in value.values())
    if isinstance(value, list):
        return any(_exact_cites(item, record) for item in value)
    return False


def _adopted_cites(value, record):
    """Join adopted revisions to their retained, SHA-bound preparation metadata."""
    if _exact_cites(value, record):
        return True
    config = value.get("configuration", value)
    if not isinstance(config, dict):
        raise ValueError("An adopted revision has unreadable configuration metadata")
    links = [(config.get("provenance") or {}).get("preparation"),
             (config.get("painted_adoption_table") or {}).get("preparation")]
    bound = [row for row in links if isinstance(row, dict) and row.get("path") and row.get("sha256")]
    for row in bound:
        path = _safe(Path(row["path"]))
        if lp._file_sha256(path) != row["sha256"]:
            raise ValueError("An adopted revision's preparation metadata changed")
        if _exact_cites(_json(path), record):
            return True
    # Older records may lack literal folder provenance. Keep their existing
    # conservative protection rather than treating an unknown identity as free.
    return not bound and _cites(value, record["sha256"])


def withdrawal_context(project, run_id, selected, *, reference_files=()):
    """Read actual approval and all saved adoption proofs before confirmation."""
    record = _checked(project, run_id, selected, content=False)
    approval = lp.map_approval(_events(project, run_id), record["sha256"],
                               version=Path(record["folder"]).name)
    if approval is None or approval.get("files") != record["files"]:
        raise ValueError("This exact saved version has no active approval to withdraw")
    from .geometry_facies import GeometryRevisionStore
    store = GeometryRevisionStore(project)
    for path in (store.current_path, store.revisions_path):
        path = _safe(path)
        if path.is_file() and path.stat().st_size > _LIMIT:
            raise ValueError("Geometry revision metadata exceeds the bounded read")
    current = store.current()  # Original current-pointer/history-fallback selection.
    if _adopted_cites(current, record):
        raise ValueError("This version is adopted for the series; adopt another version first.")
    placements = []
    for state, value, path in _reference_metadata(project, reference_files):
        if state == "placed" and _exact_cites(value, record):
            placements.append(path)
    return dict(record=record, approval=approval, placements=placements)


def withdraw_approval(project, run_id, selected):
    """Append one withdrawal; change no saved version, placement or old log line."""
    with _action_lock(project, run_id):
        context = withdrawal_context(project, run_id, selected)
        return _event(project, context["record"], "approval_withdrawn",
            withdrawn_approval=context["approval"], withdrawn_placements=context["placements"])


def archive_blockers(project, run_id, record, *, reference_files=()):
    """Actual package proofs, independent of the version table's cached status."""
    record = _checked(project, run_id, record)
    reasons = []
    events = _events(project, run_id)
    literal = Path(record["folder"]).name
    withdrawn = lp.map_approval_withdrawal(events, record["sha256"], version=literal)
    if lp.map_approval(events, record["sha256"], version=literal if withdrawn else None) is not None:
        reasons.append("This map SHA-256 is approved and cannot be archived")
    for state, value, path in _reference_metadata(project, reference_files):
        if _cites(value, record["sha256"]):
            if withdrawn and state in {"placed", "prepared for adoption"}:
                continue
            if withdrawn and state == "adopted" and not _adopted_cites(value, record):
                continue
            reasons.append(f"This version is {state}: {path}")
    return reasons


def _move(project, run_id, selected, *, restore, reference_files):
    with _action_lock(project, run_id):
        record = _checked(project, run_id, selected, deleted=restore)
        if not restore:
            reasons = archive_blockers(project, run_id, record, reference_files=reference_files)
            if reasons:
                raise ValueError("; ".join(reasons))
        versions, deleted = _roots(project, run_id)
        source = _child(Path(record["folder"]), deleted if restore else versions, exists=True)
        root = versions if restore else deleted
        root.mkdir(parents=True, exist_ok=True)
        target = _child(root / source.name, root, exists=False)
        if target.exists():
            raise FileExistsError("The destination already contains this version; nothing was overwritten")
        action = "version_restored" if restore else "version_archived"
        _event(project, record, action + "_intent", source_folder=str(source), destination_folder=str(target))
        # Recheck the actual approval log immediately before the move. The GUI
        # serializes approve/place/adopt and version actions while this runs.
        if not restore:
            events = _events(project, run_id)
            literal = Path(record["folder"]).name
            withdrawn = lp.map_approval_withdrawal(events, record["sha256"], version=literal)
            if lp.map_approval(events, record["sha256"], version=literal if withdrawn else None) is not None:
                raise ValueError("This map became approved and cannot be archived")
        _child(source, deleted if restore else versions, exists=True)
        _child(target, root, exists=False)
        if target.exists():
            raise FileExistsError("The destination already contains this version; nothing was overwritten")
        source.rename(target)
        moved = _checked(project, run_id, dict(record, folder=str(target)), deleted=not restore)
        event = _event(project, moved, action, source_folder=str(source), destination_folder=str(target))
        return dict(moved, archived=not restore, action_event=event)


def archive_version(project, run_id, record, *, reference_files=()):
    return _move(project, run_id, record, restore=False, reference_files=tuple(reference_files))


def restore_deleted_version(project, run_id, record):
    return _move(project, run_id, record, restore=True, reference_files=())


def _new_folder(project, run_id, digest):
    # Saving an explicitly requested first version also creates the package's
    # missing directory in lightweight local projects, as the original saver did.
    _safe(Path(project.root)).mkdir(parents=True, exist_ok=True)
    versions, _deleted = _roots(project, run_id)
    versions.mkdir(parents=True, exist_ok=True)
    while True:
        folder = _child(versions / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{digest[:12]}-{uuid.uuid4().hex[:12]}",
                        versions, exists=False)
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue


def save_drawing_version(project, session, *, name):
    name = str(name).strip()
    if not name:
        raise ValueError("A drawing version needs a name")
    payload = _drawing_payload(session)
    digest = _drawing_digest(payload)
    folder = _new_folder(project, session.run_id, digest)
    _write(folder / "session.json", payload)
    record = dict(schema=1, format=lp.LAYER_VERSION_FORMAT, run_id=session.run_id,
        name=name, kind=DRAWING_KIND, sha256=digest, created_utc=utc_now(),
        settings=deepcopy(payload["proposal_settings"]),
        files={"session.json": lp._file_sha256(folder / "session.json")})
    _write(folder / "version.json", record)
    return dict(record, folder=str(folder))


def load_drawing_version(project, run_id, record):
    checked = _checked(project, run_id, record)
    if checked["kind"] != DRAWING_KIND:
        raise ValueError("The selected version is not a drawing")
    return lp.PaintingSession.from_dict(_json(Path(checked["folder"]) / "session.json"))


def load_map_version(project, run_id, record, *, proposal_loader=None):
    """Verify an ordinary or explicitly supported alternate proposal version."""
    checked = _checked(project, run_id, record, proposal_loader=proposal_loader)
    if checked["kind"] in {DRAWING_KIND, manual.MANUAL_MAP_KIND}:
        raise ValueError("The selected version is not a proposal or edited map")
    return _load_map_payload(checked, proposal_loader)


def use_drawing_version(project, current_session, record, *, backup_name="Drawing before loading a version"):
    """Return the selected drawing after saving the current drawing first.

    The caller applies the returned session with its existing GUI/session
    workflow. No current proposal, correction list or session file is replaced.
    """
    loaded = load_drawing_version(project, current_session.run_id, record)
    backup = save_drawing_version(project, current_session, name=backup_name)
    return loaded, backup


def save_map_version(project, session, proposal, corrections, *, name,
                     method=None, settings=None, proposal_writer=None):
    """An ordinary immutable map version with explicit proposal method metadata."""
    name = str(name).strip()
    if not name:
        raise ValueError("A map version needs a name")
    digest = lp.map_digest(session, proposal, corrections)
    folder = _new_folder(project, session.run_id, digest)
    # Keep the original proposal/session/correction serialization by default.
    with (folder / "session.json").open("x", encoding="utf8") as stream:
        stream.write(lp._session_text(session))
    writer = proposal.save if proposal_writer is None else lambda path: proposal_writer(proposal, path)
    writer(folder / "proposal.npz")
    with (folder / "corrections.json").open("x", encoding="utf8") as stream:
        stream.write(lp._corrections_text(session.run_id, corrections))
    record = dict(schema=1, format=lp.LAYER_VERSION_FORMAT, run_id=session.run_id,
        name=name, kind="edited map" if corrections else "proposal", sha256=digest,
        created_utc=utc_now(), proposal_digest=proposal.digest, correction_count=len(corrections),
        proposal_method=str(method or lp.PROPOSAL_METHOD),
        settings=deepcopy(session.to_dict()["proposal_settings"] if settings is None else settings),
        files={leaf: lp._file_sha256(folder / leaf) for leaf in lp.VERSION_FILES})
    if method:
        from . import layer_painting_polygons_only as polygons
        if method == polygons.METHOD:
            record["uncovered_pixels"] = polygons.gap_count(
                proposal, corrections, {row.layer_id for row in session.layers})
    _write(folder / "version.json", record)
    return dict(record, folder=str(folder))


def rename_version(project, run_id, record, name):
    name = str(name).strip()
    if not name:
        raise ValueError("A version needs a nonempty display name")
    with _action_lock(project, run_id):
        checked = _checked(project, run_id, record)
        return _event(project, checked, "version_renamed", display_name=name)


def set_version_note(project, run_id, record, note):
    with _action_lock(project, run_id):
        checked = _checked(project, run_id, record)
        return _event(project, checked, "version_note", note=str(note))


def list_version_metadata(project, run_id, *, deleted=False, reference_files=()):
    """The unified table's immutable identities and append-only display labels."""
    versions, archived = _roots(project, run_id)
    root = archived if deleted else versions
    if not root.is_dir():
        return []
    history_error = ""
    try:
        events = _events(project, run_id)
    except (OSError, ValueError) as error:
        # Keep verified immutable rows visible, with explicitly unknown
        # history. No action may treat this display fallback as unapproved.
        events = []
        history_error = str(error)
    references = list(_reference_metadata(project, reference_files))
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        _safe(folder)
        if not (folder / "version.json").is_file():
            continue
        row = _checked(project, run_id, {"folder": str(folder)}, deleted=deleted, content=False)
        record_sha = lp._file_sha256(folder / "version.json")
        applicable = [event for event in events if event.get("version") == folder.name
            and event.get("version_sha256") == row["sha256"]
            and event.get("version_record_sha256") == record_sha]
        row.update(display_name=row["name"], note="", archived=deleted)
        for event in applicable:
            if event.get("event") == "version_renamed":
                row["display_name"] = event["display_name"]
            elif event.get("event") == "version_note":
                row["note"] = event["note"]
        settings = row.get("settings")
        if settings is None:
            settings = _json(folder / "session.json").get("proposal_settings", {})
        method = row.get("proposal_method")
        if "proposal.npz" in row["files"] and method is None:
            with np.load(folder / "proposal.npz", allow_pickle=False) as archive:
                method = json.loads(str(archive["meta"])).get("method", lp.PROPOSAL_METHOD)
        approval = lp.map_approval(events, row["sha256"], version=folder.name)
        withdrawn = lp.map_approval_withdrawal(events, row["sha256"], version=folder.name)
        states = [state for state, value, _path in references if _cites(value, row["sha256"])]
        placed = [value for state, value, _path in references if state == "placed" and _exact_cites(value, row)]
        placement_withdrawn = bool(placed) and all(lp.approval_withdrawal(events,
            (value.get("map_version") or {}).get("approval")) for value in placed)
        row.update(settings=settings, proposal_method=method,
            approved=None if history_error else approval is not None,
            history_error=history_error, approval_known=not bool(history_error),
            placed="placed" in states, adopted="adopted" in states,
            approval_withdrawn_utc=(withdrawn.get("recorded_utc") if withdrawn and approval is None else None),
            placement_withdrawn=placement_withdrawn,
            protection_reasons=states)
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row.get("created_utc", "")), row["folder"]))
