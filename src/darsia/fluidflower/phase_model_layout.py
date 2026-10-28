"""Preview and copy phase-model files into their series-owned folders.

Original model, review and withdrawal documents keep their exact bytes. Only
the series' storage paths and the existing review-storage alias registry route
future reads and saves to the verified copies. No model is adopted or trained.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tomllib

FORMAT = "darsia-phase-model-layout-plan-v1"
COMPLETE = "darsia-phase-model-layout-completed-v1"
_LEGACY_LAYOUT = {
    "phase_model": "PhaseModel", "phase_annotations": "PhaseModel/Annotations",
    "reviews": "PhaseModel/Annotations", "review_collections": "PhaseModel/Annotations/review_collections",
    "evidence_actions": "PhaseModel/Annotations/evidence_actions",
    "evidence_audits": "PhaseModel/Annotations/evidence_audits",
    "training_enrollments": "PhaseModel/Annotations/training_enrollments",
    "phase_models": "PhaseModel/Model", "phase_model_library": "PhaseModel/Model/library",
}
_PREVIOUS_LAYOUT = {
    "phase_model": "PhaseModels", "phase_annotations": "PhaseModels/TrainingInput",
    "reviews": "PhaseModels/TrainingInput", "review_collections": "PhaseModels/TrainingInput/review_collections",
    "evidence_actions": "PhaseModels/TrainingInput/evidence_actions",
    "evidence_audits": "PhaseModels/TrainingInput/evidence_audits",
    "training_enrollments": "PhaseModels/TrainingInput/training_enrollments",
    "phase_models": "PhaseModels/Models", "phase_model_library": "PhaseModels/Models/library",
}
_LAYOUT = {key: "ExpertKnowledge/" + path for key, path in _PREVIOUS_LAYOUT.items()}


def _key(path):
    """Normalise an absolute path for location comparisons."""
    return os.path.normcase(os.path.abspath(path))


def _digest(value):
    """Hash the deterministic JSON representation of a plan."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pin(path):
    """Record exact bytes and timestamps while detecting concurrent changes."""
    path = Path(path)
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(block)
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if stamp(before) != stamp(after):
        raise ValueError("A file changed while planning its copy: " + str(path))
    return dict(path=str(path), sha256=sha.hexdigest(), size_bytes=after.st_size,
                mtime_ns=after.st_mtime_ns)


def _verify(bound, path=None, *, timestamps=False):
    """Check a planned source or copy against its recorded identity."""
    current = _pin(bound["path"] if path is None else path)
    keys = ("sha256", "size_bytes", "mtime_ns") if timestamps else ("sha256", "size_bytes")
    if any(current[key] != bound[key] for key in keys):
        raise ValueError("A planned source or copy changed: " + str(current["path"]))
    return current


def _json(path):
    """Read bounded metadata through the strict JSON decoder."""
    from .pipeline.job_inputs import _json_bytes, MAX_METADATA_BYTES
    if Path(path).stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Layout metadata exceeds its size limit")
    return _json_bytes(Path(path).read_bytes())


def _write(path, value):
    """Create and flush a fresh metadata file without replacing existing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
            allow_nan=False, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return _pin(path)


def _inventory(root, *, excluding=()):
    """Regular files only; never follow a symbolic link or directory junction."""
    root = Path(root)
    result = []
    excluded = {_key(path) for path in excluding}
    if not root.exists():
        return result
    if root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("A phase-model source root cannot be a link or junction: " + str(root))
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(folder) / name
            info = path.lstat()
            if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("A phase-model copy cannot follow a link or junction: " + str(path))
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("A phase-model copy needs regular files: " + str(path))
        result.extend(_pin(Path(folder) / name) for name in sorted(files)
                      if _key(Path(folder) / name) not in excluded)
    return sorted(result, key=lambda row: _key(row["path"]))


def _new_manifest(raw, *, layout=None):
    """Replace just [paths]; all other original TOML bytes remain verbatim."""
    import toml
    from .series_method import _same_values
    value = tomllib.loads(raw.decode("utf-8"))
    paths = value.get("paths") or {}
    if not isinstance(paths, dict) or any(not isinstance(v, str) for v in paths.values()):
        raise ValueError("Series storage paths must be explicit text paths")
    merged = dict(paths, **(_LAYOUT if layout is None else layout))
    text = raw.decode("utf-8")
    headings = list(re.finditer(r"(?m)^[ \t]*\[paths\][ \t]*(?:#[^\n]*)?\r?$", text))
    if len(headings) > 1:
        raise ValueError("Series manifest has ambiguous [paths]")
    block = toml.dumps({"paths": merged})
    if headings:
        start = headings[0].start()
        following = re.search(r"(?m)^[ \t]*\[", text[headings[0].end():])
        end = headings[0].end() + following.start() if following else len(text)
        changed = text[:start] + block + "\n" + text[end:]
    else:
        changed = text + ("\n" if not text.endswith("\n") else "") + "\n" + block
    before_other, after_other = deepcopy(value), tomllib.loads(changed)
    before_other.pop("paths", None)
    after_other.pop("paths", None)
    if not _same_values(before_other, after_other):
        raise ValueError("Storage planning changed non-storage series settings")
    return changed.encode("utf-8")


def _manifest_series(settings):
    """Resolve the explicitly declared workspace identity to its canonical key."""
    from .series_paths import series_key
    series = (settings.get("series") or {}).get("id") or settings.get("dataset") or settings.get("series_id")
    if not isinstance(series, str) or not series.strip():
        raise ValueError("The selected series manifest must declare its own series ID")
    return series_key(series)


def _external_session(path, series):
    """Validate an external session's scope and original save-root routing."""
    from .review_storage import resolve_review_path
    root = resolve_review_path(path)
    session = _json(root / "review_session.json")
    if (session.get("format") != "fluidflower-local-review-session-v1"
            or not session.get(series + "_runs")
            or any(value for key, value in session.items()
                   if key.endswith("_runs") and key != series + "_runs")):
        raise ValueError("An external session must record only the selected series' run scope: " + str(root))
    original_save = Path(session["local_review_root"])
    old_save = resolve_review_path(original_save, context=root)
    try:
        suffix = root.relative_to(old_save)
    except ValueError:
        raise ValueError("The original session is outside its recorded save root") from None
    return root, original_save, old_save, suffix


def preview_layout(series_root, *, external_sessions=(), model_roots=None):
    """Read-only dry run. Return every exact copy and the proposed path updates."""
    from .series_paths import series_folder_path, series_key
    from .phase_model_library import discover_series_sources, source_identity, _metadata, NotPhaseModelSource
    from .review_storage import FORMAT as ALIAS_FORMAT, resolve_review_path
    from .phase_review_retractions import ledger_folder, load_retractions
    root = Path(os.path.abspath(series_root))
    manifest = root / "series.toml"
    original = _pin(manifest)
    raw = manifest.read_bytes()
    settings = tomllib.loads(raw.decode("utf-8"))
    series = _manifest_series(settings)
    if os.environ.get("DARSIA_REVIEW_ALIAS_FILES"):
        raise ValueError("Custom review alias files are configured; resolve that explicit override before organising storage")
    ledger_source = ledger_folder(root)
    if (ledger_source / ".write.lock").exists():
        raise ValueError("Another evidence change is being saved; refresh and preview again")
    if any(ledger_source.glob("*.pending")):
        raise ValueError("An unfinished evidence change remains in the original ledger; review it before organising storage")
    phase_root = root / _LAYOUT["phase_model"]
    annotations, models = root / _LAYOUT["phase_annotations"], root / _LAYOUT["phase_models"]
    trees, aliases, model_checks, notices = [], [], [], []

    def add(source, target, kind, *, original_save=None, save_root=None):
        source, target = Path(source), Path(target)
        if not source.exists() or _key(source) == _key(target):
            return
        if not source.is_dir():
            raise ValueError("A planned phase-model source is not a folder: " + str(source))
        if target.is_relative_to(source):
            raise ValueError("New storage cannot be inside its own original source")
        for old in trees:
            parent, destination = Path(old["source"]), Path(old["destination"])
            if source.is_relative_to(parent) and target == destination / source.relative_to(parent):
                return  # Already copied byte-for-byte by the owning series tree.
        files = _inventory(source)
        # Compare the physical locations only after the explicitly chosen source
        # has been opened. An ancestor junction can give one tree two spellings.
        if target.resolve().is_relative_to(source.resolve()):
            raise ValueError("New storage cannot be physically inside its original source")
        trees.append(dict(source=str(source), destination=str(target), kind=kind, files=files))
        aliases.append(dict(source=str(source), destination=str(target),
            original_save_root=str(original_save or source), save_root=str(save_root or target)))

    for key in ("reviews", "review_collections", "evidence_actions", "evidence_audits", "training_enrollments", "phase_model_library"):
        try:
            source = resolve_review_path(series_folder_path(root, key))
        except KeyError:
            continue
        add(source, root / _LAYOUT[key], key)
    for selected in external_sessions:
        selected, original_save, old_save, suffix = _external_session(selected, series)
        new_save = annotations / "legacy_sessions" / hashlib.sha256(_key(old_save).encode()).hexdigest()[:16]
        add(selected, new_save / suffix, "external_session", original_save=original_save, save_root=new_save)
    candidates = discover_series_sources(root) if model_roots is None else model_roots
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if _key(candidate) in seen:
            continue
        seen.add(_key(candidate))
        try:
            identity = source_identity(candidate)
        except NotPhaseModelSource:
            if model_roots is not None:
                raise
            continue
        scope = _metadata(candidate, series="").get("datasets") or []
        if not scope:
            raise ValueError("Model ownership is not recorded for the selected series: " + str(candidate))
        if {series_key(value) for value in scope} != {series}:
            notices.append("A model without exact selected-series ownership was left in place: " + str(candidate))
            continue
        if candidate.is_relative_to(models):
            continue
        # Keep the original leaf name: legacy model identity can use it when
        # immutable summary/request metadata has no declared model ID.
        target = models / (identity["version_sha256"][:12] + "-"
                           + hashlib.sha256(_key(candidate).encode()).hexdigest()[:8]) / candidate.name
        add(candidate, target, "model")
        model_checks.append(dict(source=str(candidate), destination=str(target), identity=identity))

    registry = resolve_review_path(series_folder_path(root, "reviews") / "storage_aliases.json")
    old_aliases = _json(registry) if registry.is_file() else dict(format=ALIAS_FORMAT, aliases=[])
    if old_aliases.get("format") != ALIAS_FORMAT:
        raise ValueError("Original review aliases use an unsupported format")
    rows, destinations = [], {}
    for tree in trees:
        for source in tree["files"]:
            target = Path(tree["destination"]) / Path(source["path"]).relative_to(tree["source"])
            if _key(target) == _key(annotations / "storage_aliases.json"):
                target = annotations / "storage_aliases_history" / (source["sha256"] + ".json")
                aliases.append(dict(source=source["path"], destination=str(target),
                    original_save_root=str(Path(source["path"]).parent), save_root=str(annotations)))
            previous = destinations.get(_key(target))
            if previous is not None:
                if previous["source"] != source:
                    raise ValueError("Two original files map to the same destination")
                continue
            row = dict(source=source, destination=str(target), kind=tree["kind"])
            if target.exists():
                _verify(source, target)
                row["action"] = "Reuse exact copy"
            else:
                row["action"] = "Copy"
            destinations[_key(target)] = row
            rows.append(row)
    def relocated(path):
        original = Path(path)
        matches = [row for row in aliases if original.is_relative_to(Path(row["source"]))]
        if not matches:
            return str(original)
        row = max(matches, key=lambda value: len(Path(value["source"]).parts))
        return str(Path(row["destination"]) / original.relative_to(row["source"]))
    # Session-specific save-root routing must match the copied context too.
    # Keep the immutable original source/save-root identities, moving only
    # their authoritative current destination and writable save root.
    merged = {_key(row["source"]): dict(row,
        destination=relocated(row["destination"]), save_root=relocated(row["save_root"]))
        for row in old_aliases["aliases"]}
    for row in aliases:
        previous = merged.get(_key(row["source"]))
        if previous is not None and previous != row:
            raise ValueError("Existing review alias conflicts with the proposed exact copy")
        merged[_key(row["source"])] = row
    alias_document = dict(old_aliases, aliases=[merged[key] for key in sorted(merged)])
    canonical_aliases = annotations / "storage_aliases.json"
    if canonical_aliases.is_file() and _json(canonical_aliases) != alias_document:
        raise ValueError("This first-migration tool cannot extend an existing destination alias registry; its original bytes are preserved")
    after = _new_manifest(raw)
    result = dict(format=FORMAT, series=series, series_root=str(root), phase_root=str(phase_root),
        series_manifest=original, manifest_after=base64.b64encode(after).decode(),
        manifest_after_sha256=hashlib.sha256(after).hexdigest(), paths=dict(_LAYOUT),
        trees=trees, files=rows, model_checks=model_checks, alias_document=alias_document,
        alias_path=str(annotations / "storage_aliases.json"),
        withdrawal_snapshot=load_retractions(root, series=series), ledger_source=str(ledger_source), notices=notices,
        originals_preserved=True, adoption=False, deletion=False,
        requires_writer_quiescence=True,
        concurrency_limit="Close all other review and evidence writers before copying. Delayed stale clients are not supported during this storage cutover.",
        bytes_to_copy=sum(row["source"]["size_bytes"] for row in rows if row["action"] == "Copy"))
    result["plan_sha256"] = _digest(result)
    _verify(original, timestamps=True)
    return result


def _verify_trees(plan, *, excluding=()):
    """Refuse changed source inventories before or during copying."""
    for tree in plan["trees"]:
        if _inventory(tree["source"], excluding=excluding) != tree["files"]:
            raise ValueError("A source tree changed after the dry run: " + tree["source"])


def _safe_target(path, phase_root):
    """Keep destinations inside the selected folder without traversing links."""
    path, phase_root = Path(os.path.abspath(path)), Path(os.path.abspath(phase_root))
    if not path.is_relative_to(phase_root):
        raise ValueError("A layout destination leaves the selected phase-model folder")
    for ancestor in (path, *path.parents):
        if ancestor.exists() and (ancestor.is_symlink()
                or getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("A layout destination cannot pass through a link or junction")
        if ancestor == phase_root.parent:
            break


def apply_layout(plan, *, writers_quiescent=False):
    """Apply only an explicitly reviewed plan. Preserve originals and every partial attempt."""
    from .phase_review_retractions import ledger_folder, load_retractions, _ledger_write
    from .series_paths import series_key
    data = dict(plan)
    digest = data.pop("plan_sha256", None)
    if (data.get("format") != FORMAT or not isinstance(data.get("series"), str)
            or not data["series"] or series_key(data["series"]) != data["series"] or _digest(data) != digest
            or data.get("adoption") is not False or data.get("deletion") is not False
            or data.get("requires_writer_quiescence") is not True):
        raise ValueError("The explicit storage dry-run plan changed")
    if writers_quiescent is not True:
        raise ValueError("Close all other review and evidence writers, then explicitly confirm that they are closed before copying")
    root = Path(plan["series_root"])
    layout = plan.get("paths")
    if layout not in (_LAYOUT, _PREVIOUS_LAYOUT, _LEGACY_LAYOUT):
        raise ValueError("The storage plan uses an unknown phase-model layout")
    if (_key(plan["phase_root"]) != _key(root / layout["phase_model"])
            or _key(plan["series_manifest"]["path"]) != _key(root / "series.toml")
            or _key(plan["alias_path"]) != _key(root / layout["phase_annotations"] / "storage_aliases.json")):
        raise ValueError("Layout paths do not belong to the selected series")
    for row in plan["files"]:
        _safe_target(row["destination"], plan["phase_root"])
    _safe_target(plan["alias_path"], plan["phase_root"])
    receipt_parent = (root / layout["phase_annotations"]
        if layout in (_LAYOUT, _PREVIOUS_LAYOUT) else Path(plan["phase_root"]))
    receipt_root = receipt_parent / ".layout" / digest
    completed = receipt_root / "completed.json"
    for controlled in (receipt_root, completed, receipt_root / "started.json", receipt_root / "error.json",
                       receipt_root / "series.toml.before", receipt_root / "series.toml.after"):
        _safe_target(controlled, plan["phase_root"])
    current_manifest = _pin(root / "series.toml")
    if _manifest_series(tomllib.loads((root / "series.toml").read_text(encoding="utf-8"))) != plan["series"]:
        raise ValueError("The storage plan belongs to another selected series")
    if completed.is_file():
        result = _json(completed)
        if (result.get("format") != COMPLETE or result.get("plan_sha256") != digest
                or result.get("complete") is not True or result.get("originals_preserved") is not True
                or result.get("adoption") is not False or result.get("deletion") is not False
                or result.get("writers_quiescent_confirmed") is not True
                or result.get("concurrency_limit") != plan["concurrency_limit"]):
            raise ValueError("Existing layout receipt belongs to another plan")
        if current_manifest["sha256"] != plan["manifest_after_sha256"]:
            raise ValueError("Series storage settings changed after the completed copy")
        for row in plan["files"]:
            _verify(row["source"], row["destination"])
        _verify_trees(plan)
        if _json(plan["alias_path"]) != plan["alias_document"]:
            raise ValueError("Review storage aliases changed after the completed copy")
        if load_retractions(root, series=plan["series"]) != plan["withdrawal_snapshot"]:
            raise ValueError("Review withdrawals changed after the completed copy")
        return result
    if receipt_root.exists():
        raise ValueError("A storage attempt already exists; its partial files are preserved. Prepare a new dry run before an explicit retry")
    if _key(ledger_folder(root)) != _key(plan["ledger_source"]):
        raise ValueError("The authoritative evidence ledger changed after the dry run")
    # This canonical lock refuses a currently active writer. The explicit
    # quiescence prerequisite also excludes clients paused with an old lock path;
    # this bounded GUI operation does not replace the evidence-writer protocol.
    with _ledger_write(root):
        lock = Path(plan["ledger_source"]) / ".write.lock"
        return _copy_locked(plan, receipt_root, completed, digest, lock)


def _copy_locked(plan, receipt_root, completed, digest, lock):
    """Copy verified bytes and switch storage paths while holding the ledger lock."""
    from .phase_model_library import source_identity
    from .phase_review_retractions import load_retractions
    from .review_storage import review_aliases
    root = Path(plan["series_root"])
    _verify(plan["series_manifest"], timestamps=True)
    if base64.b64decode(plan["manifest_after"], validate=True) != _new_manifest(
            (root / "series.toml").read_bytes(), layout=plan["paths"]):
        raise ValueError("The proposed change is not the exact storage-only path update")
    _verify_trees(plan, excluding=(lock,))
    if load_retractions(root, series=plan["series"]) != plan["withdrawal_snapshot"]:
        raise ValueError("Review withdrawals changed after the dry run")
    receipt_root.mkdir(parents=True, exist_ok=False)
    _write(receipt_root / "started.json", dict(plan_sha256=digest, originals_preserved=True,
        automatic_retry=False, adoption=False, deletion=False, writers_quiescent_confirmed=True,
        concurrency_limit=plan["concurrency_limit"]))
    written = []
    try:
        for row in plan["files"]:
            source, target = row["source"], Path(row["destination"])
            _verify(source, timestamps=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                _verify(source, target)
            else:
                with Path(source["path"]).open("rb") as incoming, target.open("xb") as output:
                    shutil.copyfileobj(incoming, output, length=8 * 1024**2)
                    output.flush()
                    os.fsync(output.fileno())
                shutil.copystat(source["path"], target, follow_symlinks=False)
                written.append(str(target))
                _verify(source, target)
        for model in plan["model_checks"]:
            if source_identity(model["destination"]) != model["identity"]:
                raise ValueError("Copied model identity differs from the original")
        _verify_trees(plan, excluding=(lock,))
        _verify(plan["series_manifest"], timestamps=True)
        if load_retractions(root, series=plan["series"]) != plan["withdrawal_snapshot"]:
            raise ValueError("Review withdrawals changed during copying")
        aliases = Path(plan["alias_path"])
        if aliases.exists():
            if _json(aliases) != plan["alias_document"]:
                raise ValueError("An existing destination alias registry differs; it is preserved")
        else:
            _write(aliases, plan["alias_document"])
        original_bytes = Path(plan["series_manifest"]["path"]).read_bytes()
        before_copy = receipt_root / "series.toml.before"
        with before_copy.open("xb") as stream:
            stream.write(original_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        after = base64.b64decode(plan["manifest_after"], validate=True)
        if hashlib.sha256(after).hexdigest() != plan["manifest_after_sha256"]:
            raise ValueError("The proposed series storage bytes changed")
        temporary = receipt_root / "series.toml.after"
        with temporary.open("xb") as stream:
            stream.write(after)
            stream.flush()
            os.fsync(stream.fileno())
        _verify(plan["series_manifest"], timestamps=True)
        os.replace(temporary, root / "series.toml")
        review_aliases(refresh=True)
        if load_retractions(root, series=plan["series"]) != plan["withdrawal_snapshot"]:
            raise ValueError("The relocated withdrawal snapshot differs; originals and the previous series manifest remain preserved")
        result = dict(format=COMPLETE, complete=True, plan_sha256=digest,
            copied_files=written, reused_files=len(plan["files"]) - len(written),
            series_manifest=_pin(root / "series.toml"), original_series_manifest=_pin(before_copy),
            originals_preserved=True, model_bytes_unchanged=True, withdrawal_digest=plan["withdrawal_snapshot"]["digest"],
            adoption=False, deletion=False, writers_quiescent_confirmed=True,
            concurrency_limit=plan["concurrency_limit"])
        _write(completed, result)
        return result
    except Exception as error:
        _write(receipt_root / "error.json", dict(plan_sha256=digest, reason=str(error),
            written_files=written, partials_preserved=True, originals_preserved=True, automatic_retry=False))
        raise
