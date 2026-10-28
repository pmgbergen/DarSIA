"""Native-only coloured-map file input, retaining the editor's save workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
import uuid
import tomllib

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
)

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.coloured_map_import import (
    ImportedColouredMap,
    import_coloured_map,
)
from darsia.fluidflower.crop_setup_ui import _same_path
from darsia.fluidflower.series_geometry import default_actor, _merge, _storage_path
from darsia.fluidflower.setup_geometry_apply import ReadBinding, _baseline_from_config


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _origin_path(project, run_id):
    return lp.session_path(project, run_id).with_suffix(".import-origin.json")


def _classes_binding(proposal):
    return {
        "digest": proposal.digest,
        "sha256": _sha(proposal.classes.tobytes()),
        "shape": list(proposal.classes.shape),
        "dtype": str(proposal.classes.dtype),
    }


def _durable_import(project, imported):
    """Copy only this explicit import's owned cache, once, beside its draft."""
    folder = lp.painting_folder(project) / "ui_import_cache"
    if imported.work_directory.resolve().is_relative_to(folder.resolve()):
        return imported
    destination = folder / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for source in sorted(imported.work_directory.rglob("*")):
        if not source.is_file():
            continue
        if source.is_symlink():
            raise ValueError(
                "The import cache must contain ordinary owned files, not symbolic links."
            )
        relative = source.relative_to(imported.work_directory)
        raw = source.read_bytes()
        _write_new(destination / relative, raw)
        records.append({"relative": str(relative), "sha256": _sha(raw)})
    _write_new(
        destination / "cache-files.json",
        (json.dumps(records, indent=2) + "\n").encode("utf-8"),
    )
    return ImportedColouredMap(
        imported.session,
        imported.proposal,
        destination / imported.receipt.relative_to(imported.work_directory),
        destination,
        destination / imported.native_labels.relative_to(imported.work_directory),
    )


def _origin_bytes(view):
    imported = getattr(view, "_ui_import", None)
    if imported is None or view.session is None or view.proposal is None:
        return None
    if view.session.run_id != imported.session.run_id:
        raise RuntimeError(
            "The imported draft origin belongs to another reference run."
        )
    baseline_sha = _sha(Path(view.session.image_path).read_bytes())
    initial_receipt = json.loads(imported.receipt.read_bytes())
    if initial_receipt.get("raw_baseline_sha256", baseline_sha) != baseline_sha:
        raise RuntimeError(
            "The imported raw reference photograph changed before its draft origin could be saved."
        )
    path = _origin_path(view.project, view.session.run_id)
    payload = {
        "format": "darsia-ui-coloured-map-draft-origin-v1",
        "run_id": view.session.run_id,
        "raw_baseline": view.session.image_path,
        "raw_baseline_sha256": baseline_sha,
        "cache_relative": str(
            imported.work_directory.relative_to(lp.painting_folder(view.project))
        ),
        "receipt_relative": str(imported.receipt.relative_to(imported.work_directory)),
        "receipt_sha256": _sha(imported.receipt.read_bytes()),
        "cache_files_sha256": _sha(
            (imported.work_directory / "cache-files.json").read_bytes()
        ),
        "installation_relative": str(
            Path(view._ui_import_installation).relative_to(imported.work_directory)
        ),
        "installation_sha256": _sha(Path(view._ui_import_installation).read_bytes()),
        "proposal": _classes_binding(view.proposal),
        "subsequent_propose_boundaries_fit_run": bool(
            getattr(view, "_ui_import_later_fit", False)
        ),
    }
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return path, raw


def persist_import_origin(view):
    prepared = _origin_bytes(view)
    if prepared is None:
        return None
    path, raw = prepared
    before = path.read_bytes() if path.is_file() else None
    if before is not None:
        _write_new(
            view._ui_import.work_directory
            / "origin-history"
            / (uuid.uuid4().hex + ".json"),
            before,
        )
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    _write_new(temporary, raw)
    if (path.read_bytes() if path.is_file() else None) != before:
        raise RuntimeError(
            "The imported draft origin changed before it could be saved."
        )
    os.replace(temporary, path)
    return path


def _current_bytes(path):
    return path.read_bytes() if path.is_file() else None


def _replace_expected(path, raw, expected):
    """Replace only the bytes explicitly observed by this Import attempt."""
    staged = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    _write_new(staged, raw)
    if _current_bytes(path) != expected:
        raise RuntimeError(f"The saved draft changed before installation: {path}")
    os.replace(staged, path)


def _restore_attempt(before, intended, cache):
    """Restore only attempt-owned bytes; preserve every displaced failed file."""
    records = []
    for path, written in reversed(tuple(intended.items())):
        old = before[path]
        try:
            current = _current_bytes(path)
            if current == old:
                records.append({"path": str(path), "state": "already-original"})
                continue
            if current != written:
                records.append(
                    {
                        "path": str(path),
                        "state": "external-change-preserved",
                        "current_sha256": (
                            _sha(current) if current is not None else None
                        ),
                    }
                )
                continue
            displaced = (
                cache / "displaced-failed-draft" / (uuid.uuid4().hex + "-" + path.name)
            )
            _write_new(displaced, current)
            if old is None:
                # A newly created canonical file is moved into the receipt's
                # cache rather than deleted. Recheck just before the rename.
                if _current_bytes(path) != written:
                    raise RuntimeError(
                        "The newly installed draft changed during restoration."
                    )
                moved = displaced.with_name(displaced.name + ".moved")
                os.replace(path, moved)
            else:
                _replace_expected(path, old, written)
            records.append(
                {
                    "path": str(path),
                    "state": "restored",
                    "displaced": str(displaced),
                    "displaced_sha256": _sha(current),
                }
            )
        except Exception:
            records.append(
                {
                    "path": str(path),
                    "state": "restore-failed",
                    "traceback": traceback.format_exc(),
                }
            )
    return records


def read_import_origin(project, session, proposal):
    """Read one explicit per-run pointer; never scan or select old histories."""
    if session is None:
        return None
    pointer = _origin_path(project, session.run_id)
    if not pointer.is_file():
        return None
    payload = json.loads(pointer.read_bytes())
    if (
        payload.get("format") != "darsia-ui-coloured-map-draft-origin-v1"
        or payload.get("run_id") != session.run_id
    ):
        raise RuntimeError(
            "The saved coloured-map draft-origin pointer has the wrong format or run."
        )
    if proposal is None or payload.get("proposal") != _classes_binding(proposal):
        raise RuntimeError(
            "The loaded proposal no longer matches its explicit coloured-map draft-origin pointer."
        )
    if not _same_path(payload.get("raw_baseline", ""), session.image_path):
        raise RuntimeError(
            "The loaded baseline no longer matches its coloured-map draft origin."
        )
    if _sha(Path(session.image_path).read_bytes()) != payload["raw_baseline_sha256"]:
        raise RuntimeError(
            "The raw baseline bytes no longer match the saved coloured-map draft origin."
        )
    base = lp.painting_folder(project).resolve()
    folder = (base / payload["cache_relative"]).resolve()
    if not folder.is_relative_to((base / "ui_import_cache").resolve()):
        raise RuntimeError(
            "The draft-origin cache is outside this project's owned import cache."
        )
    receipt = (folder / payload["receipt_relative"]).resolve()
    installation = (folder / payload["installation_relative"]).resolve()
    if not receipt.is_relative_to(folder) or not installation.is_relative_to(folder):
        raise RuntimeError("The draft-origin receipt is outside its owned cache.")
    if _sha(receipt.read_bytes()) != payload["receipt_sha256"]:
        raise RuntimeError("The durable coloured-map import receipt changed.")
    if _sha(installation.read_bytes()) != payload["installation_sha256"]:
        raise RuntimeError("The durable coloured-map installation receipt changed.")
    cache_bytes = (folder / "cache-files.json").read_bytes()
    if _sha(cache_bytes) != payload["cache_files_sha256"]:
        raise RuntimeError("The durable coloured-map source manifest changed.")
    cache_files = json.loads(cache_bytes)
    for record in cache_files:
        source = (folder / record["relative"]).resolve()
        if (
            not source.is_relative_to(folder)
            or _sha(source.read_bytes()) != record["sha256"]
        ):
            raise RuntimeError("A durable coloured-map import source file changed.")
    imported = ImportedColouredMap(
        session, proposal, receipt, folder, folder / "labels.npz"
    )
    return (
        imported,
        installation,
        bool(payload.get("subsequent_propose_boundaries_fit_run")),
    )


def restore_import_origin(view):
    view._ui_import = view._ui_import_installation = None
    view._ui_import_later_fit = False
    restored = read_import_origin(view.project, view.session, view.proposal)
    if restored is not None:
        view._ui_import, view._ui_import_installation, view._ui_import_later_fit = (
            restored
        )
    return restored


@dataclass(frozen=True)
class ImportContext:
    series_id: str
    root: Path
    project_root: Path
    reference: object
    frame: object
    config: dict
    bindings: tuple[ReadBinding, ...]


def reference_config(root: Path, reference) -> tuple[dict, tuple[ReadBinding, ...]]:
    """Read the actual reference run, never the unrelated open run's settings."""
    bindings = []

    def read(path):
        data = Path(path).read_bytes()
        bindings.append(ReadBinding(Path(path), data))
        return tomllib.loads(data.decode("utf-8-sig"))

    manifest = read(root / "series.toml")
    paths = manifest.get("paths") or {}
    common_path = _storage_path(
        paths.get("common_config") or "config/common.toml", root
    )
    common = (
        read(common_path) if common_path is not None and common_path.is_file() else {}
    )
    run = read(reference.config_path)
    actual = _merge(common, run)
    merged_folder = _storage_path(paths.get("merged_configs") or "config/merged", root)
    merged = merged_folder / f"{reference.run_id}.toml"
    if merged.is_file() and not _same_path(merged, reference.config_path):
        actual = read(merged)
    baseline = _baseline_from_config(actual, root)
    if baseline is None or not _same_path(baseline, reference.baseline_image):
        raise ValueError(
            "The reference run's actual configuration identifies another raw baseline. Import was not started."
        )
    return actual, tuple(bindings)


def install_imported_draft(view, imported: ImportedColouredMap) -> Path:
    """Explicit Import uses original draft serializers and preserves old bytes."""
    initial_receipt = json.loads(imported.receipt.read_bytes())
    if initial_receipt.get(
        "raw_baseline_sha256", _sha(Path(imported.session.image_path).read_bytes())
    ) != _sha(Path(imported.session.image_path).read_bytes()):
        raise RuntimeError(
            "The imported raw reference photograph changed before draft installation."
        )
    imported = _durable_import(view.project, imported)
    before_dir = imported.work_directory / "before-editor-draft"
    before_dir.mkdir(exist_ok=False)
    canonical_session = lp.session_path(view.project, imported.session.run_id)
    canonical_proposal = lp.proposal_path(view.project, imported.session.run_id)
    canonical_origin = _origin_path(view.project, imported.session.run_id)
    before = {
        path: _current_bytes(path)
        for path in (canonical_session, canonical_proposal, canonical_origin)
    }
    records = []
    for path, data in before.items():
        if data is not None:
            backup = before_dir / path.name
            _write_new(backup, data)
            records.append(
                {"path": str(path), "backup": str(backup), "sha256": _sha(data)}
            )
        else:
            records.append({"path": str(path), "previously_absent": True})
    installation = {
        "format": "darsia-ui-coloured-map-draft-install-v1",
        "import_receipt": str(imported.receipt),
        "import_receipt_sha256": _sha(imported.receipt.read_bytes()),
        "project_root": str(view.project.root),
        "run_id": imported.session.run_id,
        "before": records,
        "installed": [],
    }
    installed_receipt = imported.work_directory / "installation.json"
    previous = {
        name: getattr(view, name, None)
        for name in (
            "session",
            "proposal",
            "_undo_stack",
            "_redo_stack",
            "last_export",
            "_dirty",
            "_ui_import",
            "_ui_import_installation",
            "_ui_import_later_fit",
        )
    }
    show_proposal_before = view.show_proposal.isChecked()
    loading = view._loading
    intended = {}
    try:
        # Serialize with the unchanged original functions before touching any
        # canonical draft file. These exact bytes identify our own writes if
        # a later install/receipt/origin operation fails.
        staged_project = SimpleNamespace(
            root=imported.work_directory / "staged-project"
        )
        staged_session = lp.save_painting_session(
            staged_project, imported.session, actor=default_actor()
        )
        staged_proposal = lp.proposal_path(staged_project, imported.session.run_id)
        imported.proposal.save(staged_proposal)
        for path, staged in (
            (canonical_session, staged_session),
            (canonical_proposal, staged_proposal),
        ):
            raw = staged.read_bytes()
            intended[path] = raw
            _replace_expected(path, raw, before[path])
            installation["installed"].append({"path": str(path), "sha256": _sha(raw)})
        installation["state"] = "complete"
        _write_new(
            installed_receipt,
            (json.dumps(installation, indent=2, ensure_ascii=False) + "\n").encode(
                "utf-8"
            ),
        )
        snapshot = SimpleNamespace(
            project=view.project,
            session=imported.session,
            proposal=imported.proposal,
            _ui_import=imported,
            _ui_import_installation=installed_receipt,
            _ui_import_later_fit=False,
        )
        origin_path, origin_raw = _origin_bytes(snapshot)
        intended[origin_path] = origin_raw
        _replace_expected(origin_path, origin_raw, before[origin_path])
        # Only a coherent stored draft + durable origin replace editor state.
        view._save_timer.stop()
        view.session, view.proposal = imported.session, imported.proposal
        view._undo_stack, view._redo_stack = [], []
        view.last_export = None
        view._dirty = False
        view._ui_import = imported
        view._ui_import_installation = installed_receipt
        view._ui_import_later_fit = False
        view._loading = True
        view._sync_settings()
        view._refresh_layers()
        view._refresh_facies_table()
        view._render_marks()
        view.show_proposal.setChecked(True)
        view._render_proposal()
        view._refresh_export_info()
        view._refresh_status()
        view.propose_status.setText(
            "Imported the coloured map on the raw reference grid. No classifier fit was run; edit layers/facies or use Propose boundaries separately."
        )
    except Exception:
        failure_trace = traceback.format_exc()
        for name, value in previous.items():
            setattr(view, name, value)
        view.show_proposal.setChecked(show_proposal_before)
        restored = _restore_attempt(before, intended, imported.work_directory)
        unsafe = [
            row
            for row in restored
            if row["state"] not in {"restored", "already-original"}
        ]
        failure = dict(
            installation,
            state=(
                "partial-write-failure" if unsafe else "failed-and-originals-restored"
            ),
            traceback=failure_trace,
            restoration=restored,
        )
        failed_receipt = imported.work_directory / "installation.failed.json"
        try:
            _write_new(
                failed_receipt,
                (json.dumps(failure, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        except Exception:
            raise RuntimeError(
                f"Import installation failed and its failure receipt could not be written: {failed_receipt}\n"
                f"Original failure:\n{failure_trace}\nRestoration: {restored}\nReceipt failure:\n{traceback.format_exc()}"
            ) from None
        if unsafe:
            paths = ", ".join(row["path"] for row in unsafe)
            raise RuntimeError(
                f"Import installation has a partial write; external changes were preserved or restoration failed. "
                f"Inspect these paths before reopening: {paths}. Receipt: {failed_receipt}\n"
                f"Original failure:\n{failure_trace}"
            ) from None
        raise
    finally:
        view._loading = loading
    view.changed.emit()
    return installed_receipt


def bind_import_to_export(view, export) -> Path | None:
    """Persist honest provenance beside the actual saved revision/export."""
    worker = getattr(view, "_save_worker", None)
    if (
        worker is None
        or getattr(worker, "session", None) is None
        or getattr(worker, "proposal", None) is None
    ):
        raise RuntimeError(
            "The existing Save/export snapshot is unavailable for import provenance."
        )
    origin = read_import_origin(view.project, worker.session, worker.proposal)
    if origin is None:
        return None
    imported, installation_path, later_fit = origin
    installation = json.loads(Path(installation_path).read_bytes())
    import_bytes = imported.receipt.read_bytes()
    if _sha(import_bytes) != installation["import_receipt_sha256"]:
        raise RuntimeError(
            "The import provenance receipt changed before geometry export."
        )
    current = Path(view.project.root) / "overrides/geometry_facies/current.json"
    revision = json.loads(current.read_bytes())
    if revision.get("revision_id") != export.revision_id:
        raise RuntimeError(
            "The saved revision changed before import provenance could be bound to the export."
        )
    final_digest = (
        (revision.get("configuration") or {})
        .get("facies_generation", {})
        .get("proposal_digest")
    )
    if final_digest != worker.proposal.digest:
        raise RuntimeError(
            "The exported revision is not bound to the Save/export proposal snapshot."
        )
    destination = export.destination / "ui_import_provenance" / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    origin_bytes = _origin_path(view.project, worker.session.run_id).read_bytes()
    _write_new(destination / "draft-origin.json", origin_bytes)
    # The source directory was created fresh by this import and contains only
    # its frozen input, native labeling outputs/config, receipt and old drafts.
    for source in sorted(imported.work_directory.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(imported.work_directory)
        raw = source.read_bytes()
        target = destination / "import-cache" / relative
        _write_new(target, raw)
        files.append(
            {"path": str(target), "source_relative": str(relative), "sha256": _sha(raw)}
        )
    payload = {
        "format": "darsia-native-export-import-provenance-v1",
        "revision_id": export.revision_id,
        "revision_json_sha256": _sha(current.read_bytes()),
        "export_json_sha256": _sha((export.destination / "export.json").read_bytes()),
        "native_setup_toml_sha256": _sha(export.setup_toml.read_bytes()),
        "initial_import_receipt_sha256": _sha(import_bytes),
        "draft_origin_sha256": _sha(origin_bytes),
        "initial_import_interpretation": "Original native palette segmentation; no ExtraTrees fit was run by Import.",
        "initial_import_extra_trees_fit_run": False,
        "subsequent_propose_boundaries_fit_run": later_fit,
        "legacy_kernel_method_tag": lp.PROPOSAL_METHOD,
        "legacy_method_tag_is_retained_serializer_metadata": True,
        "exported_proposal_digest": final_digest,
        "save_snapshot_session_sha256": _sha(
            json.dumps(
                worker.session.to_dict(), ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ),
        "save_snapshot_proposal_classes_sha256": _sha(
            worker.proposal.classes.tobytes()
        ),
        "save_snapshot_proposal_shape": list(worker.proposal.classes.shape),
        "save_snapshot_proposal_dtype": str(worker.proposal.classes.dtype),
        "files": files,
    }
    sidecar = destination / "provenance.json"
    _write_new(
        sidecar,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    view.import_export_provenance = sidecar
    return sidecar


class _ImportWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, context, source, work_root):
        super().__init__()
        self.context, self.source, self.work_root = context, source, work_root

    @Slot()
    def run(self):
        try:
            result = import_coloured_map(
                self.source,
                self.context.reference,
                self.context.config,
                input_grid="raw",
                work_root=self.work_root,
                frame=self.context.frame,
            )
            for binding in self.context.bindings:
                binding.require_unchanged()
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.completed.emit(result)


class NativeColouredMapWorkflow(QObject):
    finished = Signal(bool, str)

    def __init__(self, window, *, work_root=None):
        super().__init__(window)
        self.window = window
        self.work_root = (
            Path(work_root)
            if work_root is not None
            else Path(tempfile.gettempdir()) / "darsia-coloured-map-imports"
        )
        self._thread = self._worker = self._context = self._view = None
        self._outcome = None
        window.installEventFilter(self)

    @property
    def running(self):
        return self._thread is not None

    def eventFilter(self, watched, event):
        if (
            watched is self.window
            and event.type() == QEvent.Type.Close
            and self.running
        ):
            event.ignore()
            self.window.print_log(
                "The coloured-map import is still running. Let its current operation finish before closing."
            )
            return True
        return super().eventFilter(watched, event)

    def install(self, view):
        if (
            view is None
            or getattr(view, "import_coloured_map_button", None) is not None
        ):
            return
        button = QPushButton("Import coloured map…")
        button.setToolTip(
            "Choose an existing coloured map on the raw reference photograph's exact pixel grid; no resizing or corrected-grid conversion."
        )
        note = QLabel(
            "Import a coloured map on the raw reference photograph's grid, or mark layers and use Propose boundaries."
        )
        note.setTextFormat(Qt.TextFormat.PlainText)
        note.setWordWrap(True)
        layout = view.import_map_start.layout()
        layout.insertWidget(0, button)
        layout.insertWidget(1, note)
        view.import_coloured_map_button = button
        view.import_coloured_map_note = note
        button.clicked.connect(lambda: self.start(view))
        original_reload = getattr(view, "reload", None)
        if original_reload is not None:

            def reload():
                result = original_reload()
                self._restore(view)
                return result

            reload.__wrapped__ = original_reload
            view.reload = reload
        original_ready = view._proposal_ready

        def proposal_ready(proposal):
            result = original_ready(proposal)
            if (
                getattr(view, "_ui_import", None) is not None
                and view.proposal is proposal
            ):
                view._ui_import_later_fit = True
                try:
                    persist_import_origin(view)
                except Exception:
                    self.window.print_log(
                        "Coloured-map draft provenance could not be updated.\n"
                        + traceback.format_exc()
                    )
            return result

        proposal_ready.__wrapped__ = original_ready
        view._proposal_ready = proposal_ready
        view.exported.connect(lambda export: self._exported(view, export))
        self._restore(view)

    def install_workspace(self, workspace):
        """Keep the same import action reachable from the retained Advanced view."""
        if getattr(workspace, "advanced_import_coloured_map_button", None) is not None:
            return
        button = QPushButton("Import coloured map…")
        button.setToolTip(
            "Open the raw-reference layer editor and import an existing coloured map there."
        )
        controls = workspace.layout().itemAt(0).layout()
        controls.insertWidget(max(0, controls.count() - 1), button)
        workspace.advanced_import_coloured_map_button = button

        def sync_button(*_args):
            button.setVisible(
                workspace.advanced_button.isChecked()
                and workspace.simple_view is not None
            )

        sync_button()
        workspace.advanced_button.toggled.connect(sync_button)
        # The original series-discard and programmatic view switch block
        # toggle signals. Follow those same instance lifecycle methods too.
        for name in ("_discard_page", "show_advanced"):
            original = getattr(workspace, name, None)
            if original is None:
                continue

            def update(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                sync_button()
                return result

            update.__wrapped__ = original
            setattr(workspace, name, update)

        def open_simple_import():
            if workspace.busy() or self.running:
                return
            workspace.show_advanced(False)
            button.hide()
            view = workspace.simple_view
            if view is not None:
                self.install(view)
                self.start(view)

        button.clicked.connect(open_simple_import)

    def _restore(self, view):
        try:
            restored = restore_import_origin(view)
        except Exception:
            message = (
                "Saved coloured-map draft provenance could not be restored.\n"
                + traceback.format_exc()
            )
            view.import_coloured_map_note.setText(message)
            self.window.print_log(message)
        else:
            if restored is not None:
                view.import_coloured_map_note.setText(
                    "Restored this reference run's saved coloured-map import origin. Initial import used original palette segmentation, without a classifier fit."
                )

    def _context_for(self, view):
        selected = self.window.phase_model_workspace
        root = selected.series_roots.get(selected.series_id)
        geometry = self.window.geometry_workspace
        if (
            root is None
            or geometry.simple_view is not view
            or not _same_path(geometry._loaded_root, root)
        ):
            raise ValueError("The layer editor is not bound to the selected series.")
        reference = view.reference()
        if reference is None or reference.baseline_image is None or view.image is None:
            raise ValueError(
                "Choose a reference run with its raw baseline photograph first."
            )
        if not _same_path(view.image.path, reference.baseline_image):
            raise ValueError(
                "The displayed raw photograph does not belong to the reference run."
            )
        config, bindings = reference_config(Path(root), reference)
        return ImportContext(
            selected.series_id,
            Path(root),
            Path(view.project.root),
            reference,
            deepcopy(view.current_frame()),
            config,
            bindings,
        )

    def start(self, view):
        if (
            self.running
            or view.busy()
            or self.window.geometry_workspace.busy()
            or self.window.process_runner.active
            or getattr(self.window.crop_workflow, "running", False)
        ):
            return False
        if getattr(
            getattr(getattr(self.window, "depth_workflow", None), "sequence", None),
            "running",
            False,
        ):
            return False
        try:
            context = self._context_for(view)
        except Exception as error:
            self.window.print_log(str(error))
            QMessageBox.warning(self.window, "Map was not imported", str(error))
            return False
        source, _ = QFileDialog.getOpenFileName(
            self.window,
            "Import coloured map — RAW reference photograph grid",
            str(context.reference.baseline_image.parent),
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)",
        )
        if not source:
            return False
        # Preserve current normal draft edits before replacing a draft through
        # this explicit Import action. Keep the existing autosave semantics.
        view.save_session()
        if view._dirty:
            QMessageBox.warning(
                self.window,
                "Map was not imported",
                "The existing draft could not be saved. Resolve that error before importing another map.",
            )
            return False
        self._context, self._view = context, view
        self._outcome = None
        self._previous_enabled = self.window.isEnabled()
        self.window.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        view.import_coloured_map_note.setText(
            "Importing the coloured map on the raw reference grid…"
        )
        thread = QThread(self)
        worker = _ImportWorker(context, Path(source), self.work_root)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._capture_complete)
        worker.failed.connect(self._capture_failure)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._finish)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()
        return True

    @Slot(object)
    def _capture_complete(self, result):
        self._outcome = (result, "")

    @Slot(str)
    def _capture_failure(self, trace):
        self._outcome = (None, trace)

    @Slot()
    def _finish(self):
        result, error = self._outcome or (
            None,
            "The import worker exited without a completion result.",
        )
        view, context = self._view, self._context
        if self._thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._thread.wait()
        self._thread = self._worker = None
        self.window.setEnabled(self._previous_enabled)
        QApplication.restoreOverrideCursor()
        if result is not None:
            try:
                now = self._context_for(view)
                if (
                    now.series_id != context.series_id
                    or not _same_path(now.root, context.root)
                    or now.reference.run_id != context.reference.run_id
                    or not _same_path(now.project_root, context.project_root)
                    or lp._frame_key(now.frame) != lp._frame_key(context.frame)
                ):
                    raise RuntimeError(
                        "The selected series or reference changed during import. The existing editor draft was preserved."
                    )
                if now.config != context.config or [
                    (str(row.path), row.data) for row in now.bindings
                ] != [(str(row.path), row.data) for row in context.bindings]:
                    raise RuntimeError(
                        "The reference configuration inputs changed during import. The existing editor draft was preserved."
                    )
                for binding in context.bindings:
                    binding.require_unchanged()
                receipt = install_imported_draft(view, result)
            except Exception:
                error = traceback.format_exc()
            else:
                message = f"Imported the raw-grid coloured map as an editable draft. No geometry revision, crop Apply or classifier fit was started. Receipt and previous draft bytes: {receipt}"
                view.import_coloured_map_note.setText(message)
                self.window.print_log(message)
                self.finished.emit(True, message)
                return
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        message = f"Coloured-map import stopped at {stamp}.\n{error}"
        view.import_coloured_map_note.setText(message)
        self.window.print_log(message)
        QMessageBox.warning(self.window, "Coloured-map import stopped", message)
        self.finished.emit(False, message)

    def _exported(self, view, export):
        try:
            sidecar = bind_import_to_export(view, export)
        except Exception:
            message = (
                "Geometry export completed, but import provenance could not be attached.\n"
                + traceback.format_exc()
            )
            self.window.print_log(message)
            QMessageBox.warning(
                self.window, "Import provenance was not attached", message
            )
        else:
            if sidecar is not None:
                self.window.print_log(
                    f"Import provenance bound to geometry revision {export.revision_id}: {sidecar}"
                )
