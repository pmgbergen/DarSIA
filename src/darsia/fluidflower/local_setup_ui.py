"""Local selected-run Setup using the original Apply and RAW Build functions.

The caller supplies an existing explicit RAW request. This adapter changes only
its post-Apply configuration descriptors; scientific inputs and output authority
remain those selected in that request. It does not prepare or publish Jobs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import traceback
import uuid
import weakref

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal, Slot


def _local_snapshot_path(path, snapshot_root):
    """Keep writes inside the explicit snapshot, away from network aliases."""
    from .rig_raw_write import _canonical

    for candidate in (snapshot_root, path):
        if str(candidate).replace("/", "\\").startswith("\\\\"):
            raise ValueError("Local Setup cannot use a UNC or device snapshot path")
    # The original path check rejects reparse ancestors and resolves mapped
    # drives. A mapped network path is refused after that resolution as well.
    root, target = _canonical(snapshot_root), _canonical(path)
    if any(str(candidate).replace("/", "\\").startswith("\\\\") for candidate in (root, target)):
        raise ValueError("Local Setup cannot use a mapped network snapshot path")
    if not root.is_dir():
        raise ValueError("Local Setup requires its supplied local snapshot root")
    if not target.is_relative_to(root):
        raise ValueError("Local Setup cannot write or bind configurations outside its owned snapshot: " + str(path))
    return target


def _local_configuration_bundle(project, reference, request, snapshot_root):
    """Admit only the original selected-run targets in the supplied snapshot."""
    from . import corrected_setup as original
    from .series_geometry import native_series_link

    _local_snapshot_path(project.root, snapshot_root)
    _local_snapshot_path(project.manifest_path, snapshot_root)
    series = _local_snapshot_path(native_series_link(project)["series_root"], snapshot_root)
    current_path = _local_snapshot_path(original.GeometryRevisionStore(project).current_path, snapshot_root)
    current = json.loads(original._small(current_path))
    config, revision = current["configuration"], current["revision_id"]
    saved = config["frame_binding"]["runs"][reference.run_id]
    _root, destination, _display, _planned = original._canonical_location(
        project.root, config, revision, reference.run_id)
    _local_snapshot_path(destination, snapshot_root)
    receipt_path = destination / "export.json"
    receipt = json.loads(original._small(receipt_path)) if receipt_path.exists() else None
    bundle = original._configuration_bundle(
        series, reference.run_id, saved["configuration"], destination=destination,
        receipt=receipt, project_root=project.root, revision=revision,
    )
    selected = {_local_snapshot_path(path, snapshot_root)
                for path in bundle["configuration_bundles"][reference.run_id]}
    allowed = set(selected)
    if bundle["common_configuration"] is not None:
        allowed.add(_local_snapshot_path(bundle["common_configuration"]["path"], snapshot_root))
    paths = [_local_snapshot_path(row["path"], snapshot_root) for row in request["configuration_files"]]
    if len(set(paths)) != len(paths) or not set(paths) <= allowed or not set(paths) & selected:
        raise ValueError("The local RAW request is not this selected run's original configuration bundle")


def apply_local_setup(project, reference, request_path, *, snapshot_root):
    """Apply one exact request's run and return its original Build handoff."""
    from . import corrected_setup as original
    from .native.config import configure_parameters, load_parameters, reset_parameters
    from .rig_raw_build import _descriptor, capture_configuration_sources, read_build_request
    from .rig_raw_input import camera_binding
    from .series_geometry import native_series_link
    from .series_native_context import camera_u_profile

    bound = deepcopy(request_path) if isinstance(request_path, dict) else _descriptor(Path(request_path))
    _local_snapshot_path(bound["path"], snapshot_root)
    request, _pins = read_build_request(bound, artifact_kind="raw_rig_baseline")
    if reference is None or request["run_id"] != reference.run_id:
        raise ValueError("The explicit RAW request must name the selected run")
    series = Path(native_series_link(project)["series_root"])
    if Path(request["series_context"]["root"]).resolve() != series.resolve():
        raise ValueError("The explicit RAW request belongs to another series")
    profile = camera_u_profile(series, reference.run_id)
    if (profile is None or Path(request["profile"]["path"]).resolve() != profile.path.resolve()
            or request["profile"]["sha256"] != profile.sha256):
        raise ValueError("The RAW request must use the series' exact selected camera profile")
    camera = camera_binding(dict(configuration=str(profile.path), sha256=profile.sha256,
                                 profile_id=profile.profile_id, run_id=reference.run_id))
    settings = dict(run_id=reference.run_id, raw_camera=camera,
                    rig_path=request["output"]["destination"])
    for key in ("native_product", "analysis_exclusions"):
        if key in request:
            settings[key] = deepcopy(request[key])
    token = configure_parameters(load_parameters(request["profile"]["path"], request["profile"]["sha256"]))
    try:
        _local_configuration_bundle(project, reference, request, snapshot_root)
        exported = original.export_corrected_setup(project, reference=reference)
        for key in ("native_product", "analysis_exclusions"):
            if not original._typed_equal(request.get(key), exported.receipt["binding"].get(key)):
                raise ValueError("The RAW request differs from this run's adopted " + key)
        completion = original.apply_corrected_selected_run(
            project, exported, rebuild_facies_by_name=True, raw_setup_settings=settings,
        )
    except BaseException:
        reset_parameters(token)
        raise
    try:
        if _descriptor(Path(bound["path"])) != bound:
            raise ValueError("The original RAW request changed")
        paths = [Path(row["path"]) for row in request["configuration_files"]]
        for path in paths:
            _local_snapshot_path(path, snapshot_root)
        original.require_corrected_export_for_build(paths)
        post_apply = deepcopy(request)
        post_apply["configuration_files"] = list(capture_configuration_sources(paths))
        request_path = exported.destination / ("local-build-request-" + uuid.uuid4().hex + ".json")
        _local_snapshot_path(request_path, snapshot_root)
        original._exclusive(request_path, (json.dumps(post_apply, indent=2, allow_nan=False) + "\n").encode())
        selected = _descriptor(request_path)
        read_build_request(selected, artifact_kind="raw_rig_baseline")
    except Exception as error:
        raise RuntimeError("Apply completed; no Build started. Apply receipt: " + str(completion)
                           + ". Build handoff failed: " + str(error)) from error
    finally:
        reset_parameters(token)
    return dict(state="applied", run_id=reference.run_id, apply_receipt=str(completion),
                raw_build_request=selected, original_request=bound, snapshot_root=str(snapshot_root))


def build_local_setup(handoff):
    """Forward the exact post-Apply request to the ordinary RAW setup entry."""
    from .native.config import configure_parameters, load_parameters, reset_parameters
    from .rig_raw_build import CERTIFICATE_NAME, _descriptor, read_build_request
    from .setup_rig import setup_rig
    from darsia.presets.workflows.rig import Rig

    bound = handoff["raw_build_request"]
    _local_snapshot_path(bound["path"], handoff["snapshot_root"])
    request, _pins = read_build_request(bound, artifact_kind="raw_rig_baseline")
    for row in request["configuration_files"]:
        _local_snapshot_path(row["path"], handoff["snapshot_root"])
    if request["run_id"] != handoff["run_id"]:
        raise ValueError("The local Build handoff names another run")
    token = configure_parameters(load_parameters(request["profile"]["path"], request["profile"]["sha256"]))
    try:
        setup_rig(Rig, [Path(row["path"]) for row in request["configuration_files"]],
                  show=False, raw_build_request=bound)
    finally:
        reset_parameters(token)
    certificate = _descriptor(Path(request["output"]["destination"]) / CERTIFICATE_NAME)
    return dict(handoff, state="built", baseline_renderer=certificate,
                output=request["output"]["destination"])


class _LocalSetupWorker(QObject):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, project, reference, request_path, snapshot_root):
        super().__init__()
        self.project, self.reference, self.request_path = project, reference, request_path
        self.snapshot_root = snapshot_root

    @Slot()
    def run(self):
        handoff = None
        try:
            self.progress.emit("Applying the selected run's corrected geometry locally…")
            handoff = apply_local_setup(self.project, self.reference, self.request_path,
                                        snapshot_root=self.snapshot_root)
            self.progress.emit("Apply complete. Building the selected RAW run model locally…")
            self.completed.emit(build_local_setup(handoff))
        except Exception:
            prefix = ""
            if handoff is not None:
                prefix = ("Apply completed; Build did not complete. Retained request: "
                          + handoff["raw_build_request"]["path"] + "\n")
            self.failed.emit(prefix + traceback.format_exc())
        finally:
            self.finished.emit()


class LocalSetupController(QObject):
    """One local operation, retaining the existing view's navigation boundary."""

    def __init__(self, workspace, view):
        super().__init__(view)
        self.workspace, self.view = weakref.ref(workspace), weakref.ref(view)
        self._original_busy = weakref.WeakMethod(view.busy)
        self._thread = self._worker = None
        self._local_window = None
        self._previous_enabled = True
        self._outcome = "Apply and Build this selected run on this computer."
        view.busy = self.busy
        view.apply_build_button.setText("Apply and Build selected run locally")
        view.apply_build_button.setToolTip("Use the selected run's prepared local Build inputs.")
        view.apply_build_button.clicked.connect(self.start)
        view.busy_changed.connect(self.refresh)
        workspace.reference_changed.connect(self.refresh)
        adoption = getattr(view, "adoption_controller", None)
        changed = getattr(adoption, "findings_changed", None)
        if changed is not None:
            changed.connect(self.refresh)
        self.refresh()

    def busy(self):
        original = self._original_busy()
        return self._thread is not None or bool(original and original())

    def set_request(self, request, *, snapshot_root=None):
        """Accept the Main/local preparation owner's explicit request handoff.

        A path names an existing request. A descriptor additionally retains its
        selected SHA and size; the original reader verifies them in the worker.
        No RAW request or scientific input is inferred here.
        """
        if request is not None and not isinstance(request, (str, Path, dict)):
            raise TypeError("The local Build handoff must be a request path or exact descriptor")
        workspace = self.workspace()
        if workspace is not None:
            workspace.local_setup_build_request = deepcopy(request)
            workspace.local_setup_snapshot_root = snapshot_root
            self.refresh()

    @Slot()
    def refresh(self, *_):
        workspace, view = self.workspace(), self.view()
        if workspace is None or view is None:
            return
        check = getattr(getattr(view, "adoption_controller", None), "findings_block_reason", None)
        reason = check() if callable(check) else ""
        if (getattr(workspace, "local_setup_build_request", None) is None
                or getattr(workspace, "local_setup_snapshot_root", None) is None):
            reason = reason or "Build inputs not ready for the selected run."
        window = workspace.window()
        runner = getattr(window, "process_runner", None)
        depth = getattr(window, "depth_workflow", None)
        if (not window.isEnabled() or (runner is not None and runner.active)
                or (depth is not None and depth.sequence.running)):
            reason = reason or "Wait for the current local operation to finish."
        ready = (workspace.project is not None and workspace.reference() is not None
                 and workspace._loaded_series == workspace.series_id and not reason and not self.busy())
        view.apply_build_button.setEnabled(ready)
        if self._thread is None:
            view.apply_build_reason.setText(reason or self._outcome)

    @Slot()
    def start(self):
        self.refresh()
        workspace, view = self.workspace(), self.view()
        if workspace is None or view is None or not view.apply_build_button.isEnabled():
            return
        request = deepcopy(getattr(workspace, "local_setup_build_request", None))
        snapshot_root = getattr(workspace, "local_setup_snapshot_root", None)
        if request is None or snapshot_root is None:
            return
        thread = QThread(self)
        worker = _LocalSetupWorker(workspace.project, workspace.reference(), request, snapshot_root)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(view.apply_build_reason.setText)
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        # Q185: keep the worker until the GUI owner joins the native thread.
        # Deleting it in its own thread can deadlock Python/Qt signal teardown.
        thread.finished.connect(self._closed)
        self._thread, self._worker = thread, worker
        self._local_window = workspace.window()
        self._previous_enabled = self._local_window.isEnabled()
        self._local_window.installEventFilter(self)
        self._local_window.setEnabled(False)
        view.apply_build_button.setEnabled(False)
        view.busy_changed.emit(True)
        try:
            thread.start()
        except Exception:
            self._failed(traceback.format_exc())
            self._closed()

    def eventFilter(self, watched, event):
        if watched is self._local_window and event.type() == QEvent.Type.Close:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    @Slot(object)
    def _completed(self, result):
        self._outcome = "Local Build completed: " + result["output"]
        self.view().apply_build_reason.setText(self._outcome)

    @Slot(str)
    def _failed(self, message):
        self._outcome = message
        self.view().apply_build_reason.setText(message)
        log = getattr(self.workspace().window(), "print_log", None)
        if callable(log):
            log(message)

    @Slot()
    def _closed(self):
        thread = self._thread
        if thread is not None and not thread.wait():
            return  # No retirement barrier: retain the worker and Main boundary.
        self._thread = self._worker = None
        if thread is not None:
            thread.deleteLater()
        window, self._local_window = self._local_window, None
        if window is not None:
            window.removeEventFilter(self)
            window.setEnabled(self._previous_enabled)
        view = self.view()
        if view is not None:
            view.busy_changed.emit(False)
            self.refresh()
