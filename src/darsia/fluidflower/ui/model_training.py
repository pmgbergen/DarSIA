"""Project-independent Qt work surface for shared optical phase-model training."""

from __future__ import annotations

from darsia.fluidflower.series_paths import configured_series, tool_path, tool_arguments, local_cache_root, model_versions_folder, series_path, series_root

import json
import importlib
import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import os
import signal
import subprocess
import sys
from pathlib import Path
from threading import Thread
from time import monotonic
import uuid

from PySide6.QtCore import QProcess, QSettings, QStandardPaths, QTimer, Qt, QUrl, Signal
from darsia.fluidflower.review_storage import resolve_review_path
from PySide6.QtGui import QCloseEvent, QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.execution import default_execution_cache_root
from darsia.fluidflower.model_display import created_utc_text, friendly_model_name
from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from contextvars import copy_context

from darsia.fluidflower.phase_model_history import ModelHistoryEntries, registration_context_available, run_with_registration_context, REGISTRATION_ROOTS_UNAVAILABLE, deduplicate_model_roots, discover_phase_model_history, inspect_phase_model_history_entry, registered_phase_model_defaults, project_phase_model_roots, project_result_model_roots, metadata_only_catalog
from darsia.fluidflower.phase_workflow import (
    build_phase_review_command,
    build_phase_review_resume_command,
    build_phase_training_command,
    default_phase_review_roots,
    discover_phase_review_sessions,
    parse_training_bundles,
    phase_python_executable,
    prepare_phase_training_request,
)
from darsia.fluidflower.project import default_project_root, sha256_file
from darsia.fluidflower.reviewer_preferences import load_reviewer_name, reviewer_identifier, save_reviewer_name
from darsia.fluidflower.ui.common import FILE_DIALOG_OPTIONS, PathField


def _canonical_series(value: object) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _model_series(entry: Mapping[str, object]) -> set[str]:
    """Use recorded scope, including legacy shared models' dataset coverage."""
    scope = _canonical_series(entry.get("series_scope"))
    if scope and scope not in {"all", "shared", "crossseries"}:
        return {scope}
    # Transfer evaluation keeps its source training inventory. Its actual
    # evaluation target is separate metadata, not the source model's coverage.
    datasets = (entry.get("evaluation_datasets") if entry.get("state") == "transfer_evaluation"
                else entry.get("datasets")) or []
    if isinstance(datasets, str):
        datasets = [datasets]
    return {value for item in datasets if (value := _canonical_series(item))}


def _models_for_series(entries, series: str):
    selected = _canonical_series(series)
    return sorted(
        (entry for entry in entries if not selected or selected in _model_series(entry)),
        key=lambda entry: (float(entry.get("created_timestamp") or 0.), str(entry.get("model_id") or "")),
        reverse=True,
    )


def _newest_review_model(entries):
    """Prefer a prepared model at startup, independently of its QA status."""
    return next((entry for entry in entries if entry.get("validation_binding")),
                entries[0] if entries else None)


def _registered_saved_phase_pipeline(model_root: Path, profile, identity) -> str | None:
    """Read the sealed build pipeline for an explicitly registered G3 campaign.

    These inference-only candidates use a build-plan identity instead of the
    legacy model/runtime formula. Accept it only through the intact registration
    and its hash-bound completion receipt, never from the map binding itself.
    """
    if (not isinstance(profile, Mapping)
            or not (profile.get("approved_g3_gas")
                    or profile.get("format") in {"ffac-rig-camera-physics-v1", "ffac-selected-policy-a-review-physics-v1"})):
        return None
    from darsia.fluidflower.phase_saved_registration import (
        registration_path, registration_warning,
    )

    files = {
        "approved_g3_downstream": ("production_candidate_receipt.json", "approved_g3_candidate_plan.json"),
        "approved_g3_top_surface": ("top_surface_candidate_receipt.json", "top_surface_candidate_plan.json"),
        "approved_g3_barrier_rule_b": ("barrier_rule_b_candidate_receipt.json", "barrier_rule_b_candidate_plan.json"),
        "approved_g3_rig_camera": ("rig_camera_candidate_receipt.json", "rig_camera_inference_plan.json"),
        "approved_g3_area_revision": ("physical_scale_receipt.json", "physical_scale_plan.json"),
        "approved_g3_old_domain_seam_revision": ("rig_camera_candidate_receipt.json", "old_domain_seam_plan.json"),
        "ffac_g3_t2_floor_review": ("rig_camera_candidate_receipt.json", "t2_floor_plan.json"),
        "ffac_g3_t2_l1zone_review": ("rig_camera_candidate_receipt.json", "t2_l1zone_plan.json"),
        "ffac_g3_t3_floor_review": ("rig_camera_candidate_receipt.json", "t3_floor_plan.json"),
        "ffac_g3_t4_floor_review": ("rig_camera_candidate_receipt.json", "t4_floor_plan.json"),
        "ffac_g3_t4b_floor_review": ("rig_camera_candidate_receipt.json", "t4b_floor_plan.json"),
        "ffac_g3_t4c_floor_review": ("rig_camera_candidate_receipt.json", "t4c_floor_plan.json"),
        "ffac_g3_t4c_p1_final_review": ("rig_camera_candidate_receipt.json", "p1_final_plan.json"),
        "ffac_g3_t4c_p1_final2_review": ("rig_camera_candidate_receipt.json", "p1_final2_plan.json"),
        "ffac_g3_t4c_p1_final3_review": ("rig_camera_candidate_receipt.json", "p1_final3_plan.json"),
        "ffac_l1_neighbourhood_floor_review": ("rig_camera_candidate_receipt.json", "l1_neighbourhood_plan.json"),
    }
    # The retained readers are byte-bound scientific sources. Apply storage
    # relocation at this GUI boundary and use their existing results argument.
    from darsia.fluidflower.phase_saved_review_metadata import configured_parameters
    registration_file = resolve_review_path(registration_path(identity[0]))
    directory_parts = len(Path(configured_parameters("registry")["directory"]).parts)
    registered_results = registration_file.parents[directory_parts]
    registration = json.loads(registration_file.read_text(encoding="utf-8-sig"))
    kind = registration.get("registration_kind")
    if kind == "ffac_selected_camera_policy_a_p1_final3_review":
        from darsia.fluidflower.phase_saved_selected_registration import selected_saved_review
        checked = selected_saved_review(model_root, results=registered_results)
        if checked["model_identity"] != list(identity):
            raise ValueError("Selected Policy A saved maps identify another model revision")
        return checked["pipeline_sha256"]
    names = files.get(kind)
    warning = registration_warning(model_root, results=registered_results)
    seam_review = (kind == "approved_g3_old_domain_seam_revision"
                   and warning.startswith("Studio review candidate:"))
    combined_review = None
    if kind == "ffac_g3_t4c_p1_final3_review":
        from darsia.fluidflower.phase_saved_contact_review import p1_final3_saved_review
        checked = p1_final3_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("P1-final3 saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_g3_t4c_p1_final2_review":
        from darsia.fluidflower.phase_saved_categorical_review import p1_final2_saved_review
        checked = p1_final2_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("P1-final2 saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_g3_t4c_p1_final_review":
        from darsia.fluidflower.phase_saved_boundary_review import p1_final_saved_review
        checked = p1_final_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("P1-final saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_g3_t4c_floor_review":
        from darsia.fluidflower.phase_saved_floor_review import t4c_saved_review
        checked = t4c_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("T4c saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_g3_t4b_floor_review":
        from darsia.fluidflower.phase_saved_revision_review import t4b_saved_review
        checked = t4b_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("T4b saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_g3_t4_floor_review":
        from darsia.fluidflower.phase_saved_revision_review import t4_saved_review
        checked = t4_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("T4 saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_l1_neighbourhood_floor_review":
        from darsia.fluidflower.phase_saved_revision_review import l1_neighbourhood_saved_review
        checked = l1_neighbourhood_saved_review(model_root)
        if checked["model_identity"] != list(identity):
            raise ValueError("L1-neighbourhood saved maps identify another model revision")
        return checked["pipeline_sha256"]
    if kind == "ffac_g3_t3_floor_review":
        from darsia.fluidflower.phase_saved_revision_review import t3_saved_review
        review = t3_saved_review(model_root)
        if review["model_identity"] != list(identity):
            raise ValueError("T3 saved maps identify another model revision")
        return review["pipeline_sha256"]
    if kind == "ffac_g3_t2_l1zone_review":
        from darsia.fluidflower.phase_saved_revision_review import combined_saved_review
        combined_review = combined_saved_review(model_root)
    if names is None or not (warning.startswith("Studio review candidate: approved G3") or seam_review or combined_review):
        raise ValueError("Saved phase maps have no intact approved G3 registration")
    if combined_review:
        if combined_review["model_identity"] != list(identity):
            raise ValueError("Combined saved maps identify another model revision")
        return combined_review["pipeline_sha256"]
    receipt_path, plan_path = (model_root / name for name in names)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    pipeline = plan.get("pipeline_sha256")
    if kind == "approved_g3_old_domain_seam_revision":
        plan_binding = receipt.get("old_domain_seam_plan") or {}
        try:
            policy = json.loads((model_root / "rig_bottom_review_policy.json").read_text(encoding="utf-8-sig"))
            plan_sha = sha256_file(plan_path)
            plan_matches = (resolve_review_path(plan_binding["path"]).samefile(plan_path)
                            and plan_sha == plan_binding["sha256"] == registration.get("source_plan_sha256")
                            and receipt.get("source_plan_sha256") == policy["source_plan"]["sha256"])
        except (OSError, KeyError, TypeError, ValueError):
            plan_matches = False
    elif kind == "approved_g3_rig_camera":
        plan_binding = receipt.get("inference_plan") or {}
        try:
            plan_matches = (resolve_review_path(plan_binding["path"]).samefile(plan_path)
                            and sha256_file(plan_path) == plan_binding["sha256"]
                            and receipt.get("source_plan_sha256")
                            == registration.get("source_plan_sha256"))
        except (OSError, KeyError, TypeError):
            plan_matches = False
    else:
        plan_matches = sha256_file(plan_path) == receipt.get("source_plan_sha256")
    if (sha256_file(receipt_path) != registration.get("production_receipt_sha256")
            or receipt.get("complete") is not True
            or receipt.get("model_identity") != list(identity)
            or not plan_matches
            or plan.get("model_identity") != list(identity)
            or not isinstance(pipeline, str) or not re.fullmatch(r"[0-9a-f]{64}", pipeline)):
        raise ValueError("Saved phase maps differ from their sealed build pipeline")
    return pipeline


def _validate_saved_phase_binding(model_root: Path, binding, manifest=None) -> None:
    """Check saved maps against their fitted model, not today's implementation.

    Code hashes remain part of the recorded pipeline. An implementation update
    may not invalidate existing maps while their weights and complete physical
    profile are unchanged. This check authorizes inspection, never inference.
    """
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower.phase_validation import _model_identity

    identity = _model_identity(model_root)
    profile = load_physics_profile(model_root)
    current_runtime = physics_runtime_identity(profile)
    runtime = binding.get("physics_runtime")
    if (binding.get("model_id"), binding.get("model_sha256")) != identity:
        raise ValueError("Completed phase maps identify another model revision")
    # Loaded profiles contain derived, process-local caches (including tuple
    # keys and arrays). Only the public settings belong to the saved snapshot.
    def public(value):
        return {key: item for key, item in value.items() if not key.startswith("_")} if isinstance(value, Mapping) else value
    snapshot_matches = "physics_profile" in binding and public(binding["physics_profile"]) == public(profile)
    if ("physics_profile" in binding and not snapshot_matches
            or runtime != current_runtime and not snapshot_matches):
        raise ValueError("Saved phase maps identify another physical profile")
    pipeline = identity[1] if runtime is None else hashlib.sha256(
        f"{identity[1]}:{runtime}".encode("ascii")).hexdigest()
    if registered := binding.get("v5_registered_runtime"):
        if not isinstance(registered, str) or not re.fullmatch(r"[0-9a-f]{64}", registered):
            raise ValueError("Saved phase maps have an invalid registered runtime")
        pipeline = hashlib.sha256(f"{pipeline}:{registered}".encode("ascii")).hexdigest()
    if registered_pipeline := _registered_saved_phase_pipeline(model_root, profile, identity):
        pipeline = registered_pipeline
    if binding.get("pipeline_sha256") != pipeline or binding.get("output_mode", "review") != "review":
        raise ValueError("Completed phase maps identify another model revision")
    if manifest is not None:
        for key in ("model_id", "model_sha256", "physics_runtime", "pipeline_sha256"):
            if manifest.get(key) != binding.get(key):
                raise ValueError("Saved phase-map manifest differs from its binding")
        if (manifest.get("output_mode", "review") != "review"
                or "physics_profile" in binding and manifest.get("physics_profile") != binding["physics_profile"]):
            raise ValueError("Saved phase-map manifest identifies another physical profile")
        for job in manifest.get("jobs") or []:
            if (job.get("params_sha256") != pipeline
                    or "optical_model_sha256" in job and job["optical_model_sha256"] != identity[1]
                    or "physics_runtime" in job and job["physics_runtime"] != runtime):
                raise ValueError("Saved phase-map candidate differs from its binding")


def _existing_saved_regional_review(model_root: Path) -> Path | None:
    """Reuse an immutable published campaign without rewriting its metadata."""
    from darsia.fluidflower.phase_evidence_audit import verified_review_binding
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness

    source = resolve_review_path(model_root / "validation")
    if not (source / "validation_binding.json").is_file():
        return None
    binding = verified_review_binding(model_root, saved_comparison=True)
    prepared = resolve_review_path(binding["combined_root"])
    manifest = json.loads((prepared / "postflight_manifest.json").read_text(encoding="utf-8-sig"))
    _validate_saved_phase_binding(model_root, binding, manifest)
    source_sha = sha256_file(resolve_review_path(model_root / "review_source/postflight_manifest.json"))
    if any(value.get("source_manifest_sha256") != source_sha for value in (binding, manifest)):
        raise ValueError("Saved phase maps identify another image-source campaign")
    require_reference_readiness(prepared, allow_pending=True)
    print(f"Reusing {binding['run_count']} saved experiments from their recorded phase pipeline; no maps recomputed",
          flush=True)
    return prepared


def _recover_ready_validation_packages(model_root: Path) -> Path | None:
    """Publish completed experiments on explicit Open; never infer missing maps.

    Interrupted preparation can leave valid run packages without the campaign
    binding that Studio discovers. This runs in the preparation subprocess,
    keeping storage reads and metadata finalization off the GUI thread.
    """
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower.tools.phase_reference_readiness import RECEIPT, require_reference_readiness
    from darsia.fluidflower.phase_saved_package_recovery import assemble_saved_packages as assemble, read

    model_root = Path(model_root)
    packages_root = resolve_review_path(model_root / "validation") / "run_packages"
    if not packages_root.is_dir():
        return None
    runtime = physics_runtime_identity(load_physics_profile(model_root))
    source = read(model_root / "review_source/postflight_manifest.json")
    expected = {str(job["run"]) for job in source["jobs"]}
    packages = []
    for package in sorted(packages_root.iterdir()):
        if not package.is_dir():
            continue
        receipt_path = package / RECEIPT
        if not receipt_path.is_file():
            continue
        receipt = read(receipt_path)
        if receipt.get("status") == "preparing" and not all(
                key in receipt for key in ("binding", "campaign")):
            continue
        # A corrupt completed package must be reported, never quietly replaced
        # with an expensive rebuild or published for another model revision.
        require_reference_readiness(package, allow_pending=True)
        binding = read(package / "validation_binding.json")
        manifest = read(package / "postflight_manifest.json")
        try:
            _validate_saved_phase_binding(model_root, binding, manifest)
        except ValueError as error:
            raise ValueError(f"{package.name}: {error}") from error
        if binding.get("physics_runtime") != runtime:
            raise ValueError(f"{package.name}: saved maps use an earlier implementation and need a published review binding")
        jobs = manifest.get("jobs") or []
        if (package.name not in expected or len(jobs) != 1
                or jobs[0].get("run") != package.name
                or binding.get("selected_candidates") != {package.name: jobs[0].get("candidate_id")}):
            raise ValueError(f"{package.name}: completed phase maps identify another experiment")
        source_sha = sha256_file(model_root / "review_source" / package.name / "postflight_manifest.json")
        if any(record.get("source_manifest_sha256") != source_sha for record in (binding, manifest)):
            raise ValueError(f"{package.name}: completed phase maps identify another image-source campaign")
        packages.append(package)
    if not packages:
        return None
    if len({read(package / "validation_binding.json")["pipeline_sha256"] for package in packages}) != 1:
        raise ValueError("Completed experiments identify different phase pipelines")
    # Validate every selected identity before any presentation metadata changes.
    from darsia.fluidflower.tools.calibration_pairwise_review import _candidate_from_job
    for package in packages:
        print(f"Reusing completed phase maps: {package.name.upper()}", flush=True)
        job = read(package / "postflight_manifest.json")["jobs"][0]
        if _candidate_from_job(package, job) is None:
            raise ValueError(f"{package.name}: completed maps have no admitted saved presentation metadata; publish that metadata before opening")
    assemble(model_root, packages, partial=len(packages) < len(expected))
    prepared = read(model_root / "studio_preparation.json")
    print(f"Ready for validation: {len(packages)}/{len(expected)} existing experiments; no maps recomputed",
          flush=True)
    return Path(prepared["validation"])


def _prepare_regional_review(model_root: Path, output_root: Path) -> None:
    """Recover interrupted packages before using the shared optimized preparer."""
    from darsia.fluidflower.phase_validation import _atomic_json, _model_identity
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness

    model_root, output_root = Path(model_root), Path(output_root)
    identity = _model_identity(model_root)
    prepared = _existing_saved_regional_review(model_root)
    if prepared is None:
        prepared = _recover_ready_validation_packages(model_root)
    if prepared is None:
        raise ValueError("This model has no completed saved phase-map package to review")
    require_reference_readiness(prepared, allow_pending=True)
    binding = json.loads((prepared / "validation_binding.json").read_text(encoding="utf-8-sig"))
    if (identity != _model_identity(model_root)
            or (binding.get("model_id"), binding.get("model_sha256")) != identity):
        raise ValueError("Prepared phase maps belong to another model revision")
    _validate_saved_phase_binding(model_root, binding)
    # QProcess completion reads this small receipt; actual assets stay in their
    # immutable prepared set on the series storage root.
    _atomic_json(output_root / "validation_binding.json", binding)


def _read_model_retraction_state(model_root, series):
    """Worker-side storage read; never treat an unavailable ledger as empty."""
    from ..phase_review_retractions import (
        FreshCalibrationRequired, assert_calibration_can_be_inherited,
        empty_snapshot, ledger_folder, load_retractions, model_retractions, resolve_project_root,
        snapshot_from_events,
    )
    live = load_retractions(series=series)
    applied = model_retractions(model_root, series=series) or empty_snapshot(series)
    if applied.get("series") == "all":
        applied = snapshot_from_events(series, [event for event in applied["events"]
            if event["target"]["series"] == _canonical_series(series)])
    calibration_required = ""
    if live["digest"] != applied["digest"]:
        try:
            assert_calibration_can_be_inherited(model_root, live)
        except FreshCalibrationRequired as error:
            calibration_required = str(error)
    return dict(changed=live["digest"] != applied["digest"],
                calibration_required=calibration_required,
                reclassified_count=sum(event.get("action") == "reclassify" for event in live.get("events", [])
                    if event["event_id"] not in {row["event_id"] for row in applied.get("events", [])}),
                current_digest=live["digest"], applied_digest=applied["digest"],
                withdrawn_count=sum(target not in applied.get("targets", []) for target in live["targets"]),
                ledger_path=str(ledger_folder(resolve_project_root(series))))


def _secondary(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("secondaryText")
    return label


def _setting_field(control: QWidget, explanation: str) -> QWidget:
    field = QWidget()
    layout = QVBoxLayout(field)
    layout.setContentsMargins(0, 0, 0, 5)
    layout.setSpacing(3)
    layout.addWidget(control, 0, Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(_secondary(explanation))
    control.setToolTip(explanation)
    return field


class _RefreshingModelComboBox(QComboBox):
    """Paint existing choices before requesting a background catalogue refresh."""

    about_to_open = Signal()

    def showPopup(self) -> None:
        super().showPopup()
        QTimer.singleShot(0, self.about_to_open.emit)


class SharedPhaseModelTrainingPage(QWidget):
    """Collect evidence and train immutable cross-series phase models."""

    training_completed = Signal(str)
    review_opened = Signal(object)
    _model_catalog_ready = Signal(object)
    _model_metadata_ready = Signal(object)
    _model_binding_ready = Signal(object)
    _contributions_ready = Signal(object)
    _retractions_ready = Signal(object)
    _pending_evidence_ready = Signal(object)
    _review_sessions_ready = Signal(object)
    _saved_experiments_ready = Signal(object)

    def __init__(self, project_root: Path | str | None = None, *, metadata_preview_only=False) -> None:
        super().__init__()
        self._native_metadata_preview_only = bool(metadata_preview_only)
        self._reviewer_project_root: Path | None = None
        self._reviewer_save_timer = QTimer(self)
        self._reviewer_save_timer.setSingleShot(True)
        self._reviewer_save_timer.setInterval(500)
        self._reviewer_save_timer.timeout.connect(self._save_reviewer_preference)
        self._project_root_timer = QTimer(self)
        self._project_root_timer.setSingleShot(True)
        self._project_root_timer.setInterval(500)
        self._project_root_timer.timeout.connect(self._project_root_changed)
        self.training_process: QProcess | None = None
        self.training_request: dict[str, object] | None = None
        self.last_output_root: Path | None = None
        self.validation_process: QProcess | None = None
        self.series_batch_pid: int | None = None
        self._series_batch_active = False
        self._series_batch_job: tuple[str, str] | None = None
        self._series_batch_status_path: Path | None = None
        self._series_batch_prior_invocation = ""
        self._series_batch_launch_error: tuple[str, str] | None = None
        self.pending_validation_mode = ""
        self.pending_validation_session_id = ""
        self.pending_validation_output: Path | None = None
        self.pending_validation_model_root: Path | None = None
        self.pending_validation_model_identity: tuple[str, str] | None = None
        self.pending_validation_physics_runtime: str | None = None
        self._training_event_buffer = ""
        self._transfer_event_buffer = ""
        self._transfer_job: dict[str, object] | None = None
        self._transfer_result: dict[str, object] | None = None
        self._transfer_sources_for = ""
        self._transfer_source_rows: list[dict[str, object]] = []
        # One discovery pass parses every prepared manifest and hashes every native input, which
        # is ~27 s for a 57-experiment series on slower storage. Switching series repeated
        # all of it for an answer that had not changed, so the result is kept per model until
        # something could have changed it.
        self._discovery_results: dict[tuple, list[dict[str, object]]] = {}
        self._pending_checked_runs: set[str] = set()
        self._contributions = {}
        self._contributions_root = ""
        self._contributions_generation = 0
        self._contributions_ready.connect(self._receive_contributions)
        self._pending_evidence = {}
        self._pending_evidence_key = None
        self._pending_evidence_inflight = None
        self._pending_evidence_refresh_queued = False
        self._pending_evidence_ready.connect(self._receive_pending_evidence)
        self._pending_evidence_timer = QTimer(self)
        self._pending_evidence_timer.setInterval(30000)
        self._pending_evidence_timer.timeout.connect(self._poll_pending_evidence)
        self._pending_evidence_timer.start()
        self._retraction_status = {}
        self._retraction_inflight = set()
        self._retractions_ready.connect(self._receive_retraction_status)
        self._retraction_poll_timer = QTimer(self)
        self._retraction_poll_timer.setInterval(5000)
        self._retraction_poll_timer.timeout.connect(self._update_action_states)
        self._retraction_poll_timer.start()
        self._completed_transfer: dict[str, object] | None = None
        self._extra_transfer_manifests: list[str] = []
        self.validation_error_output = bytearray()
        self.review_sessions: list[dict[str, object]] = []
        self.visible_review_sessions: list[dict[str, object]] = []
        self._review_sessions_loading = False
        self.model_history: list[dict[str, object]] = []
        self.visible_model_history: list[dict[str, object]] = []
        self._model_filter_settings = QSettings("DarSIA", "DarSIA Phase Model Studio")
        try:
            self._transfer_review_roots = list(json.loads(self._model_filter_settings.value("review/transferRoots", "[]")))
        except (TypeError, ValueError):
            self._transfer_review_roots = []
        self._model_series_labels = {"": "All series"}
        self._configured_series_roots: dict[str, Path] = {}
        try:
            for key, name, root in configured_series():
                self._model_series_labels[key] = name
                self._configured_series_roots[key] = Path(root)
        except (OSError, ValueError, KeyError):
            # The catalog can still display series recorded in saved model metadata.
            pass
        saved_series = _canonical_series(self._model_filter_settings.value("modelHistory/seriesFilter", ""))
        if saved_series and re.fullmatch(r"[a-z][a-z0-9]*", saved_series):
            self._model_series_labels.setdefault(saved_series, saved_series.upper())
        self._active_model_series = saved_series if saved_series in self._model_series_labels else ""
        self._model_selection_by_series: dict[str, str] = {}
        self._experiment_discovery_timer = QTimer(self)
        self._experiment_discovery_timer.setSingleShot(True)
        self._experiment_discovery_timer.timeout.connect(self._auto_find_experiments)
        self._experiment_discovery_requested = ""
        self._completed_model_roots: dict[str, str] = {}
        self._model_catalog_generation = 0
        self._model_catalog_inflight = False
        self._model_catalog_pending = False
        self._model_binding_refresh_pending = False
        self._model_catalog_root = ""
        self._model_catalog_requested_root = ""
        self._model_catalog_priority_root = ""
        self._model_catalog_metadata_generation = 0
        self._model_catalog_auto_selection_root = ""
        self._model_catalog_diagnostics = ()
        self._saved_model_preference_key = None
        self._saved_model_preference_root = ""
        self._saved_model_preference_pending = False
        self._model_catalog_ready.connect(self._apply_async_model_catalog)
        self._model_metadata_ready.connect(self._apply_async_model_metadata)
        self._model_binding_ready.connect(self._apply_async_model_binding)
        self._review_sessions_ready.connect(self._apply_async_review_sessions)
        self._saved_experiments_generation = 0
        self._saved_experiments_ready.connect(self._apply_saved_experiments)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        heading = QLabel("Optical phase models by experimental series")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.catalog_loading_status = _secondary(
            "Loading saved reviews and model versions in the background…"
        )
        layout.addWidget(self.catalog_loading_status)
        layout.addWidget(
            _secondary(
                "Train and validate visible-2D models for the selected experimental "
                "series using one common workflow. Region labeling, "
                "open correction, and model fitting are separate operations. "
                "Concentration, depth conversion, and mass remain specific to each "
                "experimental series."
            )
        )

        project_identity = QHBoxLayout()
        project_identity.addWidget(QLabel("Project folder for reviewer"))
        self.reviewer_project = PathField(
            directory=True,
            placeholder="Choose a project folder to remember the reviewer",
        )
        project_identity.addWidget(self.reviewer_project, 1)
        self.reviewer_project.changed.connect(self._project_root_timer.start)
        layout.addLayout(project_identity)

        identity = QHBoxLayout()
        identity.addWidget(QLabel("Researcher / reviewer"))
        self.reviewer = QLineEdit()
        self.reviewer.setMinimumWidth(260)
        self.reviewer.setPlaceholderText("Required before reviewing or training")
        self.reviewer.setToolTip(
            "Stored on every region label, correction, training request, and model version."
        )
        identity.addWidget(self.reviewer)
        identity.addWidget(
            _secondary("Your name is provenance. It does not change model weighting."),
            1,
        )
        layout.addLayout(identity)
        self.reviewer_preference_status = _secondary(
            "Choose a project folder to remember this reviewer for that project."
        )
        layout.addWidget(self.reviewer_preference_status)

        self.tabs = QTabWidget()
        self.evidence_tab = self._build_evidence_tab()
        self.training_tab = self._build_training_tab()
        self.history_tab = self._build_history_tab()
        from .evidence_audit_library import EvidenceAuditLibrary
        self.evidence_audit_tab = EvidenceAuditLibrary(self)
        self._install_evidence_sections()
        self._output_parent_custom = bool(os.environ.get("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY"))
        self._model_library_custom = self._output_parent_custom
        self.output_parent.changed.connect(self._mark_output_parent_custom)
        self.model_library.changed.connect(self._mark_model_library_custom)
        self.start_tab = self._build_start_tab()
        self.tabs.addTab(self.start_tab, "Start: choose a model")
        self.tabs.addTab(self.evidence_tab, "Review and label evidence")
        self.tabs.addTab(self.training_tab, "Train a new model version")
        self.mass_workspace_tab = QWidget()
        self._mass_workspace_layout = QVBoxLayout(self.mass_workspace_tab)
        self._mass_workspace = None
        self.tabs.addTab(self.mass_workspace_tab, "Concentration, mass and overflow")
        self.phase_comparison_tab = QWidget()
        self._phase_comparison_layout = QVBoxLayout(self.phase_comparison_tab)
        self._phase_comparison_panel = None
        self.tabs.addTab(self.phase_comparison_tab, "Inspect / compare phase maps")
        self.tabs.currentChanged.connect(self._open_phase_comparison_tab)
        self.tabs.currentChanged.connect(self._open_mass_workspace_tab)
        self.tabs.currentChanged.connect(self._refresh_models_on_tab_change)
        layout.addWidget(self.tabs, 1)

        self.reviewer.textChanged.connect(self._update_action_states)
        self.validation_model.currentIndexChanged.connect(self._note_model_catalog_selection)
        self.reviewer.textChanged.connect(self._queue_reviewer_save)
        self.reviewer.editingFinished.connect(self._flush_reviewer_save)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._flush_reviewer_save)
        self.bundles.textChanged.connect(self._update_action_states)
        self.radiometry_mode.currentIndexChanged.connect(self._update_action_states)
        self.radiometry_mode.currentIndexChanged.connect(self._update_training_effect_summary)
        self.radiometry_profile_path.changed.connect(self._update_action_states)
        self.radiometry_profile_path.changed.connect(self._update_training_effect_summary)
        for field in (
            self.ffac_campaign,
            self.ffum_campaign,
            self.combined_campaign,
            self.review_save_root,
        ):
            field.changed.connect(self._update_action_states)
        for control in (
            self.backend,
            self.stage_workers,
            self.extraction_workers,
            self.pixels,
            self.mil_iterations,
            self.change_ablation,
        ):
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else (
                    control.stateChanged
                    if isinstance(control, QCheckBox)
                    else control.valueChanged
                )
            )
            signal.connect(self._update_training_effect_summary)

        self._update_training_effect_summary()
        self._update_action_states()
        if project_root is not None:
            self.set_project_root(project_root)
        self._start_review_session_discovery()
        # A saved, model-bound collection can be opened without first scanning
        # every historical human review on the network storage.
        # Let the native host bind its series and saved selection before the
        # initial worker captures that context.
        QTimer.singleShot(0, self._model_popup_opened)

    def set_project_root(self, root: Path | str | None, *, fixed: bool = False) -> None:
        """Bind the reviewer preference to the selected project folder."""
        if self._reviewer_save_timer.isActive():
            self._reviewer_save_timer.stop()
            self._save_reviewer_preference()
        self._project_root_timer.stop()
        self.reviewer_project.blockSignals(True)
        self.reviewer_project.setText(str(root or ""))
        self.reviewer_project.blockSignals(False)
        self.reviewer_project.setEnabled(not fixed)
        self._reviewer_project_root = None
        if not root:
            self.reviewer_preference_status.setText(
                "Choose a project folder to remember this reviewer for that project."
            )
            return
        try:
            project_path = Path(root).expanduser().resolve()
            name = load_reviewer_name(project_path)
        except (OSError, ValueError) as exc:
            self.reviewer_preference_status.setText(
                f"Could not load reviewer from this project folder: {exc}"
            )
            return
        self._reviewer_project_root = project_path
        if self._active_model_series:
            self._configured_series_roots[self._active_model_series] = project_path
            if fixed:
                self.training_series.setText(self._active_model_series)
            previous_library = self.model_library.text()
            self._sync_default_model_libraries()
            if self.model_library.text() != previous_library:
                self._request_async_model_refresh()
        self.reviewer.blockSignals(True)
        self.reviewer.setText(name)
        self.reviewer.blockSignals(False)
        self._update_action_states()
        self.reviewer_preference_status.setText(
            f"Reviewer name is saved in {project_path / 'config' / 'phase_model_studio_reviewer.json'}."
        )

    def _project_root_changed(self) -> None:
        self.set_project_root(self.reviewer_project.text())

    def _queue_reviewer_save(self) -> None:
        if self._reviewer_project_root is not None:
            self._reviewer_save_timer.start()

    def _flush_reviewer_save(self) -> None:
        if self._reviewer_save_timer.isActive():
            self._reviewer_save_timer.stop()
            self._save_reviewer_preference()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._flush_reviewer_save()
        super().closeEvent(event)

    def _save_reviewer_preference(self) -> None:
        if self._reviewer_project_root is None:
            return
        try:
            save_reviewer_name(self._reviewer_project_root, self.reviewer.text())
        except (OSError, ValueError) as exc:
            self.reviewer_preference_status.setText(f"Could not save reviewer name: {exc}")

    def _models_folder_for_series(self, selected: str) -> Path:
        if selected in self._configured_series_roots:
            root = self._configured_series_roots[selected]
            try:
                if series_root(selected) == root.resolve():
                    return series_path(selected, "phase_models")
            except (OSError, ValueError, KeyError):
                pass
            try:
                return model_versions_folder(root)
            except (OSError, ValueError):
                return root / "models"
        return series_path(selected, "phase_models")

    def _default_model_library(self) -> Path:
        selected = self._active_model_series
        if not selected and len(self._configured_series_roots) == 1:
            selected = next(iter(self._configured_series_roots))
        if selected in self._configured_series_roots:
            return self._models_folder_for_series(selected)
        return default_project_root().parent / "Shared Phase Models"

    def _mark_output_parent_custom(self) -> None:
        self._output_parent_custom = True

    def _mark_model_library_custom(self) -> None:
        self._model_library_custom = True

    def _sync_default_model_libraries(self) -> None:
        default = str(self._default_model_library())
        if not self._output_parent_custom:
            self.output_parent.blockSignals(True)
            self.output_parent.setText(default)
            self.output_parent.blockSignals(False)
        if not self._model_library_custom:
            self.model_library.blockSignals(True)
            self.model_library.setText(default)
            self.model_library.blockSignals(False)

    def _build_start_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(14)
        heading = QLabel("How would you like to start?")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        layout.addWidget(_secondary(
            "Choose an existing model to inspect its predictions and add reviewed labels, "
            "or create a new model from training evidence. Training always saves a new version."
        ))

        existing = QFrame()
        existing.setFrameShape(QFrame.Shape.StyledPanel)
        existing_layout = QVBoxLayout(existing)
        existing_layout.addWidget(QLabel("Continue with an existing model"))
        existing_layout.addWidget(_secondary(
            "Choose a saved model in Review and label evidence. You can inspect older "
            "versions and their validation details there."
        ))
        self.start_existing_model = QPushButton("Choose an existing model")
        self.start_existing_model.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.evidence_tab)
        )
        existing_layout.addWidget(self.start_existing_model)
        layout.addWidget(existing)

        new = QFrame()
        new.setFrameShape(QFrame.Shape.StyledPanel)
        new_layout = QVBoxLayout(new)
        new_layout.addWidget(QLabel("Start from scratch"))
        new_layout.addWidget(_secondary(
            "Select reviewed evidence and training settings, then fit a first "
            "immutable model version."
        ))
        self.start_new_model = QPushButton("Train a new model from scratch")
        self.start_new_model.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.training_tab)
        )
        new_layout.addWidget(self.start_new_model)
        layout.addWidget(new)
        layout.addStretch(1)
        return content

    def _install_evidence_sections(self) -> None:
        self.history_toggle = QPushButton("Show model history and validation details")
        self.history_toggle.setCheckable(True)
        self.history_toggle.toggled.connect(self._toggle_model_history)
        self._evidence_sections.addWidget(self.history_toggle)
        self._evidence_sections.addWidget(self.history_tab)
        self.history_tab.hide()

        self.disagreement_toggle = QPushButton("Show disagreement reviews")
        self.disagreement_toggle.setCheckable(True)
        self.disagreement_toggle.toggled.connect(self._toggle_disagreement_reviews)
        self._evidence_sections.addWidget(self.disagreement_toggle)
        self._evidence_sections.addWidget(self.evidence_audit_tab)
        self.evidence_audit_tab.hide()

    def _toggle_model_history(self, visible: bool) -> None:
        self.history_tab.setVisible(visible)
        self.history_toggle.setText(
            "Hide model history and validation details" if visible
            else "Show model history and validation details"
        )
        if visible:
            self._request_async_model_refresh()
            self.evidence_scroll.ensureWidgetVisible(self.history_tab)

    def _toggle_disagreement_reviews(self, visible: bool) -> None:
        self.evidence_audit_tab.setVisible(visible)
        self.disagreement_toggle.setText(
            "Hide disagreement reviews" if visible
            else "Show disagreement reviews"
        )
        if visible:
            self.evidence_scroll.ensureWidgetVisible(self.evidence_audit_tab)


    def _open_mass_workspace_tab(self, index: int) -> None:
        if self.tabs.widget(index) is not self.mass_workspace_tab:
            return
        if self._mass_workspace is not None:
            self._mass_workspace.ensure_selected_model()
            return
        try:
            from darsia.fluidflower.tools.phase_studio_series_ui import SeriesMassWorkspace
            self._mass_workspace = SeriesMassWorkspace(self.mass_workspace_tab,
                selected_model=lambda: self._selected_validation_model())
            self._mass_workspace_scroll = QScrollArea(self.mass_workspace_tab)
            self._mass_workspace_scroll.setWidgetResizable(True)
            self._mass_workspace_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._mass_workspace_scroll.setWidget(self._mass_workspace)
            self._mass_workspace_layout.addWidget(self._mass_workspace_scroll)
            self._mass_workspace.ensure_selected_model()
        except (ImportError, FileNotFoundError) as error:
            label = _secondary("Concentration workspace could not be loaded: " + str(error))
            self._mass_workspace_layout.addWidget(label)

    def _open_model_comparison(self) -> None:
        self.tabs.setCurrentWidget(self.phase_comparison_tab)
        self._open_phase_comparison_tab(self.tabs.currentIndex())

    def _comparison_entries(self):
        return [dict(entry, label=self._validation_model_label(entry))
                for entry in _models_for_series(self.model_history, self._active_model_series)]

    def _open_phase_comparison_tab(self, index) -> None:
        if self.tabs.widget(index) is not self.phase_comparison_tab:
            return
        from darsia.fluidflower.tools.phase_model_comparison_view import PhaseModelComparisonDialog
        selected = self._selected_validation_model() or {}
        selected_root = str(selected.get("root") or "")
        if self._phase_comparison_panel is not None:
            reset = self._phase_comparison_series != self._active_model_series
            follow = reset or self._phase_comparison_context_root != selected_root
            self._phase_comparison_panel.set_entries(self._comparison_entries(),
                selected_root if follow else "", reset=reset)
            self._phase_comparison_series = self._active_model_series
            self._phase_comparison_context_root = selected_root
            return
        panel = PhaseModelComparisonDialog(self._comparison_entries(),
            selected_root, self.phase_comparison_tab, embedded=True)
        self._phase_comparison_panel = panel
        self._phase_comparison_series = self._active_model_series
        self._phase_comparison_context_root = selected_root
        self._phase_comparison_layout.addWidget(panel)
        # Model-bound corrections keep the existing autosave and review semantics.
        def open_correction(root):
            if self.select_validation_model(root):
                self._launch_review("open_correction")
        panel.review_requested.connect(open_correction)
        self._request_async_model_refresh()

    def _build_evidence_tab(self) -> QWidget:
        # Keep the model choice outside the scrollable review details. Long
        # readiness/warning text and automatic scrolling to a review section
        # must never move the selector out of sight.
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(4)

        selector = QFrame()
        selector.setFrameShape(QFrame.Shape.StyledPanel)
        selector_layout = QGridLayout(selector)
        selector_layout.addWidget(QLabel("Experimental series"), 0, 0)
        self.model_series_filter = self._build_model_series_filter()
        selector_layout.addWidget(self.model_series_filter, 0, 1, 1, 2)
        selector_layout.addWidget(QLabel("Model to validate · newest first"), 1, 0)
        self.validation_model = _RefreshingModelComboBox()
        self.validation_model.setPlaceholderText("Loading models…")
        self.validation_model.setMinimumContentsLength(58)
        self.validation_model.setToolTip(
            "Select the exact immutable model version to review. If its phase maps "
            "do not exist yet, DarSIA generates them from the prepared corrected "
            "frames before opening a new model-bound session."
        )
        self.validation_model.currentIndexChanged.connect(
            self._validation_model_changed
        )
        self.validation_model.about_to_open.connect(self._model_popup_opened)
        selector_layout.addWidget(self.validation_model, 1, 1)
        refresh_models = QPushButton("Refresh models")
        refresh_models.clicked.connect(self._request_async_model_refresh)
        selector_layout.addWidget(refresh_models, 1, 2)
        selector_layout.setColumnStretch(1, 1)
        page_layout.addWidget(selector)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(
            _secondary(
                "Use blind phase labeling when you want an independent human label "
                "for a model-selected region. Use open model correction when the "
                "prediction itself must be inspected and localized errors corrected. "
                "Both retain the full-resolution timeline and continuous local autosave."
            )
        )

        model = QFrame()
        model.setFrameShape(QFrame.Shape.StyledPanel)
        model_layout = QGridLayout(model)
        self.validation_model_summary = _secondary(
            "No model selected. Model choice is separate from saved review sessions."
        )
        model_layout.addWidget(self.validation_model_summary, 0, 0, 1, 3)
        self.train_further_button = QLabel("Saved corrections")
        self.train_further_button.setToolTip(
            "Retrain both CO2 and gas heads as a new version using earlier evidence "
            "and saved corrections for this model's lineage. Keep the selected "
            "series and physical settings, then generate native review maps."
        )
        self.train_further_button.setText("Train new version from saved corrections")
        self.training_update_card = QFrame()
        self.training_update_card.setObjectName("trainingUpdateCard")
        self.training_update_card.setStyleSheet(
            "QFrame#trainingUpdateCard { border: 1px solid #468cae; border-radius: 5px; }"
        )
        update_layout = QGridLayout(self.training_update_card)
        update_heading = QLabel("Update this model with your saved labels")
        update_heading.setStyleSheet("font-weight: 600; font-size: 14px;")
        update_layout.addWidget(update_heading, 0, 0, 1, 2)
        self.training_update_counts = QLabel("Checking saved labels…")
        self.training_update_counts.setWordWrap(True)
        self.training_update_counts.setStyleSheet("font-weight: 600; font-size: 15px;")
        update_layout.addWidget(self.training_update_counts, 1, 0, 1, 2)
        self.training_update_details = _secondary(
            "Counts are compared with the selected model's recorded training inputs."
        )
        update_layout.addWidget(self.training_update_details, 2, 0, 1, 2)
        self.update_model_button = QLabel("Saved model evidence")
        self.update_model_button.setMinimumHeight(38)
        self.update_model_button.setEnabled(False)
        update_layout.addWidget(self.update_model_button, 3, 0)
        self.refresh_new_evidence_button = QPushButton("Refresh saved labels")
        self.refresh_new_evidence_button.clicked.connect(
            lambda: self._load_pending_evidence(force=True, discover=True)
        )
        update_layout.addWidget(self.refresh_new_evidence_button, 3, 1)
        model_layout.addWidget(self.training_update_card, 1, 0, 1, 3)
        self.compare_phase_models_button = QPushButton("Compare two phase models side by side")
        self.compare_phase_models_button.setToolTip(
            "Inspect two saved models on matching source photographs, with synchronized "
            "timeline, zoom and phase layers. Existing annotations remain with their model."
        )
        self.compare_phase_models_button.clicked.connect(self._open_model_comparison)
        model_layout.addWidget(self.compare_phase_models_button, 3, 0, 1, 3)
        self.train_further_summary = _secondary(
            "Full retraining with cumulative saved labels. New versions appear here "
            "when fitted; phase maps follow automatically."
        )
        
        layout.addWidget(model)
        self._evidence_sections = QVBoxLayout()
        layout.addLayout(self._evidence_sections)

        transfer = QFrame()
        transfer.setFrameShape(QFrame.Shape.StyledPanel)
        transfer_layout = QGridLayout(transfer)
        transfer_layout.addWidget(QLabel("Experiments for the selected model"), 0, 0, 1, 3)
        self.transfer_review_filter = QComboBox()
        self.transfer_review_filter.addItem("All experiments", "all")
        self.transfer_review_filter.addItem("Not manually reviewed", "not_reviewed")
        self.transfer_review_filter.addItem("Manually reviewed", "reviewed")
        self.transfer_review_filter.setToolTip(
            "Filter by saved human phase-map review for this model. Generated predictions "
            "alone do not count as manual review. In-progress experiments stay under "
            "Not manually reviewed until their frames have a completed assessment. "
            "Experiments that still need image correction remain visible."
        )
        self.transfer_review_filter.currentIndexChanged.connect(self._rebuild_transfer_targets)
        transfer_layout.addWidget(self.transfer_review_filter, 1, 0, 1, 2)
        self.transfer_count = _secondary("Loading experiment availability…")
        transfer_layout.addWidget(self.transfer_count, 1, 2)
        self.transfer_target = QComboBox()
        self.transfer_target.setMinimumContentsLength(50)
        self.transfer_target.currentIndexChanged.connect(self._transfer_target_changed)
        self.transfer_target.setParent(content)
        self.transfer_target.hide()
        self.refresh_transfer_targets = QPushButton("Find experiments")
        self.refresh_transfer_targets.setToolTip(
            "Find experiments in the selected series and refresh saved manual-review "
            "progress after reviewing phase maps."
        )
        self.refresh_transfer_targets.clicked.connect(self._refind_transfer_targets)
        transfer_layout.addWidget(self.refresh_transfer_targets, 2, 2)
        self.transfer_physics_mode = QComboBox()
        self.transfer_physics_mode.addItem("Full phase model required", "")
        self.transfer_physics_mode.currentIndexChanged.connect(self._transfer_physics_changed)
        self.transfer_physics_mode.setParent(content)
        self.transfer_physics_mode.hide()
        self.choose_transfer_campaign = QPushButton("Choose prepared campaign...")
        self.choose_transfer_campaign.setToolTip(
            "Add a campaign's postflight_manifest.json when its corrected images are "
            "prepared. Experiments that still need image correction remain listed, "
            "but phase maps cannot be generated for them yet."
        )
        self.choose_transfer_campaign.clicked.connect(self._choose_transfer_campaign)
        
        self.test_transfer_button = QPushButton("Open selected ready experiment")
        self.test_transfer_button.setToolTip("Run the frozen model and open predictions directly. This does not retrain the model or add test labels to training.")
        self.test_transfer_button.clicked.connect(self._open_selected_experiment)
        self.blind_transfer_button = QLabel("Blind labeling (optional)")
        
        self.transfer_comparison_button = QPushButton("Open transfer comparison")
        self.transfer_comparison_button.setToolTip("After blind labeling, open the same evaluation with predictions visible. Labels remain outside automatic training.")
        self.transfer_comparison_button.clicked.connect(self._open_transfer_comparison)
        
        self.include_transfer_training_button = QLabel("Saved training inclusion")
        self.include_transfer_training_button.setToolTip(
            "Include this completed full-model experiment's saved and future corrections in the next Train further run. No training starts now.")
        
        self.transfer_status = _secondary(
            "Choose a prepared experiment and test the frozen model with predictions visible. "
            "Optical training and calibration exposure are reported separately; unverified does not mean held-out. "
            "Blind labeling is optional. Test labels stay outside automatic retraining."
        )
        transfer_layout.addWidget(self.transfer_status, 5, 0, 1, 3)
        columns = QHBoxLayout()
        self.ready_experiments = QTableWidget(0, 4)
        self.pending_experiments = QTableWidget(0, 4)
        self.select_all_pending_button = QPushButton("Select all")
        self.clear_pending_button = QPushButton("Clear selection")
        self.select_all_pending_button.clicked.connect(lambda: self._check_pending_experiments(True))
        self.clear_pending_button.clicked.connect(lambda: self._check_pending_experiments(False))
        for title, table in (("Needs preparation", self.pending_experiments),
                             ("Ready for validation", self.ready_experiments)):
            box = QVBoxLayout(); box.addWidget(QLabel(title)); box.addWidget(table)
            table.setHorizontalHeaderLabels(["Experiment", "Details", "Training regions", "New / changed"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            for column in (2, 3):
                table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            table.verticalHeader().hide(); table.setMaximumHeight(220)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.itemSelectionChanged.connect(lambda t=table: self._experiment_selected(t))
            actions = QHBoxLayout()
            if table is self.pending_experiments:
                actions.addWidget(self.select_all_pending_button)
                actions.addWidget(self.clear_pending_button)
            else:
                actions.addWidget(self.test_transfer_button)
            box.addLayout(actions)
            columns.addLayout(box)
        self.pending_experiments.itemChanged.connect(self._pending_experiment_checked)
        transfer_layout.addLayout(columns, 3, 0, 1, 3)
        statistics_controls = QHBoxLayout()
        self.experiment_sort = QComboBox()
        self.experiment_sort.addItem("Experiment number", "run")
        self.experiment_sort.addItem("Most training regions first", "trained")
        self.experiment_sort.addItem("Most new / changed regions first", "pending")
        self.experiment_sort.currentIndexChanged.connect(self._rebuild_transfer_targets)
        self.show_training_statistics = QCheckBox("Show validation statistics")
        statistics_controls.addWidget(QLabel("Sort experiments"))
        statistics_controls.addWidget(self.experiment_sort)
        statistics_controls.addWidget(self.show_training_statistics)
        self.refresh_training_statistics = QPushButton("Refresh statistics")
        self.refresh_training_statistics.clicked.connect(lambda: self._load_contributions(force=True))
        statistics_controls.addWidget(self.refresh_training_statistics)
        transfer_layout.addLayout(statistics_controls, 4, 0, 1, 3)
        self.training_statistics_panel = QWidget()
        statistics_layout = QVBoxLayout(self.training_statistics_panel)
        self.training_statistics_summary = _secondary("Select an experiment to see its saved training evidence.")
        statistics_layout.addWidget(self.training_statistics_summary)
        self.training_statistics_table = QTableWidget(0, 3)
        self.training_statistics_table.setHorizontalHeaderLabels(["Validation category", "Used in optical training", "New / changed"])
        self.training_statistics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.training_statistics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        statistics_layout.addWidget(self.training_statistics_table)
        self.training_statistics_panel.hide()
        self.show_training_statistics.toggled.connect(self.training_statistics_panel.setVisible)
        transfer_layout.addWidget(self.training_statistics_panel, 4, 3, 9, 1)
        self.advanced_experiments = QWidget()
        advanced_layout = QHBoxLayout(self.advanced_experiments)
        for widget in (self.choose_transfer_campaign, self.blind_transfer_button,
                       self.transfer_comparison_button, self.include_transfer_training_button):
            advanced_layout.addWidget(widget)
        advanced_toggle = QCheckBox("Additional review options")
        advanced_toggle.toggled.connect(self.advanced_experiments.setVisible)
        self.advanced_experiments.hide()
        transfer_layout.addWidget(advanced_toggle, 11, 0, 1, 3)
        transfer_layout.addWidget(self.advanced_experiments, 12, 0, 1, 3)
        self.series_batch_button = QLabel(
            "Generate phase maps for all experiments in this series"
        )
        self.series_batch_button.setToolTip(
            "Apply the selected frozen model to all experiments; no model fitting "
            "and no automatic QC pass. Existing completed maps are reused when resumed."
        )
        self.series_batch_workers = QSpinBox()
        self.series_batch_workers.setRange(0, 14)
        try:
            saved_batch_workers = int(self._model_filter_settings.value(
                "seriesBatch/cpuWorkers", self._model_filter_settings.value("seriesBatch/workers", 4)))
        except (TypeError, ValueError):
            saved_batch_workers = 4
        self.series_batch_workers.setValue(max(0, min(14, saved_batch_workers)))
        self.series_batch_workers.setToolTip(
            "Up to 14 shared CPU compute slots. With the GPU worker selected they run the ordered history/physics "
            "steps, plus only those optical frames the GPU backend cannot compute (two slots per experiment prepared "
            "at once, at most four experiments); without it they also compute every optical frame. Change during "
            "preparation; active work finishes before a decrease takes effect."
        )
        self.series_batch_gpu = QCheckBox("1 GPU")
        self.series_batch_gpu.setChecked(
            str(self._model_filter_settings.value("seriesBatch/gpuWorker", "true")).lower()
            in {"true", "1", "yes"})
        self.series_batch_gpu.setToolTip("One shared GPU worker. While selected it computes every optical frame it supports, so that "
            "CPU slots stay free for the ordered history/physics steps. Switch off during preparation to finish its "
            "current image and then use CPU workers.")
        self.series_batch_status = _secondary(
            "Choose one series and a model to prepare its phase maps for manual QC."
        )
        transfer_layout.addWidget(self.series_batch_status, 8, 0, 1, 3)
        self.series_batch_progress = QProgressBar()
        self.series_batch_progress.setRange(0, 1)
        self.series_batch_progress.setValue(0)
        self.series_batch_progress.setFormat("0/0 experiments")
        # Nothing to report until a batch runs in this window. A status file left behind by an
        # earlier one is history, and the two lists above already show where the series stands.
        self.series_batch_progress.hide()
        self._series_batch_reported_for: tuple[str, str] | None = None
        transfer_layout.addWidget(self.series_batch_progress, 9, 0, 1, 3)
        # A running batch reports each experiment in the two lists above, not in a third table.
        # A separate table fed from status.json kept the previous batch's rows after it finished,
        # and a batch started for part of a series showed only that part, which reads as an
        # experiment stuck in progress. The lists come from discovery and are always current.
        self._batch_run_states: dict[str, str] = {}
        layout.addWidget(transfer)

        recent = QFrame()
        recent.setFrameShape(QFrame.Shape.StyledPanel)
        recent_layout = QGridLayout(recent)
        recent_layout.addWidget(QLabel("Saved review sessions"), 0, 0, 1, 3)
        self.saved_sessions_status = _secondary("")
        recent_layout.addWidget(self.saved_sessions_status, 1, 0, 1, 3)
        self.saved_sessions = QComboBox()
        self.saved_sessions.setMinimumContentsLength(58)
        self.saved_sessions.currentIndexChanged.connect(self._session_changed)
        recent_layout.addWidget(self.saved_sessions, 2, 0, 1, 2)
        self.refresh_review_sessions_button = QPushButton("Refresh saved review sessions")
        self.refresh_review_sessions_button.clicked.connect(self._refresh_review_sessions)
        recent_layout.addWidget(self.refresh_review_sessions_button, 2, 2)
        self.session_summary = _secondary("No local review session found.")
        recent_layout.addWidget(self.session_summary, 3, 0, 1, 3)
        self.resume_session_button = QPushButton("Resume selected session")
        self.resume_session_button.setToolTip(
            "Reopens the original workspace and decision directory without converting "
            "or replacing existing annotations."
        )
        self.resume_session_button.clicked.connect(self._resume_selected_session)
        recent_layout.addWidget(self.resume_session_button, 4, 0, 1, 2)
        self.add_session_evidence = QPushButton("Add this session to training inputs")
        self.add_session_evidence.setToolTip("Adds this session to the manual Training tab. It does not change the automatic Train further evidence selection.")
        self.add_session_evidence.clicked.connect(self._add_selected_session_as_bundle)
        recent_layout.addWidget(self.add_session_evidence, 4, 2)
        layout.addWidget(recent)

        sources = QFrame()
        sources.setFrameShape(QFrame.Shape.StyledPanel)
        source_form = QFormLayout(sources)
        self.review_source_form = source_form
        self.ffac_campaign = PathField(
            directory=True,
            placeholder="First prepared source campaign",
        )
        self.ffum_campaign = PathField(
            directory=True,
            placeholder="Second prepared source campaign",
        )
        self.combined_campaign = PathField(
            directory=True,
            placeholder="Local combined review package",
        )
        self.review_save_root = PathField(
            directory=True,
            placeholder="Reviewer-owned local save root",
        )
        roots = default_phase_review_roots()
        if roots:
            self.review_save_root.setText(str(roots[0]))
        self.review_gpu = QCheckBox("Use GPU-accelerated image display")
        self.review_gpu.setChecked(True)
        self.review_gpu.setToolTip(
            "Accelerates zooming and overlays only. It does not change image pixels, "
            "phase predictions, or saved labels."
        )
        source_form.addRow(
            "Prepared source campaign 1",
            _setting_field(
                self.ffac_campaign,
                "Folder containing corrected frames, phase-map assets, and its "
                "postflight manifest. This is read-only source evidence.",
            ),
        )
        source_form.addRow(
            "Prepared source campaign 2",
            _setting_field(
                self.ffum_campaign,
                "Second read-only source campaign when the review combines separate "
                "packages. Series identity is recorded in each package manifest.",
            ),
        )
        source_form.addRow(
            "Prepared review package",
            _setting_field(
                self.combined_campaign,
                "The manifest that lists the included experiments and records "
                "which immutable model candidates and frames belong together.",
            ),
        )
        source_form.addRow(
            "Local autosave location",
            _setting_field(
                self.review_save_root,
                "Your reviewer-specific JSON, recovery journal, and session record "
                "are written here continuously. Source images are never changed.",
            ),
        )
        source_form.addRow(
            "Display",
            _setting_field(
                self.review_gpu,
                "GPU display makes zooming and overlays smoother. It has no effect "
                "on source resolution, predictions, or scientific output.",
            ),
        )
        layout.addWidget(sources)

        modes = QGridLayout()
        blind = QFrame()
        blind.setFrameShape(QFrame.Shape.StyledPanel)
        blind_layout = QVBoxLayout(blind)
        blind_heading = QLabel("Blind phase labeling")
        blind_heading.setObjectName("sectionTitle")
        blind_layout.addWidget(blind_heading)
        blind_layout.addWidget(
            _secondary(
                "The selected model ranks regions by predictive uncertainty computed "
                "from its phase probabilities: boxes with overlapping class "
                "probabilities are prioritized, while phase and position are balanced "
                "for diversity. This is an active-learning score, not physical "
                "uncertainty. "
                "Missing region suggestions are prepared when you open an experiment here. "
                "Its proposed phase remains hidden until you answer. You can also draw "
                "your own boxes or freehand evidence without revealing the phase map."
            )
        )
        self.open_blind_labeling = QPushButton("Open blind phase labeling")
        self.open_blind_labeling.clicked.connect(
            lambda: self._launch_review("blind_region_labeling")
        )
        blind_layout.addWidget(self.open_blind_labeling)
        modes.addWidget(blind, 0, 0)

        correction = QFrame()
        correction.setFrameShape(QFrame.Shape.StyledPanel)
        correction_layout = QVBoxLayout(correction)
        correction_heading = QLabel("Open model correction")
        correction_heading.setObjectName("sectionTitle")
        correction_layout.addWidget(correction_heading)
        correction_layout.addWidget(
            _secondary(
                "The complete phase map from the version selected under Model to "
                "validate is visible. Inspect it through time, draw boxes or freehand "
                "regions around specific errors, and assign the correct phase or "
                "artifact class. Every new session records the exact selected model."
            )
        )
        self.open_model_correction = QPushButton("Open model correction")
        self.open_model_correction.clicked.connect(
            lambda: self._launch_review("open_correction")
        )
        correction_layout.addWidget(self.open_model_correction)
        modes.addWidget(correction, 0, 1)
        modes.setColumnStretch(0, 1)
        modes.setColumnStretch(1, 1)
        layout.addLayout(modes)

        self.review_status = _secondary(
            "Select a saved session or choose the three prepared campaign locations."
        )
        layout.addWidget(self.review_status)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        self.evidence_scroll = scroll
        page_layout.addWidget(scroll, 1)
        return page

    def _build_training_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(
            _secondary(
                "Training never modifies an existing model. It freezes selected "
                "review files and creates a new immutable version. Complete-run "
                "holdouts test whether the model transfers between experiments "
                "rather than only memorizing reviewed frames. The image normalization "
                "selected below applies to this new version; one mode is saved per fit."
            )
        )

        continuation = QFrame()
        continuation_layout = QVBoxLayout(continuation)
        continuation_layout.addWidget(QLabel("New saved validation evidence"))
        continuation_layout.addWidget(self.train_further_summary)
        self.pending_training_sessions = QTableWidget(0, 3)
        self.pending_training_sessions.setHorizontalHeaderLabels(["Experiment / session", "Saved labels", "New evidence"])
        self.pending_training_sessions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pending_training_sessions.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pending_training_sessions.setMaximumHeight(180)
        continuation_layout.addWidget(self.pending_training_sessions)
        continuation_layout.addWidget(QLabel("Prepare phase maps after training"))
        self.post_training_scope = QComboBox()
        self.post_training_scope.addItem("Experiments with saved validation / annotations", "reviewed")
        self.post_training_scope.addItem("All experiments in the series", "all")
        self.post_training_scope.setCurrentIndex(max(0,self.post_training_scope.findData(
            str(self._model_filter_settings.value("training/postViewScope","reviewed")))))
        self.post_training_scope.currentIndexChanged.connect(lambda: self._model_filter_settings.setValue(
            "training/postViewScope", self.post_training_scope.currentData()))
        continuation_layout.addWidget(self.post_training_scope)
        continuation_layout.addWidget(_secondary("This selects the experiments to prepare for viewing, not the training data. Already prepared maps for the new model are reused."))
        continuation_layout.addWidget(self.train_further_button)
        layout.addWidget(continuation)
        evidence = QFrame()
        evidence.setFrameShape(QFrame.Shape.StyledPanel)
        evidence_layout = QFormLayout(evidence)
        self.bundles = QTextEdit()
        self.bundles.setPlaceholderText(
            "One reviewed evidence package per row:\n"
            "DECISIONS_JSON | COMBINED_MANIFEST_JSON"
        )
        self.bundles.setMaximumHeight(118)
        self.bundles.setToolTip(
            "Each row joins human decisions to the immutable image/candidate manifest "
            "that gives those decisions their exact provenance."
        )
        bundle_actions = QHBoxLayout()
        self.add_bundle_button = QPushButton("Add evidence files...")
        self.add_bundle_button.clicked.connect(self._add_bundle)
        bundle_actions.addWidget(self.add_bundle_button)
        self.add_selected_session_button = QPushButton("Add selected saved session")
        self.add_selected_session_button.clicked.connect(self._add_selected_session_as_bundle)
        bundle_actions.addWidget(self.add_selected_session_button)
        bundle_actions.addStretch(1)
        bundle_widget = QWidget()
        bundle_widget.setLayout(bundle_actions)
        evidence_layout.addRow("Reviewed evidence packages", self.bundles)
        evidence_layout.addRow("", bundle_widget)
        self.annotator_selection = QWidget()
        annotator_layout = QVBoxLayout(self.annotator_selection)
        annotator_layout.setContentsMargins(0, 0, 0, 0)
        annotator_layout.addWidget(QLabel("Annotators"))
        self.annotators = QListWidget()
        self.annotators.setMaximumHeight(96)
        self.annotators.setToolTip("People whose saved reviews were found. Unticked people's sessions are hidden.")
        self.annotators.itemChanged.connect(self._annotators_changed)
        annotator_layout.addWidget(self.annotators)
        annotator_layout.addWidget(_secondary(
            "Only the ticked people's boxes, blind answers and evidence changes enter the new model. "
            "All are ticked by default."))
        evidence_layout.addRow(self.annotator_selection)
        self.output_parent = PathField(directory=True)
        self.output_parent.setText(
            str(os.environ.get("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY") or
                self._default_model_library())
        )
        self.output_parent.setToolTip(
            "Single-series training writes new versions to that series' models folder. "
            "This directory applies to shared models; earlier versions are never overwritten."
        )
        self.cache_parent = PathField(directory=True)
        self.cache_parent.setText(
            str(default_execution_cache_root() / "phase-model-training")
        )
        self.cache_parent.setToolTip(
            "Local training workspace. Model updates reuse the series' existing feature "
            "cache under the configured execution-cache root."
        )
        evidence_layout.addRow(
            "New model versions",
            _setting_field(
                self.output_parent,
                "Every training run creates a separate timestamped folder. V5 and "
                "all earlier models remain unchanged and reproducible.",
            ),
        )
        evidence_layout.addRow(
            "Reusable local feature cache",
            _setting_field(
                self.cache_parent,
                "Keeps expensive image features for reuse. It can be deleted safely, "
                "but the next fit will need to extract those features again.",
            ),
        )
        layout.addWidget(evidence)

        radiometry_form = QFormLayout()
        self.radiometry_mode = QComboBox()
        self.radiometry_mode.addItem("Uniform24 - common ColorChecker reference image", "image")
        self.radiometry_mode.addItem("Uniform24 - published ColorChecker after 2014", "uniform24-after2014")
        self.radiometry_mode.addItem("Uniform24 - measured target profile", "custom")
        self.radiometry_mode.addItem("Legacy - unchanged prepared review images", "legacy")
        self.radiometry_mode.setToolTip(
            "All 24 chart colors define one common target across series. Each frame, "
            "baseline and previous image is corrected independently before feature "
            "extraction. The published target requires a matching physical chart. "
            "Old models retain their original input basis."
        )
        self.radiometry_profile_path = PathField(directory=False)
        self.radiometry_mode.currentIndexChanged.connect(
            lambda: self.radiometry_profile_path.setEnabled(self.radiometry_mode.currentData() in {"image", "custom"})
        )
        self.radiometry_profile_path.setEnabled(True)
        self.radiometry_profile_path.setToolTip(
            "Choose a stable RGB image with the ColorChecker visible, "
            "or a frozen profile JSON when 'measured target profile' is selected. "
            "One target is shared by all selected series; each input frame is fitted independently."
        )
        self.training_series = QLineEdit()
        self.training_series.setPlaceholderText("One registered experimental series")
        self.training_series.setText(self._active_model_series)
        self.training_series.setToolTip(
            "Enter exactly one series for separate training. Only that series' "
            "annotations enter the new model, even when a saved session contains "
            "several series. Original sessions remain unchanged."
        )
        self.training_scope = QComboBox()
        self.training_scope.addItem("Separate model for one experimental series", "single-series")
        self.training_scope.addItem("Shared model across series (experimental)", "shared")
        self.training_scope.setToolTip("Separate models share code and physics methods, not fitted weights or training labels.")
        self.training_scope.currentIndexChanged.connect(self._update_action_states)
        self.training_series.textChanged.connect(self._update_action_states)
        radiometry_form.addRow("Model scope", self.training_scope)
        self.training_evaluation = QComboBox()
        self.training_evaluation.addItem("Whole-experiment cross-validation", "whole-run")
        self.training_evaluation.addItem("Single-experiment exploratory candidate (not validated)", "fit-only-single-experiment")
        self.training_evaluation.setToolTip(
            "Cross-validation holds out complete experiments. The exploratory option "
            "requires exactly one reviewed experiment and cannot establish generalization."
        )
        radiometry_form.addRow("Evaluation", self.training_evaluation)
        radiometry_form.addRow("New-model image normalization", self.radiometry_mode)
        radiometry_form.addRow("Common reference image / profile", self.radiometry_profile_path)
        radiometry_form.addRow("Experimental series", self.training_series)
        layout.addLayout(radiometry_form)

        parameter_frame = QFrame()
        parameter_frame.setFrameShape(QFrame.Shape.StyledPanel)
        parameter_form = QFormLayout(parameter_frame)
        self.backend = QComboBox()
        self.backend.addItem("Automatic (recommended)", "auto")
        self.backend.addItem("NVIDIA CUDA · float32", "cuda")
        self.backend.addItem("CPU reference · float64", "cpu")
        parameter_form.addRow(
            "Model-fitting device",
            _setting_field(
                self.backend,
                "Automatic uses CUDA when available and otherwise CPU. CUDA is faster; "
                "CPU float64 is the numerical reference. Backend parity is checked and "
                "the selected device is recorded in the model.",
            ),
        )
        self.stage_workers = QSpinBox()
        self.stage_workers.setRange(1, 64)
        self.stage_workers.setValue(4)
        parameter_form.addRow(
            "Evidence preparation workers",
            _setting_field(
                self.stage_workers,
                "Copies and verifies independent run assets in parallel. More workers "
                "can shorten preparation but increase storage traffic and RAM use; this "
                "setting does not change the fitted model.",
            ),
        )
        self.extraction_workers = QSpinBox()
        self.extraction_workers.setRange(1, 64)
        self.extraction_workers.setValue(4)
        parameter_form.addRow(
            "Feature extraction workers",
            _setting_field(
                self.extraction_workers,
                "Builds RGB, texture, and DarSIA change features for different runs in "
                "parallel. More workers improve speed only while CPU, RAM, and source "
                "storage have capacity.",
            ),
        )
        self.pixels = QSpinBox()
        self.pixels.setRange(64, 1_000_000)
        self.pixels.setValue(1024)
        self.pixels.setSingleStep(256)
        parameter_form.addRow(
            "Maximum pixels sampled per region",
            _setting_field(
                self.pixels,
                "Controls how densely each reviewed box or freehand area represents its "
                "internal color and texture variation. Higher values cost memory and "
                "time; they do not increase source-image resolution or replace more "
                "independent annotations.",
            ),
        )
        self.mil_iterations = QSpinBox()
        self.mil_iterations.setRange(20, 2_000)
        self.mil_iterations.setValue(160)
        self.mil_iterations.setSingleStep(20)
        parameter_form.addRow(
            "Mixed-region fitting iterations",
            _setting_field(
                self.mil_iterations,
                "Upper limit for resolving mixed labels into phase evidence. Raising it "
                "may help a difficult fit converge but cannot repair incorrect or "
                "unbalanced human labels, and increases fitting time.",
            ),
        )
        self.change_ablation = QCheckBox(
            "Train a control without DarSIA image-change evidence"
        )
        self.change_ablation.setChecked(True)
        parameter_form.addRow(
            "Scientific control",
            _setting_field(
                self.change_ablation,
                "Fits one additional non-deployable control on the same cached pixels. "
                "The complete-run holdout difference measures whether DarSIA's native "
                "change detector adds transferable information. It adds fitting time "
                "but does not alter the main model.",
            ),
        )
        layout.addWidget(parameter_frame)

        self.training_effect_summary = _secondary("")
        layout.addWidget(self.training_effect_summary)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Train new immutable model version")
        self.start_button.clicked.connect(self._start_training)
        actions.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop training now")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_training)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status = _secondary("Add reviewed evidence and enter a researcher name.")
        layout.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        layout.addWidget(self.log)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_history_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(
            _secondary(
                "Every training attempt remains an immutable model version. This "
                "view distinguishes training completion, internal complete-run "
                "holdout results, human validation, and active production use. "
                "Opening or inspecting a model never changes its files."
            )
        )

        location = QFrame()
        location.setFrameShape(QFrame.Shape.StyledPanel)
        location_layout = QFormLayout(location)
        self.model_library = PathField(
            directory=True,
            placeholder="Folder containing immutable phase-model versions",
        )
        configured = str(
            os.environ.get("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY") or ""
        ).strip()
        self.model_library.setText(configured or self.output_parent.text())
        location_layout.addRow(
            "Model library",
            _setting_field(
                self.model_library,
                "The parent folder may contain model versions directly or in "
                "campaign subfolders. Only training requests, summaries, metrics, "
                "and small model files are read.",
            ),
        )
        location_actions = QHBoxLayout()
        refresh = QPushButton("Refresh model history")
        refresh.clicked.connect(self._request_async_model_refresh)
        location_actions.addWidget(refresh)
        use_output = QPushButton("Use new-model location")
        use_output.setToolTip(
            "Show the model versions stored under the output location selected "
            "on the training tab."
        )
        use_output.clicked.connect(self._use_training_output_as_history_root)
        location_actions.addWidget(use_output)
        location_actions.addStretch(1)
        location_layout.addRow("", location_actions)
        self.history_series_filter = self._build_model_series_filter()
        location_layout.addRow("Experimental series", self.history_series_filter)
        layout.addWidget(location)

        self.history_status = _secondary("No model library inspected yet.")
        layout.addWidget(self.history_status)
        self.history_table = QTableWidget(0, 10)
        self.history_table.setHorizontalHeaderLabels(
            [
                "Model version",
                "Created",
                "Lifecycle state",
                "Validation assets",
                "Series",
                "Runs",
                "Regions",
                "CO2 holdout",
                "Gas holdout",
                "Fit iterations",
            ]
        )
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setMinimumHeight(220)
        self.history_table.setMaximumHeight(320)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, self.history_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.itemSelectionChanged.connect(
            self._model_history_selection_changed
        )
        layout.addWidget(self.history_table, 1)

        actions = QHBoxLayout()
        self.open_model_folder = QPushButton("Open selected model folder")
        self.open_model_folder.setEnabled(False)
        self.open_model_folder.clicked.connect(self._open_selected_model_folder)
        actions.addWidget(self.open_model_folder)
        self.use_model_for_validation = QPushButton("Review selected model")
        self.use_model_for_validation.setEnabled(False)
        self.use_model_for_validation.clicked.connect(
            self._use_selected_history_model_for_validation
        )
        actions.addWidget(self.use_model_for_validation)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.model_history_details = QTextEdit()
        self.model_history_details.setReadOnly(True)
        self.model_history_details.setMinimumHeight(260)
        self.model_history_details.setPlaceholderText(
            "Select a model version to inspect training and validation provenance."
        )
        layout.addWidget(self.model_history_details)
        return content

    @staticmethod
    def _metric_text(value: object) -> str:
        return f"{100.0 * float(value):.2f}%" if isinstance(value, (int, float)) else "-"

    @staticmethod
    def _history_created_text(entry: Mapping[str, object]) -> str:
        raw = str(entry.get("created_utc") or "").strip()
        if not raw:
            return "Not recorded"
        try:
            return (
                datetime.fromisoformat(raw.replace("Z", "+00:00"))
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
        except ValueError:
            return raw

    @staticmethod
    def _fit_iteration_text(entry: Mapping[str, object]) -> str:
        rows = [
            row
            for row in entry.get("heads") or []
            if isinstance(row, Mapping) and row.get("adapter") == "generic"
        ]
        if not rows:
            rows = [row for row in entry.get("heads") or [] if isinstance(row, Mapping)]
        values = []
        for row in rows:
            head = str(row.get("head") or "head").replace("_within_co2", "")
            iterations = int(row.get("iterations") or 0)
            suffix = "" if bool(row.get("converged")) else " max"
            values.append(f"{head}: {iterations}{suffix}")
        return "; ".join(values) if values else "-"

    @staticmethod
    def _read_manifest(root: object) -> dict[str, object]:
        path = resolve_review_path(Path(str(root or "")).expanduser()) / "postflight_manifest.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, TypeError, ValueError):
            return {}
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _validation_binding(
        self, entry: Mapping[str, object],
        *, review_sessions: Sequence[Mapping[str, object]] | None = None,
    ) -> dict[str, object] | None:
        """Find precomputed review assets that explicitly reference one model root."""

        if entry.get("library_available") is False:
            # A portable reference cannot fall back to a different source ID or
            # older session when its exact indexed target is absent/stale.
            return None

        model_token = Path(str(entry.get("root") or "")).name.casefold()
        if not model_token:
            return None
        if isinstance(entry, dict):
            entry.pop("validation_collection_error", None)
        reference_readiness = Path(str(entry["root"])) / "validation/phase_reference_readiness.json"
        if reference_readiness.is_file():
            try:
                from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness
                require_reference_readiness(reference_readiness.parent, allow_pending=True)
            except (OSError, TypeError, ValueError) as error:
                if isinstance(entry, dict):
                    entry["validation_collection_error"] = str(error)
                return None
        try:
            from darsia.fluidflower.phase_model_catalog import saved_studio_review_source
            source = saved_studio_review_source(Path(str(entry["root"])))
            if source is not None:
                from darsia.fluidflower.phase_evidence_audit import verified_review_binding
                from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness
                binding = verified_review_binding(Path(str(entry["root"])), source, saved_comparison=True)
                manifest = self._read_manifest(source)
                _validate_saved_phase_binding(Path(str(entry["root"])), binding, manifest)
                require_reference_readiness(source, allow_pending=True)
                return dict(binding)
            collection = self._normal_review_collection(Path(str(entry["root"])))
            if collection is not None:
                if (collection.get("format") != "fluidflower-phase-review-collection-v1"
                        or collection.get("source_model_id") != entry.get("model_id")):
                    raise ValueError("Prepared review collection identifies another selected model")
                return collection
        except (OSError, ValueError, ImportError) as error:
            # A broken matching collection must not quietly reopen the older,
            # smaller experiment list or trigger fresh prediction generation.
            if isinstance(entry, dict):
                entry["validation_collection_error"] = str(error)
            return None
        prepared = Path(str(entry["root"])) / "validation" / "validation_binding.json"
        if prepared.is_file():
            try:
                from darsia.fluidflower.phase_evidence_audit import verified_review_binding
                binding = verified_review_binding(Path(str(entry["root"])), saved_comparison=True)
                manifest = self._read_manifest(binding.get("combined_root"))
                _validate_saved_phase_binding(Path(str(entry["root"])), binding, manifest)
                return dict(binding)
            except (OSError, TypeError, ValueError):
                # Legacy sessions below retain their original current-runtime
                # checks; a changed fitted profile never gains this exception.
                pass
        from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
        try:
            expected_physics = physics_runtime_identity(load_physics_profile(Path(str(entry["root"]))))
        except (OSError, TypeError, ValueError):
            # Model history already exposes the invalid profile. Do not crash
            # the entire library refresh or bind it to old optical-only maps.
            return None
        sessions = list(self.review_sessions if review_sessions is None else review_sessions)
        # A newly prepared evaluation can be opened before its first human session.
        # Discovery must not depend on fabricating an already completed review.
        prepared = Path(str(entry["root"])) / "validation" / "validation_binding.json"
        if prepared.is_file():
            try:
                from darsia.fluidflower.phase_validation import _model_identity

                binding = json.loads(prepared.read_text(encoding="utf-8-sig"))
                _, current_sha = _model_identity(Path(str(entry["root"])))
                if (isinstance(binding, Mapping)
                        and binding.get("output_mode", "review") == "review"
                        and binding.get("model_sha256") == current_sha
                        and binding.get("physics_runtime") == expected_physics):
                    sessions.insert(0, binding)
            except (OSError, TypeError, ValueError):
                pass
        for session in sessions:
            selected: dict[str, str] = {}
            selected_roles: set[str] = set()
            roots_ready = True
            for key in ("ffac_root", "ffum_root"):
                manifest = self._read_manifest(session.get(key))
                if not manifest or manifest.get("output_mode", "review") != "review":
                    roots_ready = False
                    break
                if manifest.get("physics_runtime") != expected_physics:
                    roots_ready = False
                    break
                models = manifest.get("models") or {}
                jobs = manifest.get("jobs") or []
                if not isinstance(models, Mapping) or not isinstance(jobs, list):
                    roots_ready = False
                    break
                matching_roles = []
                for role, raw_model in models.items():
                    if not isinstance(raw_model, Mapping):
                        continue
                    paths = (
                        str(raw_model.get("co2_path") or "").casefold(),
                        str(raw_model.get("gas_path") or "").casefold(),
                    )
                    if paths and all(model_token in path for path in paths):
                        matching_roles.append(str(role))
                if not matching_roles:
                    roots_ready = False
                    break
                matching_roles.sort(
                    key=lambda role: (
                        "physics" not in role.casefold(),
                        "generic" not in role.casefold(),
                        role,
                    )
                )
                selected_role = matching_roles[0]
                selected_roles.add(selected_role)
                for job in jobs:
                    if not isinstance(job, Mapping) or str(job.get("role")) != selected_role:
                        continue
                    run_id = str(job.get("run") or "").strip().casefold()
                    candidate_id = str(job.get("candidate_id") or "").strip()
                    if run_id and candidate_id:
                        selected[run_id] = candidate_id
            combined_root = resolve_review_path(
                Path(str(session.get("combined_root") or "")).expanduser()
            )
            if (
                roots_ready
                and selected
                and (combined_root / "postflight_manifest.json").is_file()
            ):
                return {
                    "campaign_id": str(session.get("campaign_id") or ""),
                    "ffac_root": str(resolve_review_path(session.get("ffac_root") or "")),
                    "ffum_root": str(resolve_review_path(session.get("ffum_root") or "")),
                    "combined_root": str(combined_root.resolve()),
                    "selected_candidates": selected,
                    "roles": sorted(selected_roles),
                    "run_count": len(selected),
                }
        return None

    def _normal_review_collection(self, model_root: Path) -> dict[str, object] | None:
        from darsia.fluidflower.tools.phase_review_collections import discover_collection
        return discover_collection(model_root.expanduser().resolve(), saved_comparison=True)

    def _validation_source_binding(self) -> dict[str, str] | None:
        """Return the newest prepared corrected-frame campaign as source evidence."""

        for session in self.review_sessions:
            roots = {
                key: str(session.get(key) or "").strip()
                for key in ("ffac_root", "ffum_root", "combined_root")
            }
            if all(
                value
                and (Path(value).expanduser() / "postflight_manifest.json").is_file()
                for value in roots.values()
            ):
                return roots
        return None

    @staticmethod
    def _model_version_text(entry: Mapping[str, object]) -> str:
        identifier = str(entry.get("model_id") or "Unnamed model")
        iteration = str(entry.get("iteration") or "").strip()
        return f"{iteration} | {identifier}" if iteration else identifier

    @staticmethod
    def _validation_model_label(entry: Mapping[str, object]) -> str:
        if entry.get("library_available") is False:
            name = str(entry.get("display_name") or entry.get("model_id") or "Phase model")
            return f"{name} · Unavailable library reference · {created_utc_text(entry)}"
        if (entry.get("p1_final_catalog_registered") is True
                or entry.get("p1_final2_catalog_registered") is True
                or entry.get("p1_final3_catalog_registered") is True
                or entry.get("selected_camera_catalog_registered") is True):
            # A recorded name is display metadata, not map admission. The
            # separate readiness state remains pending until the binding passes.
            return str(entry.get("display_name") or entry.get("model_id") or "P1 final")
        if (entry.get("p1_final3_revision") is True
                and entry.get("p1_final3_registered_complete") is not True):
            name = str(entry.get("display_name") or "FF_AC · G3-T4c P1 final 3")
            identity = str(entry.get("model_id") or Path(str(entry.get("root") or "")).name)
            return f"{name} · Incomplete P1-final3 attempt · {created_utc_text(entry)} · {identity}"
        if (entry.get("p1_final2_revision") is True
                and entry.get("p1_final2_registered_complete") is not True):
            name = str(entry.get("display_name") or "FF_AC · G3-T4c P1 final 2")
            identity = str(entry.get("model_id") or Path(str(entry.get("root") or "")).name)
            return (f"{name} · Incomplete P1-final2 attempt · "
                    f"{created_utc_text(entry)} · {identity}")
        if (entry.get("p1_final_revision") is True
                and entry.get("p1_final_registered_complete") is not True):
            name = str(entry.get("display_name") or "FF_AC · G3-T4c P1 final")
            identity = str(entry.get("model_id") or Path(str(entry.get("root") or "")).name)
            return (f"{name} · Incomplete P1-final attempt · "
                    f"{created_utc_text(entry)} · {identity}")
        if friendly := friendly_model_name(entry):
            if (entry.get("p1_final_registered_complete") is True
                    or entry.get("p1_final2_registered_complete") is True
                    or entry.get("p1_final3_registered_complete") is True):
                # The approved specification's title already includes its date
                # and optical scope. Do not append inherited state/name parts.
                return friendly
            raw_created = str(entry.get("created_utc") or "")
            try:
                created = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
                created_text = created.astimezone().strftime("%d.%m.%Y")
            except ValueError:
                created_text = "date not recorded"
            state = ("Unpublished review candidate"
                     if entry.get("state") == "human_validation_required"
                     and not entry.get("publication_scope")
                     else str(entry.get("state_label") or "Unknown state"))
            return f"{friendly} · {created_text} · {state}"
        raw = str(entry.get("training_recorded_utc") or "")
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone().strftime("%d.%m.%Y %H:%M")
            training = f"Training: {stamp}"
        except ValueError:
            training = "Training time not recorded"
        state = str(entry.get("state_label") or "Unknown state")
        warning = "QA warning | " if entry.get("support_review_warning") else ""
        physics = (" · Unified phase decision v1" if entry.get("unified_phase_decision") else
                   " · Variant B completion" if entry.get("aqueous_context_completion") else
                   " · Inherited physical support" if entry.get("gas_head_retraining") and entry.get("physics_runtime") else
                   " · Aqueous edge refinement" if entry.get("native_boundary_refinement") else
                   " · Physical support" if entry.get("physics_runtime") else "")
        series = " + ".join((f"FF_{value[2:].upper()}" if value.startswith("ff") else value.upper())
                            for value in sorted(_model_series(entry)))
        series_label = f" · {series}" if series else ""
        if entry.get("unified_phase_decision"):
            revision = SharedPhaseModelTrainingPage._history_created_text(entry)
            method = entry.get("decision_method_label") or "Unified phase decision v1"
            return (f"{series or 'Phase model'} · Updated: {revision} · {method}"
                    f" · {warning}Review required")
        return f"{training} · {warning}{SharedPhaseModelTrainingPage._model_version_text(entry)}{series_label}{physics} · {state}"

    def _build_model_series_filter(self) -> QComboBox:
        control = QComboBox()
        for value, label in self._model_series_labels.items():
            control.addItem(label, value)
        control.setCurrentIndex(control.findData(self._active_model_series))
        control.setToolTip(
            "Filter by the model's recorded experimental series. Shared models appear "
            "in each series they cover. Models without series metadata remain in All series."
        )
        control.currentIndexChanged.connect(lambda _index: self.set_model_series(str(control.currentData() or "")))
        return control

    def _register_model_series(self, series: str, label: str | None = None) -> None:
        if not series or series in self._model_series_labels:
            return
        self._model_series_labels[series] = label or series.upper()
        for control in (getattr(self, "model_series_filter", None),
                        getattr(self, "history_series_filter", None)):
            if control is not None and control.findData(series) < 0:
                control.addItem(self._model_series_labels[series], series)

    def set_model_series(self, series: str) -> None:
        selected = _canonical_series(series)
        if selected and not re.fullmatch(r"[a-z][a-z0-9]*", selected):
            raise ValueError(f"Invalid model-series filter: {series}")
        self._register_model_series(selected)
        if selected == self._active_model_series:
            return
        old_root = str(self.validation_model.currentData() or "")
        if old_root:
            self._model_selection_by_series[self._active_model_series] = old_root
        self._active_model_series = selected
        if not self.training_series.text().strip():
            self.training_series.setText(selected)
        previous_library = self.model_library.text()
        self._sync_default_model_libraries()
        for control in (self.model_series_filter, self.history_series_filter):
            control.blockSignals(True)
            control.setCurrentIndex(control.findData(selected))
            control.blockSignals(False)
        self._model_filter_settings.setValue("modelHistory/seriesFilter", selected)
        self._rebuild_model_history_view(
            preferred_root=self._model_selection_by_series.get(selected, old_root)
        )
        if self.model_library.text() != previous_library:
            self._request_async_model_refresh()

    def set_saved_model_preference(self, root, *, context_key):
        """Apply one explicit saved series choice; do not override later clicks."""
        selected = str(resolve_review_path(root)) if root else ""
        key = (context_key, selected)
        if key == self._saved_model_preference_key:
            return
        self._saved_model_preference_key = key
        self._saved_model_preference_root = selected
        self._saved_model_preference_pending = bool(selected)
        self._model_catalog_auto_selection_root = ""
        self._model_catalog_priority_root = selected
        if selected:
            self._populate_validation_models(preferred_root=selected)
        # Two series may share one catalogue path. Root-text equality alone
        # cannot authorize old-context metadata/binding callbacks after a switch.
        self._model_catalog_generation += 1
        self._request_async_model_refresh()

    def _catalog_notice(self, message=""):
        notices = ([message] if message else []) + list(self._model_catalog_diagnostics)
        if self._saved_model_preference_pending:
            notices.append(("Loading the saved series model: " if self._model_catalog_inflight
                            else "The saved series model is not in the loaded catalogue: ")
                           + self._saved_model_preference_root)
        return "\n".join(dict.fromkeys(notices))

    def select_validation_model(self, root: Path | str, *, series_context: bool = False) -> bool:
        """Select an explicit model even when a remembered filter hides it."""
        requested = str(resolve_review_path(root)).casefold()
        entry = next((row for row in self.model_history
                      if str(row.get("root") or "").casefold() == requested), None)
        if entry is None:
            return False
        coverage = _model_series(entry)
        if series_context and len(coverage) == 1:
            self.set_model_series(next(iter(coverage)))
        elif self._active_model_series and self._active_model_series not in coverage:
            self.set_model_series("")
        index = self.validation_model.findData(str(entry.get("root") or ""))
        if index < 0:
            return False
        self.validation_model.setCurrentIndex(index)
        return True

    def _populate_validation_models(self, *, preferred_root: str | None = None) -> None:
        selected_root = (str(self.validation_model.currentData() or "")
                         if preferred_root is None else preferred_root)
        entries = _models_for_series(self.model_history, self._active_model_series)
        explicit_pending = self._saved_model_preference_pending
        if explicit_pending:
            selected_root = self._saved_model_preference_root
        if not explicit_pending and not any(str(row.get("root") or "") == selected_root for row in entries):
            preferred = _newest_review_model(entries)
            selected_root = str(preferred.get("root") or "") if preferred else ""
        self.validation_model.blockSignals(True)
        self.validation_model.clear()
        selected_index = -1
        for index, entry in enumerate(entries):
            root = str(entry.get("root") or "")
            self.validation_model.addItem(self._validation_model_label(entry), root)
            self.validation_model.setItemData(index,
                f"Model: {entry.get('model_id')}\n"
                f"Created UTC: {created_utc_text(entry)}\n"
                f"Recorded weight-training time: {entry.get('training_recorded_utc') or 'not recorded'}\n"
                "Recorded training-run time; older models may record "
                "the start or completion of the run. Support revisions use inherited optical training.\n"
                f"Time source: {entry.get('training_time_source') or 'not recorded'}\n"
                f"{entry.get('support_review_warning') or ''}\n{root}",
                Qt.ItemDataRole.ToolTipRole)
            if root == selected_root:
                selected_index = index
        self.validation_model.setCurrentIndex(selected_index)
        if explicit_pending and selected_index >= 0:
            self._saved_model_preference_pending = False
        self.validation_model.blockSignals(False)
        self._validation_model_changed()

    def _selected_validation_model(self) -> Mapping[str, object] | None:
        root = str(self.validation_model.currentData() or "")
        return next(
            (entry for entry in self.model_history if str(entry.get("root") or "") == root),
            None,
        )

    def _validation_model_changed(self, _index: int = -1) -> None:
        self.review_source_form.setRowVisible(0, True)
        self.review_source_form.setRowVisible(1, True)
        entry = self._selected_validation_model()
        self._model_catalog_priority_root = (self._saved_model_preference_root if self._saved_model_preference_pending
                                             else str((entry or {}).get("root") or ""))
        self._load_pending_evidence()
        selected_series = _model_series(entry) if entry else set()
        if len(selected_series) == 1 and hasattr(self, "review_save_root"):
            series = next(iter(selected_series))
            self.review_save_root.blockSignals(True)
            self.review_save_root.setText(str(series_path(series, "reviews")))
            self.review_save_root.blockSignals(False)
        if hasattr(self, "show_training_statistics"):
            self._load_contributions()
        if hasattr(self, "transfer_target") and str((entry or {}).get("root") or "") != self._transfer_sources_for:
            self.transfer_target.blockSignals(True)
            self.transfer_target.clear()
            self.transfer_target.blockSignals(False)
            self._transfer_source_rows = []
            self._pending_checked_runs.clear()
            self.transfer_count.setText("Loading experiment availability…")
            if hasattr(self, "ready_experiments"):
                self.ready_experiments.setRowCount(0)
                self.pending_experiments.setRowCount(0)
            self._transfer_sources_for = ""
        if entry is None:
            self.validation_model_summary.setText(
                "Loading models…" if self._model_catalog_inflight else "No model selected.")
            if hasattr(self, "saved_sessions"):
                self._rebuild_session_combo()
            self._update_action_states()
            return
        if entry.get("validation_binding_loading"):
            self._experiment_discovery_timer.stop()
            self.validation_model_summary.setText(
                f"Selected: {entry.get('model_id')}. Checking this model's saved phase-map binding in the background…")
            self._update_action_states()
            if not self._model_catalog_inflight:
                self._request_async_model_refresh(bindings_only=True)
            return
        if self._experiment_discovery_requested != str(entry["root"]):
            self._experiment_discovery_timer.start(500)
        # Catalog discovery already verified this on its worker. Rehashing the
        # selected NAS collection here would freeze the GUI a second time.
        if "validation_binding" not in entry:
            entry["validation_binding"] = self._validation_binding(entry)
        binding = entry["validation_binding"]
        if entry.get("validation_collection_error"):
            self.validation_model_summary.setText(
                "The prepared experiment collection could not be verified: "
                + str(entry["validation_collection_error"])
            )
        elif binding:
            unified_source = Path(str(binding["ffac_root"])).resolve() == Path(str(binding["ffum_root"])).resolve()
            self.review_source_form.setRowVisible(0, not unified_source)
            self.review_source_form.setRowVisible(1, not unified_source)
            self.ffac_campaign.setText(str(binding["ffac_root"]))
            self.ffum_campaign.setText(str(binding["ffum_root"]))
            self.combined_campaign.setText(str(binding["combined_root"]))
            if binding.get("format") == "fluidflower-phase-review-collection-v1":
                self.validation_model_summary.setText(
                    f"Selected: {entry.get('model_id')}. Prepared phase maps for "
                    f"{binding['run_count']} experiments are available in a new validation session: "
                    + ", ".join(run.upper() for run in binding["selected_candidates"])
                    + ". Each experiment retains its recorded image and physical context."
                )
            else:
                self.validation_model_summary.setText(
                    f"Selected: {entry.get('model_id')}. Prepared phase maps are available "
                    f"for {binding['run_count']} experiment(s) in {binding['campaign_id']}; "
                    f"opening validation explicitly selects {', '.join(binding.get('roles') or ['the selected model'])}."
                )
        else:
            source = self._validation_source_binding()
            if source and not self.combined_campaign.text().strip():
                self.ffac_campaign.setText(source["ffac_root"])
                self.ffum_campaign.setText(source["ffum_root"])
                self.combined_campaign.setText(source["combined_root"])
            self.validation_model_summary.setText(
                f"Selected: {entry.get('model_id')}. No cached phase maps are bound to "
                "this model yet. Starting either review mode will generate the selected "
                "model's predictions, create a new session, and then open it automatically."
                + (
                    " Corrected frames from the newest prepared campaign are selected."
                    if source
                    else " Choose a prepared combined review package first."
                )
            )
        warning = str(entry.get("support_review_warning") or "")
        if entry.get("reference_gate_fitted"):
            gate_status = "enabled" if entry.get("reference_calibration_enabled") else "disabled; C1 retained"
            gate_label = ("Inherited reference gate" if entry.get("gas_head_retraining") or entry.get("full_head_retraining") else
                          "Inherited reference gate with native-resolution edge refinement"
                          if entry.get("native_boundary_refinement") else "Review-fitted baseline-change gate")
            fit_description = ("Both optical classifiers were retrained; CO2 and gas predictions may change."
                               if entry.get("full_head_retraining") else
                               "The gas classifier was retrained; gas predictions may change."
                               if entry.get("gas_head_retraining") else
                               "Optical weights and gas labels are unchanged.")
            self.validation_model_summary.setText(
                self.validation_model_summary.text() +
                f"\n{gate_label}: {gate_status}. {fit_description}"
            )
            gate_evidence = entry.get("reference_gate_evidence") or {}
            if gate_evidence.get("regions") is not None:
                self.validation_model_summary.setText(self.validation_model_summary.text() +
                    f" Gate-development regions: {gate_evidence['regions']} "
                    f"({gate_evidence.get('new_regions', 0)} newly reviewed). Not independent validation.")
        if warning:
            self.validation_model_summary.setText(
                self.validation_model_summary.text() + "\nWARNING: " + warning
            )
        if hasattr(self, "saved_sessions"):
            self._rebuild_session_combo()
        self._update_action_states()

    def _use_training_output_as_history_root(self) -> None:
        self.model_library.setText(self.output_parent.text())
        # The library is scanned and its metadata hashed in the background.
        self._request_async_model_refresh()

    def _sync_completed_model_roots(self) -> None:
        """Read registrations made by other Studio windows and local workers."""
        self._model_filter_settings.sync()
        try:
            saved = json.loads(str(self._model_filter_settings.value(
                "modelHistory/completedModelRoots", "{}")))
        except (TypeError, ValueError):
            return
        if isinstance(saved, dict):
            self._completed_model_roots.update(saved)

    def _refresh_models_on_tab_change(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.evidence_tab:
            self._request_async_model_refresh()
            self._load_pending_evidence(force=True, discover=True)

    @staticmethod
    @metadata_only_catalog
    def _discover_model_entries(root_text: str, completed_roots: tuple[str, ...]) -> list[dict[str, object]]:
        from darsia.fluidflower.phase_model_library import (
            configured_library_records, deduplicate_versions, unavailable_history_entry, attach_library_metadata,
        )
        library_records = configured_library_records()
        root = Path(root_text).expanduser() if root_text else None
        entries = [
            dict(entry) for entry in
            (discover_phase_model_history(root) if root is not None and root.is_dir() else [])
        ]
        existing = {str(entry["root"]): entry for entry in entries}
        roots = deduplicate_versions([
            *(record["root"] for record in library_records if record["available"]),
            *(entry["root"] for entry in entries), *completed_roots,
            *registered_phase_model_defaults().values(),
            *project_phase_model_roots(), *project_result_model_roots(),
        ])
        selected = []
        valid_targets = {os.path.normcase(os.path.normpath(record["root"])) for record in library_records if record["available"]}
        blocked = {os.path.normcase(os.path.normpath(record["target_root"])) for record in library_records
                   if not record["available"] and record.get("target_root")} - valid_targets
        for root in roots:
            if os.path.normcase(os.path.normpath(str(root))) in blocked:
                continue
            if str(root) in existing:
                selected.append(existing[str(root)])
            elif (root / "summary.json").is_file() or (root / "training_request.json").is_file():
                selected.append(inspect_phase_model_history_entry(root))
        unavailable = {entry["root"] for entry in selected if entry.get("library_available") is False}
        selected.extend(unavailable_history_entry(record) for record in library_records
                        if not record["available"] and record["root"] not in unavailable)
        attach_library_metadata(selected, library_records)
        for entry in selected:
            if entry.get("state") == "transfer_evaluation":
                manifest = SharedPhaseModelTrainingPage._read_manifest(Path(str(entry["root"])) / "validation")
                matching = str(manifest.get("model_id") or manifest.get("campaign_id") or "") == str(entry.get("model_id") or "")
                entry["evaluation_datasets"] = sorted({
                    value for job in (manifest.get("jobs") or [])
                    if matching and isinstance(job, Mapping)
                    and (value := _canonical_series(job.get("dataset")))
                })
        diagnostics = (() if registration_context_available() else (REGISTRATION_ROOTS_UNAVAILABLE,))
        return ModelHistoryEntries(selected, diagnostics=diagnostics)

    def _model_popup_opened(self):
        """Repeated openings reuse the visible list instead of queueing rescans."""
        same_root = self._model_catalog_root == self.model_library.text().strip()
        if same_root and (self._model_catalog_inflight or
                monotonic() - getattr(self, "_model_catalog_last_requested", float("-inf")) < 60):
            return
        self._request_async_model_refresh()

    def _request_async_model_refresh(self, *, bindings_only: bool = False, verify_saved: bool = False) -> None:
        """Discover metadata; native saved-map admission needs an explicit review action."""
        metadata_preview = bool(getattr(self, "_native_metadata_preview_only", False)) and not verify_saved
        if bindings_only and metadata_preview:
            return
        root_text = self.model_library.text().strip()
        self._model_catalog_requested_root = root_text
        if self._model_catalog_inflight:
            if bindings_only:
                self._model_binding_refresh_pending = True
            else:
                self._model_catalog_pending = True
            return
        if bindings_only:
            selected = self._selected_validation_model()
            if (not selected or selected.get("validation_binding")
                    or selected.get("validation_collection_error")):
                return
        self._sync_completed_model_roots()
        self._model_catalog_generation += 1
        generation = self._model_catalog_generation
        completed_roots = tuple(self._completed_model_roots.values())
        review_sessions = tuple(dict(row) for row in self.review_sessions)
        self._model_catalog_inflight = True
        self.validation_model.setPlaceholderText("Loading models…")
        if not bindings_only:
            self._model_catalog_last_requested = monotonic()
        self._model_catalog_root = root_text
        self._model_catalog_priority_root = (self._saved_model_preference_root if self._saved_model_preference_pending
                                             else str(self.validation_model.currentData() or ""))
        selected_series = self._active_model_series
        saved_entries = (ModelHistoryEntries((dict(entry) for entry in self.model_history),
                                             diagnostics=self._model_catalog_diagnostics) if bindings_only else None)
        admitted_entries = {str(entry.get("root") or ""): dict(entry)
                            for entry in self.model_history if entry.get("validation_binding")}
        snapshot_roots = tuple(getattr(self, "_configured_series_roots", {}).values()) if not self.model_history else ()
        visible_entries = [dict(entry) for entry in self.model_history]

        registration_error = None
        try:
            registration_available = registration_context_available()
        except Exception as error:  # Kept as the original visible catalogue error, never absence.
            registration_available, registration_error = None, error

        def discover() -> None:
            result = (generation, root_text, [], None)
            def obsolete():
                return (generation != self._model_catalog_generation
                        or root_text != self._model_catalog_requested_root)
            try:
                if obsolete():
                    return
                if registration_error is not None:
                    raise registration_error
                first_entries, priority_verified = visible_entries, {}
                if not bindings_only and snapshot_roots:
                    from darsia.fluidflower.phase_model_catalog import indexed_catalog_snapshot
                    snapshot = indexed_catalog_snapshot(snapshot_roots)
                    if snapshot:
                        if metadata_preview:
                            for entry in snapshot:
                                entry["validation_binding_loading"] = False
                        first_entries = snapshot
                        self._model_metadata_ready.emit((generation, root_text, snapshot, ()))
                if obsolete():
                    return
                if not bindings_only and registration_available and not metadata_preview:
                    preferred_root = self._model_catalog_priority_root
                    if not preferred_root:
                        preferred_root = str((_newest_review_model(
                            _models_for_series(first_entries, selected_series)) or {}).get("root") or "")
                    if (preferred_root and generation == self._model_catalog_generation
                            and root_text == self._model_catalog_requested_root):
                        from darsia.fluidflower.phase_model_catalog import inspect_selected_catalog_entry
                        chosen = next((entry for entry in first_entries if str(entry.get("root")) == preferred_root),
                                      {"root": preferred_root, "model_id": Path(preferred_root).name})
                        try:
                            chosen = inspect_selected_catalog_entry(chosen)
                            chosen.update(validation_binding=None, validation_binding_loading=True)
                            first_entries = [entry for entry in first_entries if str(entry.get("root")) != preferred_root] + [chosen]
                            self._model_metadata_ready.emit((generation, root_text, [dict(entry) for entry in first_entries], ()))
                            chosen["validation_binding"] = self._catalog_validation_binding(chosen, review_sessions=review_sessions)
                        except Exception as error:
                            chosen = dict(chosen, validation_binding=None,
                                          validation_collection_error=f"{type(error).__name__}: {error}")
                        chosen["validation_binding_loading"] = False
                        priority_verified[preferred_root] = chosen
                        self._model_binding_ready.emit((generation, root_text, dict(chosen)))
                if obsolete():
                    return
                entries = saved_entries if bindings_only else self._discover_model_entries(root_text, completed_roots)
                if metadata_preview:
                    for entry in entries:
                        entry["validation_binding"] = None
                        entry["validation_binding_loading"] = False
                        entry["catalog_admission_pending"] = True
                        entry.pop("validation_collection_error", None)
                    self._model_metadata_ready.emit((generation, root_text,
                        [dict(entry) for entry in entries], tuple(getattr(entries, "diagnostics", ()))))
                    result = (generation, root_text, entries, None)
                    return
                pending = []
                for entry in entries:
                    if registration_available is False:
                        # Metadata remains visible and indexed identity unchanged.
                        # No original admission/cache reader may adopt a later
                        # series' global environment on behalf of this context.
                        entry["validation_binding"] = None
                        entry.pop("validation_binding_guard", None)
                        entry["catalog_admission_pending"] = True
                        entry["validation_binding_loading"] = False
                        entry["validation_collection_error"] = REGISTRATION_ROOTS_UNAVAILABLE
                        for key in ("p1_final_registered_complete", "p1_final2_registered_complete", "p1_final3_registered_complete"):
                            if key in entry:
                                entry[key] = False
                        continue
                    first = priority_verified.get(str(entry.get("root") or ""))
                    if (first is not None and entry.get("library_available") is not False
                            and entry.get("library_source_identity") == first.get("library_source_identity")
                            and not entry.get("errors")
                            and (not first.get("validation_binding_guard")
                                 or self._saved_binding_guard_matches(first["validation_binding_guard"]))):
                        for name in ("validation_binding", "validation_binding_guard", "validation_collection_error"):
                            if name in first:
                                entry[name] = first[name]
                        entry["validation_binding_loading"] = False
                        continue
                    old = admitted_entries.get(str(entry.get("root") or ""), {})
                    guard = old.get("validation_binding_guard")
                    if (not bindings_only and guard and not entry.get("errors")
                            and entry.get("library_available") is not False
                            and self._saved_binding_guard_matches(guard)):
                        # Opening the dropdown must not discard an already
                        # admitted immutable candidate. Its complete dependency
                        # stamps, including P1's parent, still match exactly.
                        for name in ("validation_binding", "validation_binding_guard",
                                     "catalog_admission_pending", "p1_final_registered_complete", "p1_final2_registered_complete", "p1_final3_registered_complete",
                                     "state", "state_label"):
                            if name in old:
                                entry[name] = old[name]
                        entry["validation_binding_loading"] = False
                        continue
                    if bindings_only and (entry.get("validation_binding")
                                          or entry.get("validation_collection_error")):
                        continue
                    entry.setdefault("validation_binding", None)
                    entry["validation_binding_loading"] = True
                    pending.append(entry)
                # Publish lightweight choices first. Scientific saved-map
                # admission is needed only for the selected model.
                self._model_metadata_ready.emit(
                    (generation, root_text, [dict(entry) for entry in entries], tuple(getattr(entries, "diagnostics", ())))
                )
                for entry in entries:
                    if str(entry.get("root") or "") in priority_verified and not entry.get("validation_binding_loading"):
                        self._model_binding_ready.emit((generation, root_text, dict(entry)))
                ordered = _models_for_series(pending, selected_series)
                priority_roots = {str(entry.get("root") or "") for entry in ordered}
                ordered += [entry for entry in pending
                            if str(entry.get("root") or "") not in priority_roots]
                while ordered:
                    if (generation != self._model_catalog_generation
                            or root_text != self._model_catalog_requested_root):
                        # The native wrapper sets the host project just after
                        # construction. Do not admit an obsolete startup model
                        # before beginning that project's pending catalog.
                        break
                    preferred_root = self._model_catalog_priority_root
                    preferred = next((entry for entry in ordered
                                      if str(entry.get("root") or "") == preferred_root), None)
                    if preferred is None:
                        if not preferred_root and not bindings_only:
                            preferred = ordered[0]
                        else:
                            break
                    ordered.remove(preferred)
                    entry = preferred
                    entry["validation_binding"] = self._catalog_validation_binding(
                        entry, review_sessions=review_sessions
                    )
                    entry["validation_binding_loading"] = False
                    self._model_binding_ready.emit((generation, root_text, dict(entry)))
                result = (generation, root_text, entries, None)
            except Exception as exc:  # noqa: BLE001 - surfaced on the UI thread
                result = (generation, root_text, None, f"{type(exc).__name__}: {exc}")
            finally:
                try:
                    self._model_catalog_ready.emit(result)
                except RuntimeError:
                    # The page may have closed while network discovery was running.
                    pass

        catalogue_context = copy_context()
        Thread(target=lambda: catalogue_context.run(
            run_with_registration_context, discover, available=registration_available
        ), name="phase-model-catalog", daemon=True).start()

    def _apply_async_model_metadata(self, result: object) -> None:
        generation, root_text, entries = result[:3]
        diagnostics = tuple(result[3]) if len(result) > 3 else tuple(getattr(entries, "diagnostics", ()))
        if (generation != self._model_catalog_generation
                or root_text != self.model_library.text().strip()):
            return
        self._model_catalog_metadata_generation = generation
        self._model_catalog_diagnostics = diagnostics
        first_selection = not self.validation_model.currentData()
        self._install_model_history(entries)
        self._select_completed_model()
        if first_selection and not self._saved_model_preference_root:
            self._model_catalog_auto_selection_root = str(self.validation_model.currentData() or "")
        if (self._selected_validation_model() or {}).get("validation_binding_loading") and not self.catalog_loading_status.text().startswith("Could not load"):
            self.catalog_loading_status.setText(
                self._catalog_notice("Model versions loaded; checking the selected model's saved phase maps in the background…")
            )
            self.catalog_loading_status.show()

    def _note_model_catalog_selection(self, _index: int = -1) -> None:
        """An explicit choice must never follow background binding results."""
        self._model_catalog_auto_selection_root = ""
        self._saved_model_preference_pending = False
        # Nor is it replaced by a just-completed model listed later.
        self._completed_model_root = None

    def _apply_async_model_binding(self, result: object) -> None:
        generation, root_text, verified = result
        if (generation != self._model_catalog_generation
                or root_text != self.model_library.text().strip()):
            return
        entry = next((entry for entry in self.model_history
                      if entry.get("root") == verified.get("root")), None)
        if entry is None:
            return
        entry["validation_binding"] = verified.get("validation_binding")
        entry["validation_binding_loading"] = False
        entry.pop("validation_binding_guard", None)
        if verified.get("validation_binding_guard"):
            entry["validation_binding_guard"] = verified["validation_binding_guard"]
        if entry.get("validation_binding") and entry.get("catalog_admission_pending"):
            entry["catalog_admission_pending"] = False
            if entry.get("state") == "catalog_verification_pending":
                entry["state"] = "human_validation_required"
                entry["state_label"] = "Saved maps ready · Review required"
        if entry.get("validation_binding") and entry.get("p1_final3_catalog_registered") is True:
            entry["p1_final3_registered_complete"] = True
            entry["catalog_admission_pending"] = False
            entry["state"] = "human_validation_required"
            from darsia.fluidflower.phase_saved_contact_review import STATE_LABEL
            entry["state_label"] = STATE_LABEL
        if entry.get("validation_binding") and entry.get("p1_final2_catalog_registered") is True:
            entry["p1_final2_registered_complete"] = True
            entry["catalog_admission_pending"] = False
            entry["state"] = "human_validation_required"
            from darsia.fluidflower.phase_saved_categorical_review import STATE_LABEL
            entry["state_label"] = STATE_LABEL
        if entry.get("validation_binding") and entry.get("p1_final_catalog_registered") is True:
            # _validation_binding validated the exact saved pipeline before
            # returning. This never promotes mass/deployment eligibility.
            entry["p1_final_registered_complete"] = True
            entry["catalog_admission_pending"] = False
            entry["state"] = "human_validation_required"
            from darsia.fluidflower.phase_saved_boundary_review import STATE_LABEL
            entry["state_label"] = STATE_LABEL
        entry.pop("validation_collection_error", None)
        if verified.get("validation_collection_error"):
            entry["validation_collection_error"] = verified["validation_collection_error"]
        for row, visible in enumerate(self.visible_model_history):
            if visible.get("root") == entry.get("root"):
                binding = entry.get("validation_binding")
                text = (f"Ready ({binding['run_count']} runs)" if binding
                        else "Review collection unavailable" if entry.get("validation_collection_error")
                        else "Predictions not generated")
                self.history_table.setItem(row, 3, QTableWidgetItem(text))
                break
        selected = self._selected_validation_model()
        if (self._model_catalog_auto_selection_root
                and str((selected or {}).get("root") or "") == self._model_catalog_auto_selection_root
                and not selected.get("validation_binding_loading")
                and not selected.get("validation_binding")):
            ready = _newest_review_model(_models_for_series(self.model_history, self._active_model_series))
            if ready and ready.get("validation_binding"):
                self._populate_validation_models(preferred_root=str(ready["root"]))
                self._model_catalog_auto_selection_root = ""
        if str(self.validation_model.currentData() or "") == str(entry.get("root") or ""):
            self._validation_model_changed()
            if (entry.get("validation_binding")
                    and not self.catalog_loading_status.text().startswith("Could not load")):
                self.catalog_loading_status.setText(
                    self._catalog_notice("Selected model's saved phase maps are ready. Other versions are checked when selected.")
                )

    def _apply_async_model_catalog(self, result: object) -> None:
        generation, root_text, entries, error = result
        self._model_catalog_inflight = False
        changed_root = root_text != self.model_library.text().strip()
        if generation == self._model_catalog_generation and not changed_root:
            if error is None:
                self._model_catalog_diagnostics = tuple(getattr(entries, "diagnostics", ()))
                if self._model_catalog_metadata_generation != generation:
                    self._install_model_history(entries)
                selected = self._selected_validation_model() or {}
                if selected.get("validation_binding_loading"):
                    self.catalog_loading_status.setText(self._catalog_notice("Checking the selected model's saved phase-map binding…"))
                    self.catalog_loading_status.show()
                elif selected.get("validation_collection_error"):
                    self.catalog_loading_status.setText(self._catalog_notice("Could not verify the selected model's saved phase maps: "
                                                        + str(selected["validation_collection_error"])))
                    self.catalog_loading_status.show()
                elif self._catalog_notice():
                    self.catalog_loading_status.setText(self._catalog_notice())
                    self.catalog_loading_status.show()
                elif not self.catalog_loading_status.text().startswith("Could not load"):
                    self.catalog_loading_status.setText("")
                    self.catalog_loading_status.hide()
                self.validation_model.setPlaceholderText(
                    "Select a model" if self.validation_model.count() else "No models found")
            else:
                self.validation_model.setPlaceholderText("Models unavailable")
                self.history_status.setText(f"Could not inspect model history: {error}")
                self.catalog_loading_status.setText(
                    f"Could not inspect model history: {error}"
                )
                self.catalog_loading_status.show()
        if self._model_catalog_pending or changed_root:
            self._model_catalog_pending = False
            self._model_binding_refresh_pending = False
            self._request_async_model_refresh()
        elif self._model_binding_refresh_pending:
            self._model_binding_refresh_pending = False
            self._request_async_model_refresh(bindings_only=True)
        elif (self._selected_validation_model() or {}).get("validation_binding_loading"):
            # Selection can change just after the worker checked its priority.
            self._request_async_model_refresh(bindings_only=True)

    def _refresh_model_history(self) -> None:
        self._model_catalog_generation += 1
        previous = _models_for_series(self.model_history, self._active_model_series)
        previous_newest_root = str((_newest_review_model(previous) or {}).get("root") or "")
        selected_root = str(self.validation_model.currentData() or "")
        follow_newest = bool(previous_newest_root and selected_root == previous_newest_root)
        self._sync_completed_model_roots()
        try:
            entries = self._discover_model_entries(
                self.model_library.text().strip(), tuple(self._completed_model_roots.values()))
        except Exception as exc:  # noqa: BLE001 - visible read-only UI boundary
            message = f"Could not inspect model history: {type(exc).__name__}: {exc}"
            self.history_status.setText(message)
            self.catalog_loading_status.setText(message)
            self.catalog_loading_status.show()
            return
        self._model_catalog_diagnostics = tuple(getattr(entries, "diagnostics", ()))
        self._install_model_history(entries, follow_newest=follow_newest,
                                    selected_root=selected_root)
        if self._catalog_notice():
            self.catalog_loading_status.setText(self._catalog_notice())
            self.catalog_loading_status.show()

    def _catalog_validation_binding(self, entry, *, review_sessions=None):
        """A failed saved-review binding belongs to one model, not the whole list."""
        try:
            cached = self._load_p1_binding_cache(entry)
            if cached is not None:
                entry["validation_binding_guard"] = cached["guard"]
                return dict(cached["binding"])
            binding = self._validation_binding(entry, review_sessions=review_sessions)
            if binding and entry.get("p1_final3_catalog_registered"):
                entry["validation_binding_guard"] = self._p1_final3_binding_guard(entry, binding)
                self._save_p1_binding_cache(entry, binding, entry["validation_binding_guard"])
            elif binding and entry.get("p1_final2_catalog_registered"):
                entry["validation_binding_guard"] = self._p1_final2_binding_guard(entry, binding)
                self._save_p1_binding_cache(entry, binding, entry["validation_binding_guard"])
            elif binding and entry.get("p1_final_catalog_registered"):
                entry["validation_binding_guard"] = self._p1_final_binding_guard(entry, binding)
                self._save_p1_binding_cache(entry, binding, entry["validation_binding_guard"])
            return binding
        except Exception as exc:  # noqa: BLE001 - retain the model and expose its exact error
            entry["validation_collection_error"] = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _p1_binding_cache_path(entry):
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
        if not base:
            return None
        root = str(resolve_review_path(str(entry["root"])))
        key = hashlib.sha256(os.path.normcase(os.path.normpath(root)).encode("utf-8")).hexdigest()
        return Path(base) / "phase-studio-bindings-v1" / (key + ".json")

    @staticmethod
    def _load_p1_binding_cache(entry):
        """Reuse only a complete, unchanged, previously admitted P1 candidate."""
        if (not (entry.get("p1_final_catalog_registered") or entry.get("p1_final2_catalog_registered") or entry.get("p1_final3_catalog_registered")) or entry.get("errors")
                or entry.get("library_available") is False):
            return None
        try:
            path = SharedPhaseModelTrainingPage._p1_binding_cache_path(entry)
            if path is None or path.stat().st_size > 2 * 1024 * 1024:
                return None
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                return None
            guard, binding = saved["guard"], saved["binding"]
            if not isinstance(guard, dict) or not isinstance(binding, dict):
                return None
            root = str(resolve_review_path(str(entry["root"])))
            final3 = bool(entry.get("p1_final3_catalog_registered"))
            final2 = bool(entry.get("p1_final2_catalog_registered"))
            expected_format = ("fluidflower-p1-final3-studio-binding-cache-v1" if final3
                               else "fluidflower-p1-final2-studio-binding-cache-v1" if final2
                               else "fluidflower-p1-final-studio-binding-cache-v1")
            if (saved.get("format") != expected_format
                    or saved.get("model_root") != root
                    or saved.get("model_id") != entry.get("model_id")
                    or not isinstance(binding, dict)
                    or saved.get("model_identity") != guard.get("model_identity")
                    or guard.get("model_identity", [None])[0] != entry.get("model_id")
                    or len(guard.get("model_versions") or []) != (4 if final3 else 3 if final2 else 2)
                    or not guard.get("pipeline_sha256")
                    or not SharedPhaseModelTrainingPage._saved_binding_guard_matches(guard)):
                return None
            bound_id = binding.get("model_id") or binding.get("source_model_id")
            if bound_id != entry.get("model_id"):
                return None
            return saved
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            # A missing, stale or damaged cache never grants admission.
            return None

    @staticmethod
    def _save_p1_binding_cache(entry, binding, guard):
        """A local performance cache; failure never changes saved admission."""
        temporary = None
        try:
            final3 = bool(entry.get("p1_final3_catalog_registered"))
            final2 = bool(entry.get("p1_final2_catalog_registered"))
            if (len(guard.get("model_versions") or []) != (4 if final3 else 3 if final2 else 2)
                    or guard.get("model_identity", [None])[0] != entry.get("model_id")):
                return
            path = SharedPhaseModelTrainingPage._p1_binding_cache_path(entry)
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            cache_format = ("fluidflower-p1-final3-studio-binding-cache-v1" if final3
                               else "fluidflower-p1-final2-studio-binding-cache-v1" if final2
                            else "fluidflower-p1-final-studio-binding-cache-v1")
            saved = dict(format=cache_format,
                         model_root=str(resolve_review_path(str(entry["root"]))),
                         model_id=entry["model_id"], model_identity=guard["model_identity"],
                         binding=dict(binding), guard=guard)
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(saved, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _saved_binding_guard_matches(guard):
        """A changed or missing admitted dependency always requires admission."""
        try:
            from darsia.fluidflower.phase_model_library import source_identity
            if any(source_identity(row["root"]) != row["identity"] for row in guard.get("model_versions", [])):
                return False
            return bool(guard.get("assets")) and all(
                tuple(row["stamp"]) == SharedPhaseModelTrainingPage._binding_file_stamp(Path(row["path"]))
                for row in guard["assets"])
        except (OSError, KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _binding_file_stamp(path):
        from darsia.fluidflower.phase_saved_review_metadata import metadata_file_stamp
        return metadata_file_stamp(path)

    @staticmethod
    def _p1_final3_binding_guard(entry, binding):
        """Include the new runtime and the complete already-checked P1 ancestry."""
        from darsia.fluidflower import phase_saved_contact_review as final
        from darsia.fluidflower.phase_saved_registration import registration_path
        from darsia.fluidflower.phase_model_library import source_identity
        root = final._resolved_path(str(entry["root"]))
        admitted = final.p1_final3_saved_review(root)
        plan, assets, heads = final._saved_dependencies(str(root), final._signature(root / final.PLAN))
        parent = final._resolved_path(plan["binding"]["parent"])
        parent_binding = final._read(parent / "validation/validation_binding.json")
        inherited = SharedPhaseModelTrainingPage._p1_final2_binding_guard(dict(root=str(parent)), parent_binding)
        paths = {Path(row["path"]) for row in inherited["assets"]}
        paths.update(root / name for name in final.FILES)
        paths.add(final._resolved_path(str(registration_path(root.name))))
        paths.update(final._bound(asset, base=root) for asset in (*assets, *heads))
        profile = final._read(root / "phase_rig_camera.json")
        runtime_path = final._bound(profile["rig_domain_revision"]["manifest"], base=root)
        paths.add(runtime_path)
        for source in final._read(runtime_path)["sources"]:
            paths.update(final._bound(source[name], base=root) for name in ("input_manifest", "native_manifest"))
        paths.update(Path(binding["combined_root"]) / name for name in ("postflight_manifest.json", "validation_binding.json"))
        rows = [dict(path=str(path), stamp=SharedPhaseModelTrainingPage._binding_file_stamp(path))
                for path in sorted(paths, key=str)]
        versions = [dict(root=str(root), identity=source_identity(root)), *inherited["model_versions"]]
        return dict(model_identity=admitted["model_identity"], pipeline_sha256=admitted["pipeline_sha256"],
                    model_versions=versions, assets=rows)

    @staticmethod
    def _p1_final2_binding_guard(entry, binding):
        """Include the new runtime and the complete already-checked P1 ancestry."""
        from darsia.fluidflower import phase_saved_categorical_review as final
        from darsia.fluidflower.phase_saved_registration import registration_path
        from darsia.fluidflower.phase_model_library import source_identity
        root = final._resolved_path(str(entry["root"]))
        admitted = final.p1_final2_saved_review(root)
        plan, assets, heads = final._saved_dependencies(str(root), final._signature(root / final.PLAN))
        parent = final._resolved_path(plan["binding"]["parent"])
        parent_binding = final._read(parent / "validation/validation_binding.json")
        inherited = SharedPhaseModelTrainingPage._p1_final_binding_guard(dict(root=str(parent)), parent_binding)
        paths = {Path(row["path"]) for row in inherited["assets"]}
        paths.update(root / name for name in final.FILES)
        paths.add(final._resolved_path(str(registration_path(root.name))))
        paths.update(final._bound(asset, base=root) for asset in (*assets, *heads))
        profile = final._read(root / "phase_rig_camera.json")
        runtime_path = final._bound(profile["rig_domain_revision"]["manifest"], base=root)
        paths.add(runtime_path)
        for source in final._read(runtime_path)["sources"]:
            paths.update(final._bound(source[name], base=root) for name in ("input_manifest", "native_manifest"))
        paths.update(Path(binding["combined_root"]) / name for name in ("postflight_manifest.json", "validation_binding.json"))
        rows = [dict(path=str(path), stamp=SharedPhaseModelTrainingPage._binding_file_stamp(path))
                for path in sorted(paths, key=str)]
        versions = [dict(root=str(root), identity=source_identity(root)), *inherited["model_versions"]]
        return dict(model_identity=admitted["model_identity"], pipeline_sha256=admitted["pipeline_sha256"],
                    model_versions=versions, assets=rows)

    @staticmethod
    def _p1_final_binding_guard(entry, binding):
        """Retain exact dependency signatures after strict P1-final admission.

        The parent plan dependencies are already cached by that admission. No
        large plan is parsed to enumerate a model dropdown, and changed bytes
        must pass the existing SHA checks before a new guard can be recorded.
        """
        from darsia.fluidflower import phase_saved_boundary_review as final
        from darsia.fluidflower import phase_saved_floor_review as parent_review
        from darsia.fluidflower.phase_saved_review_metadata import configured_parameters
        parent_plan_name = configured_parameters("floor")["plan"]
        root = Path(str(entry["root"]))
        admitted = final.p1_final_saved_review(root)
        plan, assets, heads = final._saved_dependencies(str(root), final._signature(root / final.PLAN))
        paths = {final._resolved_path(str(Path(value))) for value in
                 [root / name for name in final.FILES]}
        from darsia.fluidflower.phase_saved_registration import registration_path
        paths.add(final._resolved_path(str(registration_path(root.name))))
        paths.update(final._bound(value, base=root) for value in (*assets, *heads))
        parent = final._resolved_path(plan["binding"]["parent"])
        registration = final._bound(plan["binding"]["inputs"]["parent_registration"], base=root)
        parent_metadata = json.loads(registration.read_text(encoding="utf-8-sig"))
        paths.update(parent / name for name in parent_metadata["metadata_sha256"])
        dependencies = parent_review._plan_dependencies(str(parent), parent_review._signature(parent / parent_plan_name),
                                                        parent_metadata["source_plan_sha256"])
        from darsia.fluidflower.phase_saved_review_metadata import bound_path
        for path, sha in dependencies:
            # Strict parent admission has already SHA-verified these assets.
            # Record their current stamps without a second large-file hash.
            paths.add(bound_path(dict(path=path, sha256=sha), base=parent, verify=False))
        for folder in (root, parent):
            paths.update(folder.glob("phase_*.json"))
            paths.update((folder / "models").glob("*/*_signal.json"))
        paths.add(Path(binding["combined_root"]) / "postflight_manifest.json")
        rows = [dict(path=str(path), stamp=SharedPhaseModelTrainingPage._binding_file_stamp(path))
                for path in sorted(paths, key=str)]
        from darsia.fluidflower.phase_model_library import source_identity
        versions = [dict(root=str(folder), identity=source_identity(folder)) for folder in (root, parent)]
        return dict(model_identity=admitted["model_identity"], pipeline_sha256=admitted["pipeline_sha256"],
                    model_versions=versions, assets=rows)

    def _install_model_history(
        self, entries: list[dict[str, object]], *,
        follow_newest: bool | None = None, selected_root: str | None = None,
    ) -> None:
        for entry in entries:
            for series in _model_series(entry):
                self._register_model_series(series)
        if follow_newest is None:
            previous = _models_for_series(self.model_history, self._active_model_series)
            previous_newest_root = str((_newest_review_model(previous) or {}).get("root") or "")
            selected_root = str(self.validation_model.currentData() or "")
            follow_newest = bool(previous_newest_root and selected_root == previous_newest_root)
        self.history_table.setRowCount(0)
        self.model_history_details.clear()
        self.open_model_folder.setEnabled(False)
        self.use_model_for_validation.setEnabled(False)
        self.model_history = entries

        binding_refresh_needed = False
        for entry in self.model_history:
            if "validation_binding" not in entry:
                entry["validation_binding"] = None
                entry["validation_binding_loading"] = True
                binding_refresh_needed = True
        newest = _models_for_series(self.model_history, self._active_model_series)
        preferred_root = (str((_newest_review_model(newest) or {}).get("root") or "") if newest and follow_newest
                          else selected_root or "")
        self._rebuild_model_history_view(preferred_root=preferred_root)
        if binding_refresh_needed:
            self._request_async_model_refresh(bindings_only=True)

    def _rebuild_model_history_view(self, *, preferred_root: str | None = None) -> None:
        previous_history_root = str((self._selected_model_history() or {}).get("root") or "")
        self.visible_model_history = _models_for_series(self.model_history, self._active_model_series)
        self.history_table.blockSignals(True)
        self.history_table.setRowCount(len(self.visible_model_history))
        selected_row = 0
        for row_index, entry in enumerate(self.visible_model_history):
            if str(entry.get("root") or "") == previous_history_root:
                selected_row = row_index
            datasets = ", ".join(self._model_series_labels.get(value, value.upper())
                                 for value in sorted(_model_series(entry)))
            values = (
                self._model_version_text(entry),
                self._history_created_text(entry),
                str(entry.get("state_label") or ""),
                (
                    "Checking saved phase maps…" if entry.get("validation_binding_loading")
                    else f"Ready ({entry['validation_binding']['run_count']} runs)"
                    if entry.get("validation_binding")
                    else "Review collection unavailable" if entry.get("validation_collection_error")
                    else "Predictions not generated"
                ),
                datasets or "-",
                str(len(entry.get("runs") or [])),
                f"{int(entry.get('region_count') or 0):,}",
                self._metric_text(entry.get("co2_balanced_accuracy")),
                self._metric_text(entry.get("gas_balanced_accuracy")),
                self._fit_iteration_text(entry),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(entry.get("root") or ""))
                self.history_table.setItem(row_index, column, item)
        self.history_status.setText(
            f"{len(self.visible_model_history)} of {len(self.model_history)} model version(s) · "
            f"{self._model_series_labels[self._active_model_series]} · newest first. "
            f"Model library: {self.model_library.text().strip()}."
        )
        if self.visible_model_history:
            self.history_table.selectRow(selected_row)
        self.history_table.blockSignals(False)
        self._model_history_selection_changed()
        self._populate_validation_models(preferred_root=preferred_root)
        if self._phase_comparison_panel is not None:
            reset = self._phase_comparison_series != self._active_model_series
            self._phase_comparison_panel.set_entries(self._comparison_entries(), reset=reset)
            self._phase_comparison_series = self._active_model_series

    def _selected_model_history(self) -> Mapping[str, object] | None:
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return self.visible_model_history[index] if 0 <= index < len(self.visible_model_history) else None

    def _model_history_selection_changed(self) -> None:
        entry = self._selected_model_history()
        self.open_model_folder.setEnabled(entry is not None)
        self.use_model_for_validation.setEnabled(entry is not None)
        if entry is None:
            self.model_history_details.clear()
            return
        lines = [
            f"Model version: {entry.get('model_id')}",
            f"Location: {entry.get('root')}",
            f"Created: {self._history_created_text(entry)}",
            f"Created by: {entry.get('created_by')}",
            f"Lifecycle: {entry.get('state_label')}",
            *([f"WARNING: {entry['support_review_warning']}"]
              if entry.get("support_review_warning") else []),
            f"Internal complete-run holdout passed: {'yes' if entry.get('internal_holdout_passed') else 'no'}",
            f"Human validation: {'recorded' if entry.get('human_validation') else 'not yet linked'}",
            f"Active production deployment: {'yes' if entry.get('deployment') else 'no'}",
            (
                "Validation assets: available"
                if entry.get("validation_binding")
                else "Validation assets: phase-map predictions have not been generated"
            ),
            f"Series: {', '.join(self._model_series_labels.get(value, value.upper()) for value in sorted(_model_series(entry))) or '-'}",
            f"Model scope: {str(entry['series_scope']).upper()} only" if entry.get('series_scope')
            else "Model scope: historical/shared model; no explicit series restriction",
            f"Runs ({len(entry.get('runs') or [])}): {', '.join(str(value) for value in entry.get('runs') or []) or '-'}",
            f"Reviewed regions: {int(entry.get('region_count') or 0):,}",
            f"Sampled training pixels: {int(entry.get('pixel_rows') or 0):,}",
            f"Adapters: {', '.join(str(value) for value in entry.get('adapters') or []) or '-'}",
            f"DarSIA change evidence: {entry.get('change_detector')}",
            f"Phase interpretation: {entry.get('physics_label', 'Optical only')}",
            f"Registered physical contexts: {', '.join(entry.get('physics_runs') or []) or 'None'}",
            f"Physics inheritance receipt: {entry.get('physics_inheritance_receipt') or 'Not recorded'}",
            f"Training-image normalization: {entry.get('radiometry_label', 'Legacy / untracked')}",
            f"Correction index: {Path(str(entry.get('root'))) / 'radiometry_inventory.json'}",
            "",
            "Head validation and fit history:",
        ]
        if entry.get("evaluation_parent"):
            parent = entry["evaluation_parent"]
            lines[5:5] = [
                f"Evaluation: {entry.get('evaluation_scope')}",
                f"Unchanged source model: {parent.get('model_root')}",
                "Training counts and holdout scores below belong to the source model, not this evaluation.",
            ]
        for row in entry.get("heads") or []:
            if not isinstance(row, Mapping):
                continue
            accuracy = self._metric_text(row.get("balanced_accuracy"))
            convergence = "converged" if row.get("converged") else "iteration limit reached"
            lines.append(
                f"- {row.get('adapter')} / {row.get('head')}: holdout {accuracy}; "
                f"threshold {row.get('threshold')}; {int(row.get('iterations') or 0)} "
                f"iterations ({convergence}); {row.get('backend') or '-'} "
                f"{row.get('precision') or ''}; internal eligibility "
                f"{'passed' if row.get('deployment_allowed') else 'not passed/control'}"
            )
        delta = entry.get("change_ablation_delta")
        if isinstance(delta, (int, float)):
            lines.extend(
                [
                    "",
                    "DarSIA change-evidence ablation:",
                    f"- Mean balanced-accuracy difference: {float(delta):+.4f}",
                ]
            )
        phase_classes = entry.get("phase_classes") or {}
        if isinstance(phase_classes, Mapping) and phase_classes:
            lines.extend(["", "Training labels by canonical class:"])
            lines.extend(
                f"- {name}: {int(count):,}"
                for name, count in sorted(phase_classes.items())
            )
        bundles = entry.get("training_bundles") or []
        lines.extend(["", f"Frozen evidence packages: {len(bundles)}"])
        gate_evidence = entry.get("reference_gate_evidence") or {}
        if entry.get("reference_gate_fitted") and gate_evidence:
            lines.extend(["", "Separate reference-gate development (optical weights unchanged):",
                f"- Regions: {gate_evidence.get('regions', 'Unavailable')}; newly reviewed: {gate_evidence.get('new_regions', 0)}",
                f"- Evidence: {gate_evidence.get('path', '')}"])
        errors = entry.get("errors") or []
        if errors:
            lines.extend(["", "Inspection findings:"])
            lines.extend(f"- {value}" for value in errors)
        self.model_history_details.setPlainText("\n".join(lines))

    def _open_selected_model_folder(self) -> None:
        entry = self._selected_model_history()
        if entry is None:
            return
        root = str(entry.get("root") or "").strip()
        if not root:
            QMessageBox.warning(
                self, "Model folder unavailable",
                "The selected model has no recorded folder. Refresh the model list.",
            )
            return
        path = Path(root).expanduser()
        try:
            if not path.is_dir():
                QMessageBox.warning(
                    self, "Model folder unavailable",
                    f"The selected model folder is missing or unavailable:\n{path}\n"
                    "Check its storage connection and refresh the model list.",
                )
                return
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
                QMessageBox.warning(
                    self, "Could not open model folder",
                    f"The desktop could not open the selected model folder:\n{path}",
                )
        except OSError as error:
            QMessageBox.warning(
                self, "Model folder unavailable",
                f"Could not access the selected model folder:\n{path}\n"
                f"{type(error).__name__}: {error}",
            )

    def _use_selected_history_model_for_validation(self) -> None:
        entry = self._selected_model_history()
        if entry is None:
            return
        index = self.validation_model.findData(str(entry.get("root") or ""))
        if index >= 0:
            self.validation_model.setCurrentIndex(index)
            self.tabs.setCurrentWidget(self.evidence_tab)
            self.evidence_scroll.verticalScrollBar().setValue(0)

    def _selected_session(self) -> dict[str, object] | None:
        index = self.saved_sessions.currentIndex()
        if 0 <= index < len(self.visible_review_sessions):
            return self.visible_review_sessions[index]
        return None

    def _review_session_id_for(self, model_id: object) -> str:
        """Keep correcting in this model's own review session until that session is locked.

        A session is locked the moment its corrections are used as training input, so the lock is
        what makes "which corrections did this model see" answerable afterwards. A locked session
        is therefore never reused and never written to again: a new one is minted instead.
        """
        model_id = str(model_id or "").strip()
        editable = [
            session for session in (getattr(self, "review_sessions", None) or [])
            if isinstance(session, Mapping)
            and str(session.get("review_session_id") or "")
            and not session.get("editing_locked")
            and model_id
            and str(session.get("target_model_id") or "").strip() == model_id
        ]
        if editable:
            newest = max(editable, key=lambda row: float(row.get("modified_timestamp") or 0.0))
            return str(newest["review_session_id"])
        return (f"review-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid.uuid4().hex[:8]}")

    def _session_matches_selected_model(
        self, session: Mapping[str, object]
    ) -> bool:
        model = self._selected_validation_model()
        if model is None:
            return False
        model_id = str(model.get("model_id") or "").strip()
        explicit = str(session.get("target_model_id") or "").strip()
        if (session.get("evidence_role") == "model_transfer_evaluation"
                and session.get("source_model_id") == model_id):
            return True
        if explicit:
            return explicit == model_id
        binding = model.get("validation_binding")
        if not isinstance(binding, Mapping):
            return False
        expected = dict(binding.get("selected_candidates") or {})
        actual = dict(session.get("selected_candidates") or {})
        overlap = set(expected).intersection(actual)
        return bool(overlap) and all(expected[run] == actual[run] for run in overlap)

    @staticmethod
    def _session_label(session: Mapping[str, object]) -> str:
        modified = (
            datetime.fromtimestamp(
                float(session.get("modified_timestamp") or 0.0), tz=UTC
            )
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )
        workspace = str(
            session.get("workspace_title") or session.get("workspace") or ""
        )
        model_id = str(session.get("target_model_id") or "Legacy model binding")
        state = "LOCKED" if session.get("editing_locked") else "Editable"
        reviewer = str(session.get("reviewer") or "Unknown reviewer")
        session_id = str(session.get("review_session_id") or "")
        short_id = session_id.rsplit("-", 1)[-1] if session_id else "legacy"
        return (
            f"{modified} · {int(session.get('manual_regions') or 0)} corrections · "
            f"{int(session.get('blind_region_labels') or 0)} blind labels · "
            f"{reviewer} · {short_id} · {state} · {workspace} · {model_id}"
        )

    @staticmethod
    def _discover_review_session_entries(
        extra_roots: tuple[str, ...],
    ) -> list[dict[str, object]]:
        sessions = [dict(row) for row in discover_phase_review_sessions()]
        if extra_roots:
            extra = discover_phase_review_sessions([Path(p) for p in extra_roots])
            combined = {str(row.get("session_path")): row for row in [*sessions, *extra]}
            sessions = sorted(
                combined.values(),
                key=lambda row: float(row.get("modified_timestamp") or 0),
                reverse=True,
            )
        return sessions

    def _start_review_session_discovery(self) -> None:
        if self._review_sessions_loading:
            return
        self._review_sessions_loading = True
        self.refresh_review_sessions_button.setEnabled(False)
        self.saved_sessions_status.setText("Loading saved review sessions in the background…")
        extra_roots = tuple(self._transfer_review_roots)

        def discover() -> None:
            try:
                result = (self._discover_review_session_entries(extra_roots), None)
            except Exception as error:  # noqa: BLE001 - show discovery failures in Studio
                result = (None, f"{type(error).__name__}: {error}")
            try:
                self._review_sessions_ready.emit(result)
            except RuntimeError:
                # The window may have closed during network discovery.
                pass

        review_context = copy_context()
        Thread(target=lambda: review_context.run(discover), name="phase-review-catalog", daemon=True).start()

    def _apply_async_review_sessions(self, result: object) -> None:
        sessions, error = result
        self._review_sessions_loading = False
        self.refresh_review_sessions_button.setEnabled(True)
        if error is None:
            self._install_review_sessions(sessions)
            self.saved_sessions_status.setText("Saved review sessions are up to date.")
            if self._model_catalog_inflight:
                self.catalog_loading_status.setText(
                    self._catalog_notice("Saved reviews loaded; checking model versions in the background…")
                )
        else:
            retained = " Previously loaded sessions are still available." if self.review_sessions else ""
            self.saved_sessions_status.setText(
                f"Could not load saved review sessions: {error}.{retained} "
                "Select Refresh saved review sessions to try again."
            )
            self.catalog_loading_status.setText(
                self._catalog_notice(f"Could not load saved reviews: {error}. Model versions are still being checked.")
            )
        # Existing saved bindings were verified by the catalogue worker. Only
        # legacy, still-unbound models need the newly discovered sessions, and
        # those can be checked without rescanning the model library.
        if error is None and sessions:
            self._request_async_model_refresh(bindings_only=True)

    def _refresh_review_sessions(self) -> None:
        self._start_review_session_discovery()

    @staticmethod
    def _session_reviewer_id(session: Mapping[str, object]) -> str:
        """The reviewer_id of a saved session (the slug of its name when absent)."""
        return reviewer_identifier(str(session.get("reviewer_id") or "").strip()
                                   or str(session.get("reviewer") or ""))

    def _refresh_annotators(self) -> None:
        """List the people of the discovered sessions; newly found people start ticked."""
        annotators = getattr(self, "annotators", None)
        if annotators is None:
            return
        unticked = {annotators.item(row).data(Qt.ItemDataRole.UserRole) for row in range(annotators.count())
                    if annotators.item(row).checkState() != Qt.CheckState.Checked}
        names: dict[str, str] = {}
        for session in self.review_sessions:
            names.setdefault(self._session_reviewer_id(session), str(session.get("reviewer") or "").strip())
        annotators.blockSignals(True)
        annotators.clear()
        for reviewer_id in sorted(names, key=lambda key: ((names[key] or key).casefold(), key)):
            name = names[reviewer_id]
            item = QListWidgetItem(f"{name} ({reviewer_id})" if name and name != reviewer_id else reviewer_id)
            item.setData(Qt.ItemDataRole.UserRole, reviewer_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if reviewer_id in unticked else Qt.CheckState.Checked)
            annotators.addItem(item)
        annotators.blockSignals(False)

    def _selected_annotators(self) -> list[str] | None:
        """Ticked reviewer IDs, or None while every listed person is ticked."""
        annotators = getattr(self, "annotators", None)
        if annotators is None:
            return None
        items = [annotators.item(row) for row in range(annotators.count())]
        ticked = [item.data(Qt.ItemDataRole.UserRole) for item in items
                  if item.checkState() == Qt.CheckState.Checked]
        return None if len(ticked) == len(items) else sorted(ticked)

    def _session_matches_annotators(self, session: Mapping[str, object]) -> bool:
        """Whether a saved session belongs to a ticked person."""
        selected = self._selected_annotators()
        return selected is None or self._session_reviewer_id(session) in selected

    def _annotators_changed(self, _item: object = None) -> None:
        """Show only the ticked people's sessions."""
        if hasattr(self, "saved_sessions"):
            self._rebuild_session_combo()

    def _install_review_sessions(self, sessions: list[dict[str, object]]) -> None:
        selected_path = str((self._selected_session() or {}).get("session_path") or "")
        self.review_sessions = sessions
        self._refresh_annotators()
        model = self._selected_validation_model()
        if (model and self._transfer_sources_for == str(model["root"])
                and any(row.get("parent_review_binding") for row in self._transfer_source_rows)):
            # Late review discovery updates progress without hiding ready maps.
            key = self._discovery_key(model, own_validation_only=True)
            self._discovery_results.pop(key, None)
            self._experiment_discovery_requested = ""
        self._rebuild_session_combo(selected_path=selected_path)
        if hasattr(self, "validation_model"):
            self._validation_model_changed()
        if hasattr(self, "show_training_statistics"):
            self._load_contributions(force=True)
        self._load_pending_evidence(force=True)

    def _rebuild_session_combo(
        self,
        _index: int = -1,
        *,
        selected_path: str = "",
    ) -> None:
        if not selected_path:
            selected_path = str(
                (self._selected_session() or {}).get("session_path") or ""
            )
        # One editable session per model, so there is nothing left to scope: the list shows the
        # selected model's sessions. Earlier models are in the model-history section above.
        self.visible_review_sessions = [
            session
            for session in self.review_sessions
            if self._session_matches_selected_model(session)
            and self._session_matches_annotators(session)
            and any(int(session.get(key) or 0) for key in
                    ("manual_regions", "blind_region_labels", "frame_verdicts"))
        ]
        self.saved_sessions.blockSignals(True)
        self.saved_sessions.clear()
        selected_index = None
        for index, session in enumerate(self.visible_review_sessions):
            self.saved_sessions.addItem(self._session_label(session))
            self.saved_sessions.setItemData(
                index,
                "\n".join(str(session.get(key) or "") for key in (
                    "review_session_id", "decision_path", "session_path"
                )),
                Qt.ItemDataRole.ToolTipRole,
            )
            if selected_path and str(session.get("session_path") or "") == selected_path:
                selected_index = index
        if selected_index is None and self.visible_review_sessions:
            nonempty = [
                index for index, session in enumerate(self.visible_review_sessions)
                if int(session.get("manual_regions") or 0)
                or int(session.get("blind_region_labels") or 0)
                or int(session.get("frame_verdicts") or 0)
            ]
            selected_index = max(
                nonempty or range(len(self.visible_review_sessions)),
                key=lambda index: float(
                    self.visible_review_sessions[index].get("modified_timestamp") or 0
                ),
            )
        self.saved_sessions.setCurrentIndex(
            selected_index if self.visible_review_sessions else -1
        )
        self.saved_sessions.blockSignals(False)
        self._session_changed()

    def _session_changed(self, _index: int = -1) -> None:
        session = self._selected_session()
        if session is None:
            self.session_summary.setToolTip("")
            self.session_summary.setText(
                "No saved boxes or validations in this session selection. "
                "Empty sessions are hidden; their files are kept."
            )
            self._update_action_states()
            return
        locked = bool(session.get("editing_locked"))
        model_id = str(session.get("target_model_id") or "legacy model binding")
        trained_models = ", ".join(
            str(row.get("model_id") or "")
            for row in session.get("training_locks") or []
            if isinstance(row, Mapping) and str(row.get("model_id") or "")
        )
        self.session_summary.setText(
            f"Prediction reviewed: {model_id} · Reviewer: {session.get('reviewer')} · "
            f"{int(session.get('reviewed_experiments') or 0)} experiment(s) · "
            f"{int(session.get('manual_regions') or 0)} localized corrections · "
            f"{int(session.get('blind_region_labels') or 0)} blind region labels. "
            + (
                f"Frozen into model: {trained_models or 'recorded model version'}. "
                "Open it to audit every saved decision without changing the evidence."
                if locked
                else "Resume reopens this exact editable decision store."
            )
        )
        self.session_summary.setToolTip(str(session.get("decision_path") or session.get("session_path") or ""))
        self.resume_session_button.setText(
            "Review locked evidence (read-only)"
            if locked
            else "Resume editable session"
        )
        self._update_review_button_labels()
        self._update_action_states()

    @staticmethod
    def _session_review_mode(session: Mapping[str, object]) -> str | None:
        surface = str(session.get("review_surface") or "").strip().casefold()
        workspace = str(session.get("workspace") or "").strip().casefold()
        if surface == "blind_gas_validation":
            return "blind_gas_validation"
        if surface == "blind_regions" or workspace == "training":
            return "blind_region_labeling"
        if surface == "open_correction" or workspace in {
            "current_comparison",
            "comparison",
        }:
            return "open_correction"
        return None

    def _matching_review_session(self, mode: str) -> Mapping[str, object] | None:
        selected = self._selected_session()
        campaign_id = str((selected or {}).get("campaign_id") or "").casefold()
        reviewer = self.reviewer.text().strip().casefold()
        for session in self.review_sessions:
            if self._session_review_mode(session) != mode:
                continue
            if reviewer and str(session.get("reviewer") or "").casefold() != reviewer:
                continue
            if (
                campaign_id
                and str(session.get("campaign_id") or "").casefold() != campaign_id
            ):
                continue
            return session
        return None

    def _update_review_button_labels(self) -> None:
        busy = self.validation_process is not None
        checking = bool((self._selected_validation_model() or {}).get("validation_binding_loading"))
        suffix = " (preparing phase maps...)" if busy else ""
        self.open_blind_labeling.setText(
            "Checking saved phase maps…" if checking else "Start new blind phase-labeling session" + suffix
        )
        self.open_model_correction.setText(
            "Checking saved phase maps…" if checking else "Start new model-validation session" + suffix
        )
        for control in (self.open_blind_labeling, self.open_model_correction):
            control.setToolTip("Verifying this exact model's saved maps and source binding in the background. No new maps are being generated."
                               if checking else "Open the selected model's validation session.")

    def _receive_retraction_status(self, message):
        import time
        key, result = message
        self._retraction_inflight.discard(key)
        self._retraction_status[key] = (time.monotonic(), result)
        self._update_action_states()

    def _pending_retraction_evidence(self, model):
        import time
        series_set = _model_series(model)
        selected = self._active_model_series or (next(iter(series_set)) if len(series_set) == 1 else "")
        if not selected or selected not in series_set or not model.get("root"):
            return None
        key = (str(model["root"]), selected)
        stamp, result = self._retraction_status.get(key, (0., None))
        if key not in self._retraction_inflight and (result is None or time.monotonic() - stamp >= 5):
            self._retraction_inflight.add(key)
            def read():
                try:
                    status = _read_model_retraction_state(*key)
                except Exception as error:
                    status = dict(error=str(error))
                try:
                    self._retractions_ready.emit((key, status))
                except RuntimeError:
                    pass  # The page was closed during a network read.
            Thread(target=read, name="phase-review-withdrawals", daemon=True).start()
        if not result or not (result.get("changed") or result.get("error")):
            return None
        error = result.get("error")
        count = int(result.get("withdrawn_count", 0))
        reclassified = int(result.get("reclassified_count", 0))
        calibration_required = result.get("calibration_required", "")
        return dict(evidence_change_kind="retraction_error" if error else "retraction",
                    calibration_required=calibration_required,
                    review_session_id=f"Historical withdrawals · {selected.upper()}",
                    target_model_id=model.get("model_id"), manual_regions=0, blind_region_labels=0,
                    withdrawn_count=count, decision_path=result.get("ledger_path", ""),
                    change_summary=(f"Cannot verify historical withdrawals: {error}" if error else
                        (f"{count} withdrawn or replaced evaluation(s), including {reclassified} reclassifications. "
                         + (calibration_required if calibration_required else
                            "New model must use the updated evidence policy"))))

    def _pending_training_evidence(self, model):
        """Use the background record comparison; never hash storage files on the GUI thread."""
        if not model:
            return []
        key = self._pending_model_key(model)
        evidence = self._pending_evidence if key == self._pending_evidence_key else {}
        pending = [dict(row) for row in evidence.get("eligible_sessions", [])]
        withdrawal = self._pending_retraction_evidence(model)
        if withdrawal is not None:
            pending.append(withdrawal)
        return pending

    def _pending_model_key(self, model):
        coverage = _model_series(model) if model else set()
        series = self._active_model_series or (next(iter(coverage)) if len(coverage) == 1 else "")
        return (str((model or {}).get("root") or ""), series)

    def _poll_pending_evidence(self):
        if self.isVisible() and self.tabs.currentWidget() is self.evidence_tab:
            self._load_pending_evidence(force=True, discover=True)

    def _load_pending_evidence(self, *, force=False, discover=False):
        model = self._selected_validation_model()
        key = self._pending_model_key(model)
        changed = key != self._pending_evidence_key
        if changed:
            self._pending_evidence_key = key
            self._pending_evidence = {}
        if not key[0] or not key[1]:
            self._update_training_update_card()
            return
        if self._pending_evidence_inflight is not None:
            self._pending_evidence_refresh_queued |= changed or force
            self._update_training_update_card()
            return
        if not changed and self._pending_evidence and not force:
            return
        self._pending_evidence_inflight = key
        sessions = [dict(row) for row in self.review_sessions]
        extra_roots = tuple(self._transfer_review_roots)
        self._update_training_update_card()

        def read():
            try:
                from ..phase_pending_evidence import pending_training_evidence
                if discover:
                    discovered = list(discover_phase_review_sessions())
                    if extra_roots:
                        discovered += list(discover_phase_review_sessions([Path(p) for p in extra_roots]))
                    current_sessions = list({str(row.get("decision_path")): dict(row)
                                             for row in discovered}.values())
                else:
                    current_sessions = sessions
                result = pending_training_evidence(key[0], current_sessions, series=key[1])
            except Exception as error:
                current_sessions = sessions
                result = dict(known=False, errors=[str(error)], eligible_sessions=[])
            try:
                self._pending_evidence_ready.emit((key, result, current_sessions))
            except RuntimeError:
                pass  # Studio closed while its saved evidence was being read.

        pending_context = copy_context()
        Thread(target=lambda: pending_context.run(read), name="phase-pending-evidence", daemon=True).start()

    def _receive_pending_evidence(self, message):
        key, result, sessions = message
        self._pending_evidence_inflight = None
        if key == self._pending_evidence_key:
            self._pending_evidence = result
            self.review_sessions = sessions
            self._rebuild_session_combo()
            self._update_action_states()
            self._load_contributions(force=True)
        if self._pending_evidence_refresh_queued or key != self._pending_evidence_key:
            self._pending_evidence_refresh_queued = False
            self._load_pending_evidence(force=True, discover=True)

    def _update_training_update_card(self):
        if not hasattr(self, "update_model_button"):
            return
        model = self._selected_validation_model()
        key = self._pending_model_key(model)
        evidence = self._pending_evidence if key == self._pending_evidence_key else {}
        self.update_model_button.setEnabled(False)
        self.refresh_new_evidence_button.setEnabled(bool(key[0] and key[1])
                                                   and self._pending_evidence_inflight is None)
        if not model or not key[1]:
            self.training_update_counts.setText("Select a model and experimental series.")
            self.training_update_details.setText("Your saved labels belong to their experimental series.")
            return
        if not evidence.get("known"):
            errors = evidence.get("errors", [])
            self.training_update_counts.setText(
                "Saved-label counts unavailable" if errors else "Checking saved labels…")
            self.training_update_details.setText(
                "; ".join(errors) if errors else "Comparing saved labels with this model's training receipt…")
            return
        pending = evidence.get("pending", {})
        visible = int(pending.get("visible_regions", 0))
        blind = int(pending.get("blind_labels", 0))
        has_changes = bool(evidence.get("has_changes"))
        self.training_update_counts.setText(
            f"{visible:,} validation regions · {blind:,} blind / quiz labels not used in this model"
            if visible + blind else ("Saved label changes are waiting for retraining" if has_changes
                                    else "No new or changed training labels found"))
        details = [f"Selected model: {model.get('model_id', '')}."]
        changed = int((evidence.get("changed") or {}).get("total", 0))
        if changed:
            details.append(f"Includes {changed:,} changed labels.")
        unused_counts = evidence.get("captured_unused") or {}
        unused = int(unused_counts.get("total", 0))
        if unused:
            reasons = evidence.get("captured_unused_reasons", {})
            descriptions = {
                "no-unambiguous-supervision": "conflicting labels leave no unambiguous training pixels",
            }
            reason_text = "; ".join(
                f"{count:,}: {descriptions.get(reason, reason.replace('-', ' ').replace('_', ' '))}"
                for reason, count in sorted(reasons.items())
            )
            details.append(f"{unused:,} previously captured labels "
                           f"({int(unused_counts.get('visible_regions', 0)):,} validation regions, "
                           f"{int(unused_counts.get('blind_labels', 0)):,} blind / quiz labels) "
                           "did not contribute to fitting. "
                           + reason_text + ".")
            if "no-unambiguous-supervision" in reasons:
                details.append("Use Check disagreements to review these conflicts.")
        holdout = int(evidence.get("independent_holdout_labels", 0))
        if holdout:
            details.append(f"{holdout:,} independent blind test labels remain reserved for evaluation.")
        pending_rows = self._pending_training_evidence(model)
        withdrawals = [row for row in pending_rows if row.get("evidence_change_kind")]
        if withdrawals:
            details.extend(str(row.get("change_summary", "")) for row in withdrawals)
        removed = int(evidence.get("pending_removed_regions", 0))
        if removed:
            details.append(f"{removed:,} previously used regions have been removed or withdrawn.")
        busy = self.training_process is not None or self.validation_process is not None or self._series_batch_active
        if busy:
            details.append("A training or phase-map job is running. The update button becomes available when it finishes.")
        elif model.get("state") == "transfer_evaluation":
            details.append("Select a fitted source model to train a new version.")
        elif _model_series(model) != {key[1]}:
            details.append("Select the fitted model for this one experimental series to update it.")
        elif not self.reviewer.text().strip():
            details.append("Enter your reviewer name above to train the new version.")
        elif not visible + blind and not has_changes:
            details.append("New saved labels will appear here automatically.")
        else:
            details.append("Train a new version using earlier evidence and these saved changes.")
        self.training_update_details.setText(" ".join(details))
        self.update_model_button.setEnabled(self.train_further_button.isEnabled())
        self.update_model_button.setToolTip(self.training_update_details.text())

    #: What the one button reads when no batch is running; the checked-aware label replaces it.
    _DEFAULT_BATCH_LABEL = "Generate phase maps for all experiments in this series"

    @staticmethod
    def _batch_series_supported(series: str) -> bool:
        """Historical phase-map batch launch is retired; saved review remains available."""
        return False

    def _runs_with_a_stale_prepared_source(self, runs) -> list[str]:
        """Of these experiments, the ones the batch must rebuild instead of verify and skip."""
        wanted = {str(run).casefold() for run in runs}
        return sorted({str(row["run"]).casefold() for row in self._transfer_source_rows
                       if str(row.get("run") or "").casefold() in wanted
                       and self._stale_prepared_source(row)})


    def _review_launch_block_reason(self, model=None):
        """Explain the actual launch gate without changing scientific eligibility."""
        model = self._selected_validation_model() if model is None else model
        if not model:
            return "Select a model to validate."
        if model.get("validation_binding_loading"):
            return "Checking this exact model's saved phase maps and source binding. No new maps are being generated."
        if model.get("validation_collection_error"):
            return "The selected model's saved maps could not be verified: " + str(model["validation_collection_error"])
        if not self.reviewer.text().strip():
            return "Enter your reviewer name before starting validation."
        if not self.review_save_root.text().strip():
            return "Choose an autosave folder for your validation session."
        if not self.combined_campaign.text().strip():
            return "No prepared review package is bound to this selection yet."
        if self.validation_process is not None:
            return "Phase-map preparation is still running."
        if self._series_batch_blocks_selection():
            return "A preparation job for the selected model and series is still active."
        if (self.training_process is not None and self.training_request
                and self.training_request.get("continuation")
                and model.get("model_id") == self.training_request.get("model_id")):
            return "A training continuation is still using the selected model."
        return "Validation is currently unavailable."

    def _update_action_states(self, _value: object = None) -> None:
        if hasattr(self, "evidence_audit_tab"):
            self.evidence_audit_tab.update_check_state()
        reviewer_ready = bool(self.reviewer.text().strip())
        review_paths_ready = bool(
            self.combined_campaign.text().strip()
            and self.review_save_root.text().strip()
        )
        model = self._selected_validation_model()
        review_ready = bool(
            reviewer_ready
            and review_paths_ready
            and model
            and not model.get("validation_binding_loading")
            and not model.get("validation_collection_error")
            and self.validation_process is None
            and not self._series_batch_blocks_selection()
        )
        if (self.training_process is not None and self.training_request
                and self.training_request.get("continuation") and model
                and model.get("model_id") == self.training_request.get("model_id")):
            review_ready = False
        self.open_blind_labeling.setEnabled(review_ready)
        self.open_model_correction.setEnabled(review_ready)
        session = self._selected_session()
        self.resume_session_button.setEnabled(session is not None)
        self.add_session_evidence.setEnabled(
            bool(session and str(session.get("decision_path") or "").strip())
        )
        self.add_selected_session_button.setEnabled(self.add_session_evidence.isEnabled())
        self.add_selected_session_button.setToolTip(
            "Check that the saved decisions and campaign manifest are available, then add this session to the evidence packages."
            if self.add_selected_session_button.isEnabled()
            else "Select a saved review session with a recorded decisions path."
        )
        training_ready = reviewer_ready and bool(self.bundles.toPlainText().strip())
        if self.training_scope.currentData() == "single-series":
            training_ready = training_ready and len([v for v in self.training_series.text().split(",") if v.strip()]) == 1
        if self.radiometry_mode.currentData() in {"image", "custom"}:
            training_ready = training_ready and bool(self.radiometry_profile_path.text().strip())
        self.start_button.setEnabled(training_ready and self.training_process is None
                                     and not self._series_batch_active)
        model_series = _model_series(model) if model else set()
        continuation_series = self._active_model_series or (next(iter(model_series)) if len(model_series) == 1 else "")
        self.train_further_button.setEnabled(bool(reviewer_ready and model and continuation_series
            and model_series == {continuation_series} and self.training_process is None
            and self.validation_process is None and not self._series_batch_active
            and model.get("state") != "transfer_evaluation"))
        pending = self._pending_training_evidence(model) if model else []
        if hasattr(self, "pending_training_sessions"):
            self.pending_training_sessions.setRowCount(len(pending))
            for index, row in enumerate(pending):
                for column, value in enumerate((row.get("review_session_id") or row.get("target_model_id"),
                        str(int(row.get("pending_visible_regions", row.get("manual_regions")) or 0)
                            + int(row.get("pending_blind_labels", row.get("blind_region_labels")) or 0)),
                        row.get("change_summary") or "New or changed saved evidence; not frozen in this model")):
                    item=QTableWidgetItem(str(value));item.setToolTip(str(row.get("decision_path") or ""))
                    self.pending_training_sessions.setItem(index,column,item)
        withdrawal_errors = [row for row in pending if row.get("evidence_change_kind") == "retraction_error"]
        saved_sessions = [row for row in pending if not row.get("evidence_change_kind")]
        withdrawals = [row for row in pending if row.get("evidence_change_kind") == "retraction"]
        calibration_blocks = [row for row in withdrawals if row.get("calibration_required")]
        inventory = (self._pending_evidence if self._pending_model_key(model) == self._pending_evidence_key
                     else {})
        inventory_changes = bool(inventory.get("known") and inventory.get("has_changes"))
        self.train_further_button.setEnabled(self.train_further_button.isEnabled() and bool(pending or inventory_changes)
                                            and not withdrawal_errors and not calibration_blocks)
        if withdrawal_errors:
            self.train_further_summary.setText(" ".join(row["change_summary"] for row in withdrawal_errors))
        elif calibration_blocks:
            self.train_further_summary.setText(
                "Historical evidence was withdrawn or reclassified. "
                "This model's learned reference/physical calibration must also be fitted again. "
                "Train further cannot yet rebuild that calibration automatically; "
                "continuation is disabled until a freshly calibrated model applies these changes. "
                "Your saved changes are retained.")
        elif pending or inventory_changes:
            self.train_further_summary.setText(
                f"Selected parent: {(model or {}).get('model_id','None')}. {len(saved_sessions)} sessions contain new or changed saved evidence. "
                + " ".join(row["change_summary"] for row in withdrawals)
                + " Training combines saved corrections with frozen evidence and excludes withdrawn evaluations.")
        elif not inventory.get("known"):
            self.train_further_summary.setText(
                "Saved training evidence is being checked or could not be verified. "
                + "; ".join(inventory.get("errors", [])))
        else:
            self.train_further_summary.setText("No new saved training evidence or historical withdrawals found for the selected model and its ancestors.")
        self._update_training_update_card()
        transfer_idle = (self.validation_process is None and self.training_process is None
                         and not self._series_batch_active)
        batch_series = self._active_model_series or (next(iter(model_series)) if len(model_series) == 1 else "")
        self.series_batch_button.setEnabled(bool(
            model and self.review_save_root.text().strip()
            and len(model_series) == 1 and batch_series in model_series
            and self._batch_series_supported(batch_series) and transfer_idle
            and (self.series_batch_workers.value() > 0 or self.series_batch_gpu.isChecked())
        ))
        if model and len(model_series) == 1 and not self._batch_series_supported(batch_series):
            self.series_batch_button.setToolTip(
                "Series-wide phase-map preparation is unavailable for this registered "
                "series in the current worker. Review and model history remain available."
            )
        else:
            self.series_batch_button.setToolTip(
                "Apply the selected frozen model to all experiments; no model fitting "
                "and no automatic QC pass. Existing completed maps are reused when resumed."
            )
        self.series_batch_workers.setEnabled(transfer_idle or bool(
            self._series_batch_active and getattr(self, "_live_batch_control", None)))
        self.series_batch_gpu.setEnabled(transfer_idle or bool(
            self._series_batch_active and getattr(self, "_live_batch_control", None)))
        self.refresh_transfer_targets.setEnabled(bool(
            model and len(model_series) == 1
            and self.validation_process is None and self.training_process is None
        ))
        self.choose_transfer_campaign.setEnabled(bool(model and len(model_series) == 1 and transfer_idle))
        transfer_target_ready = bool(model and reviewer_ready and transfer_idle
            and isinstance(self.transfer_target.currentData(), dict)
            and self.transfer_target.currentData().get("preparation_status", "ready") == "ready"
            and self._transfer_sources_for == str(model.get("root") or ""))
        self.blind_transfer_button.setEnabled(transfer_target_ready
            and self.transfer_physics_mode.currentData() in {"optical-only", "registered-target", "prepared-target"})
        target = self.transfer_target.currentData() or {}
        ready = self._experiment_ready(target)
        self.test_transfer_button.setEnabled(bool(model and reviewer_ready and ready))
        checked_runs = self._checked_pending_experiments()
        # The label is only refreshed while the button carries an idle label, never over
        # "Preparing...", "Resume..." or "Retry...", which report the state of a batch.
        text = self.series_batch_button.text()
        if text == self._DEFAULT_BATCH_LABEL or text.startswith("Prepare "):
            pending = self.pending_experiments.rowCount()
            ready = self.ready_experiments.rowCount()
            self.series_batch_button.setText(
                f"Prepare {len(checked_runs)} checked experiment"
                f"{'s' if len(checked_runs) != 1 else ''}" if checked_runs
                # With nothing left to prepare the run only re-verifies what is already there.
                else f"Verify all {ready} prepared experiments in this series"
                if ready and not pending
                else f"Prepare {pending} and verify {ready} in this series" if pending and ready
                else f"Prepare all {pending} experiments in this series" if pending
                else "Prepare and verify every experiment in this series")
        if checked_runs and self._transfer_sources_for != str((model or {}).get("root") or ""):
            # The checked names come from a list built for another model.
            self.series_batch_button.setEnabled(False)
        self.select_all_pending_button.setEnabled(bool(transfer_idle and self.pending_experiments.rowCount()))
        self.clear_pending_button.setEnabled(bool(transfer_idle and checked_runs))
        self.transfer_physics_mode.setEnabled(transfer_target_ready)
        visible_test = bool(self._completed_transfer and self._completed_transfer.get("review_mode") == "visible")
        self.transfer_comparison_button.setText("Reopen test results" if visible_test else "Open transfer comparison")
        self.transfer_comparison_button.setToolTip("Reopen the same frozen model and target experiment with predictions visible. Saved labels follow this experiment's training selection."
            if visible_test else "After blind labeling, open the same evaluation with predictions visible. Labels remain outside automatic training.")
        self.transfer_comparison_button.setEnabled(bool(model and transfer_idle and self._completed_transfer
            and self._completed_transfer.get("model_root") == str(model.get("root") or "")
            and (self._completed_transfer.get("command") if visible_test else self._completed_transfer.get("correction_command"))))
        promotable = bool(self.transfer_comparison_button.isEnabled() and visible_test
            and self._completed_transfer.get("physics_mode") in {"prepared-target", "registered-target"})
        promoted = bool(self._completed_transfer and self._completed_transfer.get("promotion_receipt"))
        self.include_transfer_training_button.setText("Included for next training" if promoted else "Include in next training")
        self.include_transfer_training_button.setEnabled(promotable and not promoted)
        self._update_review_button_labels()
        blocked = self._review_launch_block_reason(model) if not review_ready else ""
        if blocked:
            self.open_blind_labeling.setToolTip(blocked)
            self.open_model_correction.setToolTip(blocked)
            if self.validation_process is None:
                self.review_status.setText("Validation unavailable: " + blocked)
        elif self.review_status.text().startswith("Validation unavailable:"):
            self.review_status.setText("Ready to open the selected model's validation session.")

    @staticmethod
    def _companion_path(name: str) -> Path:
        return tool_path(name)

    def _auto_find_experiments(self) -> None:
        model = self._selected_validation_model()
        if not model or model.get("validation_binding_loading") or len(_model_series(model)) != 1:
            return
        if self.validation_process is not None or self.training_process is not None:
            self._experiment_discovery_timer.start(1000)
            return
        self._experiment_discovery_requested = str(model["root"])
        self._find_transfer_targets(own_validation_only=True)

    FULL_PHYSICS_MODES = frozenset({"registered-target", "prepared-target"})

    @classmethod
    def _stale_prepared_source(cls, row) -> bool:
        """Phase maps exist, but the source they were made from no longer validates.

        Discovery re-checks every experiment's own prepared target and falls back to optical-only
        when that check fails -- a target context whose provenance names another source manifest,
        for instance. The review collection applies the same check and refuses the experiment, so
        calling it ready made it vanish between the lists and the review window with nothing said.
        """
        return bool(not row.get("parent_review_binding")
                    and cls.FULL_PHYSICS_MODES.intersection(row.get("completed_evaluations") or {})
                    and not cls.FULL_PHYSICS_MODES.intersection(row.get("available_physics_modes") or []))

    @classmethod
    def _experiment_ready(cls, row) -> bool:
        """Ready means the review collection will accept it, not that a map exists somewhere."""
        if row.get("parent_review_binding"):
            return True
        if cls._stale_prepared_source(row):
            return False
        return bool(cls.FULL_PHYSICS_MODES.intersection(row.get("completed_evaluations") or {}))

    def _fill_experiment_tables(self, rows) -> None:
        self._pending_checked_runs.difference_update(
            str(row["run"]).casefold() for row in rows if self._experiment_ready(row))
        for table, ready in ((self.ready_experiments, True), (self.pending_experiments, False)):
            table.blockSignals(True); table.setRowCount(0)
            # Prefer a ready result when several corrected-image sources exist for one run.
            by_run = {}
            for row in rows:
                key = self._transfer_run_identity(row)
                old = by_run.get(key)
                if old is None or (self._experiment_ready(row), row.get("preparation_status")=="ready") > (self._experiment_ready(old), old.get("preparation_status")=="ready"):
                    by_run[key] = row
            def order(row):
                run = str(row["run"]).casefold()
                kind = self.experiment_sort.currentData()
                natural = int("".join(c for c in run if c.isdigit()) or "0")
                count = sum(self._contributions.get("runs", {}).get(run, {}).get(kind, {}).values())
                return (-count if kind in {"trained", "pending"} else 0, natural, run)
            for row in sorted(by_run.values(), key=order):
                if self._experiment_ready(row) != ready: continue
                i=table.rowCount();table.insertRow(i)
                item=QTableWidgetItem(str(row["run"]).upper());item.setData(Qt.ItemDataRole.UserRole,self._transfer_row_identity(row))
                if not ready:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if str(row["run"]).casefold()
                                       in self._pending_checked_runs else Qt.CheckState.Unchecked)
                status = (f"{row.get('reviewed_frames',0)}/{row.get('total_frames',0)} frames reviewed"
                          if ready and row.get("total_frames") else "")
                if not ready:
                    status = str(row.get("error") or row.get("progress_message") or "")
                    if not status and self._stale_prepared_source(row):
                        status = ("Phase maps exist, but their prepared source no longer validates"
                                  " \u00b7 check this experiment and prepare it again")
                table.setItem(i,0,item);table.setItem(i,1,QTableWidgetItem(status))
                contribution = self._contributions.get("runs", {}).get(str(row["run"]).casefold(), {})
                for column, kind in ((2, "trained"), (3, "pending")):
                    count = sum(contribution.get(kind, {}).values())
                    cell = QTableWidgetItem(str(count) if self._contributions.get("known") else "—")
                    cell.setToolTip("Unique training regions; not pixels or causal model importance."
                                    if kind == "trained" else "Eligible saved annotations not yet used, or changed since training.")
                    table.setItem(i, column, cell)
            table.blockSignals(False)
        self._apply_batch_states_to_tables()
        self.transfer_count.setText(f"{self.ready_experiments.rowCount()} ready · {self.pending_experiments.rowCount()} not prepared")
        self._update_action_states()
        self._update_training_statistics()

    def _checked_pending_experiments(self) -> list[str]:
        return [self.pending_experiments.item(i, 0).text().casefold()
                for i in range(self.pending_experiments.rowCount())
                if self.pending_experiments.item(i, 0).checkState() == Qt.CheckState.Checked]

    def _pending_experiment_checked(self, item) -> None:
        if item.column() != 0:
            return
        run = item.text().casefold()
        if item.checkState() == Qt.CheckState.Checked:
            self._pending_checked_runs.add(run)
        else:
            self._pending_checked_runs.discard(run)
        self._update_action_states()

    def _check_pending_experiments(self, checked: bool) -> None:
        self.pending_experiments.blockSignals(True)
        for i in range(self.pending_experiments.rowCount()):
            item = self.pending_experiments.item(i, 0)
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            if checked:
                self._pending_checked_runs.add(item.text().casefold())
            else:
                self._pending_checked_runs.discard(item.text().casefold())
        self.pending_experiments.blockSignals(False)
        self._update_action_states()

    def _experiment_selected(self, table) -> None:
        row=table.currentRow()
        if row<0 or not table.selectedItems():return
        identity=table.item(row,0).data(Qt.ItemDataRole.UserRole)
        other=self.pending_experiments if table is self.ready_experiments else self.ready_experiments
        other.blockSignals(True);other.clearSelection();other.blockSignals(False)
        for index in range(self.transfer_target.count()):
            if self._transfer_row_identity(self.transfer_target.itemData(index))==identity:
                self.transfer_target.setCurrentIndex(index);break
        self._update_training_statistics()

    def _load_contributions(self, *, force=False):
        model = self._selected_validation_model()
        root = str((model or {}).get("root") or "")
        if root == self._contributions_root and not force:
            return
        self._contributions_root = root
        self._contributions = {}
        self._contributions_generation += 1
        generation = self._contributions_generation
        if not root:
            return
        sessions = [dict(s) for s in self._pending_training_evidence(model) if not s.get("evidence_change_kind")]
        def read():
            from ..phase_contributions import model_contributions
            try:
                result = model_contributions(root, sessions)
            except Exception as error:
                result = dict(known=False, runs={}, errors=[str(error)])
            try:
                self._contributions_ready.emit((generation, result))
            except RuntimeError:
                pass  # Window closed during a network read.
        Thread(target=read, name="phase-training-statistics", daemon=True).start()

    def _receive_contributions(self, message):
        generation, result = message
        if generation != self._contributions_generation:
            return
        evidence = self._pending_evidence
        if (evidence.get("known") and self._pending_evidence_key
                and self._pending_evidence_key[0] == self._contributions_root
                and "pending_categories_by_run" in evidence):
            # Use the same revision/withdrawal accounting in the experiment
            # table and front-page total; whole-session counts can disagree.
            categories = evidence["pending_categories_by_run"]
            rows = result.setdefault("runs", {})
            for run in set(rows) | set(categories):
                rows.setdefault(run, {"trained": {}})["pending"] = dict(categories.get(run, {}))
        self._contributions = result
        self._rebuild_transfer_targets()

    def _update_training_statistics(self):
        from ..phase_contributions import LABELS
        target = self.transfer_target.currentData() or {}
        run = str(target.get("run") or "").casefold()
        counts = self._contributions.get("runs", {}).get(run, {})
        trained, pending = counts.get("trained", {}), counts.get("pending", {})
        categories = list(LABELS) + sorted((set(trained) | set(pending)) - LABELS.keys())
        self.training_statistics_table.setRowCount(len(categories))
        for row, category in enumerate(categories):
            for column, value in enumerate((LABELS.get(category, category.replace("_", " ")),
                                            trained.get(category, 0), pending.get(category, 0))):
                self.training_statistics_table.setItem(row, column, QTableWidgetItem(str(value)))
        text = (f"{run.upper()}: {sum(trained.values())} training regions; "
                f"{sum(pending.values())} new / changed. "
                "Counts describe optical training evidence, not how strongly an experiment influences predictions.")
        if self._contributions.get("inherited_from"):
            text += " Inherited weights from " + str(self._contributions["inherited_from"]) + "."
        if not self._contributions.get("known"):
            text = "Training inventory unavailable or still loading; no zero contribution is assumed."
        if self._contributions.get("errors"):
            text += " " + "; ".join(self._contributions["errors"])
        self.training_statistics_summary.setText(text)

    def _open_selected_experiment(self) -> None:
        target=self.transfer_target.currentData() or {};model=self._selected_validation_model()
        if not model or not self._experiment_ready(target):return
        binding=target.get("parent_review_binding")
        if binding:
            command=build_phase_review_command(mode="open_correction",reviewer=self.reviewer.text().strip(),
                save_root=Path(self.review_save_root.text()),cache_root=default_execution_cache_root()/"phase-review",
                campaign_root=Path(binding["combined_root"]),selected_candidates={target["run"]:binding["candidate"]},
                target_model_id=model["model_id"],
                review_session_id=self._review_session_id_for(model["model_id"]),
                use_gpu=self.review_gpu.isChecked())
            started,pid=QProcess.startDetached(command[0],command[1:])
            if started:
                # Corrections saved in that window change what discovery would report.
                self._forget_discovered_experiments()
            self.transfer_status.setText(f"Opened {target['run'].upper()} for validation." if started else "Could not open review.")
        else:
            self._select_completed_target_result(target)
            self._open_transfer_comparison()

    def _discovery_key(self, model, *, own_validation_only: bool = False) -> tuple | None:
        """What a discovery answer belongs to: this model and the sources it was asked about."""
        root = str((model or {}).get("root") or "")
        if not root:
            return None
        return (root, own_validation_only, tuple(self._extra_transfer_manifests),
                tuple(self._transfer_review_roots))

    def _forget_discovered_experiments(self) -> None:
        """Something may have changed the prepared experiments; read them again next time."""
        self._discovery_results.clear()

    def _refind_transfer_targets(self, _checked: bool = False) -> None:
        """Find experiments reads the prepared sources again, whatever is remembered."""
        self._forget_discovered_experiments()
        self._find_transfer_targets()

    def _find_transfer_targets(self, *, own_validation_only: bool = False) -> None:
        model = self._selected_validation_model()
        key = self._discovery_key(model, own_validation_only=own_validation_only)
        remembered = self._discovery_results.get(key) if key is not None else None
        if remembered is not None:
            # Same model, same extra sources: reading every manifest again gives the same answer.
            self._transfer_source_rows = [dict(row) for row in remembered]
            self._transfer_sources_for = str(model["root"])
            self._rebuild_transfer_targets()
            self.transfer_status.setText(
                "Showing the experiments found earlier for this model. "
                "Press Find experiments to read the prepared sources again."
            )
            self._update_action_states()
            return
        binding = (model or {}).get("validation_binding")
        if (own_validation_only and binding and binding.get("completed_utc")
                and binding.get("format") != "fluidflower-phase-review-collection-v1"):
            self._request_saved_experiments(model, binding)
            return
        arguments = [argument for path in self._extra_transfer_manifests for argument in ("--source-manifest", path)]
        arguments += [argument for path in self._transfer_review_roots for argument in ("--review-root", path)]
        if own_validation_only:
            arguments.append("--own-validation-only")
        self._launch_transfer_worker("list", arguments)

    def _request_saved_experiments(self, model, binding) -> None:
        """Read completed source certificates without the training/transfer scan."""
        self._saved_experiments_generation += 1
        generation = self._saved_experiments_generation
        root = str(model["root"])
        sessions = tuple(dict(row) for row in self.review_sessions)
        self.transfer_count.setText("Reading saved experiments…")
        self.transfer_status.setText("Checking this candidate's saved source manifests in the background.")

        def read():
            try:
                from ..phase_saved_transfer_review import saved_validation_sources
                rows = saved_validation_sources(root, binding, sessions)
                result = (generation, root, rows, None)
            except Exception as error:
                result = (generation, root, None, f"{type(error).__name__}: {error}")
            try:
                self._saved_experiments_ready.emit(result)
            except RuntimeError:
                pass

        saved_experiments_context = copy_context()
        Thread(target=lambda: saved_experiments_context.run(read), name="phase-saved-experiments", daemon=True).start()

    def _apply_saved_experiments(self, result) -> None:
        generation, root, rows, error = result
        current = self._selected_validation_model()
        if (generation != self._saved_experiments_generation or not current
                or root != str(current["root"])):
            return
        if error:
            self.transfer_count.setText("Could not read the saved experiments")
            self.transfer_status.setText(error)
            return
        if rows is None:
            # Legacy exports without source hashes retain the original checks.
            arguments = [argument for path in self._extra_transfer_manifests
                         for argument in ("--source-manifest", path)]
            arguments += [argument for path in self._transfer_review_roots
                          for argument in ("--review-root", path)]
            self._launch_transfer_worker("list", [*arguments, "--own-validation-only"])
            return
        self._transfer_source_rows = rows
        self._transfer_sources_for = root
        key = self._discovery_key(current, own_validation_only=True)
        if key is not None:
            self._discovery_results[key] = [dict(row) for row in rows]
        self._rebuild_transfer_targets()
        self.transfer_status.setText(
            "Saved phase maps are ready to review. Training-history statistics load separately.")

    def _series_batch_blocks_selection(self) -> bool:
        """Historical phase-map batch launch is retired; saved review remains available."""
        return False

    def _selected_series_batch(self) -> tuple[str, str] | None:
        """Historical phase-map batch launch is retired; saved review remains available."""
        return None







    def _apply_batch_states_to_tables(self) -> None:
        """Write the running batch's per-experiment state into the two experiment lists."""
        states = getattr(self, "_batch_run_states", None) or {}
        if not states:
            return
        for table in (self.pending_experiments, self.ready_experiments):
            table.blockSignals(True)
            for index in range(table.rowCount()):
                name = table.item(index, 0)
                description = states.get(name.text().casefold()) if name is not None else None
                if not description:
                    continue
                detail = table.item(index, 1)
                if detail is None:
                    detail = QTableWidgetItem()
                    table.setItem(index, 1, detail)
                detail.setText(description)
                detail.setToolTip(description)
            table.blockSignals(False)


    @staticmethod
    def _transfer_run_identity(row: Mapping[str, object]) -> tuple[str, str]:
        return (
            str(row.get("dataset") or "").casefold(),
            str(row.get("run") or "").casefold(),
        )

    @staticmethod
    def _transfer_row_identity(row: Mapping[str, object]) -> tuple[str, str, str, str]:
        return (
            str(row.get("dataset") or "").casefold(),
            str(row.get("run") or ""),
            str(row.get("native_manifest_sha256") or ""),
            str(row.get("source_manifest") or ""),
        )

    @staticmethod
    def _transfer_row_label(row: Mapping[str, object], *, duplicate_run: bool = False) -> str:
        label = str(row.get("label") or str(row.get("run") or "").upper())
        if duplicate_run:
            revision = str(row.get("native_manifest_sha256") or "")[:10]
            if revision:
                label += f" · image revision {revision}"
            else:
                source = str(row.get("source_manifest") or "")
                label += f" · source {Path(source).parent.name or 'unprepared'}"
        review = str(row.get("review_status") or "")
        review_text = {
            "reviewed": "manually reviewed",
            "in_progress": "manual review in progress",
            "not_reviewed": "not manually reviewed",
        }.get(review, "manual review unverified")
        try:
            reviewed_frames = int(row.get("reviewed_frames") or 0)
            total_frames = int(row.get("total_frames") or 0)
        except (TypeError, ValueError):
            reviewed_frames = total_frames = 0
        if total_frames:
            review_text += f" ({reviewed_frames}/{total_frames} frames)"
        if row.get("preparation_status", "ready") != "ready":
            return f"{label} · Needs image correction · {review_text}"
        return f"{label} · {review_text}"

    def _rebuild_transfer_targets(self, _index: int = -1) -> None:
        previous = self.transfer_target.currentData()
        previous_identity = (
            self._transfer_row_identity(previous)
            if isinstance(previous, Mapping) else None
        )
        scope = str(self.transfer_review_filter.currentData() or "all")
        rows = [
            row for row in self._transfer_source_rows
            if scope == "all"
            or (scope == "reviewed" and row.get("review_status") == "reviewed")
            or (scope == "not_reviewed" and row.get("review_status") != "reviewed")
        ]
        self.transfer_target.blockSignals(True)
        self.transfer_target.clear()
        selected_index = 0
        run_counts = Counter(self._transfer_run_identity(row) for row in self._transfer_source_rows)
        for index, row in enumerate(rows):
            self.transfer_target.addItem(
                self._transfer_row_label(
                    row, duplicate_run=run_counts[self._transfer_run_identity(row)] > 1,
                ),
                row,
            )
            if previous_identity == self._transfer_row_identity(row):
                selected_index = index
        self.transfer_target.setCurrentIndex(selected_index if rows else -1)
        self.transfer_target.blockSignals(False)
        total_entries = len(self._transfer_source_rows)
        total_runs = len(run_counts)
        shown_runs = len({self._transfer_run_identity(row) for row in rows})
        prepared = sum(row.get("preparation_status", "ready") == "ready"
                       for row in self._transfer_source_rows)
        reviewed = sum(row.get("review_status") == "reviewed"
                       for row in self._transfer_source_rows)
        self.transfer_count.setText(
            f"{shown_runs} of {total_runs} experiments · "
            f"{len(rows)} of {total_entries} entries shown · "
            f"{prepared} image-ready entries · {reviewed} manually reviewed entries"
        )
        if not rows and total_entries:
            self.transfer_status.setText("No experiments match this review filter.")
        self._transfer_target_changed()
        self._fill_experiment_tables(rows)

    def _choose_transfer_campaign(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Choose prepared campaign manifest",
            str(Path.home()), "Campaign manifest (postflight_manifest.json);;JSON files (*.json)",
            options=FILE_DIALOG_OPTIONS)
        if not filename:
            return
        path = str(Path(filename).resolve())
        if path not in self._extra_transfer_manifests:
            self._extra_transfer_manifests.append(path)
        self._find_transfer_targets()

    def _transfer_target_changed(self) -> None:
        if not hasattr(self, "transfer_physics_mode"):
            return
        target = self.transfer_target.currentData() or {}
        if not target:
            self.transfer_physics_mode.blockSignals(True)
            self.transfer_physics_mode.clear()
            self.transfer_physics_mode.blockSignals(False)
            self._update_action_states()
            return
        modes = ([m for m in target.get("available_physics_modes", []) if m != "optical-only"]
                 if target.get("preparation_status", "ready") == "ready" else [])
        self.transfer_physics_mode.blockSignals(True)
        self.transfer_physics_mode.clear()
        for mode in modes:
            label = self._transfer_physics_label(mode, target)
            self.transfer_physics_mode.addItem(label, mode)
        recommended = target.get("physics_mode", "optical-only")
        self.transfer_physics_mode.setCurrentIndex(
            max(0, self.transfer_physics_mode.findData(recommended)) if modes else -1
        )
        self.transfer_physics_mode.blockSignals(False)
        self._transfer_physics_changed()
        self._update_action_states()

    @staticmethod
    def _transfer_physics_label(mode: str, target: Mapping[str, object]) -> str:
        if mode == "prepared-target":
            return "Full model · own prepared target context"
        if mode == "registered-target":
            return "Full model · registered target physics"
        if mode == "optical-only":
            scope = "registered domain" if target.get("has_registered_domain") is True else "full image crop"
            return f"Optical classifier only · target baseline / {scope}"
        return f"Unsupported physics mode: {mode}"

    @staticmethod
    def _transfer_exposure_text(target: Mapping[str, object]) -> str:
        names = {"seen": "seen", "unseen": "unseen", "unverified": "unverified"}
        optical = names.get(str(target.get("optical_training_status") or ""), "unverified")
        calibration = names.get(str(target.get("calibration_evidence_status") or ""), "unverified")
        return f"Optical training: {optical}. Calibration evidence: {calibration}. Unverified does not mean held-out."

    def _transfer_physics_changed(self) -> None:
        if not hasattr(self, "transfer_status"):
            return
        target = self.transfer_target.currentData()
        if not isinstance(target, dict):
            return
        if target.get("preparation_status", "ready") != "ready":
            self._completed_transfer = None
            self.transfer_status.setText(
                f"{str(target.get('run') or '').upper()}: Needs image correction "
                "before this model can generate phase maps for manual validation."
            )
            self._update_action_states()
            return
        self._select_completed_target_result(target)
        physical = (self._transfer_physics_label(str(self.transfer_physics_mode.currentData()), target) + "."
                    if self.transfer_physics_mode.currentData() else "Full-model context must be prepared for this experiment.")
        self.transfer_status.setText(self._transfer_exposure_text(target) + " " + physical
            + " Open ready maps for review, or prepare this experiment with the full model."
            + (" Training selection needs attention: " + str(self._completed_transfer["promotion_error"])
               if self._completed_transfer and self._completed_transfer.get("promotion_error") else ""))
        self._update_action_states()

    def _select_completed_target_result(self, target: Mapping[str, object]) -> None:
        model = self._selected_validation_model()
        mode = self.transfer_physics_mode.currentData()
        candidate = (target.get("completed_evaluations") or {}).get(mode)
        def matches(result):
            return bool(result and model
                and result.get("model_root") == str(model.get("root") or "")
                and result.get("model_id") == model.get("model_id")
                and result.get("target_run") == target.get("run")
                and result.get("target_native_manifest_sha256") == target.get("native_manifest_sha256")
                and result.get("physics_mode") == mode)
        if matches(candidate):
            # Results of another experiment are independent of completion time.
            if not matches(self._completed_transfer):
                self._completed_transfer = dict(candidate)
            else:
                self._restore_transfer_result(candidate, model)
        elif not matches(self._completed_transfer):
            self._completed_transfer = None
        if self._completed_transfer:
            self._remember_transfer_review_root(self._completed_transfer)

    def _remember_transfer_review_root(self, result: Mapping[str, object]) -> None:
        command = result.get("reopen_command") or result.get("command") or []
        if "--local-review-root" not in command:
            return
        index = command.index("--local-review-root") + 1
        if index >= len(command):
            return
        root = Path(str(command[index])).expanduser().resolve()
        if root == Path(root.anchor):
            return
        if str(root) not in self._transfer_review_roots:
            self._transfer_review_roots.append(str(root))
            self._model_filter_settings.setValue("review/transferRoots",json.dumps(self._transfer_review_roots))


    def _open_transfer_comparison(self) -> None:
        result = self._completed_transfer
        model = self._selected_validation_model()
        if not result or not model or result.get("model_root") != str(model.get("root") or ""):
            return
        visible_test = result.get("review_mode") == "visible"
        command = (result.get("reopen_command") or result.get("command")) if visible_test else result.get("correction_command")
        if not command:
            return
        if "--reviewer" in command and command[command.index("--reviewer") + 1] == "Phase-map preparation":
            reviewer = self.reviewer.text().strip()
            if not reviewer:
                self.transfer_status.setText("Enter your reviewer name to start validation.")
                return
            # Preparation has no human reviewer. Bind its review commands only
            # when someone opens the maps; never save labels under the placeholder.
            result = dict(result)
            for key in ("command", "reopen_command", "correction_command"):
                saved = list(result.get(key) or [])
                if "--reviewer" in saved:
                    saved[saved.index("--reviewer") + 1] = reviewer
                    result[key] = saved
            self._completed_transfer = result
            command = (result.get("reopen_command") or result.get("command")) if visible_test else result.get("correction_command")
        started, process_id = QProcess.startDetached(command[0], command[1:])
        self.transfer_status.setText(
            f"{'Test results reopened' if visible_test else 'Comparison opened'} for {result['model_id']} · {result['label']} (process {process_id}). "
            "Predictions are visible. Training inclusion is shown with the selected experiment."
            if started else "The prepared transfer comparison could not be opened.")



    def _launch_transfer_worker(self, operation: str, arguments: list[str]) -> None:
        if operation != "list":
            raise ValueError("Only stored transfer metadata may be listed here")
        from importlib.util import find_spec
        if find_spec("darsia.fluidflower.tools.phase_transfer_review") is None:
            # The tool was removed (REMOVED.md).
            self.transfer_count.setText("")
            self.transfer_status.setText(
                "Testing a saved model on other experiments is not part of this version of DarSIA.")
            return
        model = self._selected_validation_model()
        if (model is None or self.validation_process is not None
                or self.training_process is not None
                or (self._series_batch_active and operation != "list")):
            return
        try:
            script = self._companion_path("phase_transfer_review.py")
            command = [phase_python_executable(), *tool_arguments(script), "list",
                "--model", str(model["root"]), *arguments]
            process = QProcess(self)
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.setProgram(command[0])
            process.setArguments(command[1:])
            process.setWorkingDirectory(str(local_cache_root()))
            process.readyReadStandardOutput.connect(self._transfer_output)
            process.finished.connect(self._transfer_finished)
            process.errorOccurred.connect(self._transfer_error)
            self.validation_process = process
            review_mode = arguments[arguments.index("--review-mode") + 1] if "--review-mode" in arguments else "visible"
            self._transfer_job = dict(operation=operation, model_root=str(model["root"]), model_id=model["model_id"],
                review_mode=review_mode, target=dict(self.transfer_target.currentData() or {}),
                own_validation_only="--own-validation-only" in arguments)
            self._transfer_result = None
            self._transfer_event_buffer = ""
            physics_mode = arguments[arguments.index("--physics-mode") + 1] if "--physics-mode" in arguments else ""
            evaluation_label = self._transfer_physics_label(physics_mode, self._transfer_job["target"])
            review_label = "Predictions will open directly." if review_mode == "visible" else "Optional blind labeling: predictions will remain hidden."
            if operation == "list":
                self.transfer_count.setText("Reading experiments...")
            self.transfer_status.setText("Reading this model\'s prepared experiments..." if self._transfer_job["own_validation_only"] else "Finding prepared same-series experiments...")
            self._update_action_states()
            process.start()
        except Exception as exc:  # noqa: BLE001 - visible worker boundary
            self.transfer_status.setText(f"Could not prepare model test: {exc}")

    def _show_discovery_progress(self, value: Mapping[str, object]) -> None:
        """Reading the prepared sources takes tens of seconds; show that it is happening."""
        found = int(value.get("experiments") or 0)
        done, total = int(value.get("manifests_done") or 0), int(value.get("manifests_total") or 0)
        where = f" \u00b7 source {done}/{total}" if total else ""
        run = str(value.get("run") or "")
        self.transfer_count.setText(f"Reading experiments... {found} found{where}")
        scoped = bool(self._transfer_job and self._transfer_job.get("own_validation_only"))
        self.transfer_status.setText(
            (f"Reading this model's prepared experiments: {found} found{where}" if scoped else
             f"Searching prepared sources for this model: {found} experiments so far{where}")
            + (f", checking {run.upper()}" if run else "")
            + (". Source revisions are being verified." if scoped else
               ". Historical prepared manifests are being checked.")
        )

    def _show_target_source_progress(self, value: Mapping[str, object]) -> None:
        """Keep a long native-source verification visibly moving within one manifest."""
        index, total = int(value.get("job_index") or 0), int(value.get("job_total") or 0)
        run = str(value.get("run") or "").upper()
        if total:
            self.transfer_count.setText(f"Checking prepared experiment {index}/{total}")
            self.transfer_status.setText(f"Verifying source files for {run} ({index}/{total})...")

    def _transfer_output(self) -> None:
        if self.validation_process is None:
            return
        text = bytes(self.validation_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._transfer_event_buffer += text
        while "\n" in self._transfer_event_buffer:
            line, self._transfer_event_buffer = self._transfer_event_buffer.split("\n", 1)
            prefix = next((p for p in ("PHASE_TRANSFER_EVENT ","PHASE_PROMOTION_EVENT ") if line.startswith(p)),None)
            if prefix:
                value = json.loads(line[len(prefix):])
                if value.get("stage") in {"sources", "ready", "registered"}:
                    self._transfer_result = value
                elif value.get("stage") == "discovery":
                    self._show_discovery_progress(value)
                elif value.get("stage") == "target":
                    self._show_target_source_progress(value)
            elif line.strip():
                self.transfer_status.setText(line[-500:])

    def _transfer_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._transfer_finished(-1, QProcess.ExitStatus.CrashExit)

    def _restore_transfer_result(self, result: object, model: Mapping[str, object]) -> None:
        if (not isinstance(result, dict) or result.get("model_root") != str(model.get("root") or "")
                or result.get("model_id") != model.get("model_id")):
            return
        current = self._completed_transfer
        identity = ("model_root", "model_id", "target_run", "target_native_manifest_sha256", "physics_mode")
        if current and tuple(current.get(k) for k in identity) == tuple(result.get(k) for k in identity):
            def timestamp(value):
                try:
                    parsed = datetime.fromisoformat(str(value.get("completed_utc") or value.get("created_utc") or "").replace("Z", "+00:00"))
                    return None if parsed.tzinfo is None else parsed.timestamp()
                except (TypeError, ValueError):
                    return None
            previous_time, restored_time = timestamp(current), timestamp(result)
            if previous_time is None or restored_time is None or previous_time >= restored_time:
                return
        self._completed_transfer = dict(result)
        self._remember_transfer_review_root(result)

    def _transfer_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self.validation_process is None or self._transfer_job is None:
            return
        self._transfer_output()
        process, job, result = self.validation_process, self._transfer_job, self._transfer_result
        self.validation_process = None
        self._transfer_job = None
        self._transfer_result = None
        process.deleteLater()
        if exit_code != 0 or result is None:
            if job["operation"] == "list":
                # Otherwise the count is left reading "Reading experiments..." for good.
                self.transfer_count.setText("Could not read the prepared experiments")
            self.transfer_status.setText(f"Model test preparation failed (exit {exit_code}). "
                + self.transfer_status.text()[-400:])
        elif job["operation"] == "list":
            current = self._selected_validation_model()
            if current is not None and str(current["root"]) == job["model_root"]:
                self._transfer_source_rows = [dict(row) for row in result["sources"]]
                self._transfer_sources_for = str(current["root"])
                key = self._discovery_key(current, own_validation_only=bool(job.get("own_validation_only")))
                if key is not None:
                    self._discovery_results[key] = [dict(row) for row in result["sources"]]
                self._rebuild_transfer_targets()
                if (job.get("own_validation_only") and not any(
                        row.get("prepared") for row in self._transfer_source_rows)):
                    self.transfer_status.setText(
                        "This model has no prepared experiments of its own. "
                        "Press Find experiments to search historical prepared sources.")
                if not self.transfer_target.count():
                    self._restore_transfer_result(result.get("last_evaluation"), current)
                    if not self._transfer_source_rows and not job.get("own_validation_only"):
                        self.transfer_status.setText("No compatible experiments found. Choose a campaign to add another source.")
            else:
                self.transfer_status.setText("Model selection changed. Find prepared experiments for the newly selected model.")
        elif job["operation"] == "promote":
            if self._completed_transfer and self._completed_transfer.get("evaluation_root") == job.get("evaluation_root"):
                self._completed_transfer["promotion_receipt"] = result["receipt_path"]
            for index in range(self.transfer_target.count()):
                row = self.transfer_target.itemData(index)
                if not isinstance(row,dict):
                    continue
                evaluations = dict(row.get("completed_evaluations") or {})
                for mode, value in list(evaluations.items()):
                    if value.get("evaluation_root") == job.get("evaluation_root"):
                        evaluations[mode] = dict(value,promotion_receipt=result["receipt_path"])
                self.transfer_target.setItemData(index,dict(row,completed_evaluations=evaluations))
            self.transfer_status.setText("This experiment is included in the next Train further run. Its review corrections remain editable; no training has started.")
        else:
            review_mode = str(result.get("review_mode") or job.get("review_mode") or "blind")
            # A finished evaluation, and the review window about to open on it, both change what
            # discovery would report for this experiment.
            self._forget_discovered_experiments()
            self._completed_transfer = dict(result, model_root=job["model_root"], model_id=job["model_id"], review_mode=review_mode)
            self._remember_transfer_review_root(self._completed_transfer)
            target = job.get("target") or {}
            if target.get("run"):
                self._completed_transfer.update(target_run=target["run"],
                    target_native_manifest_sha256=target.get("native_manifest_sha256"),
                    physics_mode=result.get("physics_mode") or self.transfer_physics_mode.currentData())
                target.setdefault("completed_evaluations", {})[self._completed_transfer["physics_mode"]] = dict(self._completed_transfer)
                for index in range(self.transfer_target.count()):
                    row = self.transfer_target.itemData(index)
                    if isinstance(row, dict) and (row.get("run"), row.get("native_manifest_sha256")) == (target["run"], target.get("native_manifest_sha256")):
                        self.transfer_target.setItemData(index, dict(row, completed_evaluations=target["completed_evaluations"]))
            command = result["command"]
            started, process_id = QProcess.startDetached(command[0], command[1:])
            self.transfer_status.setText(
                f"{'Test results' if review_mode == 'visible' else 'Optional blind labeling'} opened for {job['model_id']} · {result['label']} (process {process_id}). "
                + ("Predictions are visible. " if review_mode == "visible" else "Predictions are hidden. ")
                + self._transfer_exposure_text(dict(job.get("target") or {}, **result))
                + " Use Include in next training to select this session for a later fit."
                if started else "The prepared model test could not be opened."
            )
        self._update_action_states()

    def _review_command(
        self,
        mode: str,
        *,
        binding_override: Mapping[str, object] | None = None,
        session_id: str = "",
    ) -> list[str]:
        model = self._selected_validation_model()
        if model and model.get("validation_collection_error"):
            raise ValueError(str(model["validation_collection_error"]))
        binding = binding_override or (
            model.get("validation_binding")
            if isinstance(model, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            raise ValueError(
                "The selected model has no generated validation phase maps. "
                "Select a model marked Ready, or generate predictions for this model first."
            )
        model_id = str(model.get("model_id") or "").strip()
        resolved_session_id = session_id or self._review_session_id_for(model_id)
        command = build_phase_review_command(
            mode=mode,
            ffac_root=Path(str(binding["ffac_root"])),
            ffum_root=Path(str(binding["ffum_root"])),
            combined_root=Path(str(binding["combined_root"])),
            reviewer=self.reviewer.text().strip(),
            save_root=Path(self.review_save_root.text()),
            cache_root=default_execution_cache_root() / "phase-review",
            use_gpu=self.review_gpu.isChecked(),
            selected_candidates=dict(binding.get("selected_candidates") or {}),
            target_model_id=model_id,
            review_session_id=resolved_session_id,
        )
        if (
            getattr(self, "_native_step_adapter", None)
            and mode == "open_correction"
            and model and model.get("root")
        ):
            from .review_version_controls import model_choice_arguments

            choices = [
                dict(entry, display_name=self._validation_model_label(entry))
                for entry in _models_for_series(self.model_history, self._active_model_series)
            ]
            command.extend(model_choice_arguments(choices, model["root"]))
        return command

    def _launch_review(self, mode: str) -> None:
        if self._series_batch_blocks_selection():
            return
        model = self._selected_validation_model()
        if model and model.get("validation_binding_loading"):
            self.review_status.setText("Checking the selected model's saved phase-map binding before opening validation…")
            return
        if (model and model.get("validation_binding_guard")
                and not self._saved_binding_guard_matches(model["validation_binding_guard"])):
            model.pop("validation_binding_guard", None)
            model["validation_binding"] = None
            model["validation_binding_loading"] = True
            self.review_status.setText("Saved candidate metadata changed; rechecking its exact binding before opening validation…")
            self._update_action_states()
            self._request_async_model_refresh(bindings_only=True, verify_saved=True)
            return
        if model and model.get("validation_collection_error"):
            message = str(model["validation_collection_error"])
            self.review_status.setText("Could not open the prepared experiment collection: " + message)
            QMessageBox.critical(self, "Could not open phase review", message)
            return
        binding = (
            model.get("validation_binding")
            if isinstance(model, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            self.review_status.setText("No prepared phase maps are bound to this model. Open an existing saved candidate for review.")
            return
        self._start_review_process(mode, binding)

    def _start_review_process(
        self,
        mode: str,
        binding: Mapping[str, object],
        *,
        session_id: str = "",
    ) -> None:
        try:
            command = self._review_command(
                mode,
                binding_override=binding,
                session_id=session_id,
            )
            Path(self.review_save_root.text()).expanduser().resolve().mkdir(
                parents=True, exist_ok=True
            )
            started, process_id = QProcess.startDetached(command[0], command[1:])
            if not started:
                raise RuntimeError("The review application did not start")
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.review_status.setText(
                f"Could not open review: {type(exc).__name__}: {exc}"
            )
            QMessageBox.critical(self, "Could not open phase review", str(exc))
            return
        label = (
            "Blind phase labeling"
            if mode == "blind_region_labeling"
            else "Open model correction"
        )
        self.review_status.setText(
            f"{label} opened as process {int(process_id)}. Every edit is autosaved "
                "to the selected series review folder."
        )
        self.review_opened.emit(dict(binding))


    def _validation_materialization_output(self) -> None:
        process = self.validation_process
        if process is None:
            return
        lines = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        ).strip().splitlines()
        if lines:
            self.review_status.setText(f"Preparing selected model: {lines[-1]}")

    def _validation_materialization_error(self) -> None:
        process = self.validation_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardError())
        self.validation_error_output.extend(chunk)
        # readyRead drains QProcess; retain the traceback for the finished signal.
        del self.validation_error_output[:-131072]
        message = self.validation_error_output.decode("utf-8", errors="replace").strip()
        if message:
            self.review_status.setText(f"Phase-map preparation: {message.splitlines()[-1]}")

    def _validation_materialization_finished(
        self, exit_code: int, _status: QProcess.ExitStatus
    ) -> None:
        process = self.validation_process
        output_root = self.pending_validation_output
        model_root = self.pending_validation_model_root
        identity = self.pending_validation_model_identity
        physics_runtime = getattr(self, "pending_validation_physics_runtime", None)
        mode = self.pending_validation_mode
        session_id = self.pending_validation_session_id
        if process is not None:
            self._validation_materialization_error()
        error = self.validation_error_output.decode("utf-8", errors="replace").strip()
        self.validation_error_output.clear()
        self.validation_process = None
        self.pending_validation_mode = ""
        self.pending_validation_session_id = ""
        self.pending_validation_output = None
        self.pending_validation_model_root = None
        self.pending_validation_model_identity = None
        self.pending_validation_physics_runtime = None
        self._update_action_states()
        if exit_code != 0 or output_root is None:
            message = error or f"Phase-map generation exited with code {exit_code}."
            last_line = message.splitlines()[-1]
            log_path = default_execution_cache_root() / "logs" / "phase-validation" / (
                f"{session_id or uuid.uuid4().hex}.log"
            )
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("x", encoding="utf-8") as stream:
                    stream.write(f"Output: {output_root}\nExit code: {exit_code}\n{message}\n")
                message += f"\n\nDiagnostic log: {log_path}"
            except OSError:
                # Failure to write diagnostics must not hide the original error.
                pass
            self.review_status.setText(f"Could not prepare validation: {last_line}")
            QMessageBox.critical(self, "Could not prepare validation", message)
            return
        try:
            binding = json.loads(
                (output_root / "validation_binding.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            if not isinstance(binding, dict):
                raise TypeError("Validation binding is not a JSON object")
            from darsia.fluidflower.phase_validation import _model_identity
            if (model_root is None or identity is None or _model_identity(model_root) != identity
                    or (binding.get("model_id"), binding.get("model_sha256")) != identity):
                raise ValueError("Prepared phase maps belong to another model revision")
            if physics_runtime is not None:
                from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
                if physics_runtime_identity(load_physics_profile(model_root)) != physics_runtime:
                    raise ValueError("The phase-method revision changed while preparing validation")
                if binding.get("physics_runtime") != physics_runtime:
                    _validate_saved_phase_binding(model_root, binding)
            model = next((row for row in self.model_history
                          if Path(str(row.get("root") or "")).resolve() == model_root), None)
            if isinstance(model, dict):
                model["validation_binding"] = binding
            # Review commands derive identity from the selected model. Restore
            # the pinned job's model before constructing that command.
            if not self.select_validation_model(model_root, series_context=True):
                raise ValueError("The model whose maps were prepared is no longer available")
            self.ffac_campaign.setText(str(binding["ffac_root"]))
            self.ffum_campaign.setText(str(binding["ffum_root"]))
            self.combined_campaign.setText(str(binding["combined_root"]))
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(self, "Could not open validation", str(exc))
            return
        self._start_review_process(mode, binding, session_id=session_id)

    def _resume_selected_session(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        try:
            command = build_phase_review_resume_command(
                session,
                cache_root=default_execution_cache_root() / "phase-review",
                use_gpu=self.review_gpu.isChecked(),
                read_only=bool(session.get("editing_locked")),
            )
            started, process_id = QProcess.startDetached(command[0], command[1:])
            if not started:
                raise RuntimeError("The saved review session did not start")
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.review_status.setText(
                f"Could not resume review: {type(exc).__name__}: {exc}"
            )
            QMessageBox.critical(self, "Could not resume saved review", str(exc))
            return
        state = (
            "Opened immutable evidence for audit"
            if session.get("editing_locked")
            else "Resumed editable session"
        )
        self.review_status.setText(
            f"{state} as process {int(process_id)}. Existing decisions remain in place."
        )

    def _add_selected_session_as_bundle(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        try:
            decisions = resolve_review_path(str(session.get("decision_path") or ""))
            combined = resolve_review_path(str(session.get("combined_root") or ""))
            manifest = combined / "postflight_manifest.json"
            if not decisions.is_file() or not manifest.is_file():
                QMessageBox.warning(
                    self,
                    "Evidence package incomplete",
                    "The selected session needs both saved decisions and its combined "
                    "postflight manifest before it can be used for training.",
                )
                return
            self._append_bundle_row(decisions, manifest)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "Evidence package unavailable",
                "Could not access the selected session's saved evidence. "
                "Check its storage connection and try again.\n"
                f"{type(error).__name__}: {error}",
            )
            return
        self.tabs.setCurrentWidget(self.training_tab)
        self.status.setText("Session added to the manual training inputs. Review the full input list and use Start training; Train further keeps its separate cumulative evidence selection.")

    def _append_bundle_row(self, decisions: Path, manifest: Path) -> None:
        row = f"{decisions.resolve()} | {manifest.resolve()}"
        existing_rows = [
            value.strip()
            for value in self.bundles.toPlainText().splitlines()
            if value.strip()
        ]
        if row not in existing_rows:
            existing_rows.append(row)
            self.bundles.setPlainText("\n".join(existing_rows))

    def _add_bundle(self) -> None:
        decisions, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose saved review decisions",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if not decisions:
            return
        manifest, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose matching combined campaign manifest",
            str(Path(decisions).parent),
            "JSON files (*.json);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if manifest:
            self._append_bundle_row(Path(decisions), Path(manifest))

    def _update_training_effect_summary(self, _value: object = None) -> None:
        backend = str(self.backend.currentData() or "auto")
        device = {
            "auto": "automatic CUDA/CPU selection",
            "cuda": "CUDA float32 fitting with CPU-float64 parity checks",
            "cpu": "CPU float64 reference fitting",
        }[backend]
        control = (
            "An additional no-change-evidence control will be fitted."
            if self.change_ablation.isChecked()
            else "No change-evidence control will be fitted."
        )
        normalization = self.radiometry_mode.currentText()
        if self.radiometry_mode.currentData() in {"image", "custom"}:
            source = self.radiometry_profile_path.text()
            normalization += f" ({Path(source).name if source else 'source not selected'})"
        self.training_effect_summary.setText(
            f"Manual new-model training uses {normalization}. Current effect: {device}; "
            f"up to {self.stage_workers.value()} runs are "
            f"prepared and {self.extraction_workers.value()} runs have features "
            f"extracted concurrently; at most {self.pixels.value():,} pixels represent "
            f"each reviewed region; mixed-region fitting may use "
            f"{self.mil_iterations.value()} iterations. {control}"
        )

    def _start_training(self) -> None:
        if self.training_process is not None or self._series_batch_active:
            return
        try:
            bundles = parse_training_bundles(self.bundles.toPlainText())
            settings = {
                "radiometry_profile": (
                    ("image:" if self.radiometry_mode.currentData() == "image" else "") + self.radiometry_profile_path.text().strip()
                    if self.radiometry_mode.currentData() in {"custom", "image"}
                    else str(self.radiometry_mode.currentData())
                ),
                "series": [v.strip() for v in self.training_series.text().split(",") if v.strip()],
                "model_scope": str(self.training_scope.currentData()),
                "evaluation_mode": str(self.training_evaluation.currentData()),
                "regularization": 1e-4,
                "chemistry": "series-specific optical response; no concentration fit",
                "fit_mode": "mil",
                "backend": str(self.backend.currentData() or "auto"),
                "stage_workers": self.stage_workers.value(),
                "extraction_workers": self.extraction_workers.value(),
                "opencv_threads": 2,
                "pixels_per_region": self.pixels.value(),
                "mil_max_iters": self.mil_iterations.value(),
                "change_evidence_ablation": self.change_ablation.isChecked(),
            }
            selected_annotators = getattr(self, "_selected_annotators", None)
            annotators = selected_annotators() if callable(selected_annotators) else None
            if annotators is not None:
                if not annotators:
                    raise ValueError("Tick at least one annotator before training")
                settings["reviewers"] = annotators
            native_photometry = getattr(self, "_native_training_photometry", None)
            if native_photometry is not None:
                settings = native_photometry.prepare_settings(settings, bundles)
            if native_photometry is None and self.radiometry_mode.currentData() in {"custom", "image"} and not self.radiometry_profile_path.text().strip():
                raise ValueError("Select a common ColorChecker reference image or target profile first")
            # A series owns its trained versions regardless of the library
            # currently selected for browsing historical/shared models.
            output_parent = (self._models_folder_for_series(_canonical_series(settings["series"][0]))
                             if len(settings["series"]) == 1
                             else Path(self.output_parent.text()))
            request = prepare_phase_training_request(
                output_parent=output_parent,
                cache_parent=Path(self.cache_parent.text()),
                bundles=bundles,
                actor=self.reviewer.text().strip(),
                settings=settings,
            )
            if native_photometry is not None:
                native_photometry.require_request(request)
            command = build_phase_training_command(request)
            self._refresh_review_sessions()
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(self, "Could not start phase training", str(exc))
            return

        self._launch_training_job(request, command, Path(request["request_path"]).parent,
            f"Training {request['model_id']} from {len(bundles)} frozen evidence package(s)...")


    def _launch_training_job(self, request, command, working_directory, status) -> None:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._training_finished)
        process.errorOccurred.connect(self._training_error)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(working_directory))
        self.training_request = request
        self.training_process = process
        if request.get("output_root"):
            self.last_output_root = Path(request["output_root"])
        self._training_event_buffer = ""
        self.log.clear()
        self.status.setText(status)
        self.stop_button.setEnabled(True)
        self._update_action_states()
        process.start()

    def _read_output(self) -> None:
        process = self.training_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()
            if self.training_request and self.training_request.get("continuation"):
                self._training_event_buffer += text
                while "\n" in self._training_event_buffer:
                    line, self._training_event_buffer = self._training_event_buffer.split("\n", 1)
                    if line.startswith("PHASE_CONTINUATION_EVENT "):
                        self._continuation_event(json.loads(line[len("PHASE_CONTINUATION_EVENT "):]))

    def _record_completed_model(self, model_id: str, root: Path) -> None:
        self._sync_completed_model_roots()
        self._completed_model_roots[model_id] = str(root.resolve())
        self._model_filter_settings.setValue("modelHistory/completedModelRoots", json.dumps(self._completed_model_roots))
        self._model_filter_settings.sync()
        # Scanning and hashing the model library takes up to a minute on the
        # shared drive: read it in the background and select the new model
        # (then prepare its post-training views) once it is listed.
        self._completed_model_root = str(root.resolve())
        self._request_async_model_refresh()

    def _select_completed_model(self) -> None:
        """Select the model that just completed, once the background catalogue lists it."""
        root = self.__dict__.get("_completed_model_root")
        if not root or not self.select_validation_model(root, series_context=True):
            return
        self._completed_model_root = None
        request = self.__dict__.setdefault("_post_training_requests", {}).pop(root.casefold(), None)
        if request is not None:
            self._prepare_post_training_views(request, Path(root))

    def _continuation_event(self, value) -> None:
        request = self.training_request
        if not request or not request.get("continuation"):
            return
        stage = value["stage"]
        root = Path(value["output_root"]).resolve()
        model_id = str(value["model_id"])
        if root.name != model_id:
            raise ValueError("Continuation event has inconsistent model identity")
        request.update(value)
        self.last_output_root = root
        if stage in {"fitted", "published", "ready"}:
            self._record_completed_model(model_id, root)
        messages = {
            "prepared": "Evidence frozen. Retraining CO2 and gas heads...",
            "fitted": "New model fitted and visible. Preparing native review maps...",
            "published": "New model is in the library. Native review maps are being prepared...",
            "materializing": "New model fitted; native phase maps are being calculated...",
            "ready": "New model and native phase maps are ready for review. Human evaluation is still required.",
        }
        message = str(value.get("message") or messages.get(stage, stage))
        self.status.setText(message)
        self.review_status.setText(message)

    @staticmethod
    def _reviewed_training_runs(root):
        request=json.loads((Path(root)/"training_request.json").read_text(encoding="utf-8-sig"))
        runs=set()
        for bundle in request.get("bundles",[]):
            document=json.loads(Path(bundle["decisions"]["snapshot_path"]).read_text(encoding="utf-8-sig"))
            for run,row in document.get("decisions",{}).items():
                if row.get("annotations") or row.get("phase_quiz_answers") or row.get("frame_verdicts"):
                    runs.add(run.casefold())
        return sorted(runs)

    def _prepare_post_training_views(self, request, root):
        """Note which reviewed experiments already have phase maps; the files are read in the background.

        Reading every training bundle and the model's saved review binding
        takes seconds to minutes on the shared drive.
        """
        scope=request.get("post_training_view_scope")
        if scope not in {"reviewed","all"}:return
        model=self._selected_validation_model()
        if not model or len(_model_series(model))!=1:
            self.status.setText("Model fitted. Choose one series to prepare review maps for this shared model.");return
        from darsia.fluidflower.ui_metadata_preview import MetadataPreview
        entry, sessions = dict(model), tuple(dict(row) for row in self.review_sessions)

        def read(_cancelled):
            runs=None if scope=="all" else self._reviewed_training_runs(root)
            binding=self._validation_binding(entry, review_sessions=sessions) or {}
            prepared=sorted((binding.get("selected_candidates") or {}).keys())
            plan=dict(model_id=entry["model_id"],scope=scope,requested_runs=runs,
                already_prepared=prepared,cpu_workers=request["post_training_cpu_workers"],
                gpu_worker=request["post_training_gpu_worker"])
            return root, runs, prepared, plan

        reader = self.__dict__.get("_post_training_reader")
        if reader is None:
            reader = self._post_training_reader = MetadataPreview(self)
            reader.completed.connect(self._post_training_views_read)
        self.status.setText("Model fitted. Checking which reviewed experiments already have phase maps…")
        reader.start(read)

    def _post_training_views_read(self, value, error):
        if error or value is None:
            self.status.setText(f"Model fitted; post-training preparation could not start: {error or 'no answer was returned'}")
            return
        root, runs, prepared, plan = value
        try:
            (Path(root)/"post_training_view_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
        except OSError as error:
            self.status.setText(f"Model fitted; post-training preparation could not start: {error}");return
        if runs is not None and not (set(runs)-set(prepared)):
            self.status.setText("New model ready. Phase maps for the selected reviewed experiments are already prepared.");return
        self.series_batch_workers.setValue(plan["cpu_workers"])
        self.series_batch_gpu.setChecked(plan["gpu_worker"])
        self.status.setText("Model fitted. Missing phase maps are not generated by the saved-review workspace.")

    def _stop_training(self) -> None:
        if self.training_process is None:
            return
        self.status.setText("Stopping this training job and its workers. Frozen inputs remain reusable.")
        process_id = int(self.training_process.processId())
        if os.name == "nt" and process_id:
            # terminate() does not reliably stop Windows console workers, and
            # extraction pools are child processes of this exact owned PID.
            try:
                result = subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"],
                    capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                self.training_process.kill()
            else:
                if result.returncode != 0:
                    self.training_process.kill()
        elif process_id and (self.training_request or {}).get("continuation"):
            try:
                os.killpg(process_id, signal.SIGTERM)
            except ProcessLookupError:
                self.training_process.kill()
        else:
            self.training_process.terminate()

    def _training_error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(
            f"Training process error: {error.name}. Frozen inputs were retained."
        )
        if error == QProcess.ProcessError.FailedToStart and self.training_process is not None:
            self._training_finished(-1, QProcess.ExitStatus.CrashExit)

    def _training_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        self._read_output()
        request = dict(self.training_request or {})
        output_root = Path(str(request.get("output_root") or ""))
        process = self.training_process
        self.training_process = None
        self.stop_button.setEnabled(False)
        self._update_action_states()
        if process is not None:
            process.deleteLater()
        if int(exit_code) != 0:
            fitted = request.get("stage") in {"fitted", "published", "materializing"}
            self.status.setText(("The fitted model remains visible; publishing or phase-map preparation failed. "
                if fitted else f"Training stopped or failed with exit code {int(exit_code)}. ")
                + f"See the job log. Frozen evidence remains in {output_root}.")
            return
        try:
            summary = inspect_phase_model_bundle(output_root)
            ablation_path = output_root / "change_evidence_ablation.json"
            ablation = (
                json.loads(ablation_path.read_text(encoding="utf-8"))
                if ablation_path.is_file()
                else {}
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.status.setText(
                f"Training completed, but model inspection failed: {exc}"
            )
            return
        delta = ablation.get("mean_balanced_accuracy_delta")
        ablation_text = (
            ""
            if not isinstance(delta, (int, float))
            else f" Mean holdout benefit from DarSIA change evidence: {delta:+.3f}."
        )
        self.status.setText(
            f"Completed {request.get('model_id')} using "
            f"{len(summary.get('runs') or [])} source runs.{ablation_text}"
        )
        self._record_completed_model(str(request.get("model_id") or output_root.name), output_root)
        self.tabs.setCurrentWidget(self.evidence_tab)
        if not request.get("continuation"):
            self.history_toggle.setChecked(True)
        if request.get("continuation"):
            self.status.setText(
                "Native maps are ready for inspection. Some fitted heads did not converge; inspect the training diagnostics."
                if request.get("nonconverged_heads") else
                "New model and native phase maps are ready for review. Human evaluation is still required.")
        self.training_completed.emit(str(output_root))
        # Prepared once the background catalogue has selected the new model.
        self.__dict__.setdefault("_post_training_requests", {})[str(output_root.resolve()).casefold()] = request
        self._select_completed_model()


__all__ = ["SharedPhaseModelTrainingPage"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare an ordinary Studio validation session")
    parser.add_argument("--prepare-review-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    arguments = parser.parse_args()
    _prepare_regional_review(arguments.prepare_review_root, arguments.receipt_root)
