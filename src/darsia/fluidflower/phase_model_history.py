"""Read-only discovery and provenance summaries for optical phase models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from functools import wraps
from datetime import datetime, timezone
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.model_display import recorded_photometry_provenance
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path


PHASE_MODEL_HISTORY_FORMAT = "darsia-fluidflower-phase-model-history-v1"
_CATALOG_METADATA_ONLY = ContextVar("phase_catalog_metadata_only", default=False)
REGISTRATION_ROOTS_UNAVAILABLE = "registration roots unavailable: no native parameter document bound (series.toml [native])"
_CATALOG_NATIVE_AVAILABLE = ContextVar("phase_catalog_captured_native_available", default=None)


class ModelHistoryEntries(list):
    """Existing list API, plus read-only discovery diagnostics for the UI."""

    def __init__(self, entries=(), *, diagnostics=()):
        super().__init__(entries)
        self.diagnostics = tuple(dict.fromkeys(diagnostics))


def registration_context_available():
    """Only an absent document is optional; a present invalid binding is not."""
    if _CATALOG_NATIVE_AVAILABLE.get() is False:
        return False  # Explicit captured absence must never consult a later env.
    from .native.config import require_configuration

    try:
        require_configuration()
    except ValueError as error:
        if str(error) != "An explicit native parameter document is required":
            raise
        if any(os.environ.get(key) for key in
               ("DARSIA_NATIVE_CONFIGURATION", "DARSIA_NATIVE_CONFIGURATION_SHA256")):
            raise ValueError("Native parameter environment needs both configuration and SHA-256") from error
        return False
    return True


def run_with_registration_context(callback, *, available=None):
    """UI catalogue-only scope; default direct callers retain their own context.

    This flag is not a native parameter document and grants no saved-map
    admission. It only preserves an explicit missing-document observation.
    """
    if available is None:
        return callback()
    token = _CATALOG_NATIVE_AVAILABLE.set(available)
    try:
        return callback()
    finally:
        _CATALOG_NATIVE_AVAILABLE.reset(token)


def metadata_only_catalog(function):
    """Keep dropdown discovery cheap; full history remains explicitly available."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _CATALOG_METADATA_ONLY.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _CATALOG_METADATA_ONLY.reset(token)
    return wrapped


def deduplicate_model_roots(roots: Iterable[Path | str]) -> list[Path]:
    """Keep the first path to each actual directory, including storage aliases.

    A model ID or matching model bytes do not identify a directory: different
    copies can own different labels and review sessions.  NAS hostnames and IP
    addresses, however, can identify the same directory.  Filesystem identity
    proves that equivalence without opening model contents or comparing every
    pair of paths.  Unavailable paths and filesystems without usable inode IDs
    remain separate unless their normalized path text is identical.
    """
    selected: list[Path] = []
    path_keys: set[str] = set()
    directory_ids: set[tuple[int, int]] = set()
    for value in roots:
        path = resolve_review_path(value)
        key = os.path.normcase(os.path.normpath(os.fspath(path)))
        if key in path_keys:
            continue
        path_keys.add(key)
        try:
            info = path.stat()
        except OSError:
            info = None
        if (info is not None and stat.S_ISDIR(info.st_mode)
                and info.st_dev and info.st_ino):
            identity = (info.st_dev, info.st_ino)
            if identity in directory_ids:
                continue
            directory_ids.add(identity)
        selected.append(path)
    return selected


def project_phase_model_roots() -> list[Path]:
    """Read each series' model library independently of process-local settings.

    Desktop launches and background workers need not share a Windows settings
    registry view. The project directories are the durable source of models.
    Only inspect direct model folders; never walk their images or caches.
    """
    from .series_paths import configured_series, series_path

    roots = []
    for series, _label, _project_root in configured_series():
        from .phase_model_library import library_records
        roots.extend(Path(record["root"]) for record in library_records(_project_root)
                     if record["available"])
        try:
            library = series_path(series, "phase_models")
            with os.scandir(library) as children:
                roots.extend(Path(child.path).resolve() for child in children
                             if child.is_dir(follow_symlinks=False)
                             and (Path(child.path) / "summary.json").is_file())
        except OSError:
            # An unavailable series must not hide other reachable libraries.
            continue
    return sorted(deduplicate_model_roots(roots))


def project_result_model_roots() -> list[Path]:
    """List model versions stored under registered results, regardless of QA.

    Model identity and review readiness are reported separately by the history
    inspector. Neither publication status nor our QA opinion controls whether
    a model version appears in the user's selector. The bounded search skips
    images and caches and includes interrupted versions when they have a
    training request but no completed model files yet.
    """
    from .series_paths import configured_series
    registration_available = registration_context_available()

    roots: list[Path] = []
    for _series, _label, project_root in configured_series():
        roots.extend(
            path for path in _model_directories(project_root / "results", max_depth=3)
            if path.name.startswith("phase-model-")
        )
        if _series == "ffac" and registration_available:
            from .phase_saved_registration import registered_roots
            roots.extend(registered_roots(results=project_root / "results"))
    return sorted(deduplicate_model_roots(roots))


def registered_phase_model_defaults() -> dict[str, str]:
    """Explicit local defaults shared by desktop launchers, independent of QSettings."""
    from .series_paths import configured_series, series_key

    path = Path(os.environ.get("DARSIA_FLUIDFLOWER_PHASE_MODEL_DEFAULTS") or
                Path.home()/"Documents/FluidFlower_Review_Local/phase_model_defaults.json")
    if not path.is_file():
        return {}
    records = _read_mapping(path)
    available = {key for key, _label, _root in configured_series()}
    defaults: dict[str, str] = {}
    for series, root in records.items():
        try:
            key = series_key(series)
        except ValueError:
            continue
        if key not in available or not isinstance(root, str):
            continue
        model_root = resolve_review_path(root)
        if (model_root / "summary.json").is_file() and (model_root / "models").is_dir():
            defaults[key] = str(model_root)
    return defaults


def registered_old_domain_seam_review(model_root, warning):
    """Recognize this sealed review kind only through its intact warning."""
    if not warning.startswith("Studio review candidate:"):
        return False
    from .phase_saved_registration import registration_path
    try:
        registration = _read_mapping(registration_path(Path(model_root).name))
        return registration.get("registration_kind") == "approved_g3_old_domain_seam_revision"
    except (OSError, ValueError, TypeError):
        return False


def registered_combined_l1_review(model_root, warning):
    """Recognize only this intact, sealed review kind; never promote its QA."""
    from .phase_saved_review_metadata import saved_revision_bindings
    WARNING = saved_revision_bindings('combined_floor')._export('WARNING')
    if warning != WARNING:
        return False
    from .phase_saved_revision_review import combined_saved_review
    try:
        combined_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_t3_review(model_root, warning):
    """Recognize only the source-bound T3 review revision, never approve mass."""
    from .phase_saved_review_metadata import saved_revision_bindings
    WARNING = saved_revision_bindings('trained_floor')._export('WARNING')
    if warning != WARNING:
        return False
    from .phase_saved_revision_review import t3_saved_review
    try:
        t3_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_t4_review(model_root, warning):
    """Recognize the sealed two-epoch gas revision without approving mass."""
    from .phase_saved_review_metadata import saved_revision_bindings
    WARNING = saved_revision_bindings('epoch_floor')._export('WARNING')
    if warning != WARNING:
        return False
    from .phase_saved_revision_review import t4_saved_review
    try:
        t4_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_t4b_review(model_root, warning):
    """Admit the sealed joint lamp-shift review without approving mass."""
    from .phase_saved_review_metadata import saved_revision_bindings
    WARNING = saved_revision_bindings('joint_floor')._export('WARNING')
    if warning != WARNING:
        return False
    from .phase_saved_revision_review import t4b_saved_review
    try:
        t4b_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_t4c_review(model_root, warning):
    """Use exact calibrated saved-map admission, including the placement."""
    if not (resolve_review_path(model_root) / "t4c_floor_plan.json").is_file():
        return False
    from .phase_saved_floor_review import t4c_saved_review
    from .phase_saved_review_metadata import configured_parameters
    try:
        checked = t4c_saved_review(model_root)
        expected = configured_parameters("floor")["warning"]
        return warning == expected
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_p1_final3_review(model_root, warning):
    from .phase_saved_contact_review import p1_final3_saved_review, WARNING, PLAN
    if warning != WARNING or not (resolve_review_path(model_root) / PLAN).is_file():
        return False
    try:
        p1_final3_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_p1_final2_review(model_root, warning):
    from .phase_saved_categorical_review import p1_final2_saved_review, WARNING, PLAN
    if warning != WARNING or not (resolve_review_path(model_root) / PLAN).is_file():
        return False
    try:
        p1_final2_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_p1_final_review(model_root, warning):
    from .phase_saved_boundary_review import p1_final_saved_review, WARNING
    if warning != WARNING or not (resolve_review_path(model_root) / "p1_final_plan.json").is_file():
        return False
    try:
        p1_final_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def registered_l1_neighbourhood_review(model_root, warning):
    """Admit this sealed post-processing revision only through its exact policy."""
    from .phase_saved_review_metadata import saved_revision_bindings
    WARNING = saved_revision_bindings('neighbourhood_floor')._export('WARNING')
    if warning != WARNING:
        return False
    from .phase_saved_revision_review import l1_neighbourhood_saved_review
    try:
        l1_neighbourhood_saved_review(model_root)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def l1_neighbourhood_history_state_label(model_root):
    """Read the variant only from the admitted immutable plan, never its name."""
    from .phase_saved_revision_review import l1_neighbourhood_saved_review
    from .phase_saved_review_metadata import saved_revision_bindings
    PLAN = saved_revision_bindings('neighbourhood_floor')._export('PLAN')
    plan_rule_settings = saved_revision_bindings('neighbourhood_floor').plan_rule_settings

    l1_neighbourhood_saved_review(model_root)
    plan = _read_mapping(resolve_review_path(model_root) / PLAN)
    threshold = plan_rule_settings(plan)["gas_fraction_threshold"]
    variant = " 0.5" if threshold == .5 else ""
    return f"G3 gas · L1 neighbourhood rule{variant} · Floor contact · Studio review only"


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected one JSON object in {path}")
    return dict(payload)


def _model_directories(root: Path, *, max_depth: int) -> list[Path]:
    """Find model-version folders without descending into their large assets."""

    discovered: list[Path] = []
    pending: list[tuple[Path, int]] = [(root, 0)]
    ignored = {
        "input_snapshot",
        "models",
        "result_staging",
        "cache",
        ".git",
        "postflight",
        "native_phase_maps",
        "manual_pairwise_review",
        "frames",
        "movies",
    }
    while pending:
        folder, depth = pending.pop()
        try:
            names = {entry.name for entry in os.scandir(folder)}
        except OSError:
            continue
        if "summary.json" in names or "training_request.json" in names:
            discovered.append(folder)
            continue
        if depth >= max_depth:
            continue
        if depth > 2 and not any(
            "model" in part.casefold() for part in (folder.name, folder.parent.name)
        ):
            continue
        try:
            children = [
                Path(entry.path)
                for entry in os.scandir(folder)
                if entry.is_dir(follow_symlinks=False) and entry.name not in ignored
            ]
        except OSError:
            continue
        pending.extend((child, depth + 1) for child in children)
    return discovered


def _review_candidate_directories(root: Path) -> list[Path]:
    """Find model artifacts outside the shared library, regardless of QA.

    Several producers write outside the shared model library. Search their
    shallow output layouts by artifact contract rather than a single campaign
    name; readiness is shown in the catalog, not used for discovery.
    """

    if not root.is_dir():
        return []
    candidates: list[Path] = []
    for campaign in root.glob("phase_*"):
        if not campaign.is_dir():
            continue
        paths = [*campaign.glob("*/phase-model-*"),
                 *campaign.glob("*/*/phase-model-*")]
        for candidate in paths:
            if not ((candidate / "summary.json").is_file()
                    and (candidate / "models").is_dir()):
                continue
            candidates.append(candidate)
    return candidates


def _head_history(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = summary.get("models") or {}
    if not isinstance(models, Mapping):
        return rows
    for adapter, raw_model in models.items():
        if not isinstance(raw_model, Mapping):
            continue
        heads = raw_model.get("heads") or {}
        if not isinstance(heads, Mapping):
            continue
        for head, raw_head in heads.items():
            if not isinstance(raw_head, Mapping):
                continue
            final_fit = raw_head.get("final_fit") or {}
            if not isinstance(final_fit, Mapping):
                final_fit = {}
            rows.append(
                {
                    "adapter": str(adapter),
                    "head": str(head),
                    "balanced_accuracy": raw_head.get(
                        "oof_threshold_balanced_accuracy"
                    ),
                    "threshold": raw_head.get("oof_threshold"),
                    "regularization": raw_head.get("selected_regularization"),
                    "deployment_allowed": bool(
                        raw_head.get("deployment_allowed", False)
                    ),
                    "iterations": int(final_fit.get("iterations") or 0),
                    "converged": bool(final_fit.get("converged", False)),
                    "backend": str(final_fit.get("backend") or ""),
                    "precision": str(final_fit.get("precision") or ""),
                    "objective": final_fit.get("objective"),
                    "model_path": str(raw_head.get("model_path") or ""),
                    "model_sha256": str(raw_head.get("model_sha256") or ""),
                }
            )
    return rows


def _primary_metric(heads: Sequence[Mapping[str, Any]], head: str) -> float | None:
    matches = [
        row
        for row in heads
        if row.get("adapter") == "generic" and row.get("head") == head
    ]
    if not matches:
        matches = [row for row in heads if row.get("head") == head]
    value = matches[0].get("balanced_accuracy") if matches else None
    return float(value) if isinstance(value, (int, float)) else None


def _created_timestamp(value: object, fallback: float) -> float:
    text = str(value or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return fallback


def _recorded_training_time(root: Path, summary: Mapping[str, Any],
                            request: Mapping[str, Any]) -> tuple[str, str]:
    """Read the training run's recorded time, never publication or file mtime.

    Legacy trainers record either request creation or fit completion, so this
    deliberately does not claim a uniform completion-time meaning.
    """
    def valid(value: object) -> str:
        try:
            parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
            return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).isoformat()
        except ValueError:
            return ""

    def inherited(document: Mapping[str, Any]) -> bool:
        return (document.get("artifact_kind") in {"phase-support-revision", "transfer-evaluation"}
                or document.get("optical_training_performed") is False)

    for key in ("training_completed_utc", "trained_utc"):
        if value := valid(summary.get(key)):
            return value, key
    if inherited(summary):
        # Read only the frozen local snapshot, not a mutable external parent path.
        try:
            parent = _read_mapping(root / "parent_provenance" / "summary.json")
        except (OSError, TypeError, ValueError):
            parent = {}
        if not inherited(parent):
            for key in ("training_completed_utc", "trained_utc", "created_utc"):
                if value := valid(parent.get(key)):
                    return value, f"parent_provenance/summary.json:{key}"
        for document in (summary, parent):
            match = re.search(r"(?<!\d)(\d{8}T\d{6}Z)(?!\d)",
                              str(document.get("inherited_optical_model_id") or ""))
            if match:
                try:
                    value = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    return value.isoformat(), "inherited_optical_model_id"
                except ValueError:
                    pass
        return "", "not recorded"
    for document, source in ((summary, "summary.created_utc"), (request, "request.created_utc")):
        if value := valid(document.get("created_utc")):
            return value, source
    return "", "not recorded"


def _full_head_retraining(
    root: Path, summary: Mapping[str, Any], request: Mapping[str, Any] | None = None,
) -> bool:
    """Read explicit fit provenance without reclassifying frozen support revisions."""
    if summary.get("artifact_kind") in {
        "phase-support-revision", "transfer-evaluation", "gas-head-retraining-candidate",
        "optical-region-training-candidate",
    }:
        return False
    if request is None:
        try:
            request = _read_mapping(Path(root) / "training_request.json")
        except (OSError, TypeError, ValueError):
            request = {}
    settings = request.get("settings") or {}
    metadata = [summary, settings if isinstance(settings, Mapping) else {}]
    heads = {"co2_presence", "gas_within_co2"}

    def names(value):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return set()
        return {str(name).rsplit("/", 1)[-1] for name in value}

    if any(row.get("inherited_heads") for row in metadata):
        return False
    if any(row.get("inherited_heads") == [] and heads <= names(row.get("fit_heads"))
           for row in metadata):
        return True
    imported = set().union(*(names(row.get("imported_completed_heads")) for row in metadata))
    return heads <= imported


def _validation_only_publication_warning(root: Path) -> str:
    """Keep a failed publication visible as failed in every Studio model view."""
    from .phase_saved_registration import registration_warning

    registered_warning = registration_warning(root)
    if registered_warning:
        return registered_warning
    receipt_path = Path(root) / "publication_receipt.json"
    if not receipt_path.is_file():
        return ""
    try:
        receipt = _read_mapping(receipt_path)
    except (OSError, TypeError, ValueError):
        return "QA publication receipt is unreadable; do not use this model for deployment."
    if receipt.get("publication_scope") != "studio_validation_only":
        return ""
    decision_name = receipt.get("failed_qa_decision_name") or "candidate_qa_decision.json"
    if decision_name not in {"candidate_qa_decision.json", "qa_decision_unpublished.json"}:
        return "QA FAILED: validation-only evidence is incomplete or changed; do not deploy."
    path = Path(root) / "validation_only_qa" / decision_name
    try:
        decision = _read_mapping(path)
        common = (receipt.get("format") == "regional-review-publication-v1"
                  and receipt.get("qa_status") == "fail"
                  and receipt.get("scientific_acceptance") is False
                  and receipt.get("deployment_allowed") is False
                  and receipt.get("explicit_failed_qa_review_exception") is True
                  and decision.get("scientific_acceptance") is False
                  and receipt.get("model_id") == decision.get("candidate_model_id") == Path(root).name
                  and receipt.get("failed_qa_decision_sha256") == sha256_file(path))
        if decision_name == "qa_decision_unpublished.json":
            if decision.get("format") == "ffum-fast-regional-review-candidate-qa-decision-v1":
                summary = _read_mapping(Path(root) / "summary.json")
                evidence = receipt.get("qa_evidence_files") or {}
                quant = Path(decision.get("quant_report_path") or "").name
                visual = Path(decision.get("visual_report_path") or "").name
                bound = (common
                         and summary.get("inventory", {}).get("datasets") == ["ffum"]
                         and decision.get("status") == "fail_unpublished"
                         and decision.get("studio_publication") is False
                         and receipt.get("model_sha256") ==
                             decision.get("candidate_model_sha256")
                         and receipt.get("runs") == decision.get("runs_checked") == 13
                         and receipt.get("frames") == decision.get("frames_checked") == 261
                         and receipt.get("source_files", {}).get(
                             "review_source/postflight_manifest.json") ==
                             decision.get("source_manifest_sha256")
                         and receipt.get("source_files", {}).get(
                             "native_source_receipt.json") ==
                             decision.get("native_source_receipt_sha256")
                         and {"run19_late_sidewall_false_aqueous",
                              "run20_late_sidewall_false_aqueous",
                              "run16_late_aqueous_positive_lost_all",
                              "run11_gas_negative_region_increase"} <=
                             set(decision.get("blocking_findings") or [])
                         and any(Path(name).name == quant and
                                 sha256_file(Path(root) / "validation_only_qa" / name) == digest
                                 for name, digest in evidence.items())
                         and any(Path(name).name == visual and
                                 sha256_file(Path(root) / "validation_only_qa" / name) == digest
                                 for name, digest in evidence.items()))
            else:
                bound = (common
                         and decision.get("format") == "ffac-review-candidate-qa-decision-v1"
                         and decision.get("status") == "fail_unpublished"
                         and decision.get("studio_publication") is False
                         and isinstance(receipt.get("model_sha256"), str)
                         and len(receipt["model_sha256"]) == 64)
        else:
            bound = (common
                     and decision.get("format") == "ffac-regional-review-candidate-qa-decision-v1"
                     and decision.get("status") == "fail"
                     and decision.get("studio_publication_allowed") is False
                     and receipt.get("model_sha256") == decision.get("candidate_model_sha256"))
    except (OSError, TypeError, ValueError):
        bound = False
    if not bound:
        return "QA FAILED: validation-only evidence is incomplete or changed; do not deploy."
    if decision_name == "qa_decision_unpublished.json":
        if decision.get("format") == "ffum-fast-regional-review-candidate-qa-decision-v1":
            return ("QA FAILED: FF_UM RUN19 and RUN20 false aqueous CO2 sidewall lines, "
                    "RUN16 late aqueous-positive loss, and RUN11 gas-negative increase. "
                    "Studio validation only; scientific acceptance "
                    "is false and deployment is not approved. "
                    "See validation_only_qa/qa_decision_unpublished.json.")
        return ("QA FAILED: false aqueous CO2 sidewall strips in AC51 and lost gas-positive "
                "regions. Studio validation only; scientific acceptance is false and "
                "deployment is not approved. See validation_only_qa/qa_decision_unpublished.json.")
    return ("QA FAILED: late AC51 broad speckled gas persists. Studio validation only; "
            "scientific acceptance is false and deployment is not approved. "
            "See validation_only_qa/candidate_qa_decision.json.")


def support_review_warning(root: Path, summary: Mapping[str, Any] | None = None) -> str:
    publication_warning = _validation_only_publication_warning(root)
    from .phase_blind_review_policy import effective_publication_warning
    publication_warning = effective_publication_warning(root, publication_warning)
    calibration_warnings = [publication_warning] if publication_warning else []
    # A provisional RAW Track-1 candidate must carry its unresolved optical
    # checks into every Studio picker, including an early partial map set.
    try:
        request_path = Path(root) / "training_request.json"
        request = _read_mapping(request_path) if request_path.is_file() else {}
        photometry = (request.get("settings") or {}).get("v5_raw_photometry") or {}
        if photometry.get("provisional_studio_training") is True:
            warning_path = Path(photometry.get(
                "provisional_photometry_warning_receipt_path") or "")
            warning_sha = photometry.get(
                "provisional_photometry_warning_receipt_sha256")
            if (warning_path.is_file() and warning_sha
                    and sha256_file(warning_path) == warning_sha):
                decision = _read_mapping(warning_path)
                if (decision.get("whole_run_B_H_pass") is True
                        and decision.get("donor_tile_cv_status") == "failed-warning"
                        and decision.get("late_yellow_status") == "failed-warning"
                        and decision.get("weak_phase_status") == "pending-diagnostic"):
                    calibration_warnings.append(
                        "PROVISIONAL STUDIO REVIEW ONLY: four whole-run start/light holdouts passed; "
                        "4×4 donor-tile CV and the late-yellow comparison failed. "
                        "Weak-phase and late-frame diagnostics are pending. "
                        "Supported sand only; unresolved pixels are excluded. "
                        "Not for mass analysis or deployment.")
                else:
                    calibration_warnings.append(
                        "PROVISIONAL Track-1 warning receipt is incomplete; "
                        "do not use this model for mass analysis or deployment.")
            else:
                calibration_warnings.append(
                    "PROVISIONAL Track-1 warning receipt is missing or changed; "
                    "do not use this model for mass analysis or deployment.")
    except (OSError, TypeError, ValueError, AttributeError):
        calibration_warnings.append(
            "PROVISIONAL Track-1 warning evidence cannot be read; "
            "do not use this model for mass analysis or deployment.")
    if summary is None:
        try:
            summary = _read_mapping(Path(root) / "summary.json")
        except (OSError, TypeError, ValueError):
            summary = {}
    gas_retraining = summary.get("artifact_kind") == "gas-head-retraining-candidate"
    full_retraining = _full_head_retraining(root, summary)
    native_refinement = (Path(root) / "phase_native_reference.json").is_file()
    unified = (Path(root) / "phase_decision.json").is_file()
    if full_retraining:
        calibration_warnings.append(
            "Both optical classifiers were retrained; CO2-presence and gas predictions may change. "
            "Human validation is required; physical support does not establish independent accuracy."
        )
    elif summary.get("artifact_kind") == "optical-region-training-candidate":
        if summary.get("regional_training_performed") is True:
            calibration_warnings.append(
                "Optical classifier heads were inherited, but regional color and gas confirmation "
                "were fitted; final gas labels can change. Human review required; no independent "
                "validation or concentration calibration."
            )
        else:
            calibration_warnings.append(
                "Trained baseline-relative color regions; raw pixel classifier and gas labels retained. "
                "Human review required; no independent validation or concentration calibration."
            )
    elif gas_retraining:
        calibration_warnings.append(
            "The gas classifier was retrained; gas predictions may change. "
            "Test candidate requiring human validation; independent accuracy has not been established."
        )
    elif unified:
        calibration_warnings.append(
            "Unified phase decision with inherited model weights. Final gas labels can change "
            "where the accepted CO2 region changes; inherited gas evidence is not independently validated."
        )
    elif native_refinement:
        calibration_warnings.append(
            "Native-resolution edge refinement is enabled for the aqueous CO2 boundary. "
            "Optical weights and gas labels remain unchanged. Development candidate awaiting review; "
            "boundary accuracy is not independently validated."
        )
    if unified:
        calibration_warnings.append(
            "The following inherited reference-provider checks describe its input evidence, "
            "not the accuracy of the final unified decision."
        )
    facies = {}
    calibration_path = Path(root) / "reference_calibration" / "report.json"
    if calibration_path.is_file():
        try:
            calibration = _read_mapping(calibration_path).get("calibration", {})
            if calibration.get("enabled") is False:
                return " ".join(calibration_warnings + [
                    "Reference-gate calibration did not pass grouped checks. C2 additions are disabled; "
                    "this candidate retains C1 classifications. Inspect reference_calibration/report.json."])
            facies = (calibration.get("development_policy") or {}).get("facies") or {}
            limited = [str(label) for label, row in facies.items()
                       if isinstance(row, Mapping) and isinstance(
                           (row.get("metrics") or {}).get("positive_region_signal_detected"), (int, float))
                       and row["metrics"]["positive_region_signal_detected"] < .7]
            if limited:
                calibration_warnings.append("Weak-signal recovery remains limited in facies " + ", ".join(limited)
                    + ". Lower thresholds also accepted reviewed negatives there. "
                    "This is a development candidate, not independent validation.")
        except (OSError, TypeError, ValueError, AttributeError):
            return " ".join(calibration_warnings + [
                "Reference-gate calibration report could not be read; inspect the model folder."])
    path = Path(root) / "qa" / "report.json"
    if not path.is_file():
        return " ".join(calibration_warnings)
    try:
        report = _read_mapping(path)
        rows = list(report.get("negative_control_increases") or [])
        rows.extend(report.get("reference_negative_increases_vs_c1") or [])
        for group in (report.get("phase_specific_negative_increases") or {}).values():
            rows.extend(group or [])
        regions = {(row.get("run"), row.get("frame"), row.get("region_id"))
                   for row in rows if isinstance(row, Mapping)}
    except (OSError, TypeError, ValueError, AttributeError):
        return " ".join(calibration_warnings + [
            "Support-verification report could not be read; inspect the model folder."])
    warnings = calibration_warnings
    if regions:
        warnings.append(f"Negative-region regressions: {len(regions)} reviewed region(s) gained "
                "unsupported CO2 or phase coverage. Inspect qa/report.json before considering "
                "this candidate as a replacement; prepared maps are not approval.")
    positives = report.get("positive_regions_emptied") or []
    if positives:
        if native_refinement:
            warnings.append(f"{len(positives)} positive reviewed region(s) still have no predicted CO2. "
                            "Inspect qa/report.json for the comparison with the previous candidate.")
        elif facies:
            warnings.append(f"{len(positives)} positive reviewed region(s) still have no predicted CO2. "
                            "This count is relative to unfiltered C2 proposals, not a new loss relative to C3.")
        else:
            warnings.append(f"Detected CO2 disappeared from {len(positives)} positive reviewed region(s). "
                            "Inspect the sensitivity loss in qa/report.json; prepared maps are not approval.")
    return " ".join(warnings)


def reference_gate_evidence(root: Path) -> dict[str, Any]:
    path = Path(root) / "reference_calibration" / "evidence.json"
    if not path.is_file():
        return {}
    try:
        evidence = _read_mapping(path)
        regions = evidence.get("regions") or []
        return {"path": str(path), "regions": len(regions) if regions else evidence.get("calibration_regions"),
                "new_regions": sum(row.get("evidence_group") != "inherited" for row in regions),
                "review_sessions": len(evidence.get("reviews") or []),
                "independent_validation": False}
    except (OSError, TypeError, ValueError, AttributeError):
        return {"path": str(path), "error": "Reference calibration evidence could not be read"}


def _physics_run_ids(profile: Mapping[str, Any] | None) -> list[str]:
    """Read contexts from the profile's own validated source contract."""
    if not profile:
        return []
    runs = profile.get("runs") or {}
    if profile.get("format") == "ffac-rig-camera-physics-v1":
        runtime = profile.get("_rig_camera_runtime") or {}
        manifest = runtime.get("manifest") or {}
        sources = manifest.get("sources")
        if sources is not None:
            return sorted(f"ffac/{source['run']}" for source in sources)
        runs = (profile.get("inherited_public_profile") or {}).get("runs") or {}
    return sorted(runs)


def inspect_phase_model_history_entry(root: Path) -> dict[str, Any]:
    """Summarize one immutable training attempt without changing any files."""
    if _CATALOG_METADATA_ONLY.get():
        from .phase_model_catalog import inspect_catalog_entry
        return inspect_catalog_entry(root)

    model_root = resolve_review_path(root)
    request_path = model_root / "training_request.json"
    summary_path = model_root / "summary.json"
    request: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    errors: list[str] = []
    if request_path.is_file():
        try:
            request = _read_mapping(request_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Training request: {type(exc).__name__}: {exc}")
    if summary_path.is_file():
        try:
            summary = _read_mapping(summary_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Summary: {type(exc).__name__}: {exc}")

    bundle: dict[str, Any] = {}
    if summary:
        try:
            bundle = inspect_phase_model_bundle(model_root)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Model bundle: {type(exc).__name__}: {exc}")

    inventory = summary.get("inventory") or {}
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    physics_profile = None
    try:
        physics_profile = load_physics_profile(model_root)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Phase physics: {type(exc).__name__}: {exc}")
    from darsia.fluidflower.phase_native_reference import VERSION as NATIVE_REFERENCE_VERSION
    native_refinement = bool(physics_profile and physics_profile.get("format") == NATIVE_REFERENCE_VERSION)
    gas_retraining = summary.get("artifact_kind") == "gas-head-retraining-candidate"
    full_retraining = _full_head_retraining(model_root, summary, request)
    if not isinstance(inventory, Mapping):
        inventory = {}
    heads = _head_history(summary)
    production_heads = [
        row for row in heads if "no-darsia-change" not in str(row.get("adapter"))
    ]
    internal_holdout_passed = bool(production_heads) and all(
        bool(row.get("deployment_allowed")) for row in production_heads
    )
    deployment = summary.get("deployment")
    human_validation = summary.get("human_validation")
    evaluation = summary.get("evaluation")
    legacy = not bool(request)
    publication_warning = _validation_only_publication_warning(model_root)
    validation_only = bool(publication_warning.startswith("QA FAILED"))
    approved_g3_review = publication_warning.startswith("Studio review candidate: approved G3")
    seam_review = registered_old_domain_seam_review(model_root, publication_warning)
    combined_l1_review = registered_combined_l1_review(model_root, publication_warning)
    t3_review = registered_t3_review(model_root, publication_warning)
    t4_review = registered_t4_review(model_root, publication_warning)
    t4b_review = registered_t4b_review(model_root, publication_warning)
    t4c_review = registered_t4c_review(model_root, publication_warning)
    p1_final3_review = registered_p1_final3_review(model_root, publication_warning)
    p1_final2_review = registered_p1_final2_review(model_root, publication_warning)
    p1_final_review = registered_p1_final_review(model_root, publication_warning)
    l1_neighbourhood_review = registered_l1_neighbourhood_review(model_root, publication_warning)
    top_surface_review = approved_g3_review and (
        ((summary.get("approved_g3_production") or {}).get("domain_revision") or {}).get("format")
        == "ffac-photo-sand-top-revision-v1"
    )
    barrier_rule_b_review = approved_g3_review and (
        ((summary.get("approved_g3_production") or {}).get("barrier_rule") or {}).get("format")
        == "ffac-gas-barrier-rule-b-v1"
    )
    if errors:
        state = "invalid"
        state_label = "Invalid or incomplete output"
    elif validation_only:
        state = "qa_failed_validation_only"
        state_label = "QA FAILED · Studio validation only"
    elif summary.get("p1_final3") is True and not p1_final3_review:
        state = "incomplete_saved_revision"
        state_label = "Incomplete P1-final3 attempt"
    elif summary.get("p1_final2") is True and summary.get("p1_final3") is not True and not p1_final2_review:
        state = "incomplete_saved_revision"
        state_label = "Incomplete P1-final2 attempt"
    elif summary.get("p1_final") is True and summary.get("p1_final2") is not True and summary.get("p1_final3") is not True and not p1_final_review:
        state = "incomplete_saved_revision"
        state_label = "Incomplete P1-final attempt"
    elif seam_review:
        state = "human_validation_required"
        state_label = "Old domain seam corrected · Studio review only"
    elif combined_l1_review:
        state = "human_validation_required"
        state_label = "T2 at L1 · Floor contact · Studio review only"
    elif t3_review:
        state = "human_validation_required"
        state_label = "G3-T3 gas · Floor contact · Studio review only"
    elif p1_final3_review:
        from .phase_saved_contact_review import STATE_LABEL
        state = "human_validation_required"
        state_label = STATE_LABEL
    elif p1_final2_review:
        from .phase_saved_categorical_review import STATE_LABEL
        state = "human_validation_required"
        state_label = STATE_LABEL
    elif p1_final_review:
        from .phase_saved_boundary_review import STATE_LABEL
        state = "human_validation_required"
        state_label = STATE_LABEL
    elif t4c_review:
        from .phase_saved_floor_review import t4c_saved_review
        state = "human_validation_required"
        state_label = t4c_saved_review(model_root)["state_label"]
    elif t4b_review:
        state = "human_validation_required"
        state_label = "G3-T4b gas · Joint with lamp shift · Floor contact · Studio review only"
    elif t4_review:
        state = "human_validation_required"
        state_label = "G3-T4 gas · Separate light epochs · Floor contact · Studio review only"
    elif l1_neighbourhood_review:
        state = "human_validation_required"
        state_label = l1_neighbourhood_history_state_label(model_root)
    elif approved_g3_review:
        state = "human_validation_required"
        state_label = ("Barrier rule B · Studio review only" if barrier_rule_b_review
                       else "Top-sand boundary corrected · Studio review only" if top_surface_review
                       else "G3 production parity passed · Studio review only")
    elif not summary:
        state = "training_incomplete"
        state_label = "Training incomplete"
    elif summary.get("artifact_kind") == "transfer-evaluation":
        state = "transfer_evaluation"
        state_label = "Transfer evaluation (unchanged weights)"
    elif summary.get("artifact_kind") == "optical-region-training-candidate":
        state = "human_validation_required"
        state_label = "Trained color regions; human review required"
    elif summary.get("artifact_kind") == "single-run-training-candidate":
        state = "human_validation_required"
        state_label = "Single-run candidate; independent validation required"
    elif gas_retraining:
        state = "human_validation_required"
        state_label = "Gas retraining test candidate; human validation required"
    elif summary.get("artifact_kind") == "phase-support-revision":
        state = "human_validation_required"
        state_label = ("Unified phase decision; inherited model weights; review required"
                       if physics_profile and "unified_phase_decision" in physics_profile else
                       "Aqueous boundary revision; inherited optical weights; review required"
                       if physics_profile and "boundary_refinement" in physics_profile else
                       "Aqueous edge refinement; unchanged optical weights and gas labels; review required"
                       if native_refinement else
                       "Reference-gate revision; unchanged optical weights; review required"
                       if summary.get("reference_gate_fitted") else
                       "Temporal support revision; unchanged weights; review required")
    elif (summary.get("artifact_kind") == "coupled-aqueous-training-candidate"
          and isinstance(evaluation, Mapping)
          and evaluation.get("kind") == "training-only"):
        state = "human_validation_required"
        state_label = "Full optical retraining candidate; independent validation required"
    elif legacy:
        state = "historical"
        state_label = "Historical model (read-only)"
    elif not internal_holdout_passed:
        state = "internal_validation_failed"
        state_label = "Internal validation failed"
    elif not human_validation:
        state = "human_validation_required"
        state_label = ("Full optical retraining candidate; human validation required"
                       if full_retraining else "Human validation required")
    elif not deployment:
        state = "validated_not_active"
        state_label = "Validated, not active"
    else:
        state = "active"
        state_label = "Active production model"

    ablation: dict[str, Any] = {}
    ablation_path = model_root / "change_evidence_ablation.json"
    if ablation_path.is_file():
        try:
            ablation = _read_mapping(ablation_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Change ablation: {type(exc).__name__}: {exc}")

    fallback_timestamp = max(
        (path.stat().st_mtime for path in (summary_path, request_path) if path.is_file()),
        default=model_root.stat().st_mtime,
    )
    created_utc = str(
        summary.get("created_utc") or request.get("created_utc") or ""
    )
    training_recorded_utc, training_time_source = _recorded_training_time(model_root, summary, request)
    bundles = request.get("bundles") or summary.get("source_snapshot") or []
    training_bundles = (
        list(bundles)
        if isinstance(bundles, Sequence) and not isinstance(bundles, (str, bytes))
        else []
    )
    decision_spec = (physics_profile or {}).get("unified_phase_decision", {})
    regional = decision_spec.get("format") == "regional-evidence-single-contour-runtime-v1"
    if regional:
        decision_spec = decision_spec["decision"]
    decision_method_label = ("Unified phase decision v2 · Local front age"
        if decision_spec.get("format") == "unified-phase-decision-local-age-runtime-v2"
        else "Unified phase decision v1")
    optical_threshold = decision_spec.get("settings", {}).get("minimum_chromatic_change")
    if optical_threshold is not None and abs(float(optical_threshold) - .006) > 1e-12:
        decision_method_label += f" · Optical threshold {float(optical_threshold):.3f}"
    if regional:
        decision_method_label = "Regional phase · 1 mm · " + decision_method_label
    photometry_arm, photometry_stage = recorded_photometry_provenance(model_root, request, summary)
    return {
        "schema": 1,
        "format": PHASE_MODEL_HISTORY_FORMAT,
        "model_id": str(
            summary.get("campaign_id") or request.get("model_id") or model_root.name
        ),
        "root": str(model_root),
        "iteration": str(request["settings"].get("iteration") or "")
        if isinstance(request.get("settings"), Mapping) else "",
        "photometry_arm": photometry_arm,
        "photometry_stage": photometry_stage,
        "created_utc": created_utc,
        "training_recorded_utc": training_recorded_utc,
        "training_time_source": training_time_source,
        "created_timestamp": _created_timestamp(created_utc, fallback_timestamp),
        "created_by": str(request.get("created_by") or "Not recorded"),
        "historical_format": legacy,
        "state": state,
        "state_label": state_label,
        "publication_scope": "studio_validation_only" if validation_only or approved_g3_review or seam_review or combined_l1_review or t3_review or t4_review or t4b_review or t4c_review or p1_final_review or p1_final2_review or p1_final3_review or l1_neighbourhood_review else "",
        "qa_status": ("fail" if validation_only else "registered_complete_review" if seam_review or combined_l1_review or t3_review or t4_review or t4b_review or t4c_review or p1_final_review or p1_final2_review or p1_final3_review or l1_neighbourhood_review
                      else "passed_barrier_rule_b" if barrier_rule_b_review
                      else "passed_top_surface_clip" if top_surface_review
                      else "passed_production_parity" if approved_g3_review else ""),
        "display_name": str(summary.get("display_name") or ""),
        "p1_final_revision": summary.get("p1_final") is True,
        "p1_final_registered_complete": p1_final_review,
        "p1_final2_revision": summary.get("p1_final2") is True,
        "p1_final2_registered_complete": p1_final2_review,
        "p1_final3_revision": summary.get("p1_final3") is True,
        "p1_final3_registered_complete": p1_final3_review,
        "area_only_revision": summary.get("area_only_revision") is True,
        "physical_scale_approved": summary.get("physical_scale_approved") is True,
        "approved_g3_production": summary.get("approved_g3_production"),
        "support_review_warning": support_review_warning(model_root, summary),
        "gas_head_retraining": gas_retraining,
        "full_head_retraining": full_retraining,
        "native_boundary_refinement": native_refinement,
        "reference_gate_fitted": bool(summary.get("reference_gate_fitted")),
        "reference_gate_evidence": reference_gate_evidence(model_root),
        "reference_calibration_enabled": summary.get("reference_calibration_enabled"),
        "reference_calibration_reason": summary.get("reference_calibration_reason"),
        "evaluation_parent": summary.get("evaluation_parent"),
        "evaluation_scope": summary.get("evaluation_scope"),
        "internal_holdout_passed": internal_holdout_passed,
        "human_validation": human_validation,
        "deployment": deployment,
        "datasets": list(bundle.get("datasets") or inventory.get("datasets") or []),
        "runs": list(bundle.get("runs") or inventory.get("runs") or []),
        "region_count": int(inventory.get("regions") or 0),
        "series_scope": inventory.get("series_scope"),
        "training_parent": (request.get("settings") or {}).get("parent_model"),
        "physics_inheritance_receipt": str(model_root / "physics_inheritance.json")
        if (model_root / "physics_inheritance.json").is_file() else None,
        "radiometry_profile": summary.get("radiometry_profile"),
        "radiometry_label": (
            "Uniform24 / " + str(summary["radiometry_profile"]["sha256"])[:16]
            if summary.get("radiometry_profile") else "Legacy / untracked prepared images"
        ),
        "pixel_rows": int(inventory.get("pixel_rows") or 0),
        "phase_classes": dict(inventory.get("phase_classes") or {}),
        "physics_runtime": physics_runtime_identity(physics_profile),
        "aqueous_context_completion": bool(physics_profile and physics_profile.get("aqueous_context_completion")),
        "unified_phase_decision": bool(physics_profile and physics_profile.get("unified_phase_decision")),
        "decision_method_label": decision_method_label,
        "physics_label": (("Trained color regions and regional gas confirmation; gas labels can change"
                           if summary.get("regional_training_performed") is True else
                           "Trained color change and connected regions; gas labels retained")
                          if summary.get("artifact_kind") == "optical-region-training-candidate" else
                          ("Unified phase decision v2 · Local front age (review required)"
                          if physics_profile["unified_phase_decision"].get("format") == "unified-phase-decision-local-age-runtime-v2"
                          else "Unified phase decision v1 (review required)")
                          if physics_profile and physics_profile.get("unified_phase_decision") else
                          "Variant B aqueous completion with physical support (review required)"
                          if physics_profile and physics_profile.get("aqueous_context_completion") else
                          "Inherited physical support (both optical classifiers retrained)"
                          if full_retraining and physics_profile and (model_root / "physics_inheritance.json").is_file() else
                          "Optical retraining with physical support (review required)"
                          if full_retraining and physics_profile else
                          "Inherited physical support (gas retraining test candidate)"
                          if gas_retraining and physics_profile else
                          "Native-resolution aqueous edge refinement (review required)"
                          if native_refinement else "Optical + physical support (manual validation required)"
                          if physics_profile else "Optical only"),
        "physics_runs": _physics_run_ids(physics_profile),
        "adapters": list(bundle.get("adapters") or []),
        "change_detector": str(bundle.get("change_detector") or "Not available"),
        "requested_backend": str(summary.get("backend_requested") or ""),
        "heads": heads,
        "co2_balanced_accuracy": _primary_metric(heads, "co2_presence"),
        "gas_balanced_accuracy": _primary_metric(heads, "gas_within_co2"),
        "change_ablation_delta": ablation.get("mean_balanced_accuracy_delta"),
        "training_bundles": training_bundles,
        "errors": errors,
    }


def discover_phase_model_history(
    library_root: Path,
    *,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    """Return newest-first model history from a user-selected model library."""

    root = Path(library_root).expanduser().resolve()
    from .phase_model_library import records_for_location, deduplicate_versions, unavailable_history_entry, attach_library_metadata
    library_records = records_for_location(root)
    directories = (_model_directories(root, max_depth=max(0, int(max_depth)))
                   if root.is_dir() else [])
    directories = [Path(record["root"]) for record in library_records if record["available"]] + directories
    directories.extend(Path(p) for p in registered_phase_model_defaults().values())
    registration_available = registration_context_available()
    if registration_available:
        from .phase_saved_registration import registered_roots
        directories.extend(registered_roots())
    review_root = str(os.environ.get("DARSIA_FLUIDFLOWER_PHASE_MODEL_REVIEW_ROOT") or "").strip()
    if review_root:
        directories.extend(_review_candidate_directories(Path(review_root).expanduser()))
    if root.name.casefold() in {"phase_models", "shared phase models"}:
        try:
            siblings = [
                Path(entry.path)
                for entry in os.scandir(root.parent)
                if entry.is_dir(follow_symlinks=False)
                and Path(entry.path, "summary.json").is_file()
                and Path(entry.path, "models").is_dir()
            ]
        except OSError:
            siblings = []
        directories.extend(siblings)
    valid_targets = {os.path.normcase(os.path.normpath(record["root"])) for record in library_records if record["available"]}
    blocked = {os.path.normcase(os.path.normpath(record["target_root"])) for record in library_records
               if not record["available"] and record.get("target_root")} - valid_targets
    inspector = (inspect_phase_model_history_entry if registration_available
                 else metadata_only_catalog(inspect_phase_model_history_entry))
    entries = [inspector(path)
               for path in deduplicate_versions(directories)
               if os.path.normcase(os.path.normpath(str(path))) not in blocked]
    entries.extend(unavailable_history_entry(record) for record in library_records if not record["available"])
    attach_library_metadata(entries, library_records)
    return ModelHistoryEntries(sorted(
        entries,
        key=lambda row: (
            float(row.get("created_timestamp") or 0.0),
            str(row.get("model_id") or ""),
        ),
        reverse=True,
    ), diagnostics=(() if registration_available else (REGISTRATION_ROOTS_UNAVAILABLE,)))


__all__ = [
    "PHASE_MODEL_HISTORY_FORMAT",
    "deduplicate_model_roots",
    "discover_phase_model_history",
    "inspect_phase_model_history_entry",
    "project_result_model_roots",
]
