"""Small-metadata dropdown entries, with saved-map admission deferred to selection."""
from __future__ import annotations
import json
from pathlib import Path

from .review_storage import resolve_review_path


def indexed_catalog_snapshot(series_roots):
    """Show recorded index names before opening any remote model or its assets.

    These are pending choices, never verified/admitted models. The ordinary
    catalogue and selected-model admission replace this first presentation.
    """
    from .phase_model_library import FORMAT, library_root, _target
    entries = {}
    for series_root in series_roots:
        try:
            references = sorted(library_root(series_root).glob("*.json"))
        except OSError:
            continue
        for reference in references:
            try:
                saved = _read(reference)
                if saved.get("format") != FORMAT or saved.get("kind") != "phase":
                    continue
                target = _target(saved, Path(series_root))
                identity, metadata = saved["source_identity"], saved.get("metadata") or {}
                entries.setdefault(str(target), dict(root=str(target), model_id=identity["model_id"],
                    display_name=metadata.get("display_name", ""),
                    created_utc=metadata.get("created_utc", ""), created_timestamp=metadata.get("created_timestamp", 0),
                    datasets=metadata.get("datasets", []), series_scope=metadata.get("series", ""),
                    library_reference=str(reference), library_source_identity=identity,
                    catalog_admission_pending=True, validation_binding=None, validation_binding_loading=True,
                    state="catalog_verification_pending", state_label="Review readiness verification pending"))
            except (OSError, KeyError, TypeError, ValueError):
                continue  # The complete catalogue retains its original error reporting.
    return list(entries.values())


def _read(path):
    value = json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one saved metadata object: {path}")
    return value


def saved_studio_review_source(model_root):
    """Resolve the selected-camera package's own saved presentation pointer.

    This locates an existing package only. The caller must pass that package
    through the ordinary saved binding, pipeline and reference readers.
    """
    from .phase_validation import _model_identity
    root = resolve_review_path(model_root)
    path = root / "studio_preparation.json"
    if not path.is_file():
        return None
    preparation = _read(path)
    if preparation.get("selected_camera_review") is not True:
        return None
    if (preparation.get("status") != "ready" or preparation.get("partial") is not False
            or preparation.get("model_identity") != list(_model_identity(root))):
        raise ValueError("Saved Studio preparation is incomplete or identifies another model")
    source = resolve_review_path(preparation.get("validation") or "")
    if source.parent != root / "selected_studio/validation/prepared_sets":
        raise ValueError("Saved Studio preparation points outside this model's prepared sets")
    return source


def inspect_selected_catalog_entry(selected):
    """Inspect just the chosen version, retaining the index's exact-source check."""
    from .phase_model_library import resolve_reference
    root = resolve_review_path(selected["root"])
    reference = selected.get("library_reference")
    if reference:
        indexed_root = resolve_reference(reference)
        if indexed_root != root:
            raise ValueError("The selected library reference now identifies another model folder")
    entry = inspect_catalog_entry(root)
    if reference:
        entry.update(library_reference=reference, library_available=True,
                     library_source_identity=_read(reference)["source_identity"])
    return entry


def _registration(root):
    """Read a small saved claim, without parsing/rehashing its scientific plans."""
    from .phase_model_library import _metadata_sha
    from .phase_validation import _model_identity
    for parent in root.parents:
        if (parent / "series.toml").is_file():
            path = resolve_review_path(parent / "results/phase_model_review_registrations" / f"{root.name}.json")
            if not path.is_file():
                return {}
            value = _read(path)
            identity = _model_identity(root)
            if (value.get("model_id"), value.get("model_sha256")) != identity:
                raise ValueError("Saved registration claims another model identity")
            # Validate cheap source metadata, while explicitly deferring large
            # plans, dense maps, runtime and source admission to selection.
            for name in ("summary.json", "training_request.json"):
                expected = (value.get("metadata_sha256") or {}).get(name)
                if expected is not None and _metadata_sha(root / name) != expected:
                    raise ValueError(f"Saved registration metadata changed: {name}")
            return value
    return {}


def inspect_catalog_entry(root):
    from .phase_learning import inspect_phase_model_bundle
    from .phase_model_history import (_head_history, _primary_metric, _recorded_training_time,
                                     PHASE_MODEL_HISTORY_FORMAT)
    from .phase_model_library import _metadata
    from .model_display import recorded_photometry_provenance
    root = resolve_review_path(root)
    errors, summary, request, bundle, registration = [], {}, {}, {}, {}
    for name, target in (("summary.json", summary), ("training_request.json", request)):
        path = root / name
        if path.is_file():
            try:
                target.update(_read(path))
            except (OSError, TypeError, ValueError) as error:
                errors.append(f"{name}: {error}")
    if summary:
        try:
            bundle = inspect_phase_model_bundle(root)
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"Model bundle: {error}")
    else:
        errors.append("Model summary is missing; incomplete training attempt")
    try:
        registration = _registration(root)
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"Saved registration: {error}")
    metadata = _metadata(root, series="")
    inventory = summary.get("inventory") or {}
    heads = _head_history(summary)
    training_time, training_time_source = _recorded_training_time(root, summary, request)
    photometry_arm, photometry_stage = recorded_photometry_provenance(root, request, summary)
    preparation = {}
    if (root / "studio_preparation.json").is_file():
        try:
            preparation = _read(root / "studio_preparation.json")
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"Saved preparation: {error}")
    publication = {}
    if (root / "publication_receipt.json").is_file():
        try:
            publication = _read(root / "publication_receipt.json")
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"Saved publication: {error}")
    p1_final3 = summary.get("p1_final3") is True
    p1_final3_registered = registration.get("registration_kind") == "ffac_g3_t4c_p1_final3_review"
    selected_camera_registered = registration.get("registration_kind") == "ffac_selected_camera_policy_a_p1_final3_review"
    p1_final2 = summary.get("p1_final2") is True
    p1_final2_registered = registration.get("registration_kind") == "ffac_g3_t4c_p1_final2_review"
    p1_final = summary.get("p1_final") is True
    p1_registered = registration.get("registration_kind") == "ffac_g3_t4c_p1_final_review"
    if errors:
        state, label = "invalid", "Invalid or incomplete output"
    elif p1_final3 and not (p1_final3_registered or selected_camera_registered):
        state, label = "incomplete_saved_revision", "Incomplete P1-final3 attempt"
    elif p1_final2 and not p1_final3 and not p1_final2_registered:
        state, label = "incomplete_saved_revision", "Incomplete P1-final2 attempt"
    elif p1_final and not p1_final2 and not p1_final3 and not p1_registered:
        state, label = "incomplete_saved_revision", "Incomplete P1-final attempt"
    elif publication.get("qa_status") == "fail":
        state, label = "qa_failed_validation_only", "QA FAILED · Studio validation only"
    elif summary.get("artifact_kind") == "transfer-evaluation":
        state, label = "transfer_evaluation", "Transfer evaluation · Review verification pending"
    else:
        state, label = "catalog_verification_pending", "Review readiness verification pending"
    warning = str(registration.get("warning") or publication.get("warning") or "")
    if publication.get("qa_status") == "fail" and not warning:
        warning = "QA FAILED: saved Studio validation-only candidate"
    if preparation and preparation.get("status") != "ready":
        warning = (warning + " · " if warning else "") + f"Saved preparation: {preparation.get('status', 'incomplete')}"
    return {"schema": 1, "format": PHASE_MODEL_HISTORY_FORMAT,
            "model_id": str(summary.get("campaign_id") or summary.get("model_id") or request.get("model_id") or root.name),
            "root": str(root), "created_utc": metadata["created_utc"], "created_timestamp": metadata["created_timestamp"],
            "training_recorded_utc": training_time, "training_time_source": training_time_source,
            "display_name": str(summary.get("display_name") or ""), "state": state, "state_label": label,
            "catalog_admission_pending": True, "saved_registration_claim": bool(registration),
            "saved_preparation_status": preparation.get("status"), "support_review_warning": warning,
            "publication_scope": registration.get("publication_scope") or publication.get("publication_scope") or "",
            "qa_status": publication.get("qa_status") or "verification_pending",
            "p1_final_revision": p1_final, "p1_final_registered_complete": False,
            "p1_final_catalog_registered": p1_registered and not errors,
            "p1_final2_revision": p1_final2, "p1_final2_registered_complete": False,
            "p1_final2_catalog_registered": p1_final2_registered and not errors,
            "p1_final3_revision": p1_final3, "p1_final3_registered_complete": False,
            "p1_final3_catalog_registered": p1_final3_registered and not errors,
            "selected_camera_catalog_registered": selected_camera_registered and not errors,
            "area_only_revision": summary.get("area_only_revision") is True,
            "approved_g3_production": summary.get("approved_g3_production"),
            "datasets": list(bundle.get("datasets") or inventory.get("datasets") or []),
            "series_scope": inventory.get("series_scope"), "runs": list(bundle.get("runs") or inventory.get("runs") or []),
            "iteration": str((request.get("settings") or {}).get("iteration") or ""),
            "photometry_arm": photometry_arm, "photometry_stage": photometry_stage,
            "human_validation": summary.get("human_validation"), "deployment": summary.get("deployment"),
            "region_count": inventory.get("regions") or 0, "pixel_rows": inventory.get("pixel_rows") or 0,
            "phase_classes": inventory.get("phase_classes") or {}, "heads": heads, "errors": errors,
            "co2_balanced_accuracy": _primary_metric(heads, "co2_presence"),
            "gas_balanced_accuracy": _primary_metric(heads, "gas_within_co2"),
            "training_bundles": request.get("bundles") or [], "adapters": bundle.get("adapters") or [],
            "physical_scale_approved": summary.get("physical_scale_approved") is True}
