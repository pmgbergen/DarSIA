"""New native labels on an exact photographic collection, without a fitted model."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def source_identity(binding, *, series):
    from .native_phase_review_session import canonical_source_identity
    return canonical_source_identity(binding, series=series)


def _document(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _validate_session(workspace, reviewer, reservation, identity):
    from .reviewer_preferences import reviewer_identifier
    if (workspace.key != "training" or workspace.review_surface != "blind_regions"
            or not workspace.single_model or not workspace.review_session_id
            or workspace.target_model_id):
        raise ValueError("Scratch labels require a new source-bound session without a target model")
    if (not reservation or reservation.get("status") != "active"
            or reservation.get("review_basis") != "scratch"
            or reservation.get("review_session_id") != workspace.review_session_id
            or reservation.get("target_model_id")
            or reservation.get("source_identity") != identity):
        raise ValueError("The scratch reservation belongs to another source collection or session")
    path = workspace.output_dir / "review_session.json"
    if not path.exists():
        if workspace.output_dir.exists() and any(workspace.output_dir.iterdir()):
            raise ValueError("The new scratch evidence directory is not empty")
        return
    saved = _read(path)
    if (saved.get("format") != "fluidflower-local-review-session-v1"
            or saved.get("review_basis") != "scratch" or saved.get("target_model_id")
            or saved.get("source_identity") != identity
            or source_identity(saved, series=identity["series_id"]) != identity
            or saved.get("review_session_id") != workspace.review_session_id
            or reviewer_identifier(saved.get("reviewer", "")) != reviewer_identifier(reviewer)
            or saved.get("reviewer_id") != reviewer_identifier(reviewer)
            or Path(saved.get("decision_dir", "")).resolve() != workspace.output_dir.resolve()
            or saved.get("workspace") != "training" or saved.get("review_surface") != "blind_regions"
            or saved.get("editing_locked") or saved.get("retired_workflow")
            or saved.get("include_in_training") is False or saved.get("independent_holdout")
            or saved.get("evidence_role") in {"blind_transfer_validation", "independent_blind_gas_validation"}):
        raise ValueError("The saved scratch session is locked, retired, or bound to another source")


def prepare(desktop, args, reviewer, save_root):
    """Prepare photographs and original evidence storage; never seed prior decisions."""
    from .native_phase_review_session import read_current_review_session
    from .raw_photographic_review import RAW_REVIEW_FORMAT, studio_rows
    workspace = desktop.local_reviewer_workspace(
        desktop.review_workspaces(args)["training"], save_root, reviewer)
    workspace = replace(workspace, seed_review_dir=None)
    identity = source_identity({name: getattr(workspace, name) for name in (
        "combined_root", "ffac_root", "ffum_root", "selected_candidates", "ffac_runs", "ffum_runs")},
        series=args.phase_workflow_series)
    reservation = read_current_review_session(save_root, series=args.phase_workflow_series, reviewer=reviewer)
    _validate_session(workspace, reviewer, reservation, identity)
    review = desktop.review
    inventory = review.discover_review_inventory(workspace.combined_root,
        blind_seed=args.blind_seed, allow_incomplete=True)
    available = {(candidate.run, candidate.candidate_id): candidate
        for pair in inventory.pairs for candidate in (pair.candidate_a, pair.candidate_b)}
    photos, pairs, pins = {}, [], [_document(inventory.manifest_path)]
    for run, candidate_id in identity["selected_candidates"].items():
        candidate = available.get((run, candidate_id))
        if candidate is None:
            raise ValueError(f"The selected photographic source is unavailable for {run}: {candidate_id}")
        allowed = identity.get(f"{candidate.dataset.casefold()}_runs")
        if (candidate.dataset.casefold() not in {"ffac", "ffum"}
                or candidate.dataset.casefold() != identity["series_id"]
                or allowed is not None and run not in allowed):
            raise ValueError(f"The photographic source is outside the selected experiment scope: {run}")
        root = Path(candidate.review_output_dir or workspace.combined_root)
        manifest_path = review._native_manifest_path(root, run)
        native = _read(manifest_path)
        reference, ready = review._native_photograph_availability(root, run, native, candidate.frame_count)
        if not ready:
            raise ValueError(f"The selected photographic collection is incomplete: {run}")
        rows = studio_rows(native) if native.get("format") == RAW_REVIEW_FORMAT else native.get("frame_sources", [])
        records = review._calibration_records(str(candidate.per_frame_csv))
        if (not rows or len(rows) != candidate.frame_count or len(records) != len(rows)
                or any(float(row["elapsed_h"]) != float(record["elapsed_h"])
                       for row, record in zip(rows, records))):
            raise ValueError(f"The photographic timeline differs from the saved source rows: {run}")
        pins.extend((_document(manifest_path), _document(candidate.per_frame_csv)))
        photos[run] = (native, rows, reference)
        # The original saver looks for quiz pools even for an empty answer list.
        # Give those lookups this session's empty namespace, never a source model's pool.
        candidate = replace(candidate, phase_map_output_dir=workspace.output_dir,
            raw_phase_map_output_dir=None, phase_quiz_proposals_path=None,
            phase_quiz_proposals_sha256="", presence_diagnostics_dir=None,
            phase_model_only=False, phase_support_format="", optical_model_sha256="")
        pairs.append(review.ReviewPair(run, candidate, candidate, comparison_mode="single_candidate"))
    if not pairs:
        raise ValueError("Select a photographic collection before starting new labels")
    binding_path = workspace.output_dir / "review_source_binding.json"
    binding = dict(format="native-phase-scratch-binding-v1", source_identity=identity,
        review_session_id=workspace.review_session_id, sources=pins)
    if binding_path.exists():
        if _read(binding_path) != binding:
            raise ValueError("The photographs or timeline changed after this scratch session was opened")
    elif (workspace.output_dir / "review_session.json").exists():
        raise ValueError("The saved scratch session has no exact photographic binding")
    desktop.ensure_writable_directory(workspace.output_dir)
    if not binding_path.exists():
        desktop.atomic_json(binding_path, binding)
        path = desktop.write_review_session_manifest(workspace, reviewer)
        desktop.atomic_json(path, dict(_read(path), review_basis="scratch", source_identity=identity))
    inventory = review.ReviewInventory(tuple(pairs), {}, inventory.manifest_path)
    store = review.ReviewStore(workspace.output_dir, inventory.pairs)

    class PhotographicReviewApp(review.ReviewApp):
        def pair_payload(self, run, reviewer=""):
            pair = self._pair(run)
            native, rows, reference = photos[pair.run]
            raw = native.get("format") == RAW_REVIEW_FORMAT
            from .phase_photographic_assets import raw_detail_available
            shape = native.get("native_shape_hw" if raw else "analysis_image_shape_hw", [])
            item = dict(label="Image collection", frame_count=len(rows),
                elapsed_hours=[float(row["elapsed_h"]) for row in rows], frame_metrics=[],
                plain_frames_ready=True, phase_quiz_ready=False, phase_quiz_proposals=[],
                phase_quiz_initial_count=0, phase_quiz_available_count=0,
                contour_overlays_ready=False, phase_fill_overlays_ready=False,
                reference_offsets=[], reference_frames_ready=reference.is_file(),
                canvas_wh=native["canvas_wh"], analysis_image_shape_hw=shape)
            return dict(schema=1, run=pair.run, dataset=self._dataset(pair),
                comparison_mode="single_candidate", reference_ready=reference.is_file(),
                reference_label="Pre-injection reference", native_resolution=True,
                canvas_wh=native["canvas_wh"], analysis_image_shape_hw=shape,
                content_box_xywh=native["content_box_xywh"], frame_sources=deepcopy(rows),
                original_reference_source=deepcopy(native.get("reference", {})),
                v5_raw_detail_available=(raw_detail_available(native) and raw_detail_available(native, reference=True)),
                **({"ffac_selected_policy_a_sources": native["ffac_selected_policy_a_sources"]}
                   if "ffac_selected_policy_a_sources" in native else {}),
                candidate_a=deepcopy(item), candidate_b=deepcopy(item),
                decision=self.store.decisions().get(pair.run), browser_draft=None,
                reveal={side: pair.candidate(side).provenance() for side in ("A", "B")})

    model = PhotographicReviewApp(inventory, store, default_reviewer=reviewer)
    return workspace, identity, model, store


def create_window(desktop, *args, role="label", **kwargs):
    """Use the original drawing, autosave worker, recovery, and close lifecycle."""
    from .ui.phase_model_correction import create_window as original_window

    class PhotographicWindow(desktop.CalibrationReviewWindow):
        def _quiz_preparation_allowed(self):
            return False

        def _model_phase_in_region(self, *args, **kwargs):
            return dict(model_phase_class="unscored", model_phase_distribution={},
                        model_phase_sample_count=0, model_phase_source="")

    class Presentation:
        CalibrationReviewWindow = PhotographicWindow

        def __getattr__(self, name):
            return getattr(desktop, name)

    window = original_window(Presentation(), *args, role=role, **kwargs)
    window.workflow_notice.setText(
        "Start from scratch · New labels on the selected image collection. "
        "No existing model or previous labels are used. Draw a region and select its phase.")
    window.workspace_notice.setText(
        "Label phases on the selected photographs to start a new model. "
        "The image collection provides photographs and their saved coordinates only; "
        "its model parameters, predictions, and previous labels are not used.")
    window.quiz_entry_help.setText("Draw new phase labels on these photographs. Model-selected regions require a fitted model.")
    return window
