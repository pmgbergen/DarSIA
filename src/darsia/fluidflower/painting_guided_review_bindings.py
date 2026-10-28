"""Connect guided presentation to the original typed human-action backend."""
from copy import deepcopy


class RegionPreviewSession:
    """Remember local human markings for suggestions, with no saved authority."""

    def __init__(self):
        self._templates = ()
        self._without_suggestion = frozenset()

    @staticmethod
    def _key(binding, run, component, row_sha):
        return (binding["path"], binding["sha256"], binding["size_bytes"], run, component, row_sha)

    @staticmethod
    def _common_anchors(review):
        from .painted_review_comparison import context_source_sha256

        anchors = {key: deepcopy(review[key]) for key in
                   ("source", "source_map_pixels", "geometry_index", "map_version", "painting_photograph")}
        context = review.get("display_context")
        anchors["display_source_sha256"] = None if context is None else context_source_sha256(context)
        return anchors

    @staticmethod
    def _source_anchors(review, run_id):
        rows = [row for row in review["runs"] if row["run_id"] == run_id]
        if len(rows) != 1:
            raise ValueError("The marked source run is missing from this review")
        return {key: deepcopy(rows[0][key]) for key in (
            "reference_model", "target_model", "native_photo", "native_photo_pixels",
            "native_to_painting_h", "contract")}

    def remember(self, human_request, *, review):
        """Called only for a validated confirmation or an explicit Sand request."""
        from .painted_gap_regions import validate_gap_region_snapshot

        snapshot = validate_gap_region_snapshot(human_request["snapshot"], review=review)
        if (snapshot.review_binding != human_request["review_binding"]
                or (snapshot.run_id, snapshot.component_id, snapshot.review_row_sha256) !=
                (human_request["run_id"], human_request["component_id"], human_request["review_row_sha256"])):
            raise ValueError("The suggested markings do not belong to the submitted question")
        # A corrected destination remains its own decision. Avoid a chain of
        # progressively cropped suggestions from previously transferred rows.
        if snapshot.preview_source_context is not None:
            return
        key = self._key(snapshot.review_binding, snapshot.run_id,
                        snapshot.component_id, snapshot.review_row_sha256)
        common = self._common_anchors(review)
        retained = tuple(item for item in self._templates if item[2] == common and item[0] != key)
        self._templates = (*retained, (key, snapshot.snapshot(), common,
                                      self._source_anchors(review, snapshot.run_id)))

    def suppress(self, request):
        """An explicit display reset affects this question only, never a proof."""
        key = self._key(request["review_binding"], request["run_id"],
                        request["component_id"], request["review_row_sha256"])
        self._without_suggestion = self._without_suggestion | {key}

    def reader(self, original_reader):
        """Prepare suggestions inside the original owned preview worker."""
        def read(review, run_id, component_id=None):
            from .painted_gap_regions import region_draft_factory, validate_gap_region_snapshot
            from .painted_gap_region_transfer import transfer_region_masks

            packet = original_reader(review, run_id, component_id)
            if component_id is None:
                return packet
            draft = region_draft_factory(review, packet)
            key = self._key(draft.review_binding, draft.run_id, draft.component_id, draft.review_row_sha256)
            result = dict(packet, region_draft=draft)
            if key in self._without_suggestion:
                result["region_transfer_note"] = "You chose to classify this area without a suggestion."
                return result
            targets = [row for row in review["runs"] if row["run_id"] == draft.run_id]
            if len(targets) != 1 or packet["native_to_painting_h"] != targets[0]["native_to_painting_h"]:
                raise ValueError("The suggestion needs this run's exact original map placement")
            best = None
            common = self._common_anchors(review)
            for old_key, snapshot, original_common, source_anchors in self._templates:
                if old_key == key or original_common != common:
                    continue
                if source_anchors != self._source_anchors(review, snapshot.run_id):
                    continue
                # TARGET may change a component while leaving its photographed
                # physical placement unchanged. These remain descriptive
                # suggestions from the old bound photo, never rebased approval.
                if snapshot.review_binding == draft.review_binding:
                    validate_gap_region_snapshot(snapshot, review=review)
                else:
                    from .painted_analysis_exclusions import _checked_gap_review
                    previous_review = dict(_checked_gap_review(snapshot.review_binding),
                                           review_binding=deepcopy(snapshot.review_binding))
                    validate_gap_region_snapshot(snapshot, review=previous_review)
                sources = [row for row in review["runs"] if row["run_id"] == snapshot.run_id]
                if len(sources) != 1:
                    raise ValueError("The marked source is missing from this reviewed cohort")
                projected = transfer_region_masks(snapshot.class_masks, snapshot.editable_mask,
                    sources[0]["native_to_painting_h"], draft.editable_mask,
                    targets[0]["native_to_painting_h"])
                if projected["matched_pixels"] and (best is None or projected["matched_pixels"] > best[1]["matched_pixels"]):
                    best = snapshot, projected
            if best is not None:
                snapshot, projected = best
                draft.set_preview_class_masks(dict(projected["class_masks"]), source_context=dict(
                    review=snapshot.review_binding, run_id=snapshot.run_id,
                    component_id=snapshot.component_id, review_row_sha256=snapshot.review_row_sha256))
                result["region_transfer_note"] = (
                    f"Suggested from {snapshot.run_id.upper()} using this run's own map placement. "
                    "Confirm or correct what you see in this photo."
                    + (" Some of this area is still unmarked; classify it before continuing."
                       if projected["remaining_pixels"] else ""))
            elif self._templates:
                result["region_transfer_note"] = "No marked area matches this deviation. Classify this area yourself."
            return result
        return read

    @staticmethod
    def draft_from_packet(review, packet):
        """Admit an opaque prepared draft without copying it into photo data."""
        from .painted_gap_regions import GapRegionDraft, region_draft_factory, validate_gap_region_snapshot

        draft = packet.get("region_draft")
        if draft is None:
            return region_draft_factory(review, packet)
        if not isinstance(draft, GapRegionDraft):
            raise ValueError("The prepared region draft has another type")
        validate_gap_region_snapshot(draft.snapshot(), review=review)
        if (draft.run_id != packet["run_id"] or draft.component_id != packet["component_id"]):
            raise ValueError("The prepared region draft belongs to another question")
        return draft


def validate_region_result(value, review, human_request):
    """Accept only C-validated rows containing this exact submitted human action."""
    from .painted_gap_regions import validate_gap_region_result, validate_gap_region_snapshot

    if review["review_binding"] != human_request["review_binding"]:
        raise ValueError("The classified areas belong to a different review")
    snapshot = validate_gap_region_snapshot(human_request["snapshot"], review=review)
    identity = (human_request["run_id"], human_request["component_id"],
                human_request["review_row_sha256"])
    if (snapshot.review_binding != human_request["review_binding"]
            or (snapshot.run_id, snapshot.component_id, snapshot.review_row_sha256) != identity):
        raise ValueError("The submitted pixels belong to a different classification question")
    checked = validate_gap_region_result(value, review, human_request["snapshot"],
        other_text=human_request.get("other_text", ""), previous=human_request.get("previous"))
    rows = [row for row in checked["runs"]
            if (row["run_id"], row["component_id"], row["review_row_sha256"]) == identity]
    if len(rows) != 1 or any(rows[0][key] != human_request[key]
                            for key in ("actor", "approved_utc", "reason", "decision_ref")):
        raise ValueError("The classified areas do not retain this exact human confirmation")
    return deepcopy(checked)


def guided_gap_dialog(source, *, preview_session=None, series_root=None, **options):
    """Use one presentation and one typed draft; missing support fails closed.

    ``series_root`` supplies the series' recorded water-photograph choice, if
    it names one; without it the water-photograph tool stays hidden.
    """
    from .painted_gap_regions import (
        REGION_CLASSIFICATIONS, REGION_CLASSIFICATION_LABELS,
        validate_gap_region_snapshot,
    )
    from .painted_camera_jpg_display import (
        read_gap_preview, read_comparison_preview, read_water_surface_comparison_preview,
        read_target_paint,
    )
    from .painting_deviation_step_ui import DeviationPaintingStep
    from .painting_guided_review_ui import GuidedGapReviewDialog

    session = preview_session or RegionPreviewSession()
    original_reader = options.pop("preview_reader", None) or read_gap_preview
    target_reader = options.pop("target_paint_reader", None) or read_target_paint
    options.setdefault("comparison_reader", read_comparison_preview)
    options.setdefault("water_surface_comparison_reader", read_water_surface_comparison_preview)
    if "clearwater_review_builder" not in options:
        from .clearwater_source_choice import clearwater_review_builder
        options["clearwater_review_builder"] = clearwater_review_builder(series_root)
    dialog = GuidedGapReviewDialog(source, preview_reader=session.reader(original_reader),
        deviation_editor_factory=DeviationPaintingStep,
        region_draft_factory=session.draft_from_packet,
        region_snapshot_collector=validate_gap_region_snapshot,
        region_result_validator=validate_region_result,
        class_choices=[(key, REGION_CLASSIFICATION_LABELS[key]) for key in REGION_CLASSIFICATIONS],
        sand_class="sand", **options)
    dialog.region_preview_session = session
    dialog.target_paint_reader = target_reader
    return dialog


def findings_source_context(review):
    """Capture explicit original producer inputs for the owned replay worker.

    New reviews already retain their complete producer context. The older
    recovered review has a bound current recapture. Its original writer's
    fixed artifact protocol supplies locators; the replay module still checks
    every source and native fixture against the approved painting and models.
    File names never establish a physical label or an outside-frame proof.
    """
    from pathlib import Path
    import numpy as np
    from .painted_analysis_exclusions import _bound, _checked, _same
    from .layer_painting_fixtures import read_underlay

    if review.get("preparation_context") is not None:
        return deepcopy(review["preparation_context"])
    capture_binding = review.get("source_recapture")
    if capture_binding is None:
        raise ValueError("The saved review has no bound original producer context")
    captured = _checked(capture_binding, metadata=True)
    folder = Path(review["source_map"]["path"]).parent
    if (captured.get("origin") != "CURRENT_VERIFIED_RECAPTURE"
            or captured.get("original_serialized_binding_claimed") is not False
            or Path(captured.get("preparation", "")).resolve() != folder.resolve()
            or not _same(captured.get("painting_binding"), review["source"]["painting_facies"])):
        raise ValueError("The recovered review does not retain its exact current source capture")
    result = dict(continuation=_bound(folder / "continuation.json"),
                  recapture=deepcopy(capture_binding), run_entries={})
    derivation = captured.get("fixture_analysis_derivation")
    if derivation is not None:
        original_path = (folder / "source_fixture_underlay.npz").resolve()
        descriptors = [item for item in captured["verified_inputs"]
                       if Path(item["path"]).resolve() == original_path]
        if len(descriptors) != 1:
            raise ValueError("The original capture must explicitly bind its source fixture underlay")
        result.update(source_fixture_underlay=deepcopy(descriptors[0]),
            painting_analysis_derivation=deepcopy(derivation),
            fixture_frame=deepcopy(captured["painting_binding"]["frame"]))
        for row in review["runs"]:
            # Exact prepare_adoption protocol, followed by the original native
            # underlay reader and D's independent source/projection replay.
            descriptor = _bound(folder / row["run_id"] / "fixture_underlay.npz")
            underlay = read_underlay(descriptor)
            ids = [int(value) for value in np.unique(underlay.drawn_layers) if value]
            result["run_entries"][row["run_id"]] = dict(
                fixture_underlay=descriptor, fixture_underlay_layer_ids=ids)
    return result


def dispatch_findings(controller, request):
    """Use the dialog's original finite reader for one captured collector job."""
    from datetime import datetime, timezone
    import uuid

    dialog = controller.gap_dialog
    if dialog is None or dialog.busy() or dialog._closing:
        raise ValueError("Wait for this findings view's current reader before refreshing")
    owned = deepcopy(request)
    project = controller.view.project

    def collect():
        from . import layer_painting as painting
        from .geometry_facies import GeometryRevisionStore
        from .painted_adoption_table import _checked, _same
        from .painted_analysis_exclusions import _bound
        from .painted_transfer_findings import collect_transfer_findings
        from .raw_shape_basis_reader import project_retained_source_context

        try:
            # ContextVars from placement/preparation do not cross into this
            # independent reader. Select the same explicit historical copies
            # and reviewed same-byte photo moves, retaining their final guards.
            with project_retained_source_context(project):
                store = GeometryRevisionStore(project)
                current_locator = _bound(store.current_path)
                current = _checked(current_locator, metadata=True)
                revision = _bound(store.artifact_root / current["revision_id"] / "revision.json")
                if not _same(_checked(revision, metadata=True), current):
                    raise ValueError("The current workspace is not its exact immutable revision")
                inputs = owned["input_bindings"]
                destination = (painting.painting_folder(project).parent / "transfer_findings" /
                    (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]))
                result = collect_transfer_findings(revision, inputs["review"],
                    explicit_actions={key: deepcopy(inputs[key]) for key in
                                      ("analysis_exclusions", "source_choices", "display_preferences")},
                    source_context=findings_source_context(owned["review"]),
                    target_correction_bindings=deepcopy(inputs["target_bindings"]),
                    residual_destination=destination)
                _checked(current_locator)
                if not _same(store.current(), current):
                    raise ValueError("The current workspace changed while the findings were being collected")
                result["report"]["source_bindings"]["current_workspace"] = current_locator
            return dict(result, generation=owned["generation"])
        except Exception as error:
            return dict(generation=owned["generation"], input_bindings=owned["input_bindings"],
                        error=f"{type(error).__name__}: {error}")

    dialog._start(("findings-catalogue", owned["generation"]), collect)
