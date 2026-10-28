"""Ordinary local reader for the saved painting and existing chart detections.

Only the selected run's old map is opened. RAWs, photographs and fitted
optical fields are not decoded. This reader grants no adoption or job status.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from .storage_paths import resolve_storage_path, same_storage_descriptor, same_storage_path

from .painting_chart_transfer import (_matrix, _project, _centres, _identity, _json,
    restore_chart_material, propose_chart_transfer, apply_chart_transfer, FORMAT,
    proposal_format_for_approval, review_proposal_format, DETECTED_FORMAT)

DEFAULT_DETECTION_MANIFEST = (Path.home() / "Documents" / "FF Albus" / "outputs" /
    "ffac-v5-portable-preview-20260929" / "ea_calibration43_fixedpoint" / "manifest.json")


def saved_chart_review_paths(project_root):
    """Find supported saved review layouts without reading maps or photographs."""
    root = resolve_storage_path(project_root) / 'annotations'
    paths = []
    for name in ('map_adoption', 'map_placement'):
        folder = root / name
        if not folder.is_dir():
            continue
        children = list(folder.iterdir())
        if len(children) > 4096:
            raise ValueError('Too many saved map reviews in this series.')
        for child in children:
            if not child.is_dir():
                continue
            for relative in ('gap_review.json', 'native_gap_review/gap_review.json'):
                path = child / relative
                if path.is_file():
                    paths.append(path)
    return paths


def selected_chart_session(project_root, record, selection, sources=()):
    """Resolve the selected map's saved review in the existing finite reader."""
    from .painted_analysis_exclusions import _read_scientific_gap_review
    from .painting_series_use_ui import matches_version
    candidates = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        bound = source.get('review_binding', source)
        if isinstance(bound.get('path'), str):
            candidates[bound['path']] = bound
    for path in saved_chart_review_paths(project_root):
        candidates.setdefault(str(path), None)
    ordered = sorted(candidates, key=lambda path: (resolve_storage_path(path).stat().st_mtime_ns, path), reverse=True)
    for path in ordered:
        review = _read_scientific_gap_review(path)
        expected = candidates[path]
        if expected is not None and not same_storage_descriptor(review['review_binding'], expected):
            raise ValueError('The saved ColorChecker review changed its bound source file.')
        bound = review['geometry_index']
        if (matches_version(review, record) and same_storage_path(bound['path'], selection.index_path)
                and (not getattr(selection, 'sha256', '') or (
                    bound.get('sha256') == selection.sha256
                    and bound.get('size_bytes') == selection.size_bytes))):
            source = review.get('chart_source_review')
            if source is not None:
                if (not isinstance(source, dict)
                        or set(source) != {'path', 'sha256', 'size_bytes'}):
                    raise ValueError('Current chart findings need their exact original source review')
                original = _read_scientific_gap_review(source['path'])
                if not same_storage_descriptor(original['review_binding'], source):
                    raise ValueError('The original chart source review changed its bound bytes')
                if (original.get('chart_source_review') is not None
                        or not matches_version(original, record)
                        or any(_json(original[key]) != _json(review[key])
                               for key in ('map_version', 'geometry_index'))
                        or _json(original['source']) != _json(review['source'])
                        or [row['run_id'] for row in original['runs']]
                           != [row['run_id'] for row in review['runs']]):
                    raise ValueError('Current chart findings lost their original approved map, geometry or cohort')
                review = original
            return ChartTransferSession(review, proposal_format=DETECTED_FORMAT)
    raise ValueError('No saved transfer review matches this selected map and series geometry. '
                     'Use Transfer map to all runs first.')


def _bound(path):
    from .painted_adoption_table import _bound as original
    return original(Path(path))


def _checked(bound, *, metadata=False):
    from .painted_adoption_table import _checked as original
    # Saved metadata has finite explicit size, before its original decoder.
    size = bound.get("size_bytes")
    if (type(size) is not int or not 0 <= size <= (16*1024*1024 if metadata else 512*1024*1024)):
        raise ValueError("The saved ColorChecker input exceeds its bounded local format")
    return original(bound, metadata=metadata)


def _named(path, sha, *, metadata=False):
    bound = _bound(path)
    if bound["sha256"] != sha:
        raise ValueError("An original ColorChecker detection input changed")
    return _checked(bound, metadata=metadata), bound


def read_detection(manifest_row, model):
    """Read actual saved xy and join its original RAW/certificate to this model.

    A baseline-image detection is direct. An original model-image detection
    is mapped with its authenticated saved rig-pose bridge; that does not
    constitute an independent detection of chart motion on the later image.
    """
    from . import preset_wasserstein_inputs as io

    run = model["run_id"]
    if manifest_row.get("run") != run:
        raise ValueError("The ColorChecker detection belongs to another run")
    baseline = Path(manifest_row["baseline_path"])
    _, archive = _named(baseline, manifest_row["baseline_sha256"])
    sidecar, side = _named(baseline.with_suffix(".json"),
                           manifest_row["sidecar_sha256"], metadata=True)
    source, source_bound = _named(manifest_row["ea_source_binding_path"],
        manifest_row["ea_source_binding_sha256"], metadata=True)
    if (sidecar.get("contract") != "ffac-v5-source-bound-baseline-calibration-v1" or
            sidecar.get("run") != run or sidecar.get("npz_sha256") != archive["sha256"] or
            source.get("contract") != "ffac-ea-native-v5-per-run-array-key-binding-v1" or
            source.get("run") != run or source.get("fitted_baseline_sha256") != archive["sha256"] or
            source.get("fitted_baseline_sidecar_sha256") != side["sha256"]):
        raise ValueError("The original ColorChecker xy lost its saved source association")
    certificate, certificate_bound = _named(source["source_certificate_path"],
        source["source_certificate_sha256"], metadata=True)
    raw_sha = sidecar["source"]["source_arw_sha256"]
    certificate_run = certificate.get("run")
    if (source.get("source_arw_sha256") != raw_sha or
            not isinstance(certificate_run, str) or certificate_run.casefold() != run.casefold() or
            certificate["sources"].get("arw_sha256") != raw_sha or
            sidecar["source"].get("geometry_certificate_sha256") != certificate_bound["sha256"]):
        raise ValueError("The ColorChecker's original RAW/certificate identities disagree")
    with np.load(io.source(archive["path"]), allow_pickle=False) as saved:
        xy = _centres(np.array(saved["xy"], copy=True))
    native_source = 8.0*xy + 3.5
    camera_u = _project(native_source, certificate["review_to_camera_h"])
    bridge_bound = None
    if raw_sha == model["reference_image"]["sha256"]:
        metric_h = _matrix(model["H_camera_u_to_metric_mm"])
        basis = "baseline photograph detection"
    elif raw_sha == (model.get("model_photograph") or {}).get("sha256"):
        bridge_bound = deepcopy(model["baseline_to_model_registration"])
        bridge = _checked(bridge_bound, metadata=True)
        j = _matrix(bridge["H_baseline_u_to_model_u"])
        metric_h = _matrix(model["H_model_photograph_u_to_metric_mm"])
        if (bridge.get("accepted") is not True or
                bridge["baseline_raw"]["sha256"] != model["reference_image"]["sha256"] or
                bridge["model_raw"]["sha256"] != raw_sha or
                not np.array_equal(metric_h @ j, _matrix(model["H_camera_u_to_metric_mm"]))):
            raise ValueError("The saved baseline/model pose bridge does not bind this detection")
        basis = "model photograph detection; saved rig pose, no independent later ColorChecker-motion observation"
    else:
        raise ValueError("The saved ColorChecker was detected on neither of this run's bound photographs")
    pins = [archive, side, source_bound, certificate_bound]
    if bridge_bound is not None:
        pins.append(bridge_bound)
    for pin in pins:
        _checked(pin)
    return _project(camera_u, metric_h), dict(basis=basis, inputs=pins,
        xy=_identity(xy), source_raw_sha256=raw_sha)


def proposal_for_approval(session, run_id, approval):
    """Recreate a saved proposal under its explicit original motion semantics."""
    proposal_format = proposal_format_for_approval(approval)
    options = ({} if proposal_format == getattr(session, 'proposal_format', FORMAT)
               else dict(proposal_format=proposal_format))
    proposal = session.proposal(run_id, correction_mm=approval.get('correction_mm'), **options)
    if hashlib.sha256(proposal.payload).hexdigest() == approval.get('proposal_sha256'):
        return proposal
    from .painting_chart_approval_store import ChartApprovalStore
    store = getattr(session, '_approval_store', None) or ChartApprovalStore.for_session(session,
        decision_proposal_format=proposal_format if proposal_format == DETECTED_FORMAT else None)
    if store is None:
        return proposal
    return store.confirmed_proposal(proposal, approval, saved_state=getattr(session, '_approval_state', None))


class ChartTransferSession:
    """Read the approved source once and propose selected runs serially.

    Construct and call this object in the GUI's existing owned finite reader.
    It does not create a thread, executor, source provider or scientific job.
    """

    def __init__(self, review_source, detection_manifest=DEFAULT_DETECTION_MANIFEST, *,
                 proposal_format=None):
        from .painted_analysis_exclusions import _read_scientific_gap_review
        from .painted_adoption_table import verify_version
        from .layer_painting_fixtures import _drawn_polygons
        from .painted_review_comparison import _model
        from . import layer_painting as lp

        descriptor = (review_source if isinstance(review_source, dict)
                      and set(review_source) == {'path', 'sha256', 'size_bytes'} else None)
        self.review = _read_scientific_gap_review(descriptor['path'] if descriptor else review_source)
        if descriptor is not None and not same_storage_descriptor(self.review['review_binding'], descriptor):
            raise ValueError('The saved ColorChecker review differs from its exact descriptor')
        self.proposal_format = review_proposal_format(self.review)
        if proposal_format is not None:
            if proposal_format != DETECTED_FORMAT:
                raise ValueError('A new chart review must use whole-chart detected motion')
            # This selects a NEW human review, never a reinterpretation of an
            # immutable historical review or one of its signed approvals.
            self.proposal_format = self.decision_proposal_format = proposal_format
        source = self.review["source"]
        self.reference_run = source.get("reference_run")
        if not isinstance(self.reference_run, str) or not self.reference_run:
            raise ValueError("ColorChecker relocation needs the original approved reference drawing")
        self.painting = deepcopy(source["painting_facies"])
        session, layers, _frame = verify_version(self.painting)
        roles = {row.layer_id: session.facies[row.facies].role for row in session.layers}
        self.roles = roles
        chart = [key for key, role in roles.items() if role == "colorchecker"]
        if len(chart) != 1:
            raise ValueError("The original drawing must identify exactly one whole ColorChecker")
        self.chart_id, self.known_ids = chart[0], sorted(roles)
        explicit = _drawn_polygons(session,
            [p for p in session.polygons if p.layer_id == self.chart_id], layers.shape)
        self.whole = (layers == self.chart_id) | (explicit == self.chart_id)
        drawn = _drawn_polygons(session,
            [p for p in session.polygons if p.layer_id != self.chart_id], layers.shape)
        # Preserve the original saved fixture's explicit drawn support. The
        # whole chart is deliberately NOT clipped; its outside rows still use
        # the original source-row LEFT rule, without invented outside drawings.
        drawn[~lp.frame_mask(_frame, layers.shape, (1., 1.))] = 0
        self.restored = restore_chart_material(layers, drawn, self.whole,
            chart_id=self.chart_id, known_ids=self.known_ids)
        self.rows = {row["run_id"]: deepcopy(row) for row in self.review["runs"]}
        if len(self.rows) != len(self.review["runs"]):
            raise ValueError("The retained ColorChecker review repeats a run")
        if self.reference_run not in self.rows:
            raise ValueError("The retained review must include its original reference run")
        reference_row = self.rows[self.reference_run]
        self.reference, self.reference_native, _ = _model(reference_row)
        self.manifest_binding = (deepcopy(detection_manifest) if isinstance(detection_manifest, dict)
                                 else _bound(detection_manifest))
        manifest = _checked(self.manifest_binding, metadata=True)
        if manifest.get("contract") != "ffac-ea-native-v5-donor-fixedpoint-v2":
            raise ValueError("Choose the original saved baseline detection manifest")
        self.detections = {row["run"]: deepcopy(row) for row in manifest["results"]}
        if len(self.detections) != len(manifest["results"]) or not set(self.rows) <= set(self.detections):
            raise ValueError("The original detection manifest lacks this exact review cohort")
        self.reference_centres, self.reference_detection = read_detection(
            self.detections[self.reference_run], self.reference.model)

    def proposal(self, run_id, *, correction_mm=None, proposal_format=None):
        """Read one exact old product, then produce a bound chart-only change."""
        from .painted_analysis_exclusions import _arrays
        from .painted_review_comparison import _model

        if run_id not in self.rows:
            raise ValueError("Choose a run from the actual retained review")
        row = self.rows[run_id]
        if _json(row["reference_model"]) != _json(self.rows[self.reference_run]["target_model"]):
            raise ValueError("The selected run does not use this exact retained reference model")
        target, native, _ = _model(row)
        fields = _arrays(row["product"])
        actual_centres, actual_detection = read_detection(self.detections[run_id], target.model)
        carrier = native @ target.metric_to_grid @ np.linalg.inv(self.reference.metric_to_grid)
        native_to_metric = np.linalg.inv(target.metric_to_grid) @ np.linalg.inv(native)
        # Carry the SAME original AC17 centres through the original source
        # placement, then compare both detections in this target metric plane.
        ref_camera = _project(self.reference_centres, self.reference.metric_to_grid)
        carried_centres = _project(_project(ref_camera, carrier), native_to_metric)
        binding = dict(review=deepcopy(self.review["review_binding"]),
            painting_facies=deepcopy(self.painting), run_id=run_id,
            reference_model=deepcopy(self.rows[self.reference_run]["target_model"]),
            target_model=deepcopy(row["target_model"]), base_product=deepcopy(row["product"]),
            detection_manifest=deepcopy(self.manifest_binding),
            reference_detection=deepcopy(self.reference_detection), detection=actual_detection,
            restoration="explicit source drawing first, else original same-row material immediately LEFT")
        return propose_chart_transfer(run_id, fields["array"], fields["inside"],
            fields["array"] == self.chart_id, self.restored["material_labels"], self.whole,
            carrier, native_to_metric, carried_centres, actual_centres, binding,
            chart_id=self.chart_id, known_ids=self.known_ids, correction_mm=correction_mm,
            proposal_format=(getattr(self, 'proposal_format', FORMAT)
                             if proposal_format is None else proposal_format))

    def approved(self, run_id, approval):
        """Fresh original reads/replay must reproduce the exact confirmed result.

        The ordinary effective-product consumer calls this only for changed
        runs; it receives newly derived labels/masks, never old mass arrays.
        """
        fresh = ChartTransferSession(self.review, self.manifest_binding)
        return apply_chart_transfer(proposal_for_approval(fresh, run_id, approval), approval)
