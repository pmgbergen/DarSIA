"""Read-only layer-edge offset candidates in an existing corrected rig frame.

This module measures and draws candidates; it never edits a label, photograph,
workspace revision, run configuration or rig model. No correction is applied.
The original saved field identities and the actual Store revision are required
by the file adapter. Array-only functions are also usable by synthetic tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


SIGN = "Positive normal displacement goes from the smaller label ID towards the larger label ID; image row increases down, column increases right."
ELIGIBLE_ROLES = frozenset({"porous_medium", "seal"})
PHOTOGRAPHED_ROLES = ELIGIBLE_ROLES | {"free_water"}
EXCLUDED_ROLES = frozenset({"frame", "colorchecker", "excluded"})
LIMIT = 1024 * 1024


@dataclass(frozen=True)
class OffsetParameters:
    band_mm: float = 25.0
    normal_neighbours: int = 9

    def validate(self):
        if not math.isfinite(self.band_mm) or self.band_mm <= 0 or self.band_mm > 25:
            raise ValueError("The explicit normal search band must be within +/-25 mm.")
        if type(self.normal_neighbours) is not int or self.normal_neighbours < 3:
            raise ValueError("At least three local boundary samples define a tangent.")
        return self


def _metadata_grid(metadata, shape):
    if len(shape) != 2 or any(type(n) is not int or n <= 1 for n in shape):
        raise ValueError("One non-empty two-dimensional native grid is required.")
    if metadata.get("space_dim") != 2 or metadata.get("indexing") != "ij":
        raise ValueError("Use the saved two-dimensional ij corrected image grid.")
    dimensions = metadata.get("dimensions")
    if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2
            or any(not math.isfinite(float(v)) or float(v) <= 0 for v in dimensions)):
        raise ValueError("Recorded metric image dimensions are required.")
    # DarSIA dimensions are ordered in matrix axes (height, width), in metres.
    return np.asarray(dimensions, dtype=float) * 1000.0 / np.asarray(shape)


def _region_roles(regions):
    rows = {}
    for source in regions:
        label = source.get("label_id")
        if type(label) is not int or label in rows:
            raise ValueError("The saved revision must bind each label role exactly once.")
        rows[label] = dict(source)
    return rows


def _interfaces(labels):
    """Native pixel faces only: no contour simplification or raster mutation."""
    interfaces = {}
    for axis in (0, 1):
        first = labels[:-1, :] if axis == 0 else labels[:, :-1]
        second = labels[1:, :] if axis == 0 else labels[:, 1:]
        rr, cc = np.nonzero(first != second)
        for row, col in zip(rr.tolist(), cc.tolist()):
            a, b = int(first[row, col]), int(second[row, col])
            pair = (min(a, b), max(a, b))
            point = (row + .5, float(col)) if axis == 0 else (float(row), col + .5)
            direction = (1.0 if a < b else -1.0, 0.0) if axis == 0 else (0.0, 1.0 if a < b else -1.0)
            interfaces.setdefault(pair, []).append((point, direction))
    return interfaces


def _normals(points_rc, face_directions, spacing_mm, neighbours):
    """Local PCA tangent in physical coordinates; orient by the saved label pair."""
    physical = points_rc * spacing_mm
    if len(physical) < 3:
        return np.full_like(physical, np.nan)
    k = min(neighbours, len(physical))
    _, indices = cKDTree(physical).query(physical, k=k)
    normals = np.full_like(physical, np.nan)
    for index, neighbours_ in enumerate(indices):
        cloud = physical[neighbours_] - physical[neighbours_].mean(axis=0)
        values, vectors = np.linalg.eigh(cloud.T @ cloud)
        if values[-1] <= values[0]:
            continue  # A point/junction without a defined tangent is not a normal.
        normal = vectors[:, 0]
        orientation = float(normal @ face_directions[index])
        if orientation == 0:
            continue
        normals[index] = normal if orientation > 0 else -normal
    return normals


def _peak_offset(gradient, distances):
    valid = np.isfinite(gradient)
    if not valid.any():
        return None, "missing_or_masked"
    maximum = float(np.max(gradient[valid]))
    if maximum <= 0:
        return None, "no_photographic_edge"
    peaks = np.flatnonzero(valid & (gradient == maximum))
    # A contiguous pixel plateau is one edge; disconnected equally strong edges
    # are not disambiguated by an invented tie-break or a guessed displacement.
    if len(peaks) > 1 and np.any(np.diff(peaks) != 1):
        return None, "ambiguous_equal_edges"
    finite_indices = np.flatnonzero(valid)
    if peaks[0] == finite_indices[0] or peaks[-1] == finite_indices[-1]:
        return None, "edge_at_search_limit"
    return float(np.mean(distances[peaks])), "candidate"


def measure_arrays(image, labels, metadata, regions, *, parameters=OffsetParameters()):
    """Measure existing layer boundaries without generating registered geometry.

    The per-normal candidate is the largest RGB gradient magnitude in the stated
    band. Native finite samples provide the median and empirical P10/P90 along
    each saved label-pair boundary. These are candidates, never confidence/PASS.
    The normal neighbourhood is a declared estimator parameter, not a physical
    calibration or a correction. All missing/ambiguous samples are counted.
    """
    parameters.validate()
    labels = np.asarray(labels)
    image = np.asarray(image)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Use the original integer label raster.")
    if image.shape != (*labels.shape, 3) or not np.issubdtype(image.dtype, np.number):
        raise ValueError("Photograph and labels must already share the exact RGB grid.")
    spacing = _metadata_grid(metadata, labels.shape)
    roles = _region_roles(regions)
    eligible = sorted(label for label, row in roles.items()
                      if row.get("enabled", True) and row.get("analysis_role") in ELIGIBLE_ROLES)
    photographed = sorted(label for label, row in roles.items()
                          if row.get("enabled", True) and row.get("analysis_role") in PHOTOGRAPHED_ROLES)
    excluded = sorted(label for label, row in roles.items()
                      if not row.get("enabled", True) or row.get("analysis_role") in EXCLUDED_ROLES)
    unknown = sorted(int(label) for label in np.unique(labels)
                     if int(label) not in roles or roles[int(label)].get("analysis_role") not in PHOTOGRAPHED_ROLES | EXCLUDED_ROLES)
    admitted = np.isin(labels, photographed)
    steps = int(math.ceil(parameters.band_mm / float(np.min(spacing))))
    distances = np.linspace(-parameters.band_mm, parameters.band_mm, 2*steps+1)
    channels = tuple(np.ascontiguousarray(image[..., channel], dtype=float) for channel in range(3))
    admitted_field = admitted.astype(np.uint8)
    label_grid = dict(shape=list(labels.shape), dtype=str(labels.dtype),
                      sha256=hashlib.sha256(memoryview(np.ascontiguousarray(labels)).cast("B")).hexdigest())
    reports, drawings = [], []
    for pair, faces in sorted(_interfaces(labels).items()):
        points = np.asarray([row[0] for row in faces], dtype=float)
        directions = np.asarray([row[1] for row in faces], dtype=float)
        identity = f"labels_{pair[0]}_{pair[1]}"
        pair_roles = {str(label): roles.get(label, {}).get("analysis_role", "unrecorded") for label in pair}
        measurable = any(label in eligible for label in pair) and all(label in photographed for label in pair)
        if not measurable:
            reports.append(dict(boundary_id=identity, labels=list(pair), roles=pair_roles,
                measurement_eligible=False, coverage_reason="At least one sand/seal side and two recorded photographed media are required; frame/chart/excluded/unknown pairs are not photo edges",
                sample_count=len(points), measured_sample_count=0,
                sample_outcomes={"not_measurable_from_saved_roles": len(points)},
                status="not_measurable_from_saved_roles", median_mm=None, p10_mm=None, p90_mm=None))
            drawings.append(dict(boundary_id=identity, measurement_eligible=False,
                                 map_points_rc=points, selected_points_rc=np.empty((0, 2))))
            continue
        normals = _normals(points, directions, spacing, parameters.normal_neighbours)
        offsets, selected, reasons = [], [], {}
        # Keep memory bounded to one local profile; no huge image/profile cube.
        for point, normal in zip(points, normals):
            if not np.isfinite(normal).all():
                reasons["normal_undefined"] = reasons.get("normal_undefined", 0)+1
                continue
            coordinates = point[:, None] + normal[:, None] * distances[None, :] / spacing[:, None]
            values = np.stack([map_coordinates(channels[channel], coordinates,
                               order=1, mode="constant", cval=np.nan, prefilter=False) for channel in range(3)], axis=1)
            mask = map_coordinates(admitted_field, coordinates, order=0,
                                   mode="constant", cval=0, prefilter=False).astype(bool)
            gradient = np.linalg.norm(np.gradient(values, distances, axis=0), axis=1)
            # A valid gradient uses only finite/nonexcluded neighbours. No mask
            # boundary is turned into a photographic edge or a hidden fill.
            valid = mask & np.isfinite(values).all(axis=1)
            valid[1:-1] &= valid[:-2] & valid[2:]
            valid[0] = valid[-1] = False
            gradient[~valid] = np.nan
            offset, reason = _peak_offset(gradient, distances)
            reasons[reason] = reasons.get(reason, 0)+1
            if offset is not None:
                offsets.append(offset)
                selected.append(point + normal * offset / spacing)
        stats = {"median_mm": None, "p10_mm": None, "p90_mm": None}
        if offsets:
            stats = dict(zip(stats, (float(v) for v in np.percentile(offsets, [50, 10, 90]))))
        reports.append(dict(boundary_id=identity, labels=list(pair), roles=pair_roles,
                            measurement_eligible=True, coverage_reason="Saved sand/seal against a permitted photographed medium, including free water", sample_count=len(points),
                            measured_sample_count=len(offsets), sample_outcomes=reasons,
                            status="candidate" if offsets else "uncertain_or_missing_edge",
                            interpretation="Read-only candidate; visual review required, no correction applied",
                            **stats))
        drawings.append(dict(boundary_id=identity, measurement_eligible=True, map_points_rc=points, selected_points_rc=np.asarray(selected).reshape(-1, 2)))
    report = dict(format="darsia-geometry-offset-candidates-v1", frame="rig-corrected",
                  sign_convention=SIGN, parameters=asdict(parameters), spacing_mm=spacing.tolist(),
                  image_shape=list(image.shape), image_precision=str(image.dtype), label_grid=label_grid,
                  declared_excluded_labels=excluded, permitted_photographed_labels=photographed, labels_without_eligible_role=unknown,
                  boundary_pair_count=len(reports), measurable_boundary_pair_count=sum(row["measurement_eligible"] for row in reports),
                  not_measurable_boundary_pair_count=sum(not row["measurement_eligible"] for row in reports),
                  exclusions="Saved enabled/analysis_role only; no new chart/fixture classification",
                  explicit_chart_fixture_completeness="Not inferred; role-mask coverage and missing roles are reported",
                  gradient_definition="Euclidean magnitude of RGB derivative along the local metric normal",
                  missing_quality="No confidence score or automatic acceptance threshold is invented",
                  boundaries=reports, correction_applied=False, registered_revision_written=False)
    return report, drawings


def measure_saved_run(project, run_id, *, parameters=OffsetParameters()):
    """Explicit read-only file adapter using the genuine saved Store revision."""
    import darsia
    from .geometry_facies import GeometryRevisionStore, GEOMETRY_REVISION_FORMAT
    from .geometry_workspace_frame import frame_kind, selected_fields
    from .corrected_setup import _small, _verified, _require_unedited_adoption
    path = GeometryRevisionStore(project).current_path
    raw = _small(path)
    current = json.loads(raw)
    if current.get("format") != GEOMETRY_REVISION_FORMAT or current.get("schema") != 1:
        raise ValueError("Use the genuine current Store revision format.")
    config = current["configuration"]
    if frame_kind(config) != "rig-corrected" or current.get("frame") != "rig-corrected":
        raise ValueError("Offset measurement requires the explicit saved corrected-frame revision.")
    if re.fullmatch(r"[0-9a-f]{32}", str(current.get("revision_id", ""))) is None:
        raise ValueError("Use a genuine Store revision UUID.")
    _require_unedited_adoption(current, config)
    fields = selected_fields(config, run_id)
    baseline = _verified(fields["shape_corrected_baseline"])
    labels_bound = _verified(fields["labels"])
    paths = [Path(baseline["path"]), Path(labels_bound["path"])]
    versions = [(p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    image = darsia.imread(paths[0])
    labels = darsia.imread(paths[1])
    if [(p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths] != versions:
        raise ValueError("Bound photograph or labels changed during the original reader call.")
    from .setup_rig import _require_exact_grid
    _require_exact_grid(Path(labels_bound["path"]), image, "labels")
    report, drawings = measure_arrays(image.img, labels.img,
        {"space_dim": image.space_dim, "indexing": image.indexing, "dimensions": list(image.dimensions)},
        config["regions"], parameters=parameters)
    if _small(path) != raw:
        raise ValueError("Workspace revision changed during offset measurement; no report is published.")
    report.update(run_id=str(run_id), revision_id=current["revision_id"], sources={
        "revision": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)},
        "baseline": baseline, "labels": labels_bound,
        "reference_shape_corrections": config["frame_binding"]["reference_shape_corrections"]})
    return report, drawings, image.img


def propose_registration(reports, comparison_runs):
    """The existing 2 mm across-run rule, never a revision/correction writer."""
    runs = tuple(str(run) for run in comparison_runs)
    if len(runs) != 3 or len(set(runs)) != 3:
        raise ValueError("Declare the three distinct comparison runs explicitly.")
    if len({row["run_id"] for row in reports}) != len(reports):
        raise ValueError("Duplicate run measurement is not a unique comparison.")
    by_run = {row["run_id"]: row for row in reports}
    present = [by_run[run] for run in runs if run in by_run]
    if present:
        identity = {key: present[0].get(key) for key in
                    ("revision_id", "frame", "parameters", "spacing_mm", "label_grid", "sign_convention")}
        if (not identity["revision_id"] or identity["frame"] != "rig-corrected"
                or identity["sign_convention"] != SIGN
                or any({key: row.get(key) for key in identity} != identity for row in present)):
            raise ValueError("Compare the same saved revision, original label grid, metric frame and estimator only.")
    boundaries = sorted({b["boundary_id"] for run in runs if run in by_run for b in by_run[run]["boundaries"]})
    rows = []
    for boundary in boundaries:
        measurements = {}
        for run in runs:
            matches = [b for b in by_run.get(run, {}).get("boundaries", []) if b["boundary_id"] == boundary]
            measurements[run] = matches[0] if len(matches) == 1 else None
        if all(m is not None and m.get("measurement_eligible") is False for m in measurements.values()):
            rows.append(dict(boundary_id=boundary, suggestion="not_measurable_from_saved_roles", measurements=measurements))
            continue
        medians = [m["median_mm"] for m in measurements.values() if m is not None and isinstance(m["median_mm"], (int, float)) and math.isfinite(m["median_mm"])]
        if len(medians) != 3:
            rows.append(dict(boundary_id=boundary, suggestion="insufficient_edge_evidence", measurements=measurements))
            continue
        spread = max(medians)-min(medians)
        rows.append(dict(boundary_id=boundary, suggestion="common_shift_per_layer_candidate" if spread <= 2.0 else "per_run_registration_candidate",
                         across_run_median_range_mm=spread, common_candidate_mm=float(np.median(medians)) if spread <= 2.0 else None,
                         measurements=measurements))
    measured_rows = [row for row in rows if row["suggestion"] != "not_measurable_from_saved_roles"]
    overall = ("insufficient_edge_evidence" if not measured_rows or any(r["suggestion"] == "insufficient_edge_evidence" for r in measured_rows)
               else "per_run_registration_candidate" if any(r["suggestion"] == "per_run_registration_candidate" for r in measured_rows)
               else "common_shift_per_layer_candidate")
    return dict(format="darsia-geometry-registration-proposal-v1", comparison_runs=list(runs),
                rule="Per-boundary median offsets across the three runs agree within 2 mm (inclusive)",
                sign_convention=SIGN, boundaries=rows, suggestion=overall,
                measurable_boundary_pair_count=len(measured_rows),
                explicitly_unmeasurable_boundary_pair_count=len(rows)-len(measured_rows),
                decision="needs_series_decision_after_overlays_and_table", correction_applied=False,
                registered_revision_written=False)


def write_readonly_outputs(target, measured, proposal, *, raw_source_figure=None):
    """Explicit fresh output only. Never writes to any existing input/revision."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from .geometry_facies import _rgb_float
    target = Path(target)
    if target.exists():
        raise FileExistsError("Use a fresh report directory; all prior reports are preserved.")
    target.mkdir(parents=True)
    table = []
    for report, drawings, image in measured:
        run = report["run_id"]
        if not run or run in {".", ".."} or any(c in run for c in ("/", chr(92), ":", chr(0))):
            raise ValueError("Unsafe report run ID.")
        with (target / f"registration_{run}.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.imshow(_rgb_float(image))
        for drawing in drawings:
            points = drawing["map_points_rc"]
            edge = drawing["selected_points_rc"]
            ax.scatter(points[:, 1], points[:, 0], s=.15, c="cyan" if drawing["measurement_eligible"] else "gray")
            if len(edge):
                ax.scatter(edge[:, 1], edge[:, 0], s=.15, c="yellow")
        ax.set_title(f"{run.upper()}: saved map and photographic edge candidates\nCyan: saved boundary | Yellow: candidate | Gray: unmeasurable pair", fontsize=12)
        ax.set_axis_off()
        fig.subplots_adjust(left=.015, right=.985, bottom=.07, top=.90)
        fig.text(.5, .025, "No correction applied. Table: offsets.csv",
                 ha="center", fontsize=10, url="offsets.csv")
        fig.savefig(target / f"overlay_{run}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        table.extend(dict(run=run, **b) for b in report["boundaries"])
    with (target / "offsets.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "boundary_id", "status", "measurement_eligible", "coverage_reason", "median_mm", "p10_mm", "p90_mm", "sample_count", "measured_sample_count"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(table)
    with (target / "registration_proposal.json").open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if raw_source_figure is not None:
        rgb, native_source_labels, source_descriptors = raw_source_figure
        if rgb.shape[:2] != native_source_labels.shape:
            raise ValueError("Raw source photograph and its original label grid differ; no resize is allowed.")
        from .geometry_facies import _layer_boundary_mask
        boundary = _layer_boundary_mask(native_source_labels, set(int(v) for v in np.unique(native_source_labels)))
        fig, ax = plt.subplots(figsize=(13, 8)); ax.imshow(_rgb_float(rgb))
        rr, cc = np.nonzero(boundary); ax.scatter(cc, rr, c="cyan", s=.1)
        ax.set_title("Original raw source photograph and original map boundaries\nOrigin figure only; no corrected-grid measurement or registration.")
        ax.set_axis_off(); fig.savefig(target / "raw_source_origin.png", dpi=160, bbox_inches="tight"); plt.close(fig)
        with (target / "raw_source_origin.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(sources=source_descriptors, purpose="raw source origin only", metric_offset_measured=False), stream, indent=2, allow_nan=False)
    return target
