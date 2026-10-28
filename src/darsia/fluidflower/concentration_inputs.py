"""Read fixed Studio phase classes and registered images without inference.

Saved categorical PNGs supply labels, never probabilities or concentrations.
The usual corrected photograph is JPEG: its bytes can be verified, but its
decoded pixels are not the original pre-encoding float32 inference array.
Measured depth, porosity and thermodynamic calibration belong to the consumer.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .phase_rendering import AQUEOUS_RGBA, GAS_RGBA, PHASE_DISPLAY_POLICY
from .review_storage import resolve_review_path
from .series_paths import series_path

FORMAT = "darsia-fixed-phase-concentration-inputs-v1"
IDENTITY_KEYS = ("dataset", "run", "candidate_id", "params_sha256", "optical_model_sha256")


def _path(value, base=None):
    path = Path(value)
    return resolve_review_path(path if path.is_absolute() or base is None else Path(base) / path)


def _bytes(path, expected=None):
    path = _path(path)
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while being read: {path}")
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Input checksum differs: {path}")
    return content, digest


def _read(path, expected=None):
    content, digest = _bytes(path, expected)
    value = json.loads(content.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an input metadata object: {path}")
    return value, digest


def discover_model_inputs(model_root: Path, *, include_saved_results: bool = True) -> dict[str, Any]:
    """List the selected model's verified ready inputs; do not load images.

    Studio may defer the independent saved-result browser scan. The default
    preserves strict discovery for calculation and other programmatic callers.
    """
    from .phase_evidence_audit import verified_review_binding
    from .phase_saved_runtime_views import mass_candidate

    root = _path(model_root)
    binding = verified_review_binding(root)
    combined = _path(binding["combined_root"])
    manifest_path = combined / "postflight_manifest.json"
    manifest, manifest_sha = _read(manifest_path)
    run_bindings = {}
    if manifest.get("format") == "fluidflower-phase-review-collection-v1":
        receipt, _ = _read(_path(manifest["collection_receipt"], combined))
        for row in receipt["sources"]:
            reference = row["binding"]
            source, _ = _read(_path(reference["path"], combined), reference["sha256"])
            for run in source["selected_candidates"]:
                key = run.casefold()
                if key in run_bindings and run_bindings[key] != source:
                    raise ValueError(f"Ambiguous concentration binding for {run}")
                run_bindings[key] = source
    jobs = []
    for run, candidate_id in binding["selected_candidates"].items():
        matches = [job for job in manifest["jobs"]
                   if str(job["run"]).casefold() == run.casefold() and job.get("candidate_id") == candidate_id]
        if len(matches) != 1:
            raise ValueError(f"No unique fixed phase candidate for {run}")
        source = matches[0]
        pinned = run_bindings.get(run.casefold(), binding)
        if not pinned.get("pipeline_sha256") or source.get("params_sha256") != pinned["pipeline_sha256"]:
            raise ValueError(f"Phase pipeline differs from its binding: {run}")
        if source.get("optical_model_sha256") != pinned.get("model_sha256"):
            raise ValueError(f"Phase model differs from its binding: {run}")
        candidate = json.loads(json.dumps(asdict(mass_candidate(combined, source)), default=str))
        review_root = _path(source.get("analysis_source_review_root") or source["review_output_dir"], combined)
        jobs.append(dict(dataset=str(source["dataset"]).casefold(), run=run.casefold(),
            model_root=str(root), model_id=binding.get("model_id") or binding["source_model_id"],
            combined_root=str(combined), source_manifest=str(manifest_path), source_manifest_sha256=manifest_sha,
            review_root=str(review_root), candidate=candidate, source_job=source, phase_binding=pinned))
    datasets = {job["dataset"] for job in jobs}
    if len(datasets) != 1:
        raise ValueError("Concentration analysis requires one selected experiment series")
    dataset = next(iter(datasets))
    catalog = dict(format=FORMAT, dataset=dataset, model_root=str(root),
        model_id=binding.get("model_id") or binding["source_model_id"], jobs=jobs,
        results_root=str(series_path(dataset, "results") / "concentration_mass"))
    catalog["results"] = discover_saved_results(catalog) if include_saved_results else []
    return catalog


def discover_saved_results(catalog: Mapping) -> list[str]:
    """List delivered native/legacy results with this exact phase identity.

    Only small result/receipt JSONs are read. Older ordinary deliveries with
    changed source stamps are conservatively omitted rather than rehashing
    dense arrays, photographs or entire rigs in the Studio discovery thread.
    """
    def identity(value):
        return tuple(value.get(key) for key in IDENTITY_KEYS)
    selected = {identity(job["candidate"]) for job in catalog["jobs"]}
    references = list((Path(catalog["results_root"]) / "catalog").glob("*.json"))
    for job in catalog["jobs"]:
        folder = _path(job["candidate"]["phase_map_output_dir"]) / "mass_results"
        references.extend(folder.glob("*.json"))
    found = set()
    for reference in references:
        try:
            receipt, _ = _read(reference)
            kind = receipt.get("format")
            if kind == "studio-mass-variant-delivery-v1":
                if identity(receipt["candidate"]) not in selected:
                    continue
                result_path = _path(receipt["result"], reference.parent)
                result, _ = _read(result_path, receipt["result_sha256"])
                if result.get("source_bindings", {}) != receipt.get("source_bindings", {}):
                    continue
            elif kind == "studio-mass-delivery-v1":
                if identity(receipt["selection"]["candidate"]) not in selected:
                    continue
                stale = False
                for path, row in receipt.get("bindings", {}).items():
                    stat = _path(path).stat()
                    if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != row["stamp"]:
                        stale = True
                        break
                if stale:
                    continue
                result_path = _path(receipt["result"], reference.parent)
                result, _ = _read(result_path)
            else:
                continue
            if (result.get("complete") is True and result.get("frames")
                    and identity(result.get("selected_inputs", {}).get("candidate", {})) in selected):
                found.add(str(result_path))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(found)


def inspect_run(job: Mapping) -> dict[str, Any]:
    """Return a JSON-safe, pinned timeline and paths without loading dense fields."""
    from .tools.calibration_pairwise_review import (
        Candidate, _native_frame_path, _native_phase_map_path, _native_reference_path,
    )
    values = dict(job["candidate"])
    for key in ("mass_plot", "contour_movie", "per_frame_csv", "summary_path", "review_output_dir",
                "phase_map_output_dir", "raw_phase_map_output_dir", "presence_diagnostics_dir"):
        if values.get(key) is not None:
            values[key] = _path(values[key], job.get("combined_root"))
    candidate = Candidate(**values)
    root = _path(job["review_root"])
    run = str(job["run"]).casefold()
    if candidate.run.casefold() != run or candidate.dataset.casefold() != job["dataset"]:
        raise ValueError("Fixed phase candidate belongs to another experiment")
    if job.get("source_manifest") and job.get("source_manifest_sha256"):
        _bytes(job["source_manifest"], job["source_manifest_sha256"])
    native_path = root / "native_frames" / run / "manifest.json"
    native, native_sha = _read(native_path)
    if str(native["run"]).casefold() != run or str(native["dataset"]).casefold() != job["dataset"]:
        raise ValueError("Native image timeline belongs to another experiment")
    canvas = tuple(native["canvas_wh"])
    box = tuple(native["content_box_xywh"])
    if (len(canvas) != 2 or len(box) != 4 or any(type(v) is not int for v in (*canvas, *box))
            or min(canvas) < 1 or min(box[:2]) < 0 or min(box[2:]) < 1
            or box[0] + box[2] > canvas[0] or box[1] + box[3] > canvas[1]):
        raise ValueError("Invalid native canvas/content geometry")
    if native.get("analysis_image_shape_hw") and tuple(native["analysis_image_shape_hw"]) != (box[3], box[2]):
        raise ValueError("Native analysis shape differs from content geometry")
    source_job = dict(job.get("source_job") or {})
    context = dict(source_job.get("phase_transfer_context") or native.get("domain") or {})
    if context.get("native_manifest_sha256") and context["native_manifest_sha256"] != native_sha:
        raise ValueError("Physical context refers to another native image timeline")
    if context.get("path"):
        context["path"] = str(_path(context["path"], root))
    rows = native.get("frame_sources") or []
    if not rows or len(rows) != native.get("frame_count"):
        raise ValueError("Native frame timeline is incomplete")
    frames, previous = [], -math.inf
    for index, row in enumerate(rows):
        hours = float(row["elapsed_h"])
        if row.get("frame_index") != index or not math.isfinite(hours) or hours <= previous:
            raise ValueError("Native frame identities/timestamps are not chronological")
        photo = _native_frame_path(root, run, index)
        phase = _native_phase_map_path(root, candidate, "A", index)
        if not photo.is_file() or not phase.is_file():
            raise FileNotFoundError(f"Fixed image/phase input unavailable for {run}, frame {index}")
        frames.append(dict(row, photo=str(photo), phase=str(phase)))
        previous = hours
    baseline = _native_reference_path(root, run)
    stages = context.get("injection_stages") or (native.get("phase_topology") or {}).get("injection_stages")
    return dict(format=FORMAT, dataset=job["dataset"], run=run,
        identity={key: job["candidate"].get(key) for key in IDENTITY_KEYS},
        model_id=job.get("model_id"), model_root=job.get("model_root"),
        native_manifest=str(native_path), native_manifest_sha256=native_sha,
        source_manifest=job.get("source_manifest"), source_manifest_sha256=job.get("source_manifest_sha256"),
        canvas_wh=list(canvas), content_box_xywh=list(box), shape_rc=[box[3], box[2]],
        frames=frames, baseline=str(baseline), baseline_reference=native.get("reference", {}), context=context,
        geometry_metadata=(context.get("provenance") or {}).get("geometry", {}),
        injection_stages=stages, correction_contract=native.get("correction_contract"),
        phase_display_policy=PHASE_DISPLAY_POLICY,
        limitations=["Categorical labels are not phase probabilities or concentrations.",
                    "Corrected JPEG pixels differ from the pre-encoding inference float array.",
                    "Depth, porosity and thermodynamic state require separate measured inputs."])


def _unchanged(run):
    _bytes(run["native_manifest"], run["native_manifest_sha256"])


def load_context(run: Mapping) -> dict[str, Any]:
    """Read only fields actually present in the registered native context."""
    _unchanged(run)
    record = run["context"]
    if not record.get("path"):
        raise ValueError("A registered native domain/context archive is required")
    content, digest = _bytes(record["path"], record.get("sha256"))
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("domain", "facies", "x_m", "y_m", "permeability")
                  if name in archive.files}
    shape = tuple(run["shape_rc"])
    domain = arrays.get("domain")
    if domain is None or domain.dtype != np.bool_ or domain.shape != shape:
        raise ValueError("Registered domain differs from the native concentration grid")
    for key in ("facies", "permeability"):
        if key in arrays and (arrays[key].shape != shape or not np.isfinite(arrays[key]).all()):
            raise ValueError(f"Registered {key} differs from the native concentration grid")
    if "facies" in arrays and not np.issubdtype(arrays["facies"].dtype, np.integer):
        raise ValueError("Registered facies must contain integer categorical IDs")
    spacing = {}
    for key, size in (("x_m", shape[1]), ("y_m", shape[0])):
        if key not in arrays:
            continue
        vector = arrays[key]
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid registered coordinate axis: {key}")
        steps = np.diff(vector)
        if steps.size and (not (np.all(steps > 0) or np.all(steps < 0))
                           or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-10)):
            raise ValueError(f"Registered {key} is not a uniform native axis")
        if steps.size:
            spacing[key] = abs(float(steps[0]))
    for key in ("x_m", "y_m", "permeability"):
        if key in arrays:
            arrays[key] = np.asarray(arrays[key], dtype=np.float32)
    for value in arrays.values():
        value.flags.writeable = False
    return {**arrays, "spacing_m_rc": [spacing["y_m"], spacing["x_m"]] if len(spacing) == 2 else None,
            "source_reference": {"path": str(_path(record["path"])), "sha256": digest,
                "checksum_previously_recorded": bool(record.get("sha256")),
                "native_manifest_sha256": run["native_manifest_sha256"]}}


def nominal_injected_mass_g(stages, time_hours: float) -> float | None:
    """Integrate explicitly recorded kg/s segments; missing protocols stay unknown."""
    if not stages:
        return None
    hours = float(time_hours)
    if not math.isfinite(hours):
        raise ValueError("Injection time must be finite")
    total = 0.0
    for stage in stages:
        segments = stage.get("rate_segments")
        if not segments:
            return None
        previous_end = -math.inf
        for segment in segments:
            start, end, rate = (float(segment[name]) for name in ("start_h", "end_h", "rate_kg_s"))
            if (not all(map(math.isfinite, (start, end, rate))) or end <= start or rate < 0
                    or start < previous_end):
                raise ValueError("Invalid or overlapping injection-rate segments")
            total += max(0.0, min(hours, end) - start) * 3600.0 * rate * 1000.0
            previous_end = end
    return total


def _crop(image, run):
    x, y, width, height = run["content_box_xywh"]
    if image.size == (width, height):
        return image
    if image.size != tuple(run["canvas_wh"]):
        raise ValueError("Image/phase dimensions differ from the registered native grid")
    return image.crop((x, y, x + width, y + height))


def decode_fixed_phase(content: bytes, run: Mapping) -> np.ndarray:
    """Decode only the exact native categorical palette, without interpolation."""
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.mode != "RGBA":
            raise ValueError("A categorical RGBA PNG is required, not a display screenshot")
        rgba = np.asarray(_crop(image, run))
    labels = np.zeros(tuple(run["shape_rc"]), dtype=np.uint8)
    known = rgba[..., 3] == 0
    for color, label in ((AQUEOUS_RGBA, 1), (GAS_RGBA, 2)):
        mask = np.all(rgba == np.asarray(color, dtype=np.uint8), axis=-1)
        labels[mask] = label
        known |= mask
    if not known.all():
        raise ValueError("Unknown or blended categorical phase colors; fixed phase labels cannot be inferred")
    labels.flags.writeable = False
    return labels


def load_baseline(run: Mapping) -> tuple[np.ndarray, dict]:
    """Read the saved corrected baseline on the same grid as current frames."""
    _unchanged(run)
    path = _path(run["baseline"])
    content, digest = _bytes(path)
    with Image.open(io.BytesIO(content)) as image:
        encoding = image.format
        if encoding not in {"JPEG", "PNG"} or image.mode != "RGB":
            raise ValueError("A registered RGB JPEG/PNG baseline is required")
        rgb = np.asarray(_crop(image, run), dtype=np.float32) / np.float32(255.)
    rgb.flags.writeable = False
    return rgb, {"path": str(path), "sha256": digest, "encoding": encoding,
                 "native_manifest_sha256": run["native_manifest_sha256"],
                 "registered_reference": run.get("baseline_reference", {}),
                 "preencoding_float_array_recovered": False}


@dataclass(frozen=True)
class NativeConcentrationFrame:
    frame_index: int
    time_hours: float
    corrected_rgb: np.ndarray
    phase_labels: np.ndarray
    nominal_injected_mass_g: float | None
    source_reference: Mapping
    pixel_area_m2: np.ndarray | float | None = None


def frame_pixel_area(run: Mapping, index: int):
    """Use a declared rig-plane area on this native mass grid, or legacy IO."""
    row = run["frames"][index]
    if not row.get("rig_domain_revision") and not row.get("pixel_area"):
        return None
    from .phase_pixel_area import load_frame_pixel_area
    return load_frame_pixel_area(run["native_manifest"], index, run["shape_rc"],
                                  expected_native_sha256=run["native_manifest_sha256"])


def _require_bottom_mass_policy(run: Mapping) -> None:
    """Fail closed while this candidate's blind bottom-strip check is pending.

    Merely carrying an exclusion descriptor cannot remove its pixels from a
    mass integral. Until that consumer exists, keep the whole affected mass
    input blocked rather than silently include the disputed strip.
    """
    location = run.get("model_root") or run.get("native_manifest")
    if location:
        from .phase_saved_contact_review import require_mass_exclusion as require_final3_mass_exclusion
        require_final3_mass_exclusion(location)
        from .phase_saved_categorical_review import require_mass_exclusion
        require_mass_exclusion(location)
        from .phase_rig_bottom_review_policy import require_bottom_mass_exclusion
        exclusion = require_bottom_mass_exclusion(location)
        if exclusion is not None:
            raise ValueError("Bottom-strip exclusion must be applied before mass calculation; "
                             "this mass reader does not apply the bound exclusion mask")


def load_frame(run: Mapping, index: int) -> NativeConcentrationFrame:
    """Read one fixed frame; no optical inference, smoothing or phase edits."""
    _require_bottom_mass_policy(run)
    _unchanged(run)
    if type(index) is not int or not 0 <= index < len(run["frames"]):
        raise IndexError("Concentration frame index is outside the registered timeline")
    row = run["frames"][index]
    photo_path = _path(row["photo"])
    expected = row.get("native_jpeg_sha256") if photo_path.suffix.casefold() in {".jpg", ".jpeg"} else row.get("native_png_sha256")
    content, photo_sha = _bytes(photo_path, expected)
    with Image.open(io.BytesIO(content)) as image:
        encoding = image.format
        if encoding not in {"JPEG", "PNG"} or image.mode != "RGB":
            raise ValueError("A registered RGB JPEG/PNG photograph is required")
        rgb = np.asarray(_crop(image, run), dtype=np.float32) / np.float32(255.0)
    del content
    content, phase_sha = _bytes(row["phase"], row.get("phase_sha256"))
    labels = decode_fixed_phase(content, run)
    rgb.flags.writeable = False
    area = frame_pixel_area(run, index)
    coverage = None
    if row.get("rig_domain_revision"):
        from .phase_pixel_area import rig_mass_coverage
        coverage = rig_mass_coverage(row["rig_domain_revision"], run["shape_rc"],
                                      base_dir=Path(run["native_manifest"]).parent)
    return NativeConcentrationFrame(index, float(row["elapsed_h"]), rgb, labels,
        nominal_injected_mass_g(run.get("injection_stages"), row["elapsed_h"]),
        {"identity": dict(run["identity"]), "native_manifest_sha256": run["native_manifest_sha256"],
         "photo": str(photo_path), "photo_sha256": photo_sha, "photo_encoding": encoding,
         "photo_checksum_previously_recorded": bool(expected),
         "phase": str(_path(row["phase"])), "phase_sha256": phase_sha,
         "phase_semantics": "fixed_categorical_0_water_1_aqueous_2_free_gas",
         "phase_inference_repeated": False, "phase_labels_modified": False,
         "corrected_array_sha256_provenance_only": row.get("corrected_array_sha256"),
         "preencoding_float_array_recovered": False,
         "source_name": row.get("source_name"), "raw_source_sha256": row.get("source_sha256"),
         **({"pixel_area": area.audit()} if area is not None else {}),
         **({"rig_domain_coverage": coverage} if coverage is not None else {})},
        area.values_m2 if area is not None else None)


def iter_frames(run: Mapping) -> Iterator[NativeConcentrationFrame]:
    """Stream native frames without retaining an image stack."""
    for index in range(len(run["frames"])):
        yield load_frame(run, index)


@dataclass(frozen=True)
class V5NativeScientificInputs:
    """One source-bound RAW frame and its baseline in the frame's native grid.

    ``valid_mask`` is the joint current/baseline optical footprint. Zeros
    outside it are transport values, never evidence for zero concentration.
    ``context`` is per-frame and provisional; its facies and physical scale
    have not acquired mass-analysis approval merely by being loaded here.
    """

    frame_index: int
    time_hours: float
    scientific_rgb: np.ndarray
    registered_baseline_rgb: np.ndarray
    valid_mask: np.ndarray
    supported_mask: np.ndarray
    phase_labels: np.ndarray
    unresolved_mask: np.ndarray
    context: Mapping[str, np.ndarray]
    nominal_injected_mass_g: float | None
    source_reference: Mapping[str, Any]


def _decode_v5_fixed_phase(content: bytes, run: Mapping
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Retain Track 1's unresolved tint without calling it water/zero mass."""
    from .phase_validation import PHASE_FILL_COLORS

    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.mode != "RGBA":
            raise ValueError("A native categorical RGBA phase PNG is required")
        rgba = np.asarray(_crop(image, run))
    labels = np.zeros(tuple(run["shape_rc"]), np.uint8)
    known = rgba[..., 3] == 0
    for name, label in (("aqueous", 1), ("gas", 2)):
        mask = np.all(rgba == np.asarray(PHASE_FILL_COLORS[name], np.uint8),
                      axis=-1)
        labels[mask] = label
        known |= mask
    unresolved = np.all(rgba == np.asarray(
        PHASE_FILL_COLORS["unresolved"], np.uint8), axis=-1)
    known |= unresolved
    if not known.all():
        raise ValueError("Unknown or blended V5 phase colors")
    labels.flags.writeable = False
    unresolved.flags.writeable = False
    return labels, unresolved


class V5NativeScientificInputReader:
    """Explicit FF_AC V5 float32 input path for future native calibration.

    Existing ``load_baseline`` / ``load_frame`` continue to read legacy JPEG
    inputs. This reader does not load a Color Path or perform a mass fit. In
    particular, historical JPEG Color Paths must not be paired with its RAW
    values. The B reference is loaded once, then registered independently to
    every frame using that frame's SHA-bound V5 geometry receipt.
    """

    def __init__(self, run: Mapping):
        from .phase_training.log_linear_input import RunV5LinearInput
        from .phase_training.frame_context import V5FrameGeometryCohort
        from .phase_training.photo_context import Track1FramePhotoCohort

        _require_bottom_mass_policy(run)
        _unchanged(run)
        if str(run.get("dataset", "")).casefold() != "ffac":
            raise ValueError("V5 scientific concentration input is FF_AC only")
        if not run.get("native_manifest_sha256") or not run.get("frames"):
            raise ValueError("V5 scientific input needs an exact native timeline")
        shape = tuple(run["shape_rc"])
        self.run = run
        self.shape = shape
        manifest, _ = _read(run["native_manifest"],
                            run["native_manifest_sha256"])
        physical = bool(manifest.get("v5_frame_geometry"))
        photo_only = bool(manifest.get("v5_track1_photo_geometry"))
        if physical == photo_only:
            raise ValueError("V5 native revision needs one exact geometry cohort")
        self.geometry = (V5FrameGeometryCohort(run["native_manifest"])
                         if physical else
                         Track1FramePhotoCohort(run["native_manifest"]))
        self.physical_context_available = physical
        self.linear = RunV5LinearInput(
            native_manifest_path=run["native_manifest"], run_id=run["run"],
            content_shape_hw=shape)
        if (self.geometry.native_manifest_sha256 != run["native_manifest_sha256"]
                or self.linear.manifest_sha256 != run["native_manifest_sha256"]
                or self.geometry.shape_hw != shape):
            raise ValueError("V5 RAW and per-frame geometry use different native revisions")
        self.baseline_path = _path(run["baseline"])
        self.baseline_rgb = self.linear.load(self.baseline_path, "pre_injection")
        self.baseline_valid, origin = self.linear.load_validity(
            self.baseline_path, "pre_injection", extended=False)
        if (origin != (0, 0) or self.baseline_rgb.shape != (*shape, 3)
                or self.baseline_rgb.dtype != np.float32
                or self.baseline_valid.shape != shape):
            raise ValueError("V5 RAW baseline is not on its exact native grid")
        self.baseline_rgb.flags.writeable = False
        self.baseline_valid.flags.writeable = False
        self.baseline_source = dict(self.linear.receipts[-1])

    def load_frame(self, index: int) -> V5NativeScientificInputs:
        """Read one unmodified phase map and two registered RAW optical inputs."""
        import cv2

        _unchanged(self.run)
        if type(index) is not int or not 0 <= index < len(self.run["frames"]):
            raise IndexError("V5 scientific frame is outside its native timeline")
        row = self.run["frames"][index]
        if row.get("frame_index") != index:
            raise ValueError("V5 scientific frame index changed")
        if self.physical_context_available:
            fields = ("physics_supported_domain", "physical_sand_domain",
                      "photo_certified_sand_domain", "facies",
                      "pixel_area_m2")
            context, binding = self.geometry.context(index, fields=fields)
            required = context["physics_supported_domain"] & context[
                "photo_certified_sand_domain"]
        else:
            sand, uncertain, binding = self.geometry.photo(index)
            context = dict(photo_certified_sand_domain=sand,
                           photo_boundary_uncertain_domain=uncertain)
            required = sand
        if binding["frame_index"] != index:
            raise ValueError("V5 frame geometry belongs to another image")
        area = frame_pixel_area(self.run, index)
        if area is not None:
            context["pixel_area_m2"] = np.broadcast_to(area.values_m2, self.shape)
        photo_path = _path(row["photo"])
        frame_id = f"frame_{index:03d}"
        # A source-validity hole may intersect even the transported safe B
        # photo core. Keep that pixel unresolved instead of turning its zero
        # transport value into concentration evidence (or discarding a whole
        # review frame). The exact per-frame mask is still SHA-verified.
        rgb = self.linear.load(photo_path, frame_id,
                               required_valid_mask=None)
        current_valid, origin = self.linear.load_validity(
            photo_path, frame_id, extended=False)
        if (origin != (0, 0) or rgb.shape != (*self.shape, 3)
                or rgb.dtype != np.float32
                or current_valid.shape != self.shape):
            raise ValueError("V5 RAW frame is not on its exact native grid")

        h = np.asarray(binding["H_v5_B_to_frame_native"], dtype=np.float64)
        baseline = cv2.warpPerspective(
            self.baseline_rgb, h, (self.shape[1], self.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0)
        coverage = cv2.warpPerspective(
            self.baseline_valid.astype(np.float32), h,
            (self.shape[1], self.shape[0]), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        valid = current_valid & (coverage >= np.float32(1 - 1e-6))
        phase_content, phase_sha = _bytes(row["phase"], row.get("phase_sha256"))
        labels, unresolved = _decode_v5_fixed_phase(phase_content, self.run)
        supported = valid & required & ~unresolved
        if (labels.shape != self.shape
                or not np.any(supported)):
            raise ValueError("V5 phase or scientific support is absent")
        for array in (rgb, baseline, valid, supported, *context.values()):
            array.flags.writeable = False
        frame_source = dict(self.linear.receipts[-1])
        provenance = dict(
            input_domain="ffac-v5-camera-linear-float32-raw-v1",
            native_manifest=str(_path(self.run["native_manifest"])),
            native_manifest_sha256=self.run["native_manifest_sha256"],
            baseline_scientific_float=self.baseline_source,
            frame_scientific_float=frame_source,
            frame_geometry_receipt=binding["receipt_path"],
            frame_geometry_receipt_sha256=binding["receipt_sha256"],
            H_v5_B_to_frame_native=h.tolist(),
            phase=str(_path(row["phase"])), phase_sha256=phase_sha,
            phase_semantics="fixed_categorical_0_water_1_aqueous_2_free_gas",
            unresolved_semantics="source-unresolved-not-water-or-zero-mass",
            phase_labels_modified=False, phase_inference_repeated=False,
            source_name=row.get("source_name"),
            raw_source_sha256=row.get("source_sha256"),
            mass_eligible=False,
            physical_context_available=self.physical_context_available,
            facies_available="facies" in context,
            historical_jpeg_color_paths_compatible=False,
            valid_pixels=int(np.count_nonzero(valid)),
            supported_pixels=int(np.count_nonzero(supported)),
            unresolved_pixels=int(np.count_nonzero(unresolved)),
            phase_pixels_outside_supported=int(np.count_nonzero(
                (labels != 0) & ~supported)),
            **({"pixel_area": area.audit()} if area is not None else {}))
        return V5NativeScientificInputs(
            index, float(row["elapsed_h"]), rgb, baseline, valid, supported,
            labels, unresolved, context,
            nominal_injected_mass_g(self.run.get("injection_stages"),
                                    row["elapsed_h"]), provenance)
