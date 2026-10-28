"""Configuration-driven native N1 whole-run computation on direct file inputs.

The public entrypoint is :func:`run_n1`. Queue ownership, code/environment
admission and process lifetime belong to its caller. This module contains no
campaign orchestration, source-code loader, fitting or equivalence validator.

Configuration format ``darsia-n1-run-v1`` has an ordered ``frames`` list,
``shared`` saved optics/physical data, one ``calibration`` descriptor, the
already measured ``physical_correction``, a genuine observed-volume ``anchor``,
``recipe``, ``dimensions_m``, ``boxes`` and provenance ``bindings``. Each frame
provides native current/baseline encodings, fixed labels/classes/areas and its
own registration, original-domain row and atmosphere/dose. Descriptors retain
their original SHA; ``data_aliases`` changes only a logical file's locator.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import csv
import gzip
import hashlib
import json
import ntpath
import os
import socket
import time
import uuid
from ..pipeline.io_ops import retry_io
from .phase_expert_knowledge import expert_knowledge_of

CONFIG_FORMAT = "darsia-n1-run-v1"
CONFIG_REQUIRED = (
    "run",
    "frames",
    "shared",
    "calibration",
    "physical_correction",
    "anchor",
    "recipe",
    "dimensions_m",
    "boxes",
    "summary",
    "bindings",
)
FRAME_REQUIRED = (
    "case_id",
    "frame_index",
    "elapsed_h",
    "selected_target_h",
    "source_binding",
    "current",
    "baseline",
    "labels",
    "classes",
    "area",
    "original_domain",
    "photo_top",
    "inner_polygon_native",
)
SHARED_REQUIRED = (
    "physical_reference",
    "floor_curve",
    "floor_origin_xy",
    "depth_points",
    "porosity_by_facies",
    "swi_by_facies",
    "clearwater",
    "optics",
    "unquantified_facies",
)


def _key(path):
    """Compare Windows logical data locators without rewriting descriptors."""
    return ntpath.normcase(ntpath.normpath(str(path)))


def _is_unc(path):
    """Recognize a genuine server/share locator without mapped drives."""
    drive, _ = ntpath.splitdrive(ntpath.normpath(str(path)))
    return drive.startswith("\\\\")


def _strict_json(raw):
    """Read UTF8 JSON with duplicate keys and nonfinite numbers rejected."""

    def pairs(items):
        """Build an unambiguous JSON mapping."""
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        """Reject a nonfinite JSON literal."""
        raise ValueError("Nonfinite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _stamp(path):
    """Actual file identity used to retain one consumed-data SHA verification."""
    from ..pipeline.job_queue import retry_transient
    value = retry_transient(path.stat)
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def asset(path):
    """Return the actual full SHA/size descriptor of a produced file."""
    from ..pipeline.io_ops import retry_io
    return retry_io(_asset_once)(path)


def _asset_once(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


class DataReader:
    """Direct data reader with explicit locator aliases and no staging route."""

    def __init__(self, aliases=(), *, unc_only=False):
        """Bind original data descriptors to declared byte-identical files."""
        self.aliases, self.verified, self.baselines = {}, {}, {}
        self.unc_only = unc_only
        from ..pipeline.run_io import current_binding
        self.io_binding = current_binding()
        if self.io_binding is not None:
            self.verified = self.io_binding.reader_verifications()
        for row in aliases:
            logical, physical = row["logical"], row["physical"]
            if logical["sha256"] != physical["sha256"]:
                raise ValueError("Data alias changes the original content SHA")
            if (
                "size_bytes" in logical
                and logical["size_bytes"] != physical["size_bytes"]
            ):
                raise ValueError("Data alias changes the original size")
            key = _key(logical["path"])
            if key in self.aliases and self.aliases[key] != physical:
                raise ValueError("Conflicting data alias")
            self.aliases[key] = dict(physical)

    def path(self, descriptor):
        """Verify a consumed data file once; a later changed identity fails."""
        if not isinstance(descriptor, dict) or not {"path", "sha256"}.issubset(
            descriptor
        ):
            raise ValueError("A bound data descriptor is required")
        physical = self.aliases.get(_key(descriptor["path"]), descriptor)
        if self.io_binding is not None:
            physical = self.io_binding.physical_descriptor(descriptor)
        if physical["sha256"] != descriptor["sha256"]:
            raise ValueError("Alias and original data identity differ")
        if self.io_binding is None and self.unc_only and not _is_unc(physical["path"]):
            raise ValueError(
                "Production inputs must resolve directly to UNC: "
                + str(physical["path"])
            )
        path = Path(physical["path"])
        stamp = _stamp(path)
        size = descriptor.get("size_bytes", physical.get("size_bytes"))
        if size is not None and (type(size) is not int or size != stamp[0]):
            raise ValueError("Data size differs: " + str(path))
        identity = (_key(path), descriptor["sha256"])
        if identity in self.verified:
            if self.verified[identity] != stamp:
                raise ValueError("Previously consumed data changed: " + str(path))
            return path
        actual = asset(path)
        if actual["sha256"] != descriptor["sha256"] or _stamp(path) != stamp:
            raise ValueError("Data SHA or identity differs: " + str(path))
        self.verified[identity] = stamp
        return path

    def json(self, descriptor):
        """Read a consumed JSON data document, never a code/control snapshot."""
        path = self.path(descriptor)
        from ..pipeline.job_queue import retry_transient
        raw = retry_transient(path.read_bytes)
        if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("Consumed JSON data changed: " + str(path))
        result = _strict_json(raw)
        self.path(descriptor)
        return result

    @retry_io
    def array(self, descriptor):
        """Read exact NPY/explicit NPZ-array storage without dtype conversion."""
        import numpy as np

        path = self.path(descriptor)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                name = descriptor.get("array_key")
                if name is None:
                    if len(archive.files) != 1:
                        raise ValueError("An explicit NPZ array key is required")
                    name = archive.files[0]
                result = archive[name].copy()
        else:
            result = np.load(path, allow_pickle=False)
        self.path(descriptor)
        return result

    @retry_io
    def archive(self, descriptor, names, *, optional=()):
        """Read declared registered fields from one physical/class archive."""
        import numpy as np

        path = self.path(descriptor)
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name].copy() for name in names}
            result.update({name: archive[name].copy() for name in optional if name in archive.files})
        self.path(descriptor)
        return result


def _geometry_fields(config, supplied=None):
    """Require explicit source and provenance field names without alias searches."""
    required = (
        "source_to_native",
        "original_domain_to_native",
        "carried_homography_provenance",
    )
    existing = config.get("geometry_fields")
    if supplied is not None:
        if existing is not None and existing != supplied:
            raise ValueError("Explicit geometry field mappings differ")
        existing = supplied
    if not isinstance(existing, dict) or set(existing) != set(required):
        raise ValueError("The complete explicit geometry_fields mapping is required")
    if any(
        not isinstance(existing[name], str) or not existing[name] for name in required
    ):
        raise ValueError("Geometry field names must be nonempty strings")
    return dict(existing)


def load_n1_run(descriptor, *, unc_only=False, geometry_fields=None):
    """Read the explicit whole-run configuration and check its metadata scope."""
    reader = DataReader(unc_only=unc_only)
    config = reader.json(descriptor)
    config = dict(config, geometry_fields=_geometry_fields(config, geometry_fields))
    if config.get("format") != CONFIG_FORMAT:
        raise ValueError("Explicit N1 run configuration required")
    for key in CONFIG_REQUIRED:
        if key not in config:
            raise ValueError("Missing N1 configuration field: " + key)
    if not isinstance(config["run"], str) or not config["run"]:
        raise ValueError("A nonempty run identifier is required")
    if not config["frames"]:
        raise ValueError("An explicitly selected nonempty run is required")
    seen, previous = set(), None
    for frame in config["frames"]:
        for key in FRAME_REQUIRED:
            if key not in frame:
                raise ValueError("Missing frame field: " + key)
        identifier = frame["case_id"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen
            or Path(identifier).name != identifier
            or "/" in identifier
            or "\\" in identifier
        ):
            raise ValueError("Unique plain frame identifiers are required")
        if type(frame["frame_index"]) is not int or frame["frame_index"] < 0:
            raise ValueError("An exact acquisition index is required")
        position = (float(frame["elapsed_h"]), frame["frame_index"])
        if previous is not None and position <= previous:
            raise ValueError(
                "Selected frames must retain chronological acquisition order"
            )
        seen.add(identifier)
        previous = position
    for key in SHARED_REQUIRED:
        if key not in config["shared"]:
            raise ValueError("Missing shared N1 data: " + key)
    if not {"elapsed_h", "affected_pore_volume_m3"}.issubset(config["anchor"]):
        raise ValueError("A genuine existing observed-volume anchor is required")
    if config["summary"].get("intervals") != 32768:
        raise ValueError("The unchanged native paired-summary quantization is required")
    return config


def _depth_points(value, reader):
    """Load the original measured triples; no geometric fit is performed here."""
    import numpy as np

    if isinstance(value, dict):
        with reader.path(value).open(encoding="utf-8-sig", newline="") as stream:
            value = [
                [float(row[name]) for name in ("x", "y", "depth")]
                for row in csv.DictReader(stream)
            ]
    return np.asarray(value, np.float64)


def _atmosphere(config, spec):
    """Use explicit measured state or the exact original linear interpolation."""
    if spec.get("atmosphere") is not None:
        return dict(spec["atmosphere"])
    import numpy as np

    record = config["atmosphere_series"]
    rows = record["rows"]
    times = np.asarray([row["time_hours"] for row in rows], np.float64)
    elapsed = float(spec["elapsed_h"])
    if not np.isfinite(elapsed) or not times[0] <= elapsed <= times[-1]:
        raise ValueError("Atmosphere is outside observed range")
    return {
        name: float(np.interp(elapsed, times, [row[name] for row in rows]))
        for name in (
            "pressure_bar",
            "temperature_c",
            "pressure_gradient_bar_per_m",
            "temperature_gradient_c_per_m",
        )
    }


def _dose(config, spec):
    """Use genuine saved dose or the unchanged measured/nominal integration."""
    if spec.get("nominal_injected_mass_g") is not None:
        return float(spec["nominal_injected_mass_g"])
    import numpy as np

    row, elapsed = config["dose"], float(spec["elapsed_h"])
    if not np.isfinite(elapsed):
        raise ValueError("Dose needs finite camera-relative time")
    if row["dose_kind"] == "measured_mfc":
        measured = row["measured_mfc_dose"]
        return (
            0.0
            if elapsed <= 0
            else float(
                np.interp(elapsed, measured["elapsed_h"], measured["cumulative_mass_g"])
            )
        )
    return float(
        sum(
            max(0.0, min(elapsed, item["end_h"]) - item["start_h"])
            * 3600
            * item["mass_rate_g_s"]
            for item in row["injection_stages"]
        )
    )


def _local_build_accounting(frame, fields):
    """Describe full-report exclusions without assigning an unseen mass value."""
    from copy import deepcopy
    import numpy as np

    reporting = frame.domain
    material = fields["material_supported_mask"]
    phi = np.asarray(frame.porosity)
    missing_material = reporting & ~material
    unobserved_material = reporting & material & (phi > 0) & (expert_knowledge_of(frame).unknown)
    unquantified = missing_material | unobserved_material
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), reporting.shape)

    def describe(mask):
        return dict(pixel_count=int(mask.sum()), area_cm2=float(np.sum(area[mask])) * 1.e4,
            phase255_pixels=int(np.count_nonzero(mask & (expert_knowledge_of(frame).unknown))),
            unknown_physical_pixels=int(np.count_nonzero(mask & frame.unknown_physical_mask)),
            unquantified_pixels=int(np.count_nonzero(mask & unquantified)))

    return dict(format="darsia-local-build-accounting-v1", run=frame.run_id,
        frame_index=frame.frame_index, reporting=describe(reporting),
        analysis=describe(fields["analysis_domain"]),
        missing_material=describe(missing_material),
        unobserved_positive_porosity_material=describe(unobserved_material),
        known_zero_porosity_material=describe(reporting & material & (phi == 0)),
        excluded={name: describe(mask) for name, mask in sorted(frame.excluded_masks.items())},
        unquantified=describe(unquantified), unknown_mass_g=None,
        whole_rig_mass_complete=False if unquantified.any() else None,
        numeric_total_scope="existing quantified analysis; unobserved mass is not zero",
        numerical_results_changed=False,
        source_binding=deepcopy(fields["source_binding"]))


class N1Pipeline:
    """One run's explicit saved native data and unchanged typed arithmetic."""

    def __init__(
        self, config, *, precision, pixel_math, unc_only=False, geometry_fields=None,
        statistics_only=False, frame_loading_only=False, current_build=None, local_build=None
    ):
        """Load saved native inputs, optionally without any optical/mass model.

        ``frame_loading_only`` retains the original measured input preparation
        for callers fitting NEW optical paths. It cannot extract optical
        statistics or evaluate mass and does not certify adoption or Build.
        """
        if type(frame_loading_only) is not bool:
            raise TypeError("Frame-loading-only mode must be explicit")
        if frame_loading_only and statistics_only:
            raise ValueError("Frame loading and optical statistics are separate modes")
        if local_build is not None:
            from ..pipeline.local_build import LocalBuildInputs
            if not isinstance(local_build, LocalBuildInputs):
                raise TypeError("The actual local Build input adapter is required")
            if current_build is not None:
                raise ValueError("Local and current-plan Build inputs cannot be combined")
            if "pixel_domain" in config:
                raise ValueError("Local Build inputs use the original N1 statistics route")
            local_build.validate_mass_config(config, statistics_only=statistics_only,
                                             frame_loading_only=frame_loading_only)
        if current_build is not None:
            from ..pipeline.current_build import CheckedCurrentBuild
            if type(current_build) is not CheckedCurrentBuild:
                raise TypeError("The original checked current Build consumer is required")
            if frame_loading_only:
                raise ValueError("Current Build closure has no original frame-loading-only consumer join")
            if "pixel_domain" in config:
                raise ValueError("Current Build closure has no original pixel-domain consumer join")
            current_build.validate_mass_config(config, statistics_only=statistics_only)
            # Refuse before any historical physical archive can be decoded or
            # Freewater facies 0 can enter the original missing-sand correction.
            current_build.require_physical_inputs()
        self._current_build = current_build
        self._local_build = local_build
        self._local_prepared = None
        self.frame_loading_only = frame_loading_only
        from darsia._opencv_runtime import scientific_thread
        scientific_thread("N1Pipeline:setup")
        import numpy as np
        import darsia
        from .n1_arithmetic import validate_parameters

        config = dict(config, geometry_fields=_geometry_fields(config, geometry_fields))
        self.config, self.math = config, pixel_math
        if "opencl_acceleration" in config and hasattr(pixel_math, "configure_opencl"):
            pixel_math.configure_opencl(config["opencl_acceleration"])
        if "pixel_domain" in config:
            from .pixel_pipeline import validate_policy

            validate_policy(config["pixel_domain"])
        if precision not in {"float32", "float64"} or pixel_math.precision != precision:
            raise ValueError("Provider and working precision differ")
        self.reader = DataReader(config.get("data_aliases", ()), unc_only=unc_only)
        if type(statistics_only) is not bool:
            raise TypeError("Statistics-only mode must be explicit")
        self.statistics_only = statistics_only
        if frame_loading_only:
            pass  # Native preparation reads no mass calibration or optical method.
        elif statistics_only:
            self.method_bindings = dict(config["bindings"])
            for name in ("round4_spec_sha256", "amendment_sha256"):
                if name not in self.method_bindings:
                    raise ValueError("Statistics require explicit method binding: " + name)
        else:
            self.calibration = self.reader.json(config["calibration"])
            self.parameters = dict(self.calibration["parameters"])
            validate_parameters(self.parameters)
            self.method_bindings = dict(config["bindings"])
            for name in ("round4_spec_sha256", "amendment_sha256"):
                if name not in self.method_bindings and name in self.calibration:
                    self.method_bindings[name] = self.calibration[name]
        shared = config["shared"]
        if local_build is None:
            correction = self.reader.json(config["physical_correction"])
            self.correction = np.asarray(correction["correction_matrix"], np.float64)
            if (
                self.correction.shape != (3, 3)
                or not np.isfinite(self.correction).all()
                or self.correction[0, 0] <= 0
                or self.correction[0, 0] != self.correction[1, 1]
                or self.correction[0, 1] != 0
                or self.correction[1, 0] != 0
                or not np.array_equal(self.correction[2], (0.0, 0.0, 1.0))
            ):
                raise ValueError("The measured single shift/scale correction is required")
            self.physical = self.reader.archive(
                shared["physical_reference"], ("facies", "domain", "porosity", "depth_m")
            )
        else:
            # Local inputs supply the actual Build fields and measured metric
            # transform; historical physical archives are not an input fallback.
            self.physical = self.correction = None
        self.floor = self.reader.array(shared["floor_curve"])
        self.depth_points = _depth_points(shared["depth_points"], self.reader)
        if local_build is not None:
            self.reference_matrix = None
        elif "reference_matrix" in shared:
            self.reference_matrix = np.asarray(shared["reference_matrix"], np.float64)
        else:
            proof = shared["physical_coordinate_provenance"]
            if "path" in proof:
                proof = self.reader.json(proof)
            if proof.get("physical_grid_2d_exact") is not True:
                raise ValueError("An explicit projective physical grid is required")
            h, w = self.physical["facies"].shape
            height, width = map(float, shared["legacy_dimensions_m"])
            pixel_to_metre = np.asarray(
                [
                    [width / w, 0, width / (2 * w)],
                    [0, -height / h, height - height / (2 * h)],
                    [0, 0, 1.0],
                ]
            )
            self.reference_matrix = pixel_to_metre @ np.linalg.inv(
                np.asarray(proof["H_legacy_to_reference"], np.float64)
            )
        clear = shared["clearwater"]
        self.clear_rgb, self.clear_valid = self.reader.array(
            clear["rgb"]
        ), self.reader.array(clear["valid"])
        if (
            self.clear_rgb.dtype != np.float32
            or self.clear_valid.dtype != bool
            or self.clear_rgb.shape != (*self.clear_valid.shape, 3)
        ):
            raise ValueError("Frozen float32 scientific clear-water storage required")
        if frame_loading_only:
            return  # load_frame below uses only the verified native/physical inputs.
        optics = shared["optics"]
        metadata = self.reader.json(optics["metadata"])
        self.models = {
            "metadata": metadata,
            "colors": {
                int(label): darsia.ColorPathInterpolation.load(
                    self.reader.path(bound).with_suffix("")
                )
                for label, bound in optics["color_paths"].items()
            },
            "signals": {
                int(label): darsia.PWTransformation.load(
                    self.reader.path(bound).with_suffix("")
                )
                for label, bound in optics["signals"].items()
            },
            "flash": darsia.SimpleFlash.load(
                self.reader.path(optics["flash"]).with_suffix("")
            ),
        }
        if statistics_only:
            return
        boxes = config["boxes"]
        if "path" in boxes:
            from darsia.presets.workflows.config.roi_registry import RoiRegistry

            self.boxes = {
                name: value.roi
                for name, value in RoiRegistry()
                .load(self.reader.path(boxes))
                .resolve_rois(["box1", "box2"])
                .items()
            }
        else:
            self.boxes = boxes

    def load_frame(self, spec):
        """Decode/register one original frame and correct only measured physics."""
        import numpy as np
        import cv2
        from PIL import Image
        from . import n1_geometry as geometry
        from .n1_models import FrameData

        config, shared, reader, math = (
            self.config,
            self.config["shared"],
            self.reader,
            self.math,
        )
        local_build = self._local_build
        if local_build is not None:
            self._local_prepared = None
        current, valid = geometry.compose_raw(spec["current"], reader, math)
        shape = current.shape[:2]
        baseline_key = json.dumps(spec["baseline"], sort_keys=True)
        if baseline_key not in reader.baselines:
            reader.baselines[baseline_key] = geometry.compose_raw(
                spec["baseline"], reader, math
            )
        baseline, baseline_valid = reader.baselines[baseline_key]

        def matrix(raw):
            """Compose the original source-to-canvas homography."""
            origin = raw["native_origin_xy"]
            h = np.asarray(
                raw["binding"]["geometry"][
                    config["geometry_fields"]["source_to_native"]
                ],
                np.float64,
            )
            if (
                h.shape != (3, 3)
                or not np.isfinite(h).all()
                or abs(np.linalg.det(h)) < 1e-12
            ):
                raise ValueError("A registered native homography is required")
            return (
                np.asarray(
                    [
                        [1.0, 0.0, -float(origin[0])],
                        [0.0, 1.0, -float(origin[1])],
                        [0.0, 0.0, 1.0],
                    ]
                )
                @ h
            )

        current_matrix, baseline_matrix = matrix(spec["current"]), matrix(
            spec["baseline"]
        )
        warp = current_matrix @ np.linalg.inv(baseline_matrix)
        baseline = math.warp_continuous(baseline, warp, shape, cv2_module=cv2)
        coverage = math.warp_continuous(baseline_valid, warp, shape, cv2_module=cv2)
        clear = shared["clearwater"]
        clear_source_valid = (
            self.clear_valid
            & np.isfinite(self.clear_rgb).all(-1)
            & (self.clear_rgb > 0).all(-1)
        )
        clear_warp = current_matrix @ np.linalg.inv(
            np.asarray(clear["H_reference_to_grid"], np.float64)
        )
        clear_rgb = math.warp_continuous(
            np.where(clear_source_valid[..., None], self.clear_rgb, 0),
            clear_warp,
            shape,
            cv2_module=cv2,
        )
        clear_coverage = math.warp_continuous(
            clear_source_valid, clear_warp, shape, cv2_module=cv2
        )
        clear_supported = (
            (clear_coverage >= 1 - 1e-6)
            & np.isfinite(clear_rgb).all(-1)
            & (clear_rgb > 0).all(-1)
        )
        physical = (geometry.transport_physical_reference(
            self.physical, current_matrix, shape, math, cv2
        ) if local_build is None else local_build.physical_fields(spec, shape, math))
        with Image.open(reader.path(spec["labels"])) as image:
            labels = np.asarray(image).copy()
        if (
            labels.dtype != np.uint8
            or labels.shape != shape
            or not np.isin(labels, (0, 1, 2, 255)).all()
        ):
            raise ValueError("Fixed categorical phase labels/grid differ")
        classes = reader.archive(
            spec["classes"],
            ("rig_domain", "modelled", "filled_observed", "filled_unobserved"),
            **({} if local_build is None else dict(optional=(
                "physical_rig_domain", "approved_analysis_gap", "colorchecker_occlusion"))),
        )
        if any(
            value.dtype != bool or value.shape != shape for value in classes.values()
        ):
            raise ValueError("Rig classes use a different native grid")
        domain = classes["rig_domain"]
        if not np.array_equal(domain, labels != 255):
            raise ValueError("Fixed labels and original rig domain disagree")
        local_context = None
        if local_build is not None:
            phase_qa = reader.json(spec["phase_qa"]) if "phase_qa" in spec else None
            local_context = local_build.frame_context(spec, labels, classes, phase_qa)
            domain = local_context["analysis_domain"]
        area_descriptor = dict(spec["area"])
        area_descriptor.setdefault("array_key", "pixel_area_mm2")
        if local_build is None:
            area = np.asarray(reader.array(area_descriptor), np.float64) / 1e6
        else:
            original_area = reader.array(area_descriptor)
            local_build.check_area(spec, original_area)
            area = np.asarray(original_area, np.float64) / 1e6
        if area.shape != shape or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("Positive original per-pixel areas required")
        if "pixel_domain" in config:
            from .pixel_pipeline import prepare_frame

            return prepare_frame(
                self, spec, current=current, valid=valid, baseline=baseline,
                coverage=coverage, clear_rgb=clear_rgb, clear_supported=clear_supported,
                labels=labels, area=area, current_matrix=current_matrix,
            )
        top_descriptor = dict(spec["photo_top"])
        top_descriptor.setdefault("array_key", "keep")
        top = reader.array(top_descriptor)
        carried, carried_proof = geometry._carried_domain(
            spec["original_domain"],
            current_matrix,
            shape,
            reader,
            fields=config["geometry_fields"],
        )
        bottom = (
            geometry._bottom_side(
                domain,
                top,
                spec["inner_polygon_native"],
                spec["current"]["native_origin_xy"],
            )
            & ~carried
        )
        below = classes["filled_unobserved"] & ~bottom
        unknown = domain & ~physical["physical_supported_mask"]
        optical = valid & (coverage >= 1 - 1e-6) & clear_supported
        photo_support = (
            domain
            & optical
            & np.isfinite(current).all(-1)
            & np.isfinite(baseline).all(-1)
            & np.isfinite(clear_rgb).all(-1)
            & (current > 0).all(-1)
            & (baseline > 0).all(-1)
            & (clear_rgb > 0).all(-1)
        )
        raw_geometry = spec["current"]["binding"]["geometry"]
        revision = {
            "phase_shape_hw": list(shape),
            "native_origin_xy": spec["current"]["native_origin_xy"],
        }
        x, y = (geometry.frame_coordinates(
            self.reference_matrix,
            raw_geometry,
            revision,
            fields=config["geometry_fields"],
        ) if local_build is None else local_build.coordinates(spec, shape))
        floor = geometry.transport_curve(
            self.floor,
            H_source_to_target=np.asarray(
                raw_geometry[config["geometry_fields"]["source_to_native"]], np.float64
            ),
            source_origin_xy=shared["floor_origin_xy"],
            target_origin_xy=revision["native_origin_xy"],
            target_width=shape[1],
        )
        good = np.flatnonzero(np.isfinite(floor))
        if not len(good):
            raise ValueError("Transported sand floor has no valid columns")
        floor[: good[0]], floor[good[-1] + 1 :] = floor[good[0]], floor[good[-1]]
        if not np.isfinite(floor).all():
            raise ValueError("Transported floor has an interior gap")
        atmosphere = _atmosphere(config, spec)
        prepared = geometry.PreparedInputs(
            config["run"],
            spec["frame_index"],
            float(spec["elapsed_h"]),
            current,
            baseline,
            clear_rgb,
            clear_supported,
            labels,
            physical["facies"],
            domain,
            (domain & optical & ~unknown) if local_build is None else photo_support,
            domain & ~bottom & ~below,
            bottom,
            below,
            unknown,
            domain & np.isin(physical["facies"], shared["unquantified_facies"]),
            physical["depth_m"],
            physical["porosity"],
            area,
            config["recipe"],
            atmosphere,
            _dose(config, spec),
            tuple(config["dimensions_m"]),
            dict(
                spec.get("provenance", {}),
                source_binding=spec["source_binding"],
                original_bottom_domain=carried_proof,
            ),
        )
        facies = None
        if local_build is None and not np.array_equal(self.correction, np.eye(3)):
            facies = cv2.warpPerspective(
                np.asarray(self.physical["facies"], np.uint8),
                current_matrix @ self.correction,
                shape[::-1],
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        measured_fields = None
        if local_build is None:
            prepared, report, _ = geometry.correct_prepared(
                prepared,
                x_m=x,
                y_m=y,
                floor_curve_canvas_y=floor,
                depth_points=self.depth_points,
                porosity_by_facies=shared["porosity_by_facies"],
                swi_by_facies=shared["swi_by_facies"],
                optical_support=photo_support,
                transported_facies=facies,
                source_binding=dict(
                    source=spec["source_binding"],
                    physical_reference=shared["physical_reference"],
                    correction_matrix=self.correction.tolist(),
                ),
                unquantified_facies=shared["unquantified_facies"],
            )
        else:
            prepared, report, measured_fields = geometry.prepare_build_physical(
                prepared, x_m=x, y_m=y, floor_curve_canvas_y=floor,
                depth_points=self.depth_points, facies=physical["facies"],
                porosity=physical["porosity"], swi=physical["swi"],
                analysis_domain=local_context["analysis_domain"],
                reporting_domain=local_context["reporting_domain"],
                material_supported_mask=local_context["material_supported_mask"],
                excluded_masks=local_context["excluded_masks"],
                source_binding=local_context["source_binding"])
        known = prepared.domain & ~prepared.unknown_physical_mask
        bottom = prepared.bottom_strip_mask & known
        below = prepared.below_photo_filled_mask & known & ~bottom
        provenance = dict(prepared.provenance)
        excluded_masks = {"bottom_strip": bottom, "below_photo_filled": below}
        if local_context is not None:
            excluded_masks = measured_fields["excluded_masks"]
            if local_context["exclusion_provenance"] is not None:
                from copy import deepcopy
                provenance["approved_analysis_gap"] = deepcopy(local_context["exclusion_provenance"])
        frame = FrameData(
            run_id=prepared.run,
            frame_index=prepared.frame_index,
            current_rgb=prepared.current_rgb,
            baseline_rgb=prepared.baseline_rgb,
            clear_water_rgb=prepared.clear_water_rgb,
            clear_water_supported_mask=prepared.clear_water_supported_mask,
            phase_labels=prepared.phase_labels,
            facies=prepared.facies,
            domain=prepared.domain,
            supported_mask=prepared.supported_mask,
            mass_eligible_mask=prepared.mass_eligible_mask,
            depth_m=prepared.depth_m,
            porosity=prepared.porosity,
            pixel_area_m2=prepared.pixel_area_m2,
            dimensions_m=prepared.dimensions_m,
            pressure_bar=atmosphere["pressure_bar"],
            temperature_c=atmosphere["temperature_c"],
            pressure_gradient_bar_per_m=atmosphere.get(
                "pressure_gradient_bar_per_m", 0.0
            ),
            temperature_gradient_c_per_m=atmosphere.get(
                "temperature_gradient_c_per_m", 0.0
            ),
            elapsed_h=prepared.elapsed_h,
            nominal_injected_mass_g=prepared.nominal_injected_mass_g,
            recipe=dict(prepared.recipe),
            provenance=provenance,
            working_dtype=math.precision,
            excluded_masks=excluded_masks,
            unknown_physical_mask=prepared.unknown_physical_mask,
        )
        if local_build is not None:
            from .n1_arithmetic import analysis_gap_scope
            analysis_gap_scope(frame)
            frame.provenance["local_build_accounting"] = _local_build_accounting(frame, measured_fields)
            self._local_prepared = (frame, measured_fields)
        return frame, x, y, photo_support, report

    def _prepare_statistics(self, spec, *, colour_coordinate=False):
        """Run the original calibration-independent native summary prefix once.

        The fixed summary baseline_gain, beta=0 and historical summary anchor
        remain inside summarize_round3_response. No fitted mass parameters are
        read here. ``colour_coordinate`` only captures the colour-path
        coordinate of the interpretation (storage_profile option).
        """
        import numpy as np
        import darsia
        from .n1_models import prepare_response, own_colour_response
        from .n1_arithmetic import evaluate_main, main_products
        from .n1_statistics import (
            build_estimate_geometry,
            summarize_round3_response,
            exact_prediction,
        )

        if getattr(self, "frame_loading_only", False):
            raise ValueError("A frame-loading-only context cannot extract optical statistics")
        frame, x, y, photo, report = self.load_frame(spec)
        local_fields = None
        if self._local_build is not None:
            retained, self._local_prepared = self._local_prepared, None
            if retained is None or retained[0] is not frame:
                raise ValueError("Local physical fields lost their single loaded frame")
            local_fields = retained[1]
        current_build = getattr(self, "_current_build", None)
        if current_build is not None:
            current_build.check_mass(spec, frame, x, y, photo, report, mass_config=self.config)
        shared, math, config = self.config["shared"], self.math, self.config
        # The default call stays exactly the original one.
        capture = dict(colour_coordinate=True) if colour_coordinate else {}
        response = prepare_response(frame, self.models, math, shared["swi_by_facies"],
                                    **capture)
        geometry = build_estimate_geometry(
            x,
            y,
            frame.facies,
            self.depth_points,
            porosity_by_facies=shared["porosity_by_facies"],
            swi_by_facies=shared["swi_by_facies"],
            source_binding=(config.get("geometry_bindings", {}) if local_fields is None
                            else local_fields["source_binding"]),
            **({} if local_fields is None else dict(prepared_fields=local_fields)),
        )
        own = own_colour_response(response, geometry, photo, self.models, math)
        if spec.get("no_outflow_evidence") is not None:
            no_outflow = spec["no_outflow_evidence"]
        else:
            occupied = frame.domain.any(0)
            first = frame.domain.argmax(0)
            contact = int(
                np.count_nonzero(
                    occupied
                    & np.isin(
                        frame.phase_labels[first, np.arange(frame.domain.shape[1])],
                        (1, 2),
                    )
                )
            )
            no_outflow = dict(
                no_co2_left_rig=contact == 0,
                co2_contact_columns=contact,
                source=spec.get("source_receipt"),
                status="inferred_locked_phase_open_top_proxy",
                spatial_margin_px=0,
                limitation="No measured CO2 outlet; fixed phase/top rules may weaken optical evidence",
                measured_water_displacement_is_not_CO2_outlet=True,
            )
        row = summarize_round3_response(
            response,
            geometry,
            own_colour=own,
            photo_support=photo,
            no_outflow_evidence=no_outflow,
            light_epoch=spec["light_epoch"],
            source_binding=spec["source_binding"],
            pixel_math=math,
            summary_settings=config["summary"],
        )
        candidate = (spec.get("provenance") or {}).get("fit_candidate")
        target = spec["selected_target_h"]
        if target is None and candidate is not False:
            raise ValueError("A missing nominal target requires explicit history-only provenance")
        row.update(
            selected_target_h=None if target is None else float(target),
            precision=math.precision,
            physical_correction=config["physical_correction"],
            scope="frozen_evaluation",
            round4_spec_sha256=self.method_bindings.get("round4_spec_sha256"),
            round4_amendment_sha256=self.method_bindings.get("amendment_sha256"),
        )
        if candidate is not None:
            if type(candidate) is not bool:
                raise ValueError("Fit candidate provenance must be explicit boolean DATA")
            row["fit_candidate"] = candidate
        if local_fields is not None:
            from copy import deepcopy
            row["local_build_accounting"] = deepcopy(frame.provenance["local_build_accounting"])
        return dict(frame=frame, x_m=x, y_m=y, response=response,
                    statistics=row, sandmap=report)

    def extract_statistics(self, spec):
        """Expose the original summary row without evaluating fitted N1 mass.

        A phase-one caller constructs this context with statistics_only=True.
        This method writes no files, commits no frame and advances no history.
        """
        self.last_opencv_ipp = None
        from darsia._opencv_runtime import scientific_thread, completed_frame
        ipp_before = scientific_thread("N1Pipeline.extract_statistics:before")
        prepared = self._prepare_statistics(spec)
        self.last_opencv_ipp = completed_frame(ipp_before, "N1Pipeline.extract_statistics:after")
        return dict(statistics=prepared["statistics"], sandmap=prepared["sandmap"])

    def _overflow_phase_split(self, spec, prepared):
        """The image's overflow-governed free gas, when its sealed plan applies one.

        None without the plan's switch. Otherwise the report records the
        image's target (its own measurement, a pressure-matched estimate or
        none); with a target, the distribution follows the existing
        probability law over the image's stored gas probabilities.
        """
        block = getattr(self, "config", {}).get("overflow_phase_split")
        if block is None:
            return None
        image = block["frames"][spec["case_id"]]
        target = image["target"]
        if target is None:
            return dict(report=dict(policy=block["policy"], kind="none", reason=image["reason"],
                                    labels_modified=False))
        from .overflow_phase_split import distribute_target
        from .overflow_statistics import frame_population
        population = frame_population(self.reader, prepared, probability_roles=image["probability_roles"],
            phase_source=image["phase_source"], case_id=spec["case_id"], labels=spec["labels"])
        distribution = distribute_target(population, target["target_volume_m3"])
        report = dict(policy=block["policy"], kind=target["kind"], estimated=target["estimated"],
            target_volume_m3=target["target_volume_m3"], achieved_volume_m3=distribution["actual_volume_m3"],
            threshold=distribution["threshold"], mode=distribution["mode"],
            extended_zero_score_pixels=distribution["extended_zero_score_pixels"],
            float32_volume_residual_m3=distribution["float32_volume_residual_m3"], labels_modified=False)
        if target["kind"] == "pressure_matched_estimate":
            report.update(donors=[dict(run=row["run"], pressure_bar=row["pressure_bar"], weight=row["weight"])
                                  for row in target["donors"]],
                          pressure_difference_bar=target["pressure_difference_bar"])
        return dict(population=population, distribution=distribution, report=report)

    def evaluate(self, spec, *, output_capture=None):
        """Compute fields and the original paired E/U_rest summary once."""
        self.last_opencv_ipp = None
        from darsia._opencv_runtime import scientific_thread, completed_frame
        ipp_before = scientific_thread("N1Pipeline.evaluate:before")
        if getattr(self, "frame_loading_only", False):
            raise ValueError("A frame-loading-only context cannot evaluate fitted mass")
        if self.statistics_only:
            raise ValueError("A statistics-only context cannot evaluate fitted mass")
        if "pixel_domain" in getattr(self, "config", {}):
            from .pixel_pipeline import evaluate

            evaluated = evaluate(self, spec, output_capture=output_capture)
            self.last_opencv_ipp = completed_frame(ipp_before, "N1Pipeline.evaluate:after")
            return evaluated
        import numpy as np
        import darsia
        from .n1_models import prepare_response, own_colour_response
        from .n1_arithmetic import evaluate_main, main_products
        from .n1_statistics import (
            build_estimate_geometry,
            summarize_round3_response,
            exact_prediction,
        )

        from ..storage_profile import stores_colour_coordinate
        stores = stores_colour_coordinate(getattr(self, "config", {}))
        capture = dict(colour_coordinate=True) if stores else {}
        prepared = self._prepare_statistics(spec, **capture)  # Default call unchanged.
        frame, x, y = prepared["frame"], prepared["x_m"], prepared["y_m"]
        response, row, report = (prepared["response"], prepared["statistics"],
                                 prepared["sandmap"])
        shared, math, config = self.config["shared"], self.math, self.config
        image = response["analysis"].scalar(frame.phase_labels, frame.elapsed_h)
        masks = {"full_rig": frame.domain}
        for name in ("box1", "box2"):
            roi = self.boxes[name]
            roi = roi.get("roi", roi) if isinstance(roi, dict) else roi
            masks[name] = darsia.roi_to_mask(
                darsia.CoordinateArray(roi), image, mode="voxels"
            ).img
        split = self._overflow_phase_split(spec, prepared)
        result = evaluate_main(
            response,
            config["anchor"],
            parameters=self.parameters,
            pixel_math=math,
            calibration_sha256=config["calibration"]["sha256"],
            domain_masks=masks,
            **({} if split is None or "distribution" not in split else
               {"gas_distribution": {key: split[key] for key in ("population", "distribution")}}),
        )
        if split is not None:
            result["overflow_phase_split"] = split["report"]
        if output_capture is not None:
            output_capture.update(frame=frame, x_m=x, y_m=y,
                                  volume_m3=response["analysis"].geometry.cached_voxel_volume)
        prediction = exact_prediction(row, result, self.parameters, config["anchor"])
        evaluated = dict(
            response=response,
            result=result,
            prediction=prediction,
            export=main_products(result, math),
            statistics=row,
            sandmap=report,
        )
        if self._local_build is not None:
            from copy import deepcopy
            for name in ("result", "prediction", "export"):
                evaluated[name]["local_build_accounting"] = deepcopy(frame.provenance["local_build_accounting"])
        self.last_opencv_ipp = completed_frame(ipp_before, "N1Pipeline.evaluate:after")
        return evaluated


def _jsonable(value):
    """Convert only scalar metadata, rejecting leaked scientific pixel arrays."""
    import numpy as np

    if isinstance(value, np.ndarray):
        raise TypeError("Pixel arrays belong in products, not result JSON")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, value):
    """Create one metadata file exclusively; never overwrite a peer's result."""
    raw = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    from ..pipeline.job_queue import _publish_bytes
    _publish_bytes(Path(path), raw, create_parents=False)
    return asset(path)


def _write_npz(path, arrays):
    """Write standard lossless level-one NPZ with original dtype/bit bytes."""
    import numpy as np
    import zipfile

    from ..pipeline.io_ops import publish_file

    def write(stream):
        with zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for name, array in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member, np.asanyarray(array), allow_pickle=False
                    )
    publish_file(path, write)
    return asset(path)


def _array_identity(value):
    """Describe actual scientific dtype/grid/bytes independently of storage."""
    import numpy as np

    value = np.asarray(value)
    digest = hashlib.sha256()
    for chunk in np.nditer(
        value,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=65536,
    ):
        digest.update(chunk.tobytes(order="C"))
    return {
        "sha256": digest.hexdigest(),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }


def _storage_facts(output):
    """Report the actual admitted output location; never an assumed NAS copy."""
    from ..pipeline.run_io import current_binding
    binding = current_binding()
    if binding is None:
        return dict(local_retained=False,direct_NAS_write=True)
    binding.require_output(output)
    from ..job_filesystem import LocalFileAdmission, ResultsRootAdmission, valid_results_admission
    admitted = binding.local_output
    if not valid_results_admission(admitted.admission,admitted.results_root):
        raise ValueError("Storage metadata requires its positive actual results-root admission")
    local = (isinstance(admitted.admission,LocalFileAdmission)
             or isinstance(admitted.admission,ResultsRootAdmission) and admitted.admission.local is True)
    return dict(local_retained=local,direct_NAS_write=not local,direct_local_write=local)


def _publish_frame(output, manifest, manifest_asset, spec, evaluated, config,
                   backend, position, compute_seconds, *, storage_context=None, lean=None):
    """Publish through the existing writer, with a lean-only metadata guard."""
    if lean is None:
        return _publish_frame_content(output, manifest, manifest_asset, spec, evaluated,
            config, backend, position, compute_seconds, storage_context=storage_context, lean=lean)
    from ..pipeline.io_ops import LEAN_METADATA_BYTES, metadata_ceiling
    with metadata_ceiling(LEAN_METADATA_BYTES):
        return _publish_frame_content(output, manifest, manifest_asset, spec, evaluated,
            config, backend, position, compute_seconds, storage_context=storage_context, lean=lean)


def _publish_frame_content(
    output,
    manifest,
    manifest_asset,
    spec,
    evaluated,
    config,
    backend,
    position,
    compute_seconds,
    *,
    storage_context=None,
    lean=None,
):
    """Publish one Main-format scientific frame directly on its owned output.

    ``lean`` is the run's explicit lean storage context, or None for the
    unchanged full layout (see darsia.fluidflower.storage_profile).
    """
    from darsia._opencv_runtime import require_ipp_off
    require_ipp_off("publish_frame:before")
    import numpy as np

    output = Path(output)
    final = output / "frames" / spec["case_id"]
    from ..pipeline.job_queue import retry_transient, _create_directory
    retry_transient(lambda: final.parent.mkdir(parents=True, exist_ok=True))
    if final.exists():
        raise FileExistsError(final)
    from darsia_cache.publication_paths import partial_name
    temporary = final.parent / partial_name(spec["case_id"], uuid.uuid4().hex)
    _create_directory(temporary)
    response, result, prediction = (
        evaluated["response"],
        evaluated["result"],
        evaluated["prediction"],
    )
    frame = response["frame"]
    products = dict(
        evaluated["export"]["products"],
        main_mass_eligible_mask=result["fields"]["main_mass_eligible_mask"],
    )
    if "pixel_domain" in evaluated:
        products.update(evaluated["extra_products"])
    for name, mask in (frame.excluded_masks or {}).items():
        products[name + "_mask"] = mask
    products["unquantified_facies_mask"] = frame.domain & np.isin(
        frame.facies, config["shared"]["unquantified_facies"]
    )
    urest = prediction["U_rest_g"]
    defined = urest is not None and bool(np.isfinite(urest))
    amendment = prediction.get("amendment_sha256") or config["bindings"].get(
        "amendment_sha256", ""
    )
    bindings = dict(
        source_sha256=spec["source_binding"]["source_sha256"],
        phase_sha256=spec["labels"]["sha256"],
        calibration_sha256=config["calibration"]["sha256"],
        code_sha256=manifest["code_sha256"],
    )
    for name in ("E_lo_g", "E_hi_g"):
        products[name] = np.asarray(prediction[name], np.float64)
    products.update(
        U_rest_g=np.asarray(urest if defined else np.nan, np.float64),
        U_rest_defined=np.asarray(defined, bool),
        U_rest_amendment_sha256=np.asarray(amendment),
        **{name: np.asarray(value) for name, value in bindings.items()},
    )
    lean_frame = None if lean is None else lean.frame(spec, evaluated)
    stored_products = products if lean_frame is None else lean_frame.products(products)
    product_asset = _write_npz(temporary / "products.npz", stored_products)
    from ..storage_profile import (COLOUR_COORDINATE_RECORD, stores_colour_coordinate,
                                   write_colour_coordinate)
    coordinate_record = (write_colour_coordinate(temporary, response)
                         if stores_colour_coordinate(config) else None)
    geometry_fields = dict(
        pixel_area_m2=np.broadcast_to(
            np.asarray(frame.pixel_area_m2), frame.domain.shape
        ),
        depth_m=np.broadcast_to(np.asarray(frame.depth_m), frame.domain.shape),
        porosity=np.broadcast_to(np.asarray(frame.porosity), frame.domain.shape),
        facies=frame.facies,
        transport_weight_l=np.asarray(
            response["analysis"].geometry.cached_voxel_volume, np.float64
        )
        * 1000.0,
    )
    if "pixel_domain" in evaluated:
        geometry_fields = evaluated["geometry_fields"]
    if lean_frame is None:
        geometry_asset = _write_npz(temporary / "native_geometry.npz", geometry_fields)
        geometry_record = {
            "path": "native_geometry.npz",
            "sha256": geometry_asset["sha256"],
            "size_bytes": geometry_asset["size_bytes"],
        }
        identities = {
            name: _array_identity(value) for name, value in geometry_fields.items()
        }
    else:
        geometry_record, identities = lean_frame.geometry(temporary, geometry_fields)
    static = dict(
        asset=geometry_record,
        arrays=identities,
        per_frame_geometry=True,
        units=dict(
            pixel_area_m2="m2",
            depth_m="m",
            porosity="1",
            facies="category",
            transport_weight_l="L pore volume per pixel",
        ),
    )
    if lean_frame is not None:
        static["storage"] = lean_frame.storage
    if "n16_producer" in evaluated:
        static["units"].update(full_physical_depth_m="m", full_physical_porosity="1",
            full_physical_pixel_pore_volume_m3="m3 pore volume per pixel",
            full_physical_facies="category", full_physical_physical_area_m2="m2", n16_solubility_g_per_l="g/L")
    record = dict(
        contract=config.get("contracts", {}).get("result", "darsia-n1-products-v1"),
        complete=True,
        run=frame.run_id,
        case_id=spec["case_id"],
        frame_index=frame.frame_index,
        elapsed_h=frame.elapsed_h,
        selected_target_h=spec["selected_target_h"],
        selected_frame_count=len(config["frames"]),
        chronological_position=position,
        working_pixel_dtype=result["precision"],
        precision=result["precision"],
        reduction_dtype="float64",
        backend=backend,
        products={
            "path": "products.npz",
            "sha256": product_asset["sha256"],
            "size_bytes": product_asset["size_bytes"],
        },
        units=evaluated["export"]["units"],
        static=static,
        bindings=bindings,
        total_mass_g=result["total_mass_g"],
        aqueous_mass_g=result["aqueous_mass_g"],
        gas_mass_g=result["gas_mass_g"],
        affected_pore_volume_m3=result["affected_pore_volume_m3"],
        injected_mass_g=prediction.get("injected_mass_g"),
        domains=result["domains"],
        excluded_mass=result["excluded_mass"],
        E_lo_g=prediction["E_lo_g"],
        E_hi_g=prediction["E_hi_g"],
        U_rest_g=urest,
        U_rest_defined=defined,
        U_rest_amendment_sha256=amendment,
        W_g=prediction["W_g"],
        W_over_d=prediction["W_over_d"],
        outside_M_parts=prediction["excluded_parts"],
        sandmap=evaluated["sandmap"],
        anchor=config["anchor"],
        source_binding=spec["source_binding"],
        source_frame_unchanged=True,
        physical_correction=config["physical_correction"],
        prediction=prediction,
        scalar_result={
            key: value
            for key, value in result.items()
            if key not in {"fields", "pixel_backend_facts"}
        },
        pixel_backend_facts=dict(result["pixel_backend_facts"]),
        phase_labels_changed=False,
        field_precision_changed=False,
        Wasserstein_performed=False,
        compute_seconds=compute_seconds,
    )
    record.update(
        round4_spec_sha256=prediction.get("round4_spec_sha256"),
        amendment_sha256=prediction.get("amendment_sha256"),
        scope="frozen_evaluation",
        opencv_ipp=evaluated.get("opencv_ipp"),
    )
    if lean_frame is not None:
        record["lean_storage"] = lean_frame.record()
    if coordinate_record is not None:
        record[COLOUR_COORDINATE_RECORD] = coordinate_record
    if "pixel_domain" in evaluated:
        domain = evaluated["pixel_domain"]
        record.update(
            scope="pixel_domain_evaluation", pixel_domain=domain,
            M_measured_g=domain["M_measured_g"], M_edge_g=domain["M_edge_g"], U_g=domain["U_g"],
            phase_unknown_no_capacity_area_cm2=domain["phase_unknown_no_capacity_area_cm2"],
        )
    for part in (() if "pixel_domain" in evaluated else (
        "added_to_M",
        "filled_without_optical_support",
        "below_sand_floor",
        "removed_from_M",
    )):
        for quantity in ("area_cm2", "co2_area_cm2"):
            record[part + "_" + quantity] = evaluated["sandmap"][part][quantity]
    from ..pipeline.io_ops import publish_file
    def write_statistics(binary):
        import io
        with gzip.GzipFile(filename="native_statistics.json", mode="wb",
                           fileobj=binary, compresslevel=3) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                json.dump(_jsonable(evaluated["statistics"]), stream,
                          allow_nan=False, separators=(",", ":"))
    publish_file(temporary / "native_statistics.json.gz", write_statistics)
    if "n16_producer" in evaluated:
        from ..storage_extensive_io import bind_n16_producer
        bind_n16_producer(evaluated, record, manifest=manifest, manifest_asset=manifest_asset,
            spec=spec, configuration=manifest["configuration"], static=static,
            products=record["products"], chronological_position=position)
    if result.get("analysis_scope") is not None:
        from .n1_arithmetic import public_analysis_scope
        record = public_analysis_scope(dict(record, analysis_scope=result["analysis_scope"]))
    _write_json(temporary / "result.json", record)
    storage_kinds = {}
    if storage_context is not None:
        from ..storage_n1_outputs import write_n1_frame
        context, capture = storage_context
        extensive_kwargs = {}
        if evaluated.get("extensive_inputs") is not None:
            extensive_kwargs["extensive_inputs"] = evaluated["extensive_inputs"]
        storage_kinds = write_n1_frame(
            temporary, final, manifest=manifest, manifest_asset=manifest_asset,
            spec=spec, config=config, evaluated=evaluated, capture=capture, context=context,
            **extensive_kwargs)
    files = {
        path.name: {key: value for key, value in asset(path).items() if key != "path"}
        for path in sorted(retry_transient(lambda: list(temporary.iterdir())))
    }
    for name, kind in storage_kinds.items():
        files[name]["kind"] = kind
    commit_id = uuid.uuid4().hex
    commit = dict(
        complete=True,
        commit_id=commit_id,
        frame_id=spec["case_id"],
        job_id=manifest["job_id"],
        manifest_sha256=manifest_asset["sha256"],
        settings=manifest["settings"],
        products=dict(precision=result["precision"], backend=backend),
        files=files,
        committed_utc=datetime.now(timezone.utc).isoformat(),
        opencv_ipp=evaluated.get("opencv_ipp"),
    )
    require_ipp_off("publish_frame:commit")
    commit_asset = _write_json(temporary / "commit.json", commit)
    registration = dict(
        manifest_sha256=manifest_asset["sha256"],
        frame_id=spec["case_id"],
        commit=dict(commit_asset, path=str(final / "commit.json")),
        files=files,
    )
    operations = output / "ops"
    retry_transient(lambda: operations.mkdir(exist_ok=True))
    _write_json(operations / ("frame_commit_" + commit_id + ".json"), registration)
    # Windows rename and POSIX rename never replace a nonempty frame directory.
    if final.exists():
        raise FileExistsError(final)
    from ..pipeline.job_queue import rename
    rename(temporary, final)
    inventory = {
        **files,
        "commit.json": {
            key: value
            for key, value in asset(final / "commit.json").items()
            if key != "path"
        },
    }
    for name, bound in inventory.items():
        actual = asset(final / name)
        if (
            actual["sha256"] != bound["sha256"]
            or actual["size_bytes"] != bound["size_bytes"]
        ):
            raise ValueError("Published frame bytes differ: " + name)
    publication = dict(
        frame_id=spec["case_id"],
        **_storage_facts(output),
        nas_directory=str(final),
        files=inventory,
        manifest_sha256=manifest_asset["sha256"],
        verified_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(operations / ("nas_copy_" + commit_id + ".json"), publication)
    return publication


def run_n1(
    task: dict,
    output: Path,
    backend: str,
    precision: str,
    on_progress: Callable | None = None,
    *,
    _runner=None,
) -> dict:
    """Run one N1 selection under measured, restored CPU thread limits.

    Thread admission is independent of the scientific formula. Import the
    numerical libraries before limiting their pools, then cover setup, every
    frame and publication in the same limit context.
    """
    from darsia._opencv_runtime import scientific_thread
    ipp_start = scientific_thread("run_n1:start")
    threads = task.get("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("task.threads must be a positive integer")
    import cv2
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits

    previous_cv2_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(threads)
        with threadpool_limits(limits=threads):
            runtime = {
                "requested_threads": threads,
                "cv2_threads": cv2.getNumThreads(),
                "threadpools": threadpool_info(),
                "opencv_ipp": ipp_start,
            }
            runner = _run_n1 if _runner is None else _runner
            return runner(
                task, output, backend, precision, on_progress, thread_runtime=runtime
            )
    finally:
        cv2.setNumThreads(previous_cv2_threads)


def _run_n1(task: dict, output: Path, backend: str, precision: str,
            on_progress: Callable | None = None, *, thread_runtime: dict, geometry_fields=None) -> dict:
    """Scope the lean metadata bound around the original numerical runner."""
    from ..storage_profile import declared_storage_profile
    if declared_storage_profile(task) != "lean":
        return _run_n1_content(task, output, backend, precision, on_progress,
            thread_runtime=thread_runtime, geometry_fields=geometry_fields)
    from ..pipeline.io_ops import LEAN_METADATA_BYTES, metadata_ceiling
    with metadata_ceiling(LEAN_METADATA_BYTES):
        return _run_n1_content(task, output, backend, precision, on_progress,
            thread_runtime=thread_runtime, geometry_fields=geometry_fields)


def _run_n1_content(
    task: dict,
    output: Path,
    backend: str,
    precision: str,
    on_progress: Callable | None = None,
    *,
    thread_runtime: dict,
    geometry_fields=None,
) -> dict:
    """Compute one configured whole run once and publish Main-format frames.

    ``task.configuration`` is a full descriptor of the explicit run data;
    ``task.run`` must agree. The output is the caller's unique owned attempt
    directory (UNC in production). An existing frame is never recomputed or
    overwritten. The caller owns queue/TTL/code/environment/process admission.
    """
    from darsia._opencv_runtime import scientific_thread, completed_frame, require_ipp_off
    from .production_precision import PixelMath

    from ..pipeline.run_io import for_task
    io_binding = for_task(task)
    configuration = task["configuration"] if io_binding is None else io_binding.configuration_descriptor
    if io_binding is None:
        unc_only = _is_unc(output) or _is_unc(configuration["path"])
        if unc_only and (not _is_unc(output) or not _is_unc(configuration["path"])):
            raise ValueError("Production configuration and output must both be UNC")
    else:
        io_binding.require_output(output)
        unc_only = True  # DataReader requires the positive run binding for local I/O.
    from ..storage_profile import run_record_files
    config = load_n1_run(
        configuration, unc_only=unc_only, geometry_fields=geometry_fields
    )
    if config["run"] != task["run"] or backend not in {"cpu", "cuda"}:
        raise ValueError("Task run and explicit numerical backend are required")
    from ..storage_profile import run_setting
    setting = run_setting(task, config)  # The job's choice; validated before any output.
    if on_progress is not None and not callable(on_progress):
        raise TypeError("Progress callback must be callable")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            "The caller-owned attempt directory must be empty: " + str(output)
        )
    clock = time.perf_counter()
    from ..pipeline.checkpoint import verify_frame_prefix, run_identity, frame_producer

    resume = verify_frame_prefix(
        task, config, output, backend, precision, task.get("resume_from")
    )
    if io_binding is not None:
        io_binding.admit_resume_root(resume["source"])
    resumed = len(resume["records"])
    storage_request = None
    if "storage" in config:
        from ..storage_outputs import parse_request
        storage_request = parse_request(config, precision)
        if storage_request["resolution_mm"] == "native":
            storage_request = None  # Native/default publication remains the original path.
    if storage_request is not None and resumed:
        raise ValueError("Coarse N1 resume requires a separately qualified saved-storage prefix; no replay")
    records = list(resume["records"])
    frame_producers = list(resume["frame_producers"])
    resume_asset = None
    if task.get("resume_from") is not None:
        resume_asset = _write_json(
            output / run_record_files(config)["resume"],
            dict(
                source=task["resume_from"],
                prefix_case_ids=resume["prefix_case_ids"],
                frame_producers=resume["frame_producers"],
                predecessor_input=resume["predecessor_input"],
            ),
        )
        if on_progress:
            on_progress(
                dict(
                    run=task["run"],
                    state="resume_verified",
                    complete=True,
                    verified_prefix_case_ids=resume["prefix_case_ids"],
                    elapsed_seconds=time.perf_counter() - clock,
                )
            )
    if backend == "cpu":
        math = PixelMath(precision)
    else:
        from .mass_cuda import CudaHostPixelMath

        math = CudaHostPixelMath(
            precision,
            device_id=int(task.get("device_id", 0)),
            pool_limit_bytes=int(task["pool_limit_bytes"]),
        )
    try:
        pipeline = N1Pipeline(
            config, precision=precision, pixel_math=math, unc_only=unc_only
        )
        storage_context = None
        if storage_request is not None:
            from ..storage_n1_outputs import Phase2Storage
            storage_context = Phase2Storage(config, pipeline.reader, storage_request)
        lean = None
        if setting["profile"] == "lean":
            from ..storage_profile import LeanRun
            lean = LeanRun(output, config, pipeline, setting, colour_intermediates_produced=False)
        settings = dict(
            host=socket.gethostname(),
            backend=backend,
            precision=precision,
            threads=task.get("threads", 1),
            thread_runtime=thread_runtime,
        )
        commit = task.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
        ):
            raise ValueError("The caller must provide the actual full Git commit")
        code_sha = hashlib.sha256(commit.encode("ascii")).hexdigest()
        manifest = dict(
            format="darsia-n1-run-manifest-v1",
            job_id=task.get("job_id", output.name),
            run=task["run"],
            configuration=task["configuration"],
            code={"commit": commit, "identity_algorithm": "sha256-git-commit-ascii"},
            code_sha256=code_sha,
            settings=settings,
            frames={frame["case_id"]: frame for frame in config["frames"]},
            task_identity=run_identity(
                task, backend, precision, config["calibration"]["sha256"]
            ),
        )
        if storage_context is not None:
            manifest["storage"] = dict(producer_kind="n1-mass-only", request=storage_request,
                phase1=storage_context.binding, ordered_case_ids=[frame["case_id"] for frame in config["frames"]])
        if resume_asset is not None:
            manifest["resume_checkpoint"] = resume_asset
        if lean is not None:
            manifest["storage_profile"] = lean.manifest_record()
        manifest_asset = _write_json(output / run_record_files(config)["manifest"], manifest)
        for position, spec in enumerate(config["frames"]):
            if position < resumed:
                continue
            ipp_before = scientific_thread("run_n1:" + spec["case_id"] + ":before")
            if on_progress:
                on_progress(
                    dict(
                        run=task["run"],
                        case_id=spec["case_id"],
                        frame_index=spec["frame_index"],
                        state="frame_started",
                        elapsed_seconds=time.perf_counter() - clock,
                    )
                )
            started = time.perf_counter()
            capture = None if storage_context is None else {}
            evaluated = (pipeline.evaluate(spec) if capture is None
                         else pipeline.evaluate(spec, output_capture=capture))
            compute_seconds = time.perf_counter() - started
            evaluated["opencv_ipp"] = completed_frame(ipp_before, "run_n1:" + spec["case_id"] + ":after")
            evaluated["opencv_ipp"]["mass_evaluation"] = pipeline.last_opencv_ipp
            publication = _publish_frame(
                output,
                manifest,
                manifest_asset,
                spec,
                evaluated,
                config,
                backend,
                position,
                compute_seconds,
                storage_context=None if storage_context is None else (storage_context, capture),
                lean=lean,
            )
            del capture
            records.append(publication)
            frame_producers.append(frame_producer(output, publication, manifest_asset))
            del evaluated
            if on_progress:
                on_progress(
                    dict(
                        run=task["run"],
                        case_id=spec["case_id"],
                        frame_index=spec["frame_index"],
                        state="frame_completed",
                        complete=True,
                        compute_seconds=compute_seconds,
                        elapsed_seconds=time.perf_counter() - clock,
                        publication=publication,
                    )
                )
        if "pixel_domain" in config:
            from ..pipeline.runner import scalar_table

            table = scalar_table(
                records, config["pixel_domain"]["table_fields"], output / run_record_files(config)["scalar_table"]
            )
        wall = time.perf_counter() - clock
        outcome = dict(
            complete=True,
            run=task["run"],
            whole_run=True,
            backend=backend,
            precision=precision,
            working_pixel_dtype=precision,
            pixel_backend=math.facts()["backend"],
            ordered_case_ids=[frame["case_id"] for frame in config["frames"]],
            case_count=len(records),
            records=records,
            frames=len(records),
            wall_seconds=wall,
            images_per_hour=(len(records) - resumed) * 3600 / wall,
            frames_computed_this_attempt=len(records) - resumed,
            resumed_frame_count=resumed,
            rate_scope="new_frames_this_attempt",
            frame_producers=frame_producers,
            resume_checkpoint=resume_asset,
            static_copy=None,
            static_geometry="per-frame native_geometry.npz",
            configuration=task["configuration"],
            manifest=manifest_asset,
            calibration=config["calibration"],
            physical_correction=config["physical_correction"],
            anchor=config["anchor"],
            elapsed_seconds=wall,
            opencv_ipp=require_ipp_off("run_n1:completed"),
        )
        if "pixel_domain" in config:
            outcome["table"] = table
            outcome["pixel_domain"] = config["pixel_domain"]
        if lean is not None:
            outcome.update(
                static_geometry="lean: per-run lean_run_static store, per-frame regeneration descriptor",
                lean_storage=lean.outcome_record(),
            )
        del pipeline
    finally:
        close = getattr(math, "close", None)
        if callable(close):
            close()
    outcome["worker_references_closed"] = True
    return outcome
