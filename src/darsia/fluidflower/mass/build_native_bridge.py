"""Bind a completed Build grid to original RAW photometry, before phase/optics.

Camera-U is the calibrated camera-equivalent sampling grid, not the rectified
rig grid. The saved Build grid and the directly sampled RAW native grid remain
separate: their RGB pixels are not claimed equal. No phase, physical-domain or
training eligibility is inferred here.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .native_optical_source import _bound, _same, load_build_physical_context
from .n1_pipeline import DataReader, _stamp, _write_json, _write_npz, asset
from ..job_filesystem import inspect_results_root
from ..native import acquisition_lens, raw_positioning, raw_rig_source, registration, source
from ..rig_raw_build import _require_quality
from ..rig_raw_write import _no_links


def _fresh(path, forbidden):
    """Reserve one admitted output directory without replacing saved inputs."""
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or path.exists()
            or not inspect_results_root(path).allowed):
        raise ValueError("A fresh absolute native-bridge directory on supported storage is required")
    _no_links(path.parent)
    actual = path.resolve()
    if any(actual == root or actual.is_relative_to(root) or root.is_relative_to(actual)
           for root in map(lambda value: Path(value).resolve(), forbidden)):
        raise ValueError("Native-bridge output overlaps a protected RAW or saved Build root")
    path.mkdir()
    return path


@dataclass(frozen=True)
class BuildNativeFrame:
    """Actual corrected RAW buffers and original runner inputs, without P1 claims."""
    native_task: dict
    products: dict
    receipt: dict


class _Inputs:
    """Retain consumed producer descriptors for cached late-identity checks."""
    def __init__(self, reader):
        self.reader, self.rows = reader, {}

    def path(self, bound):
        path = self.reader.path(bound)
        self.rows[(str(bound["path"]), bound["sha256"])] = deepcopy(bound)
        return path

    def json(self, bound):
        self.path(bound)
        return self.reader.json(bound)

    def revalidate(self):
        for row in tuple(self.rows.values()):
            self.reader.path(row)


class _MeasuredClearwaterRenderer:
    """Keep a measured wash crop independent of indicator drift features.

    The bridge has already registered the saved Build directly to this RAW
    acquisition, including the original held-out checks. A wash can have a
    different chemical appearance from the indicator baseline, so replaying
    the baseline's translation-only feature ROI is not a second valid motion
    measurement. Use the checked projective registration on the baseline's
    physical native grid instead. All photometric correction remains in the
    original processor, and this scope never applies to another acquisition.
    """

    @contextmanager
    def _clearwater_scope(self):
        if getattr(self, "_measured_clearwater", None) is not None:
            raise ValueError("A measured clear-water acquisition is already active")
        try:
            yield
        finally:
            self._measured_clearwater = None

    def _bind_clearwater_renderer(self, sensor, native_to_u, quality, *, build_to_u, bridge):
        renderer = dict(
            contract=raw_positioning.CONTRACT, source_kind="raw",
            raw_acquisition=deepcopy(sensor.source),
            camera_u_wh=list(self.parameters["registration"]["common"]["sensor"]["camera_wh"]),
            native_shape_hw=list(self.binding["native_shape_hw"]),
            native_to_camera_u_h=native_to_u.tolist(),
            positioning_inputs="saved Build to current complete RAW camera U, with held-out checks",
            shape_corrections=deepcopy(self.shape_assets), shape_chain_replayed=False,
            registration=deepcopy(quality), sensor_metadata=deepcopy(sensor.metadata),
            profile_id=sensor.profile_id, white_balance_applied=False, gamma_applied=False,
            scientific_units="black-subtracted camera DN",
            scientific_interpolation="one direct per-channel sensor-to-native bilinear float32 warp",
            measured_clearwater=dict(bridge=deepcopy(bridge),
                build_to_frame_camera_u_h=build_to_u.tolist(),
                native_grid="unchanged saved baseline physical grid",
                source_purpose="clearwater"),
            new_rig_fit=False, new_photometric_fit=False, training=False)
        self._measured_clearwater = (sensor.source["sha256"], deepcopy(renderer))
        return renderer

    def _render(self, sensor, *, camera_u=None):
        registered = getattr(self, "_measured_clearwater", None)
        if registered is None or sensor.source["sha256"] != registered[0]:
            if camera_u is None:
                return super()._render(sensor)
            return super()._render(sensor, camera_u=camera_u)
        renderer = deepcopy(registered[1])
        if camera_u is not None:
            shape = (3, *renderer["camera_u_wh"][::-1])
            if camera_u.dtype != np.float32 or camera_u.shape != shape or not np.isfinite(camera_u).all():
                raise ValueError("The retained full camera-U view differs from the measured wash photograph")
            renderer["scientific_interpolation"] = "positioning only; no cropped scientific pixels sampled"
            return None, renderer
        pixels = raw_positioning.sample_sensor(sensor,
            np.asarray(renderer["native_to_camera_u_h"], np.float64),
            shape_hw=self.binding["native_shape_hw"], parameters=self.parameters["registration"])
        return pixels, renderer


class _AcquisitionLensSensor:
    """Read each acquisition with the lens tables that it records.

    The calibrated profile holds the camera's tables at calibration time. An
    acquisition that records other tables, for example after a small zoom
    change, is read with its own tables and otherwise the same sensor checks;
    the measured registration and its held-out checks still decide whether it
    can be used. Acquisitions with the profile's tables use the original reader.
    """

    def _sensor(self, descriptor):
        lens = self.parameters.get("registration", {}).get("common", {}).get("lens")
        if lens is None or not isinstance(descriptor, dict) or "path" not in descriptor:
            return super()._sensor(descriptor)
        path = self._path(descriptor["path"])
        key = (str(path), descriptor.get("sha256"))
        if getattr(self, "_current_sensor_key", None) == key:
            return super()._sensor(descriptor)
        try:
            own = acquisition_lens.acquisition_lens(path, self.parameters["registration"],
                                                    profile_id=self.binding["profile_id"])
        except (OSError, ValueError, LookupError, TypeError):
            own = None  # The original reader reports this acquisition's actual error.
        if own is None:
            return super()._sensor(descriptor)
        baseline = getattr(self, "baseline_sensor", None)
        if baseline is None:
            raise ValueError("The baseline acquisition records other lens tables than its calibrated profile")
        sensor = raw_positioning.read_sensor(self._asset(descriptor),
            expected_sha256=descriptor["sha256"], parameters=own.parameters,
            profile_id=self.binding["profile_id"])
        if any(sensor.metadata[name] != baseline.metadata[name] for name in ("make", "model")):
            raise ValueError("An acquisition with its own lens tables must come from the calibrated camera model")
        sensor = replace(sensor, metadata={**sensor.metadata, "lens_tables": deepcopy(own.provenance)})
        self._current_sensor_key, self._current_sensor = key, sensor
        return sensor


def _bridge_processor(binding, *, parameters):
    # Resolve the actual original processor at construction, retaining its
    # complete initialization, identity and correction implementation.
    class Processor(_MeasuredClearwaterRenderer, _AcquisitionLensSensor,
                    raw_rig_source.RawRigSourceProcessor):
        pass
    return Processor(binding, parameters=parameters)


class BuildNativeBridge:
    """Produce original RAW-native inputs joined to an exact completed Build.

    The caller supplies a current, independently bound frozen-photometry source
    binding. Its models are preserved as their own inputs, never relabelled as
    Build products. All output directories must be fresh and positively local;
    a failed preparation is preserved and cannot be reused.
    """

    def __init__(self, operation_result, photometry_binding, output, *, reader=None, local_reuse=None):
        self.producer_stamp = _stamp(Path(__file__))
        self.producer = asset(Path(__file__))
        self.reader = _Inputs(DataReader() if reader is None else reader)
        self.context = load_build_physical_context(operation_result, reader=self.reader, local_reuse=local_reuse)
        self.operation_result = deepcopy(_bound(operation_result))
        self.native = deepcopy(self.context.provenance["native_configuration"])
        self.parameters = self._document(self.native)
        original = self._document(photometry_binding)
        self.photometry_binding = deepcopy(photometry_binding)
        camera = self.context.provenance["camera_binding"]
        baseline = self.context.provenance["raw_acquisition"]
        if (original["run"] != self.context.run or camera["run_id"] != self.context.run
                or original["profile_id"] != camera["profile_id"]
                or original["baseline_raw"]["sha256"] != baseline["sha256"]
                or list(original["native_shape_hw"]) != self.context.grid["num_voxels"]):
            raise ValueError("RAW photometry and completed Build differ in run/profile/baseline/grid")
        certificate = self._document(self.context.provenance["baseline_renderer"])
        request = self._document(self.context.provenance["raw_build_request"])
        self.forbidden = [*request["output"]["forbidden_raw_roots"],
                          str(Path(certificate["corrected_baseline"]["archive"]["path"]).parent)]
        renderer = certificate["renderer"]
        common = self.parameters["registration"]["common"]
        self.policies = common["review"]
        self.build_h = self._matrix(certificate["H_rig_corrected_native_to_baseline_camera_u"])
        # Homographies describe projective coordinates: their homogeneous
        # scale is not a physical difference. Normalize both saved values
        # before the existing configured matrix comparison. Comparing literal
        # JSON with only one normalized side rejects a valid h33=1-epsilon.
        if (not source._same_h(self._matrix(renderer["native_to_camera_u_h"]), self.build_h,
                               parameters=self.parameters)
                or renderer["profile_id"] != camera["profile_id"]
                or renderer["camera_u_wh"] != common["sensor"]["camera_wh"]
                or renderer["saved_shape_chain_replay_bit_identical"] is not True
                or not _same(renderer["source_npz"],
                             self.context.provenance["physical_archives"]["shape_corrected_baseline"])):
            raise ValueError("The saved Build lacks its exact camera-U registration and replay")
        _require_quality(renderer["registration"], self.policies)
        assets = renderer["saved_shape_chain_replay"]["correction_assets"]
        if not assets or not _same(assets, renderer["shape_corrections"]):
            raise ValueError("The Build lacks its ordered saved shape assets")
        owned = {row["path"]: row for row in certificate["outputs"]}
        for row in assets:
            if not _same(owned.get(row["path"]), row):
                raise ValueError("A shape asset is not an exact owned Build output")
            self._pin(row)
        old = raw_rig_source.RawRigSourceProcessor(
            self.reader.path(photometry_binding), parameters=self.parameters)
        _require_quality(old.baseline_renderer["registration"], self.policies)
        for row in [original["baseline_raw"], *original["shape_assets"],
                    *(item for model in original["models"].values() if model is not None
                      for item in (model.values() if "data" in model else (model,)))]:
            self._pin(row, resolve=old._path)
        white_range = old.baseline_sensor.metadata["white_level"] - old.baseline_sensor.metadata["black_level"]
        normalization = renderer["saved_shape_chain_replay"]["normalization"]
        if (not np.isfinite(white_range) or white_range <= 0
                or normalization != dict(black_subtracted_dn_divisor=float(white_range),
                                         white_balance_applied=False, gamma_applied=False)
                or old.baseline_renderer["camera_u_wh"] != renderer["camera_u_wh"]):
            raise ValueError("The Build and RAW source do not share the exact calibrated camera-U basis")
        self.review_green = np.asarray(self.context.baseline_rgb[..., 1], np.float32) * np.float32(white_range)
        if not np.isfinite(self.review_green).all():
            raise ValueError("The saved Build green registration input is nonfinite")
        metadata = deepcopy(certificate["inputs"]["uncorrected_grid"]["metadata"])
        metadata.pop("type", None)
        metadata.pop("color_space", None)  # native_review_rgb supplies RGB explicitly.
        if metadata.get("time") is None and isinstance(metadata.get("reference_date"), str):
            raise ValueError("The original RAW binding needs explicit time with serialized Build dates")
        runtime_metadata = deepcopy(metadata)
        for name in ("date", "reference_date"):
            if isinstance(runtime_metadata.get(name), str):
                runtime_metadata[name] = datetime.fromisoformat(runtime_metadata[name])
        _, fresh_renderer = raw_positioning.render_native(old.baseline_sensor,
            shape_assets=assets, shape_hw=original["native_shape_hw"],
            image_metadata=runtime_metadata, parameters=self.parameters["registration"])
        _require_quality(fresh_renderer["registration"], self.policies)
        hn = self._matrix(fresh_renderer["native_to_camera_u_h"])
        ho = self._matrix(old.baseline_renderer["native_to_camera_u_h"])
        frozen_reference = self._matrix(original["baseline_reference_to_native_h"])
        reference = self._matrix(np.linalg.inv(hn) @ ho @ frozen_reference)
        self.build_to_baseline = self._matrix(np.linalg.inv(hn) @ self.build_h)
        binding = deepcopy(original)
        binding.update(baseline_raw=deepcopy(baseline), shape_assets=deepcopy(assets),
                       image_metadata=metadata, baseline_reference_to_native_h=reference.tolist())
        raw_rig_source.validate_binding(binding, self.parameters)
        self._revalidate()
        self.output = _fresh(output, self.forbidden)
        self.binding = _write_json(self.output / "raw_source_binding.json", binding)
        self.processor = _bridge_processor(self.binding["path"], parameters=self.parameters)
        if not source._same_h(hn, self._matrix(self.processor.baseline_renderer["native_to_camera_u_h"]),
                              parameters=self.parameters):
            raise ValueError("The newly bound RAW baseline registration changed")
        self.reference = reference
        self.baseline_sha = baseline["sha256"]
        for value in (self.review_green, self.build_h, self.build_to_baseline, self.reference):
            value.setflags(write=False)
        self._revalidate()
        self.proof = _write_json(self.output / "bridge.json", dict(
            format="darsia-build-native-bridge-v1", operation_result=self.operation_result,
            photometry_source=photometry_binding, native_configuration=self.native,
            raw_source_binding=self.binding, build_to_baseline_native_h=self.build_to_baseline.tolist(),
            frozen_reference_to_baseline_native_h=reference.tolist(),
            producer=dict(callable="darsia.fluidflower.mass.build_native_bridge.BuildNativeBridge",
                          module=self.producer, raw_execution_identity=raw_rig_source.execution_identity()),
            saved_build_to_camera_u_h=self.build_h.tolist(),
            old_raw_baseline_renderer=old.baseline_renderer,
            new_raw_baseline_renderer=self.processor.baseline_renderer,
            input_scope="original RAW photometry and saved Build coordinate join; no phase/domain qualification"))

    def _matrix(self, value):
        return source.source_homography(value, parameters=self.parameters)

    def _pin(self, bound, *, resolve=None):
        """Track the actual consumed bytes, retaining the original declared SHA."""
        path = Path(bound["path"]) if resolve is None else resolve(bound["path"])
        row = dict(path=str(path), sha256=bound["sha256"],
                   size_bytes=bound.get("size_bytes", path.stat().st_size))
        self.reader.path(_bound(row))
        return row

    def _document(self, bound):
        return self.reader.json(_bound(bound))

    def _revalidate(self):
        # Includes every original physical archive, request, owner and plan consumed
        # by the Build reader, without decoding the unchanged archives again.
        if _stamp(Path(__file__)) != self.producer_stamp:
            raise ValueError("The native-bridge producer changed after preparation started")
        self.reader.revalidate()

    def corrected_frame(self, acquisition, *, source_purpose=None, prepared_baseline=None):
        """Return original corrected RAW buffers in memory, without phase inference.

        Optical sampling consumes these once and retains only small samples
        and their source identities. The ordinary frame publisher below uses
        the exact same calculation and still writes its original products.
        """
        with self.processor._clearwater_scope():
            return self._corrected_frame(acquisition, source_purpose=source_purpose,
                                         prepared_baseline=prepared_baseline)

    def _corrected_frame(self, acquisition, *, source_purpose=None, prepared_baseline=None):
        acquisition = self._pin(acquisition)
        self.reader.path(self.binding)
        self.reader.path(self.proof)
        self._revalidate()
        exact_baseline = acquisition["sha256"] == self.baseline_sha
        if exact_baseline:
            baseline_to_frame = np.eye(3)
            build_to_frame = self.build_to_baseline
            measured = dict(exact_baseline=True, registration="original baseline identity")
        else:
            sensor = self.processor._sensor(acquisition)
            view = raw_positioning.camera_u(sensor, parameters=self.parameters["registration"])
            j, quality = registration.review_to_camera(self.review_green, view[1],
                                                       parameters=self.parameters["registration"]["common"])
            _require_quality(quality, self.policies)
            if source_purpose == "clearwater":
                # J maps the saved Build grid into this acquisition's camera U.
                # Preserve the baseline native coordinates rather than asking
                # the chemically different wash to replay indicator drift ROIs.
                k = self._matrix(self._matrix(j) @ np.linalg.inv(self.build_to_baseline))
                renderer = self.processor._bind_clearwater_renderer(sensor, k, quality,
                    build_to_u=self._matrix(j), bridge=self.proof)
            else:
                _, renderer = self.processor._render(sensor)
            _require_quality(renderer["registration"], self.policies)
            k = self._matrix(renderer["native_to_camera_u_h"])
            if source_purpose == "clearwater":
                build_to_frame = self.build_to_baseline.copy()
                baseline_to_frame = np.eye(3)
            else:
                build_to_frame = self._matrix(np.linalg.inv(k) @ self._matrix(j))
                baseline_to_frame = self._matrix(build_to_frame @ np.linalg.inv(self.build_to_baseline))
            measured = dict(exact_baseline=False, build_to_frame_camera_u_h=self._matrix(j).tolist(),
                            registration=quality, frame_native_renderer=renderer)
        geometry = dict(contract=raw_rig_source.FRAME_CONTRACT, run=self.context.run,
            acquisition=acquisition, baseline_to_native_h=baseline_to_frame.tolist(),
            reference_to_native_h=self._matrix(baseline_to_frame @ self.reference).tolist())
        raw_rig_source.validate_frame(geometry, self.processor.binding, self.parameters)
        if prepared_baseline is not None:
            if not exact_baseline or source_purpose is not None:
                raise ValueError("Saved Setup baseline pixels cannot replace another acquisition/purpose")
            cached = self._document(prepared_baseline)
            if (cached.get("format") != "darsia-setup-corrected-baseline-v1"
                    or cached.get("run") != self.context.run
                    or cached.get("raw_acquisition") != acquisition
                    or cached.get("photometry_source") != self.photometry_binding
                    or cached.get("native_shape_hw") != list(self.context.grid["num_voxels"])
                    or not source._same_h(self._matrix(cached["native_to_camera_u_h"]),
                        self._matrix(self.processor.baseline_renderer["native_to_camera_u_h"]), parameters=self.parameters)):
                raise ValueError("Saved Setup baseline differs from this exact RAW/photometry/crop producer")
            with np.load(self.reader.path(cached["products"]), allow_pickle=False) as archive:
                log = archive["log_linear_rgb"].copy()
            audit = dict(contract=self.processor.binding["contract"], processor=self.processor.identity,
                sources=dict(camera_arw=dict(acquisition)), renderer=deepcopy(self.processor.baseline_renderer),
                correction_order=cached["correction_order"], exact_B=True,
                prepared_baseline=deepcopy(prepared_baseline), numerical_source="completed Setup image preparation",
                photometry=deepcopy(cached["correction"]),
                **{self.parameters["source"]["symbols"]["reference_to_frame_h_key"]: deepcopy(geometry["reference_to_native_h"])},
                H_v5_B_to_frame_native=deepcopy(geometry["baseline_to_native_h"]),
                source_identity=dict(kind="raw-acquisition", raw_sha256=acquisition["sha256"]),
                parent_cache=None, cpu_only=True, training=False, jpeg_inputs=False)
        else:
            from ..pipeline.series import same_raw_source
            log, audit = self.processor.correct(geometry,
                **({"source_purpose": source_purpose} if source_purpose is not None else {}))
            # The RAW reader names the file it opened and its SHA-256, without a
            # size. Once it is the same acquisition, name it exactly as given,
            # as the saved Setup baseline route above does.
            if not same_raw_source(audit["sources"]["camera_arw"], acquisition, self.processor._path):
                raise ValueError("The RAW reader opened another acquisition than the one given")
            audit["sources"] = dict(audit["sources"], camera_arw=dict(acquisition))
        expected_h = self._matrix(self.processor.baseline_renderer["native_to_camera_u_h"] if exact_baseline
                                  else measured["frame_native_renderer"]["native_to_camera_u_h"])
        if not source._same_h(expected_h, self._matrix(audit["renderer"]["native_to_camera_u_h"]),
                              parameters=self.parameters):
            raise ValueError("The frame RAW registration changed before correction completed")
        if log.dtype != np.float32 or log.shape != (3, *self.context.grid["num_voxels"]):
            raise ValueError("Original RAW correction returned another native grid or dtype")
        valid = np.isfinite(log).all(0)
        if not valid.any():
            raise ValueError("Original RAW correction has no finite native support")
        self._revalidate()
        return dict(geometry=geometry, log_linear_rgb=log, valid_mask=valid,
            build_to_frame_native_h=build_to_frame.tolist(), registration=measured,
            source_purpose=source_purpose, correction=audit)

    def frame(self, acquisition, output, *, source_purpose=None, prepared_baseline=None):
        """Publish the original corrected RAW frame and exact source/grid receipt."""
        result = self.corrected_frame(acquisition, source_purpose=source_purpose,
                                      prepared_baseline=prepared_baseline)
        target = _fresh(output, self.forbidden)
        frame_binding = _write_json(target / "raw_native_frame.json", result["geometry"])
        products = _write_npz(target / "native.npz", {key: result[key] for key in ("log_linear_rgb", "valid_mask")})
        self._revalidate()
        receipt = _write_json(target / "frame.json", dict(
            format="darsia-build-native-frame-v1", bridge=self.proof,
            raw_source_binding=self.binding, native_frame=frame_binding, products=products,
            build_to_frame_native_h=result["build_to_frame_native_h"], registration=result["registration"],
            source_purpose=source_purpose,
            correction=result["correction"], physical_context=self.context.provenance,
            scope="corrected native RAW buffers and coordinate join only; phase/history/domain remain separate"))
        return BuildNativeFrame(dict(kind="raw_rig", geometry=frame_binding,
            processor=dict(raw_binding=deepcopy(self.binding))), products, receipt)
