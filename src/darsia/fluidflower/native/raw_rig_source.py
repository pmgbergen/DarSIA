"""RAW-only acquisition bindings on the existing calibrated native grid.

This producer loads the same frozen photometry and geometry as the historical
producer. It never opens a companion photograph or retains pixels from the
JPEG-positioned reference. New RAW references and inference remain separate
validation gates.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
import json
import re

import numpy as np

from . import extension, photometry, raw_chart, raw_positioning, raw_positioning_port
from . import registration, source
from .io import resolve_asset_path, sha256_file, read_bound_json


CONTRACT = "darsia-raw-rig-source-v1"
FRAME_CONTRACT = "darsia-raw-native-frame-v1"


def execution_identity() -> dict:
    """Bind canonical source content; acquisition/model hashes stay exact bytes.

    Inside a queue attempt, RunIOBinding rechecks the admitted working copy
    and returns the real Git-blob SHA. Standalone control preparation uses its
    LF source representation, the representation tracked by these modules.
    """
    from ..pipeline.run_io import RAW_CODE_ROLES, current_binding
    from ..pipeline.job_queue import retry_transient
    package_root = Path(__file__).resolve().parents[3]
    identity = {}
    for name, relative in RAW_CODE_ROLES.items():
        path = package_root / relative.removeprefix("src/")
        if current_binding() is not None:
            identity[name] = sha256_file(path)
        else:
            data = retry_transient(path.read_bytes)
            identity[name] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    return identity


def _keys(value, required, *, where):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError("Explicit RAW-only fields required for " + where)
    return value


def validate_binding(binding: Mapping, parameters: Mapping) -> None:
    """Validate data roles before any camera/model file is opened."""
    required = {"contract", "run", "profile_id", "native_shape_hw", "baseline_raw",
                    "baseline_reference_to_native_h", "shape_assets", "image_metadata",
                    "models", "execution_identity"}
    if "correction_order" in binding:
        from .full_photometry import ORDER
        if binding["correction_order"] != ORDER:
            raise ValueError("Unknown full-photograph correction order")
        required.add("correction_order")
        required.add("normalisation")
        normalisation = binding.get("normalisation")
        if (not isinstance(normalisation, dict) or set(normalisation) != {"choice", "record"}
                or not isinstance(normalisation["choice"], dict)
                or normalisation["choice"].get("method") not in {
                    "off", "match_reference_lighting", "match_reference_lighting_equalise_sand"}):
            raise ValueError("Full-photograph processing needs its actual saved series normalisation choice")
    if "photometric_method" in binding:
        from .full_photometry import INDEPENDENT
        if binding["photometric_method"] != INDEPENDENT or "normalisation" not in binding:
            raise ValueError("Unknown independent photometric producer")
        required.add("photometric_method")
    _keys(binding, required, where="source binding")
    if binding["contract"] != CONTRACT:
        raise ValueError("Unknown RAW source contract")
    if (not isinstance(binding["run"], str)
            or re.fullmatch(parameters["source"]["source_id_pattern"], binding["run"]) is None):
        raise ValueError("RAW source requires its exact run identity")
    expected_profile = parameters["source"]["profiles"][binding["run"]]
    if binding["profile_id"] != expected_profile:
        raise ValueError("RAW source camera profile differs from its run")
    if tuple(binding["native_shape_hw"]) != tuple(parameters["source"]["native_shape_hw"]):
        raise ValueError("RAW source differs from the unchanged native grid")
    source.source_homography(binding["baseline_reference_to_native_h"], parameters=parameters)
    if not isinstance(binding["shape_assets"], list) or not binding["shape_assets"]:
        raise ValueError("RAW source needs the existing shape correction assets")
    models = _keys(binding["models"], {"fit", "reference", "old_field"}, where="frozen models")
    for name in ("fit", "reference"):
        _keys(models[name], {"data", "sidecar"}, where=name + " model")
    actual = execution_identity()
    source.verify_execution_identity(actual, binding["execution_identity"], tuple(actual))
    declared = parameters["source"].get("raw_execution_identity")
    if declared is not None and declared != binding["execution_identity"] and "photometric_method" not in binding:
        raise ValueError("RAW source execution binding differs from its native configuration")


def validate_frame(row: Mapping, binding: Mapping, parameters: Mapping) -> tuple:
    _keys(row, {"contract", "run", "acquisition", "reference_to_native_h",
                "baseline_to_native_h"}, where="native frame")
    if row["contract"] != FRAME_CONTRACT or row["run"] != binding["run"]:
        raise ValueError("RAW frame differs from its source contract/run")
    h_ref = source.source_homography(row["reference_to_native_h"], parameters=parameters)
    h_b = source.source_homography(row["baseline_to_native_h"], parameters=parameters)
    h_ref_b = source.source_homography(np.linalg.inv(h_b) @ h_ref, parameters=parameters)
    expected = source.source_homography(binding["baseline_reference_to_native_h"], parameters=parameters)
    if not source._same_h(h_ref_b, expected, parameters=parameters):
        raise ValueError("RAW frame changes the frozen reference-to-baseline geometry")
    return h_ref, h_b, h_ref_b


class RawRigSourceProcessor:
    """Correct RAW acquisitions with unchanged fits, grids and field arithmetic."""

    def __init__(self, binding_path, *, parameters: Mapping):
        from darsia._opencv_runtime import scientific_thread
        scientific_thread("RawRigSourceProcessor:setup")
        self.parameters = parameters
        self.binding_path = self._path(binding_path)
        self.binding_sha256 = sha256_file(self.binding_path)
        self.binding = read_bound_json(self.binding_path, self.binding_sha256)
        validate_binding(self.binding, parameters)
        b = self.binding
        if b.get("photometric_method"):
            self._initialize_independent()
            return
        # Both the NPZ and the exact sidecar are inputs. An internally
        # consistent replacement pair cannot substitute for the frozen model.
        loaded = {}
        for name in ("fit", "reference"):
            model = b["models"][name]
            data, sidecar = self._asset(model["data"]), self._asset(model["sidecar"])
            if sidecar != data.with_suffix(".json"):
                raise ValueError("RAW model sidecar differs from the loader's actual input")
            loaded[name] = photometry.load_baseline(data, parameters=parameters["photometry"])
            # The legacy model loader derives its sidecar name. Verify both
            # consumed files again after loading, retaining their outer pins.
            self._asset(model["data"])
            self._asset(model["sidecar"])
        fit, grid = loaded["fit"]
        reference, ref_grid = loaded["reference"]
        if b.get("correction_order"):
            choice = b["normalisation"]["choice"]
            record = b["normalisation"]["record"]
            saved = read_bound_json(self._asset(record), record["sha256"])
            if saved.get("format") != "darsia-prepared-image-normalisation-v1" or saved.get("choice") != choice:
                raise ValueError("The saved normalisation choice differs from the prepared RAW inputs")
            if choice["method"] != "off" and reference.run != choice["reference_run"]:
                raise ValueError("The actual light-reference model belongs to another selected run")
        # The virtual-reference asset has its own historical run string
        # (e.g. a donor-consensus name). Its exact bound bytes define its role.
        if fit.run != b["run"] or grid != ref_grid:
            raise ValueError("RAW source and frozen photometric models/grid differ")
        fit_sidecar = b["models"]["fit"]["sidecar"]
        fit_receipt = read_bound_json(self._asset(fit_sidecar), fit_sidecar["sha256"])
        if fit_receipt["source"]["source_arw_sha256"] != b["baseline_raw"]["sha256"]:
            raise ValueError("RAW baseline acquisition differs from the unchanged B fit")
        old_field = None
        if b["models"]["old_field"] is not None:
            with np.load(self._asset(b["models"]["old_field"]), allow_pickle=False) as archive:
                old_field = archive["field"].astype(np.float32)
            if old_field.dtype != np.float32 or old_field.shape != grid.shape_hw:
                raise ValueError("RAW source old-light field differs from its frozen grid/type")
        self.processor = SimpleNamespace(run=b["run"], fit=fit, reference=reference,
                                         grid=grid, old_field=old_field)
        self.shape_assets = [dict(item, path=str(self._asset(item))) for item in b["shape_assets"]]
        self.shape_chain = raw_positioning.load_shape_chain(self.shape_assets)
        self.image_metadata = dict(b["image_metadata"])
        if isinstance(self.image_metadata.get("date"), str):
            self.image_metadata["date"] = datetime.fromisoformat(self.image_metadata["date"])
        self.baseline_sensor = self._sensor(b["baseline_raw"])
        self.baseline_camera_u = raw_positioning.camera_u(self.baseline_sensor, parameters=parameters["registration"])
        if b.get("correction_order"):
            _, self.baseline_renderer = self._render(self.baseline_sensor, camera_u=self.baseline_camera_u)
        else:
            _, self.baseline_renderer = self._render(self.baseline_sensor)
        self.identity = dict(contract=CONTRACT, source_binding=dict(path=str(self.binding_path),
                             sha256=self.binding_sha256), execution_identity=execution_identity(),
                             models=b["models"], profile_id=b["profile_id"],
                             native_shape_hw=b["native_shape_hw"], cpu_only=True,
                             training=False, new_photometric_fit=False, jpeg_inputs=False)

    def _initialize_independent(self):
        """Use new explicitly prepared inputs, without loading a saved native fit."""
        from .full_photometry import load_independent_model
        b = self.binding
        saved = read_bound_json(self._asset(b["normalisation"]["record"]),
                                b["normalisation"]["record"]["sha256"])
        if saved.get("format") != "darsia-prepared-image-normalisation-v1" or saved.get("choice") != b["normalisation"]["choice"]:
            raise ValueError("The independent source has another saved image choice")
        model = load_independent_model(self)
        self.processor = SimpleNamespace(run=b["run"], independent=model)
        self.shape_assets = [dict(item, path=str(self._asset(item))) for item in b["shape_assets"]]
        self.shape_chain = raw_positioning.load_shape_chain(self.shape_assets)
        self.image_metadata = dict(b["image_metadata"])
        for key in ("date", "reference_date"):
            if isinstance(self.image_metadata.get(key), str):
                self.image_metadata[key] = datetime.fromisoformat(self.image_metadata[key])
        self.baseline_sensor = self._sensor(b["baseline_raw"])
        self.baseline_camera_u = raw_positioning.camera_u(self.baseline_sensor, parameters=self.parameters["registration"])
        _, self.baseline_renderer = self._render(self.baseline_sensor, camera_u=self.baseline_camera_u)
        self.identity = dict(contract=CONTRACT, source_binding=dict(path=str(self.binding_path),
            sha256=self.binding_sha256), execution_identity=execution_identity(), models=b["models"],
            profile_id=b["profile_id"], native_shape_hw=b["native_shape_hw"], cpu_only=True,
            training=False, new_photometric_fit=True, photometric_method=b["photometric_method"], jpeg_inputs=False)

    def _path(self, value):
        return resolve_asset_path(value, self.parameters["source"]["asset_aliases"])

    def _asset(self, descriptor):
        if (not isinstance(descriptor, dict) or set(descriptor) - {"path", "sha256", "size_bytes"}
                or not {"path", "sha256"} <= set(descriptor)):
            raise ValueError("Exact RAW asset descriptor required")
        path = self._path(descriptor["path"])
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("Exact RAW asset SHA-256 required")
        if (("size_bytes" in descriptor and path.stat().st_size != descriptor["size_bytes"])
                or sha256_file(path) != digest):
            raise ValueError("RAW source/model input differs from its binding: " + str(path))
        return path

    def _sensor(self, descriptor):
        path = self._asset(descriptor)
        key = (str(path), descriptor["sha256"])
        if getattr(self, "_current_sensor_key", None) == key:
            return self._current_sensor
        sensor = raw_positioning.read_sensor(path,
                    expected_sha256=descriptor["sha256"], parameters=self.parameters["registration"],
                    profile_id=self.binding["profile_id"])
        self._current_sensor_key, self._current_sensor = key, sensor
        return sensor

    def _render(self, sensor, *, camera_u=None):
        options = {} if camera_u is None else dict(camera_u_rgb=camera_u, positioning_only=True)
        linear, audit = raw_positioning.render_native(sensor, shape_assets=self.shape_assets,
                    shape_hw=self.binding["native_shape_hw"], image_metadata=self.image_metadata,
                    parameters=self.parameters["registration"], **options)
        audit.update(source_id=self.processor.run, run=self.processor.run,
                     profile=sensor.profile_id, sources=dict(arw_sha256=sensor.source["sha256"]),
                     white_level=sensor.metadata["white_level"],
                     review_to_camera_h=audit["native_to_camera_u_h"],
                     camera_jpeg_wh=audit["camera_u_wh"], raw_visible_wh=sensor.metadata["raw_visible_wh"],
                     sony_distortion_table=list(sensor.distortion), sony_ca_red_table=list(sensor.ca_red),
                     sony_ca_blue_table=list(sensor.ca_blue), legacy_numeric_grid_names_only=True)
        return linear, audit

    def correct(self, geometry_frame, *, parent_frame_index=None, parent_reference=False, source_purpose=None):
        from darsia._opencv_runtime import scientific_thread, completed_frame
        before = scientific_thread("RawRigSourceProcessor.correct:before")
        if parent_frame_index is not None or parent_reference:
            raise ValueError("JPEG-positioned parent pixels cannot be reused by the RAW-only source")
        if source_purpose not in {None, "clearwater"}:
            raise ValueError("Unknown RAW source purpose")
        if sha256_file(self.binding_path) != self.binding_sha256:
            raise ValueError("RAW source binding changed after processor setup")
        actual = execution_identity()
        source.verify_execution_identity(actual, self.binding["execution_identity"], tuple(actual))
        h_ref, h_b, h_ref_b = validate_frame(geometry_frame, self.binding, self.parameters)
        sensor = self._sensor(geometry_frame["acquisition"])
        exact_b = sensor.source["sha256"] == self.baseline_sensor.source["sha256"]
        if exact_b and not source._same_h(h_b, np.eye(3), parameters=self.parameters):
            raise ValueError("Exact RAW B acquisition has a nonidentity baseline transform")
        if self.binding.get("correction_order"):
            from .full_photometry import correct
            return correct(self, sensor, h_ref, h_b, h_ref_b, source_purpose=source_purpose)
        if source_purpose == "clearwater":
            raise ValueError("Current clear water requires the explicit full-photograph correction route")
        linear, renderer = self._render(sensor)
        xy = usable = chart = None
        p = self.processor
        if not exact_b:
            xy, usable, chart = raw_chart.register(
                baseline_sensor=self.baseline_sensor, frame_sensor=sensor,
                baseline_camera_u=self.baseline_camera_u,
                frame_camera_u=raw_positioning.camera_u(sensor, parameters=self.parameters["registration"]),
                baseline_chart_xy8=source.chart_baseline_coordinates(p.fit, p.grid, h_ref_b, parameters=self.parameters),
                baseline_chart_log_rgb=p.fit.CB,
                baseline_native_to_u_h=np.asarray(self.baseline_renderer["native_to_camera_u_h"]),
                frame_native_to_u_h=np.asarray(renderer["native_to_camera_u_h"]),
                frame_native_linear_rgb=linear, parameters=self.parameters["registration"])
        corrected, info = photometry.correct_frame_native(
            linear, p.fit, p.reference, p.grid, h_ref,
            np.ones(self.binding["native_shape_hw"], bool), exact_baseline_source=exact_b,
            frame_chart_xy8=xy, frame_chart_usable=usable, old_field_ref8=p.old_field,
            parameters=self.parameters["photometry"])
        return corrected, dict(contract=CONTRACT, processor=self.identity,
            sources=dict(camera_arw=dict(sensor.source)), renderer=renderer, chart=chart,
            photometry=info, exact_B=exact_b,
            **{self.parameters["source"]["symbols"]["reference_to_frame_h_key"]: h_ref.tolist()},
            H_v5_B_to_frame_native=h_b.tolist(), source_identity=dict(kind="raw-acquisition",
            raw_sha256=sensor.source["sha256"]), parent_cache=None, cpu_only=True, training=False,
            jpeg_inputs=False, opencv_ipp=completed_frame(before, "RawRigSourceProcessor.correct:after"))
