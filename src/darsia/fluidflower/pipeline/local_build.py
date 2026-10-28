"""Ordinary local P1 inputs from the completed Build and measured run geometry.

This adapter joins already produced DATA. It neither decodes a second frame nor
changes the original SOURCE optical/floor transforms, interpolation or history.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np

from ..geometry.context import homography
from ..mass.build_frame_fields import _canvas, registered_build_fields
from ..mass.native_optical_source import _bound, _same, load_build_physical_context
from ..mass.n1_pipeline import DataReader


class LocalBuildInputs:
    """One run's original saved material, metric model and frame receipts."""

    def __init__(self, configuration, *, operation_result, metric_model,
                 frame_receipts, chart=None, reader=None, _checked_build=None):
        self.reader = reader or DataReader(configuration.get("data_aliases", ()), unc_only=False)
        self.operation_result = deepcopy(_bound(operation_result))
        self.metric_model = deepcopy(_bound(metric_model))
        if _checked_build is None:
            self.context = load_build_physical_context(self.operation_result, reader=self.reader)
        else:
            from .current_build import CheckedCurrentBuild
            if (type(_checked_build) is not CheckedCurrentBuild
                    or not _same(configuration, _checked_build._config)
                    or not _same(self.operation_result, _checked_build._declaration["operation_result"])):
                raise ValueError("The decoded Build must belong to the exact active run")
            _checked_build._active()
            self.context = _checked_build._physical
        self.config = self.bind_configuration(configuration)
        if self.context.run != self.config["run"]:
            raise ValueError("Local physical Build belongs to another run")
        if not _same(self.config["native"]["configuration"], self.context.provenance["native_configuration"]):
            from .current_phase_recipe import require_runtime_configuration
            require_runtime_configuration(self.context.provenance["native_configuration"],
                                          self.config["native"]["configuration"], reader=self.reader)
        self.phase = self.reader.json(self.config["phase"]["configuration"])
        self.native_parameters = self.reader.json(self.config["native"]["configuration"])
        self.geometry_parameters = self.phase["geometry_parameters"]
        self.frames = {row["case_id"]: row for row in self.config["frames"]}
        if len(self.frames) != len(self.config["frames"]) or set(frame_receipts) != set(self.frames):
            raise ValueError("Local receipts must cover each selected frame exactly once")
        self.frame_receipts = {key: deepcopy(_bound(value)) for key, value in frame_receipts.items()}
        self._pending = None
        self._read_material()
        self._read_metric()
        self.chart_mask = np.zeros(self.context.labels.shape, bool)
        self.chart_provenance = None
        if chart is None and self.preparation is not None:
            preparation = self.reader.json(self.preparation)
            if self.context.run in preparation.get("chart_transfers", {}):
                chart = {"preparation": self.preparation}
        if chart is not None:
            from .local_build_chart import read_chart
            self.chart_mask, self.chart_provenance = read_chart(chart, reader=self.reader, context=self.context)
        chart_roles = np.isin(self.context.facies, [key for key, row in self.roles.items()
                                                  if row["analysis_role"] == "colorchecker"])
        if np.any(chart_roles) and chart is None:
            from ..painting_chart_transfer import MOVED_RUNS
            if self.context.run in MOVED_RUNS:
                raise ValueError("This run requires its approved transferred chart mask")
            self.chart_mask = chart_roles.copy()
            self.chart_provenance = dict(mode="retained Build chart mask",
                operation_result=deepcopy(self.operation_result))
        if not np.array_equal(chart_roles, self.chart_mask):
            raise ValueError("Actual chart material differs from its source-bound occlusion mask")

    def bind_configuration(self, configuration):
        """Retain original inputs; bind the legacy result slot to the real Build."""
        value = deepcopy(configuration)
        value.pop("current_build_binding", None)
        template = value["statistics"] if "statistics" in value else value["mass"]["template"]
        template["physical_correction"] = deepcopy(self.operation_result)
        return value

    def _read_material(self):
        from ..geometry_raster_sources import _full_csv
        from ..painted_analysis_exclusions import configuration_exclusion
        adopted = self.reader.json(self.context.provenance["adopted_revision"])
        config = adopted["configuration"]
        self.preparation = deepcopy((config.get("painted_adoption_table") or {}).get("preparation"))
        if config["grouping_mode"] != "shared_facies":
            raise ValueError("Local Build requires the actual shared facies table")
        self.table = deepcopy(_bound(adopted["artifacts"]["source_mode_files"]["facies_properties"]))
        columns, records, ids = _full_csv(self.reader.path(self.table), strict=True)
        names = {name.strip().casefold(): name for name in columns}
        self.roles = {key: dict(name=row[names["name"]], analysis_role=row[names["analysis_role"]],
                               porosity=float(row[names["porosity"]])) for key, row in zip(ids, records)}
        self.mapping, self.enabled = {}, []
        for row in config["regions"]:
            label, facies = row["label_id"], row["facies_id"]
            if (type(label) is not int or type(facies) is not int or label in self.mapping
                    or facies not in self.roles or type(row.get("enabled", True)) is not bool):
                raise ValueError("Adopted labels require unique explicit facies and activation")
            self.mapping[label] = facies
            if row.get("enabled", True):
                self.enabled.append(label)
        self.source_known = np.zeros(self.context.labels.shape, bool)
        for label, facies in self.mapping.items():
            selected = self.context.labels == label
            phi = self.roles[facies]["porosity"]
            if (not np.isfinite(phi) or not 0 <= phi <= 1
                    or np.any(self.context.facies[selected] != facies)
                    or np.any(self.context.porosity[selected] != np.asarray(phi, self.context.porosity.dtype))):
                raise ValueError("Saved Build facies/porosity differ from the actual adopted CSV")
            self.source_known |= selected
        gap = configuration_exclusion(config, self.context.run, labels=self.context.labels)
        if gap is not None:
            self.source_known &= ~gap
        # Unknown label zero is never reinterpreted as Freewater facies zero.
        self.source_enabled = np.isin(self.context.labels, self.enabled)

    def _read_metric(self):
        model = self.reader.json(self.metric_model)
        certificate = self.reader.json(self.context.provenance["baseline_renderer"])
        grid = model["baseline_native_grid"]
        if (model.get("format") != "darsia-measured-run-geometry-v1"
                or model.get("run") != self.context.run or model.get("run_id") != self.context.run
                or not _same(model["reference_image"], self.context.provenance["raw_acquisition"])
                or not _same(grid["renderer"], self.context.provenance["baseline_renderer"])
                or not _same(grid["shape_hw"], self.context.grid["num_voxels"])
                or not _same(model["shape_hw"], certificate["renderer"]["camera_u_wh"][::-1])):
            raise ValueError("Measured metric model differs from the actual Build baseline/grid")
        self.build_to_camera = self._h(certificate["H_rig_corrected_native_to_baseline_camera_u"])
        if (not np.array_equal(self.build_to_camera, self._h(grid["H_rig_corrected_native_to_baseline_camera_u"]))
                or not np.array_equal(self.build_to_camera, self._h(certificate["renderer"]["native_to_camera_u_h"]))):
            raise ValueError("Metric model and actual Build camera-U transforms differ")
        for key in ("source_model", "current_edges", "baseline_configuration", "model_photograph"):
            self.reader.path(_bound(model[key]))
        if model.get("baseline_to_model_registration") is not None:
            self.reader.path(_bound(model["baseline_to_model_registration"]))
        self.build_to_m = np.diag([.001, .001, 1.]) @ self._h(model["H_camera_u_to_metric_mm"]) @ self.build_to_camera

    def _h(self, value):
        return homography(value, parameters=self.geometry_parameters)

    def _same_h(self, left, right):
        from ..native.source import source_homography, _same_h
        return _same_h(source_homography(left, parameters=self.native_parameters),
                       source_homography(right, parameters=self.native_parameters),
                       parameters=self.native_parameters)

    def validate_mass_config(self, config, *, statistics_only, frame_loading_only=False):
        """Require the unchanged selected scientific recipe with the actual Build."""
        phase1 = "statistics" in self.config
        template = self.config["statistics"] if phase1 else self.config["mass"]["template"]
        expected = dict(template, run=self.config["run"], frames=self.config["frames"],
                        data_aliases=self.config.get("data_aliases", ()))
        mode_matches = (phase1 and not statistics_only if frame_loading_only
                        else statistics_only == phase1)
        if (type(statistics_only) is not bool or type(frame_loading_only) is not bool
                or not mode_matches or "pixel_domain" in config or not _same(config, expected)):
            raise ValueError("Local N1 inputs differ from the exact selected scientific configuration")
        shared = config["shared"]
        self.swi = {}
        for name, value in shared["swi_by_facies"].items():
            key = int(name)
            if key in self.swi or key not in self.roles or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Swi must retain explicit valid adopted facies IDs")
            self.swi[key] = value
        for name, value in shared["porosity_by_facies"].items():
            key = int(name)
            if key not in self.roles or value != self.roles[key]["porosity"]:
                raise ValueError("N1 porosity names/values differ from the actual adopted facies")

    def _warp_mask(self, value, row):
        return cv2.warpPerspective(np.asarray(value, np.uint8), row["matrix"], row["shape"][::-1],
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    def check_phase(self, spec, native, frame):
        """Check one actual upstream result and restrict only explicit exclusions."""
        self._pending = None
        if not _same(self.frames.get(spec.get("case_id")), spec):
            raise ValueError("Local phase input differs from its exact selected frame")
        receipt_binding = self.frame_receipts[spec["case_id"]]
        receipt = self.reader.json(receipt_binding)
        task = spec["native_task"]
        from .current_build import _physical_context_equal
        if (task["kind"] != "raw_rig" or receipt.get("format") != "darsia-build-native-frame-v1"
                or not _physical_context_equal(receipt["physical_context"], self.context.provenance)
                or not _same(receipt["native_frame"], task["geometry"])
                or not _same(receipt["raw_source_binding"], task["processor"]["raw_binding"])):
            raise ValueError("Local frame receipt differs from its actual Build/source inputs")
        bridge = self.reader.json(_bound(receipt["bridge"]))
        if not _same(bridge["native_configuration"], self.config["native"]["configuration"]):
            from .current_phase_recipe import require_runtime_configuration
            require_runtime_configuration(bridge["native_configuration"],
                                          self.config["native"]["configuration"], reader=self.reader)
        if (bridge.get("format") != "darsia-build-native-bridge-v1"
                or not _same(bridge["operation_result"], self.operation_result)
                or not np.array_equal(self._h(bridge["saved_build_to_camera_u_h"]), self.build_to_camera)):
            raise ValueError("Local frame bridge differs from the actual saved Build")
        actual, prior = native["audit"], receipt["correction"]
        for key in ("contract", "jpeg_inputs", "source_identity"):
            if not _same(actual[key], prior[key]):
                raise ValueError("Actual RAW audit differs from its Build frame")
        from .current_build import _source_equal
        acquisition = native["native_geometry_row"]["acquisition"]
        if (not _same(native["native_geometry_row"], self.reader.json(receipt["native_frame"]))
                or not _same(native["geometry_row"], self.reader.json(spec["geometry"]))
                or acquisition["sha256"] != spec["source"]["source_sha256"]
                or not _source_equal(actual["sources"]["camera_arw"], acquisition, self.reader)
                or not _source_equal(prior["sources"]["camera_arw"], acquisition, self.reader)):
            raise ValueError("Actual RAW decoder consumed another acquisition")
        for key in ("models", "execution_identity", "profile_id", "native_shape_hw"):
            if not _same(actual["processor"][key], prior["processor"][key]):
                raise ValueError("Actual RAW processor differs from the Build frame")
        if (not _source_equal(actual["processor"]["source_binding"], receipt["raw_source_binding"], self.reader)
                or not _source_equal(prior["processor"]["source_binding"], receipt["raw_source_binding"], self.reader)):
            raise ValueError("Actual RAW processor consumed another source binding")
        for left, right in ((actual["renderer"]["native_to_camera_u_h"], prior["renderer"]["native_to_camera_u_h"]),
                            (actual["H_v5_B_to_frame_native"], prior["H_v5_B_to_frame_native"]),
                            (frame["current_h"], spec["phase_roles"]["current_h"]),
                            (frame["reference_to_current_h"], spec["phase_roles"]["reference_to_current_h"])):
            if not self._same_h(left, right):
                raise ValueError("Actual RAW/phase transform differs from its source-bound receipt")
        reference_key = self.native_parameters["source"]["symbols"]["reference_to_frame_h_key"]
        raw_geometry = native["native_geometry_row"]
        for left, right in ((actual[reference_key], raw_geometry["reference_to_native_h"]),
                            (actual["H_v5_B_to_frame_native"], raw_geometry["baseline_to_native_h"]),
                            (receipt["build_to_frame_native_h"], self._h(raw_geometry["baseline_to_native_h"])
                             @ self._h(bridge["build_to_baseline_native_h"]))):
            if not self._same_h(left, right):
                raise ValueError("Build physical transform differs from the actual RAW coordinate join")
        context = frame["context"]
        shape = context.rig_domain.shape
        area = frame["pixel_area_mm2"]
        if (context is not native["context"] or frame["full_log"] is not native["full_log"]
                or not _same(frame["source"], spec["source"])
                or area.dtype != np.float32 or area.shape != shape
                or not np.isfinite(area).all() or np.any(area <= 0)):
            raise ValueError("Original phase buffers or measured mm2 areas differ")
        area_role = spec["phase_roles"]["area"]
        if area_role["role"] == "measured_parent":
            expected_area = self.reader.array(area_role["measured"])
        else:
            from ..geometry.areas import apply_selected_area_scale
            if not np.array_equal(frame["original_area"], context.pixel_area_mm2):
                raise ValueError("Selected frame lost its original measured area map")
            expected_area = apply_selected_area_scale(context.pixel_area_mm2, area_role["factor"])
        if expected_area.dtype != np.float32 or not np.array_equal(area, expected_area):
            raise ValueError("Phase area differs from its original measured area recipe")
        current = native["record"]["full"]
        if (not _same(current["native_origin_xy"], list(context.origin_xy))
                or not _same(current["phase_shape_hw"], list(shape))):
            raise ValueError("Native canvas differs from its original phase context")
        row = dict(case_id=spec["case_id"], receipt=receipt, receipt_binding=receipt_binding,
                   current=deepcopy(current), source=deepcopy(spec["source"]["source_binding"]),
                   shape=shape, origin=list(context.origin_xy), area=area,
                   matrix=_canvas(receipt["build_to_frame_native_h"], context.origin_xy, self.geometry_parameters))
        from ..painted_analysis_exclusions import transport_build_exclusion
        gap, gap_provenance = transport_build_exclusion(self.context, self.reader,
            receipt["build_to_frame_native_h"], context.origin_xy, shape, self.geometry_parameters)
        report_domain = context.rig_domain.copy()
        if gap is not None:
            report_domain |= gap
        chart = self._warp_mask(self.chart_mask, row) & (context.rig_domain | context.hardware_excluded)
        # A chart may overlap an original hardware exclusion; retain its report membership.
        report_domain |= chart
        exclusions, used = {}, np.zeros(shape, bool)
        for name, mask in (("approved_analysis_gap", gap), ("colorchecker_occlusion", chart)):
            if mask is not None and mask.any():
                selected = mask & report_domain & ~used
                exclusions[name] = selected
                used |= selected
        for key, role in self.roles.items():
            source = self.source_known & (self.context.facies == key)
            if role["analysis_role"] == "excluded" or role["porosity"] == 0:
                selected = self._warp_mask(source, row) & report_domain & ~used
                if selected.any():
                    exclusions["material_" + str(key)] = selected
                    used |= selected
        disabled = self._warp_mask(self.source_known & ~self.source_enabled, row) & report_domain & ~used
        if disabled.any():
            exclusions["disabled_material"] = disabled
            used |= disabled
        analysis = context.rig_domain & ~used
        updated = dict(frame, context=replace(context, rig_domain=analysis,
            modelled=context.modelled & analysis, filled_observed=context.filled_observed & analysis,
            filled_unobserved=context.filled_unobserved & analysis), physical_rig_domain=report_domain)
        row.update(frame=updated, analysis_domain=analysis, reporting_domain=report_domain,
                   excluded_masks=exclusions, gap_provenance=gap_provenance)
        self._pending = row
        return updated

    def check_inferred(self, spec, frame, inferred):
        row = self._row(spec, phase=True)
        if frame is not row["frame"]:
            raise ValueError("Local inference lacks its exact checked phase frame")
        labels = inferred["labels"]
        if (labels.dtype != np.uint8 or labels.shape != row["shape"]
                or not np.array_equal(labels != 255, row["analysis_domain"])
                or not np.array_equal(inferred["pixel_area_mm2"], row["area"])):
            raise ValueError("Phase inference changed the explicit domain or measured areas")
        inferred["classes"]["physical_rig_domain"] = row["reporting_domain"]
        for name in ("approved_analysis_gap", "colorchecker_occlusion"):
            if name in row["excluded_masks"]:
                inferred["classes"][name] = row["excluded_masks"][name]
        if row["gap_provenance"] is not None:
            inferred["qa"]["approved_analysis_gap"] = deepcopy(row["gap_provenance"])
        inferred["qa"]["local_build"] = self._source_binding(row)
        row["labels"] = labels.copy()

    def _row(self, spec, *, phase=False):
        row = self._pending
        if row is None or row["case_id"] != spec.get("case_id"):
            raise ValueError("Local physical input lacks its one checked upstream frame")
        if not phase and (not _same(row["current"], spec["current"])
                          or not _same(row["source"], spec["source_binding"])):
            raise ValueError("Local N1 input differs from the actual upstream source")
        return row

    def _source_binding(self, row):
        return dict(physical_input_mode="local_build", legacy_physical_correction_applied=False,
            operation_result=deepcopy(self.operation_result), metric_model=deepcopy(self.metric_model),
            adopted_facies_table=deepcopy(self.table), bridge_frame=deepcopy(row["receipt_binding"]),
            source=deepcopy(row["source"]), chart=deepcopy(self.chart_provenance),
            depth_method="original N1 linear measured depth; nearest outside hull")

    def physical_fields(self, spec, shape, math):
        row = self._row(spec)
        if tuple(shape) != row["shape"]:
            raise ValueError("Local physical fields differ from the current canvas")
        result = registered_build_fields(self.context, row["receipt"]["build_to_frame_native_h"],
            row["origin"], shape, row["reporting_domain"], math, self.geometry_parameters,
            source_domain=self.source_known)
        coverage = math.warp_continuous(self.source_known, row["matrix"], shape, cv2_module=cv2)
        material = (row["reporting_domain"] & self._warp_mask(self.source_known, row)
                    & (coverage >= 1 - 1e-6) & np.isfinite(result["porosity"])
                    & (result["porosity"] >= 0) & (result["porosity"] <= 1))
        row["material_supported_mask"] = material
        water = np.full(shape, np.nan, math.dtype)
        for key, value in self.swi.items():
            water[result["facies"] == key] = value
        result["swi"] = water
        return result

    def frame_context(self, spec, labels, classes, phase_qa):
        row = self._row(spec)
        if ("labels" not in row or not np.array_equal(labels, row["labels"])
                or not np.array_equal(classes["rig_domain"], row["analysis_domain"])
                or not np.array_equal(classes.get("physical_rig_domain"), row["reporting_domain"])
                or not isinstance(phase_qa, dict)
                or not _same(phase_qa.get("local_build"), self._source_binding(row))):
            raise ValueError("N1 phase labels/domain/QA differ from the actual local inference")
        for name in ("approved_analysis_gap", "colorchecker_occlusion"):
            if name in row["excluded_masks"] and not np.array_equal(classes.get(name), row["excluded_masks"][name]):
                raise ValueError("N1 phase exclusions differ from the actual local inference")
        if row["gap_provenance"] is not None and not _same(phase_qa.get("approved_analysis_gap"), row["gap_provenance"]):
            raise ValueError("N1 lost the actual approved-gap provenance")
        return dict(analysis_domain=row["analysis_domain"], reporting_domain=row["reporting_domain"],
            material_supported_mask=row["material_supported_mask"], excluded_masks=row["excluded_masks"],
            source_binding=self._source_binding(row), exclusion_provenance=row["gap_provenance"])

    def coordinates(self, spec, shape):
        from ..mass.n1_geometry import frame_coordinates
        row = self._row(spec)
        if tuple(shape) != row["shape"]:
            raise ValueError("Local metric coordinates differ from the current canvas")
        return frame_coordinates(self.build_to_m, {"build_to_native": row["receipt"]["build_to_frame_native_h"]},
            dict(phase_shape_hw=list(shape), native_origin_xy=row["origin"]), fields={"source_to_native": "build_to_native"})

    def check_area(self, spec, area):
        """Preserve the actual phase writer's float32 mm2 map before conversion."""
        row = self._row(spec)
        if area.dtype != np.float32 or not np.array_equal(area, row["area"]):
            raise ValueError("N1 area differs from the original phase measured mm2 map")
