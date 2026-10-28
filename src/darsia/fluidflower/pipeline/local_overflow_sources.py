"""Replay closed local P1 products with their original measured Build inputs.

This reads the original completion and every declared byte. It never decodes
RAW, repeats phase inference or treats a queued/partial run as completed.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from .local_build import LocalBuildInputs
from ..mass.build_frame_fields import _canvas
from ..mass.native_optical_source import _same
from ..mass.n1_pipeline import DataReader, asset, N1Pipeline
from ..mass.production_precision import PixelMath


class SavedLocalBuildInputs(LocalBuildInputs):
    """Restore one verified producer frame, retaining the same physical methods."""

    def restore(self, phase_spec, record, mass_spec):
        reader = self.reader
        if (not _same(self.frames.get(phase_spec["case_id"]), phase_spec)
                or record["case_id"] != phase_spec["case_id"]
                or not _same(record["source"], phase_spec["source"])):
            raise ValueError("Saved local Build replay identifies another acquisition")
        qa = reader.json(record["phase"]["qa"])
        receipt_binding = self.frame_receipts[phase_spec["case_id"]]
        receipt = reader.json(receipt_binding)
        from .current_build import _physical_context_equal
        if (receipt.get("format") != "darsia-build-native-frame-v1"
                or not _physical_context_equal(receipt["physical_context"], self.context.provenance)
                or not _same(receipt["native_frame"], phase_spec["native_task"]["geometry"])
                or not _same(receipt["raw_source_binding"], phase_spec["native_task"]["processor"]["raw_binding"])):
            raise ValueError("Saved frame receipt differs from the original Build/source")
        bridge = reader.json(receipt["bridge"])
        if (not _same(bridge["operation_result"], self.operation_result)
                or not _same(bridge["native_configuration"], self.config["native"]["configuration"])
                or not np.array_equal(self._h(bridge["saved_build_to_camera_u_h"]), self.build_to_camera)):
            raise ValueError("Saved frame bridge differs from the actual Build")
        current = record["native"]["full"]
        shape = tuple(current["phase_shape_hw"])
        classes = reader.archive(record["phase"]["classes"],
            ("rig_domain", "modelled", "filled_observed", "filled_unobserved"),
            optional=("physical_rig_domain", "approved_analysis_gap", "colorchecker_occlusion"))
        if ("physical_rig_domain" not in classes
                or any(mask.dtype != bool or mask.shape != shape for mask in classes.values())):
            raise ValueError("Saved local domains differ from the original native grid")
        with Image.open(reader.path(record["phase"]["labels"])) as image:
            labels = np.asarray(image).copy()
        if labels.dtype != np.uint8 or labels.shape != shape or not np.array_equal(labels != 255, classes["rig_domain"]):
            raise ValueError("Saved phase labels differ from their original analysis domain")
        area_bound = dict(record["phase"]["area"])
        area_bound.setdefault("array_key", "pixel_area_mm2")
        area = reader.array(area_bound)
        row = dict(case_id=phase_spec["case_id"], receipt=receipt, receipt_binding=receipt_binding,
            current=deepcopy(current), source=deepcopy(phase_spec["source"]["source_binding"]),
            shape=shape, origin=current["native_origin_xy"], area=area,
            matrix=_canvas(receipt["build_to_frame_native_h"], current["native_origin_xy"], self.geometry_parameters),
            labels=labels, analysis_domain=classes["rig_domain"], reporting_domain=classes["physical_rig_domain"])
        if not _same(qa.get("local_build"), self._source_binding(row)):
            raise ValueError("Saved local physical source binding changed")
        from ..painted_analysis_exclusions import transport_build_exclusion
        gap, provenance = transport_build_exclusion(self.context, reader,
            receipt["build_to_frame_native_h"], row["origin"], shape, self.geometry_parameters)
        chart = self._warp_mask(self.chart_mask, row) & row["reporting_domain"]
        exclusions, used = {}, np.zeros(shape, bool)
        for name, mask in (("approved_analysis_gap", gap), ("colorchecker_occlusion", chart)):
            if mask is not None and mask.any():
                selected = mask & row["reporting_domain"] & ~used
                exclusions[name] = selected
                used |= selected
        for key, role in self.roles.items():
            source = self.source_known & (self.context.facies == key)
            if role["analysis_role"] == "excluded" or role["porosity"] == 0:
                selected = self._warp_mask(source, row) & row["reporting_domain"] & ~used
                if selected.any():
                    exclusions["material_" + str(key)] = selected
                    used |= selected
        disabled = self._warp_mask(self.source_known & ~self.source_enabled, row) & row["reporting_domain"] & ~used
        if disabled.any():
            exclusions["disabled_material"] = disabled
        row.update(excluded_masks=exclusions, gap_provenance=provenance)
        self._pending = row
        # Existing methods independently compare all saved labels, domains,
        # measured areas, source descriptors and exclusion provenance.
        self.physical_fields(mass_spec, shape, PixelMath("float32"))
        self.frame_context(mass_spec, labels, classes, qa)
        self.check_area(mass_spec, area)


class LocalOverflowRun:
    """Exact selected local completion, original P1 contexts and probability data."""

    def __init__(self, phase_job, run, *, reader=None):
        from .refit_jobs import completed_attempt
        from .staged_data import verify_phase1_completion
        from ..phase.probability_storage import parse_request
        self.reader = reader or DataReader(unc_only=False)
        completion_binding = asset(Path(phase_job) / "results" / run / "completed.json")
        result = self.reader.json(completion_binding)
        attempt = completed_attempt(phase_job, run)
        manifest = self.reader.json(result["manifest"])
        if (result.get("execution_mode") != "local" or result.get("run") != run
                or manifest.get("execution_mode") != "local"
                or not _same(manifest["configuration"], result["configuration"])
                or Path(result["manifest"]["path"]).parent != attempt):
            raise ValueError("Overflow needs an exact completed local P1 result")
        self.verified = verify_phase1_completion(result["phase1_completion"], result["configuration"],
            expected_commit=manifest["code"]["commit"], reader=self.reader)
        self.config = self.verified["configuration"]
        if (self.config["run"] != run or self.config["precision"] != "float32"
                or parse_request(self.config) is None):
            raise ValueError("Using overflow as the free-gas target requires the original lossless float32 phase probabilities")
        self.bindings = dict(completion=completion_binding, manifest=result["manifest"],
            configuration=result["configuration"], phase1_completion=result["phase1_completion"])
        proofs = [self.reader.json(item["record"]["phase"]["qa"])["local_build"]
                  for item in self.verified["records"]]
        first = proofs[0]
        if any(not _same(proof[field], first[field]) for proof in proofs
               for field in ("operation_result", "metric_model")):
            raise ValueError("Local P1 frames bind different physical Build inputs")
        self.physical = SavedLocalBuildInputs(self.config, operation_result=first["operation_result"],
            metric_model=first["metric_model"], frame_receipts={spec["case_id"]: proof["bridge_frame"]
                for spec, proof in zip(self.config["frames"], proofs)}, reader=self.reader)
        statistics = dict(deepcopy(self.config["statistics"]), run=run,
            frames=deepcopy(self.config["frames"]), data_aliases=self.config.get("data_aliases", ()))
        self.pipeline = N1Pipeline(statistics, precision="float32", pixel_math=PixelMath("float32"),
            statistics_only=True, unc_only=False, local_build=self.physical)

    def frames(self, *, window=None, frame_indices=None):
        from .series import bind_mass_frame
        from ..mass.overflow_statistics import prepare_frame
        for spec, item in zip(self.config["frames"], self.verified["records"]):
            if window is not None and not window[0] <= spec["elapsed_h"] <= window[1]:
                continue
            if frame_indices is not None and spec["frame_index"] not in frame_indices:
                continue
            record = item["record"]
            mass = bind_mass_frame(spec, record["native"], record["phase"],
                baseline=self.config["phase"]["reference"]["full"])
            self.physical.restore(spec, record, mass)
            frame = prepare_frame(self.pipeline, mass, probability_roles=record["probabilities"],
                statistics_descriptor=record["statistics"], phase_source=spec["source"], case_id=spec["case_id"])
            yield spec, frame
            del frame

    def finish(self):
        for bound in self.bindings.values():
            self.reader.path(bound)
        for source in (self.reader, self.pipeline.reader):
            for (path, digest), stamp in list(source.verified.items()):
                source.path(dict(path=path, sha256=digest, size_bytes=stamp[0]))
