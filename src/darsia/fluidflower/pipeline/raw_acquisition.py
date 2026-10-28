"""Prepare new RAW baseline/history roles with the existing native pipeline.

Configuration selects acquisitions, frozen models, geometry and mask assets.
This route performs no fitting or phase inference, and publishes no adopted
reference. Its outputs feed the normal SeriesInputs baseline/history readers.
"""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from .series import SeriesInputs, _no_executable_configuration
from . import phase_inputs
from ..mass.n1_pipeline import DataReader, asset, _write_json
from ..native import sidecar, support
from ..phase import inputs


FORMAT = "darsia-raw-acquisition-preparation-v1"


class RawAcquisitionInputs(SeriesInputs):
    """Only native source/geometry operations; no old baseline is loaded."""

    def __init__(self, configuration, *, unc_only=True):
        _no_executable_configuration(configuration)
        required = {"format", "run", "native", "phase", "acquisitions", "temporal_pad_px"}
        if (not required <= set(configuration)
                or set(configuration) - required - {"data_aliases"}
                or configuration["format"] != FORMAT):
            raise ValueError("Explicit RAW acquisition preparation DATA required")
        frames = configuration["acquisitions"]
        if (not isinstance(frames, list) or not frames
                or len({f["case_id"] for f in frames}) != len(frames)
                or any(f["native_task"]["kind"] != "raw_rig" for f in frames)):
            raise ValueError("Distinct explicit RAW-only acquisitions required")
        self.config = configuration
        self.reader = DataReader(configuration.get("data_aliases", ()), unc_only=unc_only)
        self.aliases = {row["logical"]["path"]: row["physical"]["path"]
                        for row in configuration.get("data_aliases", ())}
        from .run_io import current_binding
        binding = current_binding()
        if binding is not None:
            self.aliases = binding.asset_aliases()
        self.phase = self.reader.json(configuration["phase"]["configuration"])
        self.parameters = deepcopy(self.phase["parameters"])
        self.native_parameters = self.reader.json(configuration["native"]["configuration"])
        self.native_parameters["source"]["asset_aliases"] = dict(self.aliases)
        self.processors = {}
        self.raw_mode = True
        pad = configuration["temporal_pad_px"]
        if type(pad) is not int or pad < 0:
            raise ValueError("The existing temporal padding convention must be explicit")

    def prepare(self, spec, stage):
        """Prepare this acquisition's native/ring and masked history arrays."""
        from .job_queue import name
        name(spec["case_id"])
        if spec["source"]["run"] != self.config["run"]:
            raise ValueError("RAW preparation source belongs to another run")
        stage = Path(stage)
        if stage.exists():
            raise FileExistsError("RAW acquisition stage already exists: " + str(stage))
        native = self.native(spec, stage)
        roles, p = spec["phase_roles"], self.parameters
        mask = roles["photo_sand_reference"]
        sand = support.source_photo_grid(self.reader.path(mask), mask["sha256"],
                                         parameters=self.native_parameters["support"])
        photo = support.project_source_photo_safe_core(
            sand, roles["reference_to_current_h"], shape_hw=tuple(p["geometry"]["native_shape_hw"]),
            guard_px=self.native_parameters["support"]["guard_px"],
            enclosed_hole_policy=self.native_parameters["support"]["fill_policy"],
            parameters=self.native_parameters["support"])
        masked = inputs.new_native_input(native["log"], native["valid"], photo,
            source_binding=spec["source"]["source_binding"],
            photo_mask_binding=roles["photo_proof"], parameters=p["inputs"])
        paths = stage / "masked_native_log.npy", stage / "masked_native_valid.npy"
        sidecar.write_log_linear(paths[0], masked.log)
        sidecar.write_log_linear(paths[1], masked.valid)
        fields = p["inputs"]["fields"]["native"]
        if len(fields) != 4:
            raise ValueError("Unsupported explicit native history field schema")
        log_asset, valid_asset = asset(paths[0]), asset(paths[1])
        audit_asset = native["record"]["audit"]
        raw_source = native["audit"]["sources"]["camera_arw"]
        row = dict(contract=phase_inputs.RAW_ACQUISITION_CONTRACT, source_kind="raw",
            raw_acquisition=raw_source, raw_producer_audit=audit_asset,
            **{p["inputs"]["source_sha_field"]: spec["source"]["source_sha256"],
               fields[0]: log_asset["path"], fields[1]: log_asset["sha256"],
               fields[2]: valid_asset["path"], fields[3]: valid_asset["sha256"]})
        pad = self.config["temporal_pad_px"]
        if pad:
            x, y, width, height = map(int, p["geometry"]["source_box_xywh"])
            full = native["full_log"]
            if min(x - pad, y - pad) < 0 or x + width + pad > full.shape[2] or y + height + pad > full.shape[1]:
                raise ValueError("Configured temporal ring escapes the unchanged phase grid")
            padded = full[:, y-pad:y+height+pad, x-pad:x+width+pad].copy()
            padded[:, pad:-pad, pad:-pad] = masked.log
            ring_parameters = self.native_parameters["extended_ring"]
            strips, _ = sidecar.split_extended_ring(padded, masked.log, pad_px=pad,
                maximum_center_delta_log=ring_parameters["maximum_center_delta_log"],
                parameters=ring_parameters)
            path = stage / "temporal_ring.npz"
            from .io_ops import publish_file
            def write_ring(stream):
                np.savez(stream, **strips, pad_px=np.asarray(pad, np.int16),
                         contract=np.asarray(ring_parameters["contract"]))
            publish_file(path, write_ring)
            ring = asset(path)
            keys = p["inputs"]["fields"]["temporal"]
            if len(keys) != 4:
                raise ValueError("Unsupported explicit temporal history field schema")
            row.update({keys[0]: ring["path"], keys[1]: ring["sha256"], keys[2]: pad, keys[3]: [-pad, -pad]})
        row_asset = _write_json(stage / "raw_native_row.json", row)
        record = dict(native_row=row_asset, full=native["record"]["full"],
                      source=deepcopy(spec["source"]), current_h=deepcopy(roles["current_h"]))
        phase_inputs.verify_raw_acquisition(record, row, native["audit"], parameters=p["inputs"])
        _write_json(stage / "raw_acquisition.json", record)
        return record


def prepare(configuration_path, output, *, expected_sha256, unc_only=True):
    """Execute only the bound, ordered acquisition DATA in a new output folder."""
    reader = DataReader(unc_only=unc_only)
    configuration = reader.json(dict(path=str(configuration_path), sha256=expected_sha256))
    owner = RawAcquisitionInputs(configuration, unc_only=unc_only)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [owner.prepare(spec, output / spec["case_id"]) for spec in configuration["acquisitions"]]
    _write_json(output / "raw_acquisitions.json", dict(format=FORMAT, run=configuration["run"],
        records=records, configuration=asset(configuration_path), adopted=False))
    return records
