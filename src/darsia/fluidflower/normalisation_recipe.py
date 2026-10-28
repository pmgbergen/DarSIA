"""Join saved training, fit and frame evidence without loading scientific data.

This is a recipe *audit*, not a replacement producer. Existing asset pointers
are retained so that a full rebuild can be tested against actual saved output,
rather than against a similarly named historical prototype.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


FORMAT = "saved-lighting-reproduction-plan-v1"
MAX_METADATA_BYTES = 2 * 1024 * 1024


def descriptor(path, digest):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("A bound source path is required.")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("An exact source SHA-256 is required.")
    return dict(path=str(path), sha256=digest)


def _json(bound):
    path = Path(bound["path"])
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Source metadata exceeds the read limit.")
    raw = path.read_bytes()
    if len(raw) > MAX_METADATA_BYTES or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError("Source metadata differs from its exact binding: " + str(path))
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Source metadata must be an object.")
    return result


def _one(rows, run):
    matches = [r for r in rows if isinstance(r, dict) and r.get("run") == run]
    if len(matches) != 1:
        raise ValueError("An unambiguous saved source is required for " + run)
    return matches[0]


def assemble_saved_method(training_request, run, *, frame_index=0, resolve_path=lambda path: path):
    """Follow the actual saved output lineage; never open RAW/NPZ/NPY payloads.

    ``resolve_path`` resolves historical relative paths explicitly. It must not
    cut over to a newer NAS copy. Returned scientific pointers are declarations;
    only the JSON records in ``verified_metadata`` were read and SHA verified.
    ``frame_index`` selects the saved output to reproduce while the baseline
    input remains bound to frame zero of the same saved run.
    """
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("Choose a non-negative saved frame index.")
    training = descriptor(training_request["path"], training_request["sha256"])
    request = _json(training)
    settings = request["settings"]["v5_raw_photometry"]
    if (settings["format"] != "ffac-v5-raw-full-native-training-v1"
            or settings.get("ac28_light_anchor") is not True):
        raise ValueError("This saved method is not the recorded AC28 lighting lineage.")
    batch = descriptor(settings["materialization_receipt_path"], settings["materialization_receipt_sha256"])
    receipt = _json(batch)
    if receipt.get("format") != "ffac-v5-native-materialization-batch-v1":
        raise ValueError("Unknown saved materialization lineage.")
    row = _one(receipt["revisions"], run)
    producer = receipt["processor_identities"][row["source_group"]]
    fit_manifest = descriptor(producer["calibration_manifest_path"], producer["calibration_manifest_sha256"])
    fit = _json(fit_manifest)
    if fit.get("contract") != "ffac-v4r5-interior-local-old-B-residual-calibration43-v1":
        raise ValueError("A different producer recipe must be audited explicitly.")
    fitted = _one(fit["results"], run)
    virtual = _one(fit["results"], "virtual_reference")
    model = descriptor(resolve_path(fitted["baseline_path"]), fitted["baseline_sha256"])
    sidecar = descriptor(Path(model["path"]).with_suffix(".json"), fitted["sidecar_sha256"])
    model_receipt = _json(sidecar)
    if model_receipt.get("npz_sha256") != model["sha256"] or model_receipt.get("run") != run:
        raise ValueError("The saved baseline and its sidecar disagree.")
    output_manifest = descriptor(row["materialized_manifest_path"], row["materialized_manifest_sha256"])
    output = _json(output_manifest)
    def saved_frame(index):
        matches = [r for r in output["frame_sources"] if r.get("frame_index") == index]
        if len(matches) != 1:
            raise ValueError(f"Saved frame {index} is missing or ambiguous.")
        row = matches[0]
        bound = descriptor(row["v5_frame_receipt_path"], row["v5_frame_receipt_sha256"])
        metadata = _json(bound)
        if (metadata.get("run") != run or metadata.get("frame_index") != index
                or metadata.get("processor") != producer
                or metadata["source"]["camera_arw_sha256"] != row["v5_raw_sha256"]
                or metadata["log_linear_rgb_sha256"] != row["v5_log_linear_rgb_sha256"]):
            raise ValueError(f"Saved RAW and output evidence do not identify frame {index}.")
        return row, bound, metadata

    baseline, baseline_receipt, baseline_metadata = saved_frame(0)
    if model_receipt["source"]["source_arw_sha256"] != baseline["v5_raw_sha256"]:
        raise ValueError("Saved RAW, fit and output evidence do not identify the same baseline.")
    frame, frame_receipt, frame_metadata = ((baseline, baseline_receipt, baseline_metadata)
        if frame_index == 0 else saved_frame(frame_index))
    old_fields = [r for r in fit["old_B_local_residual_fields"] if r["run"] == run]
    if len(old_fields) > 1 or (int(run[2:]) < 50 and not old_fields):
        raise ValueError("The saved old-light residual is missing or ambiguous.")
    old_field = (descriptor(resolve_path(old_fields[0]["field_path"]), old_fields[0]["field_sha256"])
                 if old_fields else None)
    if old_field and any(metadata["certificate"]["old_field_sha256"] != old_field["sha256"]
                         for metadata in (baseline_metadata, frame_metadata)):
        raise ValueError("The saved output used a different old-light field.")
    verified = [training, batch, fit_manifest, sidecar, output_manifest, baseline_receipt]
    if frame_index != 0:
        verified.append(frame_receipt)
    return dict(
        format=FORMAT, run=run, frame_index=frame_index, reference_run="ac28", numerical_status="not_run",
        producer_contract=producer["contract"], fit_contract=fit["contract"],
        saved_method_settings=deepcopy(settings), saved_eligibility=receipt.get("training_eligibility"),
        verified_metadata=verified,
        baseline_raw=descriptor(baseline["v5_raw_path"], baseline["v5_raw_sha256"]),
        frame_raw=descriptor(frame["v5_raw_path"], frame["v5_raw_sha256"]),
        expected_baseline=model,
        expected_reference=descriptor(resolve_path(virtual["baseline_path"]), virtual["baseline_sha256"]),
        expected_reference_sidecar=descriptor(Path(resolve_path(virtual["baseline_path"])).with_suffix(".json"), virtual["sidecar_sha256"]),
        reference_assets=descriptor(resolve_path(fit["assets_path"]), fit["assets_sha256"]),
        old_field=old_field,
        expected_log_rgb=descriptor(frame["v5_log_linear_rgb_path"], frame["v5_log_linear_rgb_sha256"]),
        expected_valid_mask=descriptor(frame["v5_log_linear_valid_mask_path"], frame["v5_log_linear_valid_mask_sha256"]),
        expected_output_shape=frame_metadata["log_linear_rgb_shape"],
        expected_output_dtype=frame_metadata["log_linear_rgb_dtype"],
        producer_identity=deepcopy(producer), fitter_sha256=fit["fitter_sha256"],
        fit_order=["source-photo interior masks and fixed donor cohorts",
                   "original observed-sand baseline fit and virtual reference",
                   "original donor consensus/flat iteration and convergence",
                   "final baseline fits and old-B scalar residual"],
        apply_order=["original source-bound RAW renderer and geometry",
                     "zero frame-light for exact B, otherwise original chart/light field",
                     "original native correction", "old-B scalar residual after frame-light",
                     "original source-valid-mask/materialization encoding"],
        current_protocol_gate="unverified: historical evidence does not establish current protocol identity",
        pending_gates=["complete RAW-to-grid recipe and all fixed cohort/source dependencies",
                       "exact original fitter operation-order port and parameter binding",
                       "current protocol/source pins revalidated before execution",
                       "all scientific payload SHA checks and exact fit/reference/field comparison",
                       "actual saved AC28 baseline and non-baseline output/valid-mask bit parity"],
        ready=False,
    )


def require_exact_array(actual, expected):
    """Require byte identity, including NaN payloads and signed zero; no tolerance."""
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes(order="C") != expected.tobytes(order="C")):
        raise ValueError("Saved lighting parity failed; source, fit or output needs review.")
