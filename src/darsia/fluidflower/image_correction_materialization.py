"""Batch materialization of versioned FluidFlower radiometric images."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import darsia
from darsia.fluidflower.execution import (
    AdapterDescriptor,
    TaskExecutionContext,
    load_task_plan,
    verify_file_reference,
)
from darsia.fluidflower.frame_quality import (
    assess_frame_quality_rows,
    frame_quality_signature,
)
from darsia.fluidflower.image_correction import (
    COLORCHECKER_ALGORITHMS,
    CORRECTION_REVISION_FORMAT,
    CORRECTION_TASK_CONTRACT_FORMAT,
    DARSIA_BASELINE_COLOR_ALGORITHM,
    IDENTITY_ALGORITHM,
    SUPPORTED_CORRECTION_ALGORITHMS,
    UNIFORM24_ALGORITHM,
    apply_darsia_baseline_array_correction,
    apply_uniform24_array_transform,
    fit_uniform24_array_transform,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now

CORRECTION_MATERIALIZER_VERSION = "1.2.0"
CORRECTION_RESULT_FORMAT = "darsia-fluidflower-image-correction-result-v1"
CORRECTED_FRAME_INDEX_FORMAT = "darsia-fluidflower-corrected-frame-index-v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)


def _safe_identifier(value: str, fallback: str = "item") -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value).strip().casefold()
    ).strip("-")
    return cleaned or fallback


def _write_corrected_image(
    path_without_suffix: Path,
    image: np.ndarray,
    encoding: str,
) -> Path:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Corrected images must be RGB arrays")
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    if encoding == "npy-float32":
        path = path_without_suffix.with_suffix(".npy")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
        return path
    if encoding != "png16":
        raise ValueError("Correction output encoding must be png16 or npy-float32")
    path = path_without_suffix.with_suffix(".png")
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)
    encoded_bgr = cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    if not cv2.imwrite(str(temporary), encoded_bgr):
        raise OSError(f"OpenCV could not write corrected image: {temporary}")
    os.replace(temporary, path)
    return path


def _artifact(root: Path, path: Path, product: str) -> dict[str, Any]:
    return {
        "product": product,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class ImageCorrectionAdapter:
    """Apply the saved radiometric contract to every inventoried source frame."""

    descriptor = AdapterDescriptor(
        key="image-correction-uniform24-v1",
        version=CORRECTION_MATERIALIZER_VERSION,
        title="Radiometric full-series correction",
        modules=("image_correction",),
        backends=("cpu-float64",),
        description=(
            "Applies an explicit no-target pass-through or fits one full-resolution, "
            "field-wise ColorChecker transform per frame, with resumable verified chunks."
        ),
    )

    def execute(
        self,
        task: Mapping[str, Any],
        context: TaskExecutionContext,
    ) -> Mapping[str, Any]:
        from darsia._opencv_runtime import scientific_thread, completed_frame
        scientific_thread("image-correction:setup")
        if task.get("module") != "image_correction":
            raise ValueError(
                "The correction adapter only accepts image-correction tasks"
            )
        if (task.get("backend") or {}).get("name") != "cpu-float64":
            raise ValueError("Image correction uses the CPU-float64 reference contract")
        plan, plan_path, plan_file_sha256 = load_task_plan(task)
        if plan.get("format") != CORRECTION_REVISION_FORMAT:
            raise ValueError("The task does not reference an image-correction revision")
        contract = dict(plan.get("task_contract") or {})
        if contract.get("format") != CORRECTION_TASK_CONTRACT_FORMAT:
            raise ValueError("Unsupported image-correction task contract")
        if not bool((plan.get("validation") or {}).get("execution_ready")):
            raise RuntimeError("The image-correction revision is not execution-ready")
        algorithm = str(contract.get("algorithm") or "")
        if algorithm not in SUPPORTED_CORRECTION_ALGORITHMS:
            raise ValueError(f"Unsupported radiometric algorithm: {algorithm!r}")

        project = FluidFlowerProject.load(Path(str(task["project"]["root"])))
        if project.project_id != str(task["project"].get("project_id") or ""):
            raise RuntimeError("Task and project identities do not match")
        reference_row = dict(contract.get("reference") or {})
        reference_image: np.ndarray | None = None
        if algorithm in COLORCHECKER_ALGORITHMS:
            reference_path = verify_file_reference(
                str(reference_row.get("path") or ""),
                str(reference_row.get("sha256") or ""),
                label="ColorChecker reference image",
            )
            reference_image = np.asarray(darsia.imread(reference_path).img)
        batch = dict(contract.get("batch") or {})
        frames = [
            dict(row) for row in batch.get("frames") or [] if isinstance(row, Mapping)
        ]
        encoding = str(batch.get("output_encoding") or "png16")
        workers = min(
            24,
            len(frames),
            max(1, int((task.get("resources") or {}).get("cpu_workers") or 1)),
        )
        context.checkpoint(
            {
                "stage": "inputs_verified",
                "frame_count": len(frames),
                "worker_count": workers,
                "result_destination": str(
                    (task.get("output") or {}).get("destination")
                ),
            }
        )

        rows_by_frame: dict[str, dict[str, Any]] = {}
        missing: list[dict[str, Any]] = []
        for frame in frames:
            frame_id = str(frame["frame_id"])
            chunk_id = f"correction-frame-{frame_id}"
            committed = context.committed_chunk(chunk_id)
            frame_result = (committed.get("state") or {}).get("frame_result")
            if committed and isinstance(frame_result, Mapping):
                if not (
                    frame_result.get("raw_signature")
                    and frame_result.get("corrected_signature")
                ):
                    raise RuntimeError(
                        "This paused image-correction task predates frame-quality "
                        "screening. Create a new task so every frame is processed "
                        "under one immutable correction and quality contract."
                    )
                context.restore_chunk(chunk_id, context.staging)
                rows_by_frame[frame_id] = dict(frame_result)
            else:
                missing.append(frame)

        def correct_frame(frame: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
            ipp_before = scientific_thread("image-correction:frame-before")
            frame_id = str(frame["frame_id"])
            run_id = str(frame.get("run_id") or "run")
            source = verify_file_reference(
                str(frame.get("source_path") or ""),
                str(frame.get("sha256") or ""),
                label=f"Correction source frame {frame_id}",
            )
            target_image = np.asarray(darsia.imread(source).img)
            if algorithm == UNIFORM24_ALGORITHM:
                if reference_image is None:
                    raise RuntimeError("Uniform24 reference image was not loaded")
                fitted = fit_uniform24_array_transform(
                    reference_image,
                    target_image,
                    corner=str(contract.get("colorchecker_corner") or "upper_right"),
                    roi_rc=contract.get("colorchecker_roi_rc"),
                    settings=dict(contract.get("settings") or {}),
                )
                corrected = apply_uniform24_array_transform(
                    target_image,
                    fitted["transform"],
                )
            elif algorithm == DARSIA_BASELINE_COLOR_ALGORITHM:
                if reference_image is None:
                    raise RuntimeError("DarSIA ColorCorrection reference was not loaded")
                fitted = apply_darsia_baseline_array_correction(
                    reference_image,
                    target_image,
                    corner=str(contract.get("colorchecker_corner") or "upper_right"),
                    roi_rc=contract.get("colorchecker_roi_rc"),
                    settings=dict(contract.get("settings") or {}),
                )
                corrected = np.asarray(fitted["corrected"], dtype=np.float32)
            else:
                corrected = np.asarray(target_image, dtype=np.float32)
                fitted = {
                    "transform": {"algorithm": IDENTITY_ALGORITHM},
                    "diagnostics": {
                        "algorithm": IDENTITY_ALGORITHM,
                        "flagged": False,
                        "flag_reasons": [],
                        "note": "No external radiometric color target was applied.",
                    },
                }
            chunk_id = f"correction-frame-{frame_id}"
            chunk_staging = context.prepare_chunk(chunk_id)
            frame_root = (
                chunk_staging / "runs" / _safe_identifier(run_id, "run") / "frames"
            )
            corrected_path = _write_corrected_image(
                frame_root / frame_id,
                corrected,
                encoding,
            )
            transform_path = frame_root / f"{frame_id}.radiometric.json"
            _atomic_json(
                transform_path,
                {
                    "schema": 1,
                    "algorithm": fitted["transform"]["algorithm"],
                    "frame_id": frame_id,
                    "run_id": run_id,
                    "source_path": str(source),
                    "source_sha256": str(frame.get("sha256") or ""),
                    "transform": fitted["transform"],
                    "diagnostics": fitted["diagnostics"],
                },
            )
            frame_result = {
                "frame_id": frame_id,
                "run_id": run_id,
                "source_path": str(source),
                "source_relative_path": str(frame.get("relative_path") or ""),
                "source_sha256": str(frame.get("sha256") or ""),
                "corrected_path": corrected_path.relative_to(chunk_staging).as_posix(),
                "corrected_sha256": sha256_file(corrected_path),
                "transform_path": transform_path.relative_to(chunk_staging).as_posix(),
                "transform_sha256": sha256_file(transform_path),
                "source_shape": list(target_image.shape),
                "output_dtype": "uint16" if encoding == "png16" else "float32",
                "output_encoding": encoding,
                "qc": deepcopy(dict(fitted["diagnostics"])),
                "raw_signature": frame_quality_signature(target_image),
                "corrected_signature": frame_quality_signature(corrected),
                "correction_qc": deepcopy(dict(fitted["diagnostics"])),
                "correction_dependency_sha256": str(
                    plan.get("dependency_sha256")
                    or contract.get("content_sha256")
                    or ""
                ),
                "analysis_error": "",
                "opencv_ipp": completed_frame(ipp_before, "image-correction:frame-after"),
            }
            context.commit_chunk(
                chunk_id,
                chunk_staging,
                state={"frame_result": frame_result},
            )
            return chunk_id, frame_result

        for offset in range(0, len(missing), workers):
            batch_rows = missing[offset : offset + workers]
            with ThreadPoolExecutor(max_workers=min(workers, len(batch_rows))) as pool:
                futures = {
                    pool.submit(correct_frame, frame): str(frame["frame_id"])
                    for frame in batch_rows
                }
                for future in as_completed(futures):
                    frame_id = futures[future]
                    chunk_id, frame_result = future.result()
                    context.restore_chunk(chunk_id, context.staging)
                    rows_by_frame[frame_id] = frame_result
            context.checkpoint(
                {
                    "stage": "frame_batch_complete",
                    "completed_frames": len(rows_by_frame),
                    "frame_count": len(frames),
                }
            )
            context.raise_if_interrupted()

        ordered_rows = [rows_by_frame[str(frame["frame_id"])] for frame in frames]
        quality_assessment = assess_frame_quality_rows(ordered_rows)
        quality_assessment.update(
            {
                "generated_utc": utc_now(),
                "correction_revision_id": str(plan.get("revision_id") or ""),
                "correction_dependency_sha256": str(
                    plan.get("dependency_sha256")
                    or contract.get("content_sha256")
                    or ""
                ),
                "source": "image_correction_materialization",
            }
        )
        quality_path = context.staging / "frame_quality.json"
        _atomic_json(quality_path, quality_assessment)
        quality_by_frame = {
            str(row.get("frame_id") or ""): row
            for row in quality_assessment.get("frames") or []
        }
        indexed_rows: list[dict[str, Any]] = []
        for raw_row in ordered_rows:
            row = deepcopy(raw_row)
            quality = quality_by_frame.get(str(row.get("frame_id") or ""), {})
            row["frame_quality"] = {
                "automatic_status": str(quality.get("automatic_status") or ""),
                "automatic_flags": list(quality.get("automatic_flags") or []),
                "blacklist_candidate": bool(quality.get("blacklist_candidate")),
                "automatically_excluded": False,
                "raw_metrics": deepcopy(dict(quality.get("raw_metrics") or {})),
                "corrected_metrics": deepcopy(
                    dict(quality.get("corrected_metrics") or {})
                ),
            }
            row.pop("raw_signature", None)
            row.pop("corrected_signature", None)
            row.pop("correction_qc", None)
            row.pop("correction_dependency_sha256", None)
            row.pop("analysis_error", None)
            indexed_rows.append(row)
        index_path = context.staging / "corrected_frames.json"
        index = {
            "schema": 1,
            "format": CORRECTED_FRAME_INDEX_FORMAT,
            "task_id": str(task["task_id"]),
            "correction_revision_id": str(plan.get("revision_id") or ""),
            "algorithm": str(contract.get("algorithm") or ""),
            "reference": reference_row,
            "frame_count": len(indexed_rows),
            "flagged_frame_count": sum(
                bool((row.get("qc") or {}).get("flagged")) for row in indexed_rows
            ),
            "quality_assessment": {
                "path": quality_path.name,
                "sha256": sha256_file(quality_path),
                "advisory_only": True,
                "automatic_exclusion": False,
            },
            "frames": indexed_rows,
        }
        _atomic_json(index_path, index)
        provenance_path = context.staging / "provenance.json"
        _atomic_json(
            provenance_path,
            {
                "schema": 1,
                "generated_utc": utc_now(),
                "task_id": str(task["task_id"]),
                "task_sha256": str(task.get("content_sha256") or ""),
                "correction_revision_id": str(plan.get("revision_id") or ""),
                "correction_revision_sha256": str(plan.get("content_sha256") or ""),
                "correction_dependency_sha256": str(
                    plan.get("dependency_sha256")
                    or contract.get("content_sha256")
                    or ""
                ),
                "plan_path": str(plan_path),
                "plan_file_sha256": plan_file_sha256,
                "adapter": {
                    "key": self.descriptor.key,
                    "version": self.descriptor.version,
                },
                "backend": deepcopy(dict(task.get("backend") or {})),
                "hardware": deepcopy(dict(context.inventory)),
                "workers": workers,
                "scientific_contract": {
                    "resolution": "native_source_resolution",
                    "geometric_registration": "must_precede_this_adapter",
                    "color_reference": (
                        (
                            "all_24_colorchecker_fields_uniform24"
                            if algorithm == UNIFORM24_ALGORITHM
                            else "all_24_colorchecker_fields_original_darsia"
                        )
                        if algorithm in COLORCHECKER_ALGORITHMS
                        else "none_explicit_identity"
                    ),
                    "tone_reference": (
                        "six_neutral_colorchecker_fields"
                        if algorithm == UNIFORM24_ALGORITHM
                        else (
                            "original_darsia_white_and_affine_balance"
                            if algorithm == DARSIA_BASELINE_COLOR_ALGORITHM
                            else "none"
                        )
                    ),
                    "indicator_buffer_reference_used_for_radiometry": False,
                    "flagged_frame_policy": "preserve_and_require_review",
                    "silent_neighbor_replacement": False,
                },
            },
        )
        result_path = context.staging / "image_correction_result.json"
        result = {
            "schema": 1,
            "format": CORRECTION_RESULT_FORMAT,
            "complete": True,
            "task_id": str(task["task_id"]),
            "correction_revision_id": str(plan.get("revision_id") or ""),
            "correction_revision_sha256": str(plan.get("content_sha256") or ""),
            "correction_dependency_sha256": str(
                plan.get("dependency_sha256") or contract.get("content_sha256") or ""
            ),
            "correction_algorithm": str(contract.get("algorithm") or ""),
            "frame_count": len(indexed_rows),
            "flagged_frame_count": index["flagged_frame_count"],
            "artifacts": [
                _artifact(context.staging, index_path, "corrected_frame_index"),
                _artifact(context.staging, quality_path, "frame_quality_assessment"),
                _artifact(context.staging, provenance_path, "provenance_json"),
            ],
        }
        _atomic_json(result_path, result)
        _atomic_json(context.staging / "task.json", task)
        _atomic_json(context.staging / "plan.json", plan)
        context.checkpoint(
            {
                "stage": "image_correction_ready",
                "frame_count": len(indexed_rows),
                "flagged_frame_count": index["flagged_frame_count"],
            }
        )
        return {
            "result_manifest": result_path.name,
            "frame_count": len(indexed_rows),
            "flagged_frame_count": index["flagged_frame_count"],
            "scientific_results_created": True,
        }


__all__ = [
    "CORRECTED_FRAME_INDEX_FORMAT",
    "CORRECTION_MATERIALIZER_VERSION",
    "CORRECTION_RESULT_FORMAT",
    "ImageCorrectionAdapter",
]
