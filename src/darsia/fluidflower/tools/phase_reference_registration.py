"""Measured baseline-to-frame translation for the aligned pre-injection reference.

A target prepared by ``prepare_ffum_phase_target`` publishes corrected review frames but
never measures how far the rig moved between the pre-injection baseline and each later
photograph. The native manifest therefore carries no ``reference_registration``, and
``calibration_pairwise_review.pair_payload`` keeps the aligned pre-injection reference
switched off, because it cannot place the photograph over the selected frame. The
campaign review roots these evaluations were bound to before carried the measurement,
which is why the switch worked there and stopped working once each evaluation moved
onto its own prepared target.

The measurement is display-only. It positions an existing photograph on screen and never
reaches a phase map, a training label, a timeline or a model parameter. It is kept beside
the reference photograph rather than inside the hashed native manifest -- exactly where
the acquisition inventory lives -- so every existing evaluation stays verifiable against
the inputs it was produced from. The record names the manifest digest it was measured
against, and a record that names another digest is ignored rather than painted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

FORMAT = "native-reference-registration-v1"
SIDECAR = "reference_registration.json"
SOURCE = "published_native_review_frames"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def manifest_path(review_root: Path | str, run: str) -> Path:
    return Path(review_root) / "native_frames" / str(run).casefold() / "manifest.json"


def sidecar_path(review_root: Path | str, run: str) -> Path:
    return Path(review_root) / "native_references" / str(run).casefold() / SIDECAR


def frame_path(review_root: Path | str, run: str, index: int) -> Path:
    return Path(review_root) / "native_frames" / str(run).casefold() / f"frame_{index:03d}.jpg"


def read(review_root: Path | str, run: str, native_manifest: Path | str | None = None) -> list[dict]:
    """Return the measured rows, or nothing at all when they do not match this manifest."""
    path = sidecar_path(review_root, run)
    try:
        record = _read(path)
    except (OSError, ValueError):
        return []
    if record.get("format") != FORMAT:
        return []
    try:
        digest = _sha(Path(native_manifest or manifest_path(review_root, run)))
    except OSError:
        return []
    if record.get("native_manifest_sha256") != digest:
        return []
    rows = record.get("reference_registration")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return []
    return [dict(row) for row in rows]


def measure(review_root: Path | str, run: str, *,
            progress: Callable[[str], None] | None = None) -> dict:
    """Measure every published frame against the published baseline photograph.

    The same estimator and the same canvas fractions as ``prepare_phase_transfer_target``,
    read from the frames the target already published instead of from the raw source.
    """
    import cv2

    from darsia.fluidflower.tools import calibration_pairwise_review as review

    root, run = Path(review_root), str(run).casefold()
    path = manifest_path(root, run)
    manifest, digest = _read(path), _sha(path)
    if str(manifest.get("run", run)).casefold() != run:
        raise ValueError("Native manifest names another experiment")
    frames = manifest.get("frame_sources") or []
    canvas = list(manifest.get("canvas_wh") or [])
    box = list(manifest.get("content_box_xywh") or []) or None
    if len(canvas) != 2 or not frames:
        raise ValueError("Native manifest carries no review canvas or published frames")
    width, height = int(canvas[0]), int(canvas[1])
    baseline = cv2.imread(str(frame_path(root, run, 0)), cv2.IMREAD_COLOR)
    if baseline is None:
        raise ValueError(f"Published baseline frame is unreadable: {frame_path(root, run, 0)}")
    rows = [dict(frame_index=0, dx_px=0.0, dy_px=0.0, response=1.0,
                 region="identical_baseline", source=SOURCE,
                 dx_canvas_fraction=0.0, dy_canvas_fraction=0.0)]
    for index in range(1, len(frames)):
        target = frame_path(root, run, index)
        image = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Published review frame is unreadable: {target}")
        if image.shape != baseline.shape:
            raise ValueError(f"Published review frames differ in size: {target}")
        dx, dy, response, region = review._estimate_review_translation(
            baseline, image, header_height=0, content_box=box)
        rows.append(dict(frame_index=index, dx_px=float(dx), dy_px=float(dy),
                         response=float(response), region=str(region), source=SOURCE,
                         dx_canvas_fraction=float(dx) / width,
                         dy_canvas_fraction=float(dy) / height))
        if progress:
            progress(f"{run.upper()}: measured reference alignment {index}/{len(frames) - 1}")
    if _sha(path) != digest:
        raise ValueError("Native manifest changed while measuring; reopen and retry")
    return dict(format=FORMAT, dataset=str(manifest.get("dataset") or ""), run=run,
                native_manifest=str(path), native_manifest_sha256=digest,
                canvas_wh=[width, height], content_box_xywh=box,
                frame_count=len(frames), display_only=True,
                phase_maps_touched=False, timeline_touched=False, model_touched=False,
                measured_from="published native review frames of this prepared target",
                estimator="calibration_pairwise_review._estimate_review_translation",
                created_utc=datetime.now(timezone.utc).isoformat(),
                reference_registration=rows)


def ensure(review_root: Path | str, run: str, *,
           progress: Callable[[str], None] | None = None) -> list[dict]:
    """Give this target a usable alignment measurement, measuring it only if needed."""
    root, run = Path(review_root), str(run).casefold()
    path = manifest_path(root, run)
    manifest = _read(path)
    published = manifest.get("reference_registration") or []
    expected = len(manifest.get("frame_sources") or [])
    if len(published) == expected and expected:
        return [dict(row) for row in published]
    existing = read(root, run, path)
    if len(existing) == expected and expected:
        return existing
    record = measure(root, run, progress=progress)
    _write(sidecar_path(root, run), record)
    return [dict(row) for row in record["reference_registration"]]
