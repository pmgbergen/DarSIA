"""Render saved phase PNG/CSV diagnostics without phase inference or fitting.

The native manifest is a historical review manifest or a RAW photographic
review (``raw_photographic_review``); a RAW run takes its timeline from the
review's frames and its video photographs from the review's natural PNGs.
"""
from __future__ import annotations
import csv
import json
import os
import time
import uuid
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .project import sha256_file
from .review_storage import resolve_review_path
from .phase_saved_review_metadata import configured_parameters
from .raw_photographic_review import timeline_rows

def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .project import _atomic_json as _write_metadata
    return _write_metadata(path, payload, copy_payload=True)

@contextmanager
def _worker_lock(path):
    # Advisory OS lock automatically releases if a worker exits unexpectedly.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + 3600
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Another exact reference worker is still running")
                    time.sleep(.5)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

def _area_domain(model_root, job, native):
    from .phase_runtime import load_physics_profile, load_run_context
    _, _, width, height = native["content_box_xywh"]
    profile = load_physics_profile(model_root)
    if profile is not None:
        entry = profile["runs"][f"{job['dataset']}/{job['run']}"]
        context, _ = load_run_context(model_root, profile, job["dataset"], job["run"],
            native_manifest_sha256=entry["native_manifest_sha256"])
        domain = np.asarray(context.domain, dtype=bool)
        source = str(model_root / entry["path"])
    elif (native.get("domain") or {}).get("path"):
        source = Path(native["domain"]["path"])
        if not source.is_absolute():
            # Radiometry display assets may live elsewhere; domain paths belong
            # to the target's original native-review root.
            source = Path(job.get("analysis_source_review_root") or job["review_output_dir"]) / source
        source = resolve_review_path(source)
        with np.load(source, allow_pickle=False) as archive:
            domain = np.asarray(archive["domain"], dtype=bool)
        source = str(source.resolve())
    else:
        return np.ones((height, width), dtype=bool), dict(kind="image-crop", pixels=width * height,
            axis_label="Visible image crop area [%]", source=None)
    if domain.shape != (height, width) or not domain.any():
        raise ValueError("Transfer area domain must be nonempty and match the target crop")
    return domain, dict(kind="porous-domain", pixels=int(domain.sum()),
        axis_label="Visible porous area [%]", source=source)

def _plot_area(path, times, counts, area, run):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(14, 5))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    if area.get("kind") in {"frame-native-porous-domain",
                              "frame-native-observation-domain"}:
        denominator = np.asarray(area.get("per_frame_pixels"), dtype=np.float64)
        if (denominator.shape != (len(counts),) or not np.isfinite(denominator).all()
                or np.any(denominator <= 0)):
            raise ValueError("Per-frame phase area denominator is incomplete")
    else:
        denominator = int(area["pixels"])
        if denominator <= 0:
            raise ValueError("Porous area denominator must be positive")
    values = [np.asarray([row[key] for row in counts]) * 100 / denominator
              for key in ("aqueous", "gas")]
    ax.stackplot(times, *values, colors=["#31d9a0", "#f24c9d"], labels=["Aqueous", "Free gas"])
    ax.set(xlabel="Time since first injection [h]", ylabel=area["axis_label"],
           title=f"{run.upper()} predicted 2D phase area - not concentration or mass")
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(path)

def validate_deferred_presentation(summary, *, run, candidate_id, params_sha256,
                                   review_output_dir, phase_map_output_dir, per_frame_csv):
    """Accept absent display media only for an explicit, identity-bound native package."""
    descriptor = summary.get("deferred_presentation") or {}
    if summary.get("presentation_status") != "deferred" or descriptor.get("format") != configured_parameters("presentation")["deferred_format"]:
        raise ValueError("Missing valid deferred presentation metadata")
    for key, expected in (("run", run), ("candidate_id", candidate_id), ("params_sha256", params_sha256)):
        if descriptor.get(key) != expected:
            raise ValueError(f"Deferred presentation identifies another {key}")
    native_path = resolve_review_path(review_output_dir) / "native_frames" / run / "manifest.json"
    if (resolve_review_path(descriptor["native_manifest"]) != native_path.resolve()
            or sha256_file(native_path) != descriptor["native_manifest_sha256"]
            or resolve_review_path(descriptor["phase_map_output_dir"]) != resolve_review_path(phase_map_output_dir)):
        raise ValueError("Deferred presentation native identity changed")
    native = _read(native_path)
    count = int(descriptor["frame_count"])
    with Path(per_frame_csv).open(encoding="utf-8-sig", newline="") as stream:
        times = [float(row["elapsed_h"]) for row in csv.DictReader(stream)]
    expected = [float(row["elapsed_h"]) for row in timeline_rows(native, run)]
    if count < 1 or len(expected) != count or int(native["frame_count"]) != count or times != expected:
        raise ValueError("Deferred presentation has another or incomplete timeline")
    if not np.isfinite(times).all():
        raise ValueError("Deferred presentation has a nonfinite timeline")
    if int((summary.get("calibration_review_movie") or {}).get("calibration_frame_count", 0)) != count:
        raise ValueError("Deferred presentation frame count changed")
    return descriptor, native, times

def ensure_candidate_presentation(candidate, kind="plot", progress=None):
    """Build one explicitly requested display asset, leaving scientific data untouched."""
    if kind not in {"plot", "video"}:
        raise ValueError("Presentation kind must be plot or video")
    target = Path(candidate.mass_plot if kind == "plot" else candidate.contour_movie)
    summary_path = Path(candidate.summary_path)
    summary = _read(summary_path)
    if not summary.get("deferred_presentation"):
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    descriptor, native, times = validate_deferred_presentation(summary, run=candidate.run,
        candidate_id=candidate.candidate_id, params_sha256=candidate.params_sha256,
        review_output_dir=candidate.review_output_dir, phase_map_output_dir=candidate.phase_map_output_dir,
        per_frame_csv=candidate.per_frame_csv)
    identity = sha256_file(summary_path)
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    with _worker_lock(target.with_suffix(target.suffix + ".lock")):
        if receipt_path.is_file():
            receipt = _read(receipt_path)
            if (receipt.get("summary_sha256") != identity or not target.is_file()
                    or receipt.get("sha256") != sha256_file(target)):
                raise ValueError("Generated presentation asset changed")
            return target
        temporary = target.with_name(target.stem + ".building-" + uuid.uuid4().hex + target.suffix)
        try:
            if progress:
                progress(f"Preparing {candidate.run.upper()} {'area diagnostics' if kind == 'plot' else 'video'}")
            job = {"run": candidate.run, "dataset": candidate.dataset,
                   "review_output_dir": str(candidate.review_output_dir)}
            if kind == "plot":
                counts, area = descriptor.get("frame_counts"), descriptor.get("area_denominator")
                domain = None
                if not area or not counts:
                    domain, resolved_area = _area_domain(Path(descriptor["model_root"]), job, native)
                    area = area or resolved_area
                if not counts:
                    counts = []
                    x, y, width, height = native["content_box_xywh"]
                    for index in range(len(times)):
                        if progress:
                            progress(f"Reading phase areas {index+1}/{len(times)}")
                        with Image.open(_presentation_layer(candidate, index)) as image:
                            if image.size != tuple(native["canvas_wh"]):
                                raise ValueError("Phase-map geometry changed")
                            rgba = np.asarray(image.convert("RGBA"))[y:y+height, x:x+width]
                        visible = (rgba[..., 3] > 0) & domain
                        counts.append({phase: int((visible & np.all(rgba[..., :3] == color[:3], axis=-1)).sum())
                                       for phase, color in configured_parameters("presentation")["phase_fill_colors"].items()})
                _plot_area(temporary, times, counts, area, candidate.run)
            else:
                movie = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 2., (1280, 720))
                try:
                    if not movie.isOpened():
                        raise RuntimeError("Cannot create prediction video")
                    for index in range(len(times)):
                        if progress:
                            progress(f"Encoding video {index+1}/{len(times)}")
                        photo_path = _presentation_photo(candidate, native, index)
                        with Image.open(_presentation_layer(candidate, index)) as layer, Image.open(photo_path) as photo:
                            if layer.size != tuple(native["canvas_wh"]) or photo.size != layer.size:
                                raise ValueError("Prediction layer and photo have different native geometry")
                            image = Image.alpha_composite(photo.convert("RGBA"), layer.convert("RGBA")).convert("RGB")
                            image.thumbnail((1280, 720))
                            canvas = Image.new("RGB", (1280, 720))
                            canvas.paste(image, ((1280-image.width)//2, (720-image.height)//2))
                            movie.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
                finally:
                    movie.release()
            if not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError("Presentation produced no output")
            if progress:
                progress("Publishing requested presentation")
            if sha256_file(summary_path) != identity:
                raise ValueError("Candidate changed while preparing presentation")
            temporary.replace(target)
            _atomic_json(receipt_path, {"summary_sha256": identity, "sha256": sha256_file(target)})
        finally:
            temporary.unlink(missing_ok=True)
    return target

def _presentation_photo(candidate, native, index):
    """The photograph under one phase layer: the RAW review's natural PNG or the historical frame."""
    from .raw_photographic_review import RAW_REVIEW_FORMAT, resolve_raw_photograph
    if native.get("format") == RAW_REVIEW_FORMAT:
        return resolve_raw_photograph(candidate.review_output_dir, candidate.run, index, mode="natural")
    photo_path = Path(candidate.review_output_dir) / "native_frames" / candidate.run / f"frame_{index:03d}.png"
    if not photo_path.is_file():
        photo_path = photo_path.with_suffix(".jpg")
    return photo_path

def _presentation_layer(candidate, index):
    return Path(candidate.phase_map_output_dir) / "native_phase_maps" / candidate.run / (
        "A_" + candidate.params_sha256[:16]) / f"frame_{index:03d}.png"
