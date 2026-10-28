"""Verified, display-only FF_AC free-gas physics overlays.

The exported mask is independent evidence.  Loading it never changes saved
phase maps, model probabilities, source masks, or annotations.  A missing or
changed export fails closed instead of silently displaying the optical map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile

import numpy as np

from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA


CONTRACT = "ffac-gas-physics-v6-studio-layer-export-v1"
PHYSICS_VERSION = "ffac-gas-physics-v6"
TARGET_CANDIDATE = "phase-model-20260930-track1-natural-lifts-invalidfilled-ffac"
# The export folder is configured (Studio's 'physics_layer_exports' tool path).
UNRESOLVED_RGB = (30, 210, 255)
ADDED_RGBA = (40, 110, 255, 255)
REJECTED_RGBA = (110, 40, 160, 179)  # Dark violet at 70%; never a white gas cue.
DIFFERENCE_LEGEND = (
    "Difference: pink at 60% = gas in both maps; blue = physics-added gas; "
    "dark violet at 70% = optical gas rejected by physics. "
    "Aqueous CO2 is hidden. Review rejected injection-path gas manually."
)
COMMON_GAS_RGBA = (*GAS_RGBA[:3], 153)  # 60% opacity


class PhysicsLayerUnavailable(ValueError):
    """A precise reason that Studio can show beside a disabled layer choice."""


@dataclass(frozen=True)
class VerifiedPhysicsFrame:
    """Native-grid mask plus the verified placement needed for display."""

    mask: np.ndarray
    photo_sand: np.ndarray
    content_box_xywh: tuple[int, int, int, int]
    canvas_wh: tuple[int, int]
    known_limitations: tuple[str, ...]
    run: str
    frame_index: int
    physics_version: str = PHYSICS_VERSION


def _sha256(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise PhysicsLayerUnavailable(f"{label} cannot be read: {exc}") from exc
    if not isinstance(value, dict):
        raise PhysicsLayerUnavailable(f"{label} is not a JSON object")
    return value


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PhysicsLayerUnavailable(reason)


def _hex_digest(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdefABCDEF" for c in value),
             f"{label} has no valid SHA-256")
    return value.lower()


def _verified(path: Path, digest: object, label: str) -> Path:
    expected = _hex_digest(digest, label)
    try:
        actual = _sha256(path)
    except OSError as exc:
        raise PhysicsLayerUnavailable(f"{label} is missing or unreadable: {exc}") from exc
    _require(actual == expected, f"{label} SHA-256 differs from its manifest")
    return path


def _manifest_child(root: Path, relative: object, expected: Path, label: str) -> Path:
    """Accept only the declared fixed export location, not an arbitrary path."""
    _require(isinstance(relative, str) and relative.replace("\\", "/") == relative,
             f"{label} has an invalid relative path")
    parts = relative.split("/")
    _require(all(part not in ("", ".", "..") and ":" not in part for part in parts),
             f"{label} has an unsafe relative path")
    actual = root.joinpath(*parts)
    _require(actual.resolve() == expected.resolve(),
             f"{label} points outside its expected run/frame location")
    return actual


def _source_and_candidate_binding(
    global_manifest: dict, run_manifest: dict, model_root: Path,
    review_root: Path, run: str,
) -> tuple[dict, tuple[int, int], tuple[int, int, int, int], tuple[int, int]]:
    """Bind the independent export to this exact v2 candidate and source run."""
    source_candidate = str(global_manifest.get("source_candidate") or "").replace("\\", "/")
    expected_candidate_tail = f"/{model_root.parent.name}/{model_root.name}"
    _require(source_candidate.endswith(expected_candidate_tail)
             and source_candidate.startswith("FF_AC/DarSIA/results/"),
             "Physics export belongs to another phase-model candidate")
    _require(str(run_manifest.get("candidate") or "") == source_candidate.removeprefix("FF_AC/DarSIA/results/"),
             "Physics run manifest belongs to another candidate")
    source_batch = str(global_manifest.get("source_prepared_review") or "").replace("\\", "/")
    _require(review_root.name == "manual_pairwise_review"
             and review_root.parent.parent.name == "revisions"
             and review_root.parent.parent.parent.name == source_batch.rsplit("/", 1)[-1]
             and source_batch.startswith("FF_AC/DarSIA/data/prepared_review/"),
             "Physics export belongs to another prepared-review batch")
    _require(str(run_manifest.get("run") or "").casefold() == run,
             "Physics run manifest identifies another experiment")
    _require(str(run_manifest.get("revision") or "") == review_root.parent.name,
             "Physics run revision differs from the selected source revision")

    binding_path = model_root / "inference_run_bindings" / f"{run}.json"
    binding = _read_json(binding_path, "Candidate run binding")
    identity = binding.get("model_identity") or []
    _require(binding.get("run") == run and isinstance(identity, list)
             and identity and identity[0] == model_root.name,
             "Candidate run binding does not identify this model and experiment")
    _require(binding.get("source_manifest_sha256") == run_manifest.get("revision"),
             "Physics run revision differs from the candidate's source revision")
    native_path = review_root / "native_frames" / run / "manifest.json"
    _require(Path(str(binding.get("materialized_manifest_path") or "")).resolve()
             == native_path.resolve(),
             "Candidate run binding points to another native source manifest")
    _verified(native_path, binding.get("materialized_manifest_sha256"),
              "Candidate native source manifest")
    native = _read_json(native_path, "Candidate native source manifest")
    _require(str(native.get("run") or "").casefold() == run,
             "Native source manifest identifies another experiment")
    try:
        shape = tuple(map(int, native["analysis_image_shape_hw"]))
        box = tuple(map(int, native["content_box_xywh"]))
        canvas = tuple(map(int, native["canvas_wh"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicsLayerUnavailable("Native source geometry is incomplete") from exc
    _require(len(shape) == 2 and len(box) == 4 and len(canvas) == 2
             and shape[0] > 0 and shape[1] > 0,
             "Native source geometry is invalid")
    x, y, width, height = box
    _require((height, width) == shape and x >= 0 and y >= 0
             and x + width <= canvas[0] and y + height <= canvas[1],
             "Native analysis grid does not fit its recorded canvas")
    return native, shape, box, canvas


def load_verified_physics_frame(
    export_root: str | Path,
    model_root: str | Path,
    review_root: str | Path,
    run: str,
    frame_index: int,
) -> VerifiedPhysicsFrame:
    """Verify global/run/frame SHA and native geometry, then unpack gas.

    This is intended for an image-loading worker, not the GUI thread.  The
    returned mask is native-grid boolean and is not a substitute for a saved
    model prediction or any mass-analysis input.
    """
    export_root = Path(export_root).resolve()
    model_root = Path(model_root).resolve()
    review_root = Path(review_root).resolve()
    _require(model_root.name == TARGET_CANDIDATE,
             "Gas physics layer v6 is bound only to its Track1 display-modes v2 candidate")
    run = str(run).casefold()
    _require(run.isascii() and run.startswith("ac") and run[2:].isdigit(),
             "Physics layer requires an FF_AC experiment identifier")
    _require(isinstance(frame_index, int) and not isinstance(frame_index, bool)
             and frame_index >= 0, "Physics frame index is invalid")

    global_manifest = _read_json(export_root / "manifest.json", "Physics global manifest")
    _require(global_manifest.get("contract") == CONTRACT
             and global_manifest.get("physics") == PHYSICS_VERSION,
             "Physics export contract or version is unsupported")
    run_entry = (global_manifest.get("runs") or {}).get(run)
    _require(isinstance(run_entry, dict), f"Physics export has no {run.upper()} run")
    run_path = _manifest_child(export_root, run_entry.get("manifest"),
                               export_root / run / "manifest.json", "Physics run manifest")
    _verified(run_path, run_entry.get("sha256"), "Physics run manifest")
    run_manifest = _read_json(run_path, "Physics run manifest")
    _require(run_manifest.get("physics") == PHYSICS_VERSION,
             "Physics run version differs from the global manifest")
    native, shape, box, canvas = _source_and_candidate_binding(
        global_manifest, run_manifest, model_root, review_root, run)

    rows = run_manifest.get("frames")
    source_rows = native.get("frame_sources")
    _require(isinstance(rows, list) and isinstance(source_rows, list)
             and len(rows) == len(source_rows) == run_entry.get("frames"),
             "Physics and candidate frame timelines disagree")
    _require(frame_index < len(rows),
             f"Physics layer has no frame {frame_index} in {run.upper()}")
    row, source_row = rows[frame_index], source_rows[frame_index]
    _require(isinstance(row, dict) and isinstance(source_row, dict)
             and row.get("frame_index") == source_row.get("frame_index") == frame_index,
             "Physics frame index is absent or differs from the candidate")
    try:
        elapsed = float(row["elapsed_h"])
        source_elapsed = float(source_row["elapsed_h"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicsLayerUnavailable("Physics or candidate frame time is missing") from exc
    _require(math.isfinite(elapsed) and math.isfinite(source_elapsed)
             and math.isclose(elapsed, source_elapsed, abs_tol=1e-6),
             "Physics frame time differs from the candidate timeline")

    frame_path = _manifest_child(export_root, f"{run}/{row.get('file')}",
                                 export_root / run / f"frame_{frame_index:03d}.physics_gas_mask.npz",
                                 "Physics frame archive")
    _verified(frame_path, row.get("sha256"), "Physics frame archive")
    try:
        with np.load(frame_path, allow_pickle=False) as archive:
            _require(set(archive.files) == {"gas", "shape", "frame_index", "elapsed_h"},
                     "Physics frame archive has unexpected fields")
            packed = archive["gas"]
            stored_shape = archive["shape"]
            stored_index = archive["frame_index"]
            stored_elapsed = archive["elapsed_h"]
    except (OSError, BadZipFile, ValueError, KeyError) as exc:
        raise PhysicsLayerUnavailable(f"Physics frame archive cannot be decoded: {exc}") from exc
    _require(stored_shape.shape == (2,) and np.issubdtype(stored_shape.dtype, np.integer)
             and tuple(map(int, stored_shape)) == shape,
             "Physics frame shape differs from analysis_image_shape_hw")
    _require(stored_index.shape == () and np.issubdtype(stored_index.dtype, np.integer)
             and int(stored_index) == frame_index,
             "Physics archive frame index differs from the candidate")
    _require(stored_elapsed.shape == () and np.issubdtype(stored_elapsed.dtype, np.number)
             and math.isfinite(float(stored_elapsed))
             and math.isclose(float(stored_elapsed), elapsed, abs_tol=1e-6),
             "Physics archive frame time differs from the run manifest")
    _require(packed.dtype == np.uint8 and packed.shape == (shape[0], (shape[1] + 7) // 8),
             "Physics packed mask has an invalid native grid")
    gas = np.unpackbits(packed, axis=-1, count=shape[1]).astype(bool, copy=False)
    _require(int(np.count_nonzero(gas)) == row.get("gas_px"),
             "Physics gas pixel count differs from the run manifest")

    sand_path = review_root / "native_frames" / run / f"frame_{frame_index:03d}.photo_sand_mask.npy"
    _require(Path(str(source_row.get("v5_photo_sand_mask_path") or "")).resolve()
             == sand_path.resolve(), "Candidate photo-sand mask path differs from its manifest")
    _verified(sand_path, source_row.get("v5_photo_sand_mask_sha256"),
              "Candidate photo-sand mask")
    try:
        photo_sand = np.load(sand_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PhysicsLayerUnavailable(f"Candidate photo-sand mask cannot be decoded: {exc}") from exc
    _require(photo_sand.dtype == np.bool_ and photo_sand.shape == shape,
             "Candidate photo-sand mask has an invalid native grid")
    _require(not np.any(gas & ~photo_sand),
             "Physics gas extends outside the verified photo-sand domain")
    limitations = global_manifest.get("known_limitations")
    _require(isinstance(limitations, list) and all(isinstance(item, str) for item in limitations),
             "Physics export does not document its known limitations")
    return VerifiedPhysicsFrame(gas, photo_sand, box, canvas,
                                tuple(limitations), run, frame_index)


def try_load_verified_physics_frame(*args, **kwargs) -> tuple[VerifiedPhysicsFrame | None, str | None]:
    """UI helper: never silently substitute the optical layer on a bad export."""
    try:
        return load_verified_physics_frame(*args, **kwargs), None
    except (PhysicsLayerUnavailable, OSError, TypeError, KeyError, ValueError) as exc:
        return None, f"Gas physics layer unavailable: {exc}"


def render_physics_overlay(
    optical_rgba: np.ndarray,
    frame: VerifiedPhysicsFrame,
    layer: Literal["physics", "difference"],
) -> np.ndarray:
    """Compose a display copy on native grid or the recorded full canvas."""
    rgba = np.asarray(optical_rgba)
    _require(rgba.dtype == np.uint8 and rgba.ndim == 3 and rgba.shape[-1] == 4,
             "Optical map is not a categorical uint8 RGBA image")
    x, y, width, height = frame.content_box_xywh
    if rgba.shape[:2] == frame.mask.shape:
        optical = rgba
        full_canvas = False
    elif rgba.shape[:2] == (frame.canvas_wh[1], frame.canvas_wh[0]):
        optical = rgba[y:y + height, x:x + width]
        full_canvas = True
    else:
        raise PhysicsLayerUnavailable("Optical map and physics mask have different native grids")
    active = optical[..., 3] > 0
    gas = active & np.all(optical[..., :3] == GAS_RGBA[:3], axis=-1)
    aqueous = active & np.all(optical[..., :3] == AQUEOUS_RGBA[:3], axis=-1)
    unresolved = active & np.all(optical[..., :3] == UNRESOLVED_RGB, axis=-1)
    _require(not np.any(active & frame.photo_sand & ~(gas | aqueous | unresolved)),
             "Optical map contains noncategorical pixels inside photo-sand")
    support = frame.photo_sand
    gas &= support
    aqueous &= support
    unresolved &= support
    physical = frame.mask
    out = np.zeros_like(rgba) if full_canvas else np.zeros_like(optical)
    crop = out[y:y + height, x:x + width] if full_canvas else out
    if layer == "physics":
        crop[unresolved & ~physical] = optical[unresolved & ~physical]
        crop[(aqueous | gas) & ~physical] = AQUEOUS_RGBA
        crop[physical] = GAS_RGBA
    elif layer == "difference":
        crop[gas & physical] = COMMON_GAS_RGBA
        crop[physical & ~gas] = ADDED_RGBA
        crop[gas & ~physical] = REJECTED_RGBA
    else:
        raise PhysicsLayerUnavailable(f"Unknown gas physics display layer: {layer}")
    return out
