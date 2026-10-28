"""Resolve source-bound V5 photographic displays without changing phase evidence.

The selected image is a presentation artifact on the same native canvas.  The
float32 sidecar, not either JPEG, remains the model's numerical input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


V5_SOURCE_FORMAT = "ffac-v5-native-source-materialization-v1"
CHANGE_DISPLAY_CONTRACT = "ffac-track1-display-change-flatfield-v1"
SELECTED_SOURCE_KEY = "ffac_selected_policy_a_sources"

def _selected_rows(manifest, reference=False):
    rows = [manifest.get("reference") or {}] if reference else manifest.get("frame_sources") or []
    return rows if SELECTED_SOURCE_KEY in manifest or (rows and all(row.get("source_label_identity") and row.get("photographs") for row in rows)) else []

def _raw_review(review_root, run):
    """The run's RAW photographic review manifest, or None for a historical root."""
    if not review_root or not run:
        return None
    from .raw_photographic_review import read_raw_review_manifest
    return read_raw_review_manifest(review_root, run)


def _raw_rows(manifest, reference=False):
    """The rows of a RAW review manifest (or of RAW rows passed alone), else None."""
    from .raw_photographic_review import RAW_REVIEW_FORMAT
    if manifest.get("format") == RAW_REVIEW_FORMAT:
        return [manifest.get("reference") or {}] if reference else list(manifest.get("frames") or [])
    rows = [manifest.get("reference") or {}] if reference else (manifest.get("frame_sources") or [])
    from .raw_photographic_review import raw_review_row
    if rows and all(raw_review_row(row) for row in rows):
        return list(rows)
    return None


def _raw_modes(rows, modes):
    """Whether every RAW row declares each of ``modes``; no file I/O."""
    return bool(rows) and all(all(mode in row["assets"] for mode in modes) for row in rows)


def _selected_manifest(review_root, run):
    from .phase_selected_policy_review import _native_document, _READER
    path = Path(review_root)/"native_frames"/str(run).casefold()/"manifest.json"
    native = _native_document(path,_READER)
    return native if SELECTED_SOURCE_KEY in native else None

BASE_DISPLAY_MODES = ("natural", "lift1", "lift2")
DISPLAY_MODES = (*BASE_DISPLAY_MODES, "change")
DISPLAY_MODE_LABELS = {
    "natural": "Brightness: natural",
    "lift1": "Brightness: lift ×4.3",
    "lift2": "Brightness: lift ×7.1",
    "change": "Change (flat field)",
}


def _sha256(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


def raw_detail_available(manifest: Mapping, *, reference: bool = False) -> bool:
    """Whether a V5 manifest advertises the optional display asset."""
    if (raw := _raw_rows(manifest, reference)) is not None:
        return _raw_modes(raw, ("raw_detail",))
    if rows := _selected_rows(manifest,reference):
        return all(row.get("photographs",{}).get("raw_detail") for row in rows)
    if (manifest.get("v5_photometry") or {}).get("format") != V5_SOURCE_FORMAT:
        return False
    rows = [manifest.get("reference") or {}] if reference else (manifest.get("frame_sources") or [])
    return bool(rows) and all(
        row.get("v5_raw_detail_jpeg_path") and row.get("v5_raw_detail_jpeg_sha256")
        for row in rows
    )


def display_modes_available(manifest: Mapping, *, reference: bool = False) -> bool:
    """Inspect mode declarations without filesystem work on the GUI thread."""
    if (raw := _raw_rows(manifest, reference)) is not None:
        return _raw_modes(raw, BASE_DISPLAY_MODES)
    rows = [manifest.get("reference") or {}] if reference else (manifest.get("frame_sources") or [])
    return bool(rows) and all(
        isinstance(row.get("v5_display_modes"), Mapping)
        and all(isinstance(row["v5_display_modes"].get(mode), Mapping)
                and row["v5_display_modes"][mode].get("jpeg_path")
                and row["v5_display_modes"][mode].get("jpeg_sha256")
                for mode in BASE_DISPLAY_MODES)
        for row in rows
    )


def original_display_available(manifest: Mapping, *, reference: bool = False) -> bool:
    """Inspect source-bound camera JPEG declarations without filesystem I/O.

    A RAW review root declares its camera-RAW original as an asset instead.
    """
    if (raw := _raw_rows(manifest, reference)) is not None:
        return _raw_modes(raw, ("original",))
    if rows := _selected_rows(manifest,reference):
        return all(row.get("photographs",{}).get("original") for row in rows)
    if (manifest.get("v5_photometry") or {}).get("format") != V5_SOURCE_FORMAT:
        return False
    canvas, box, shape = (manifest.get("canvas_wh"),
                          manifest.get("content_box_xywh"),
                          manifest.get("analysis_image_shape_hw"))
    try:
        width, height = map(int, canvas)
        x, y, box_width, box_height = map(int, box)
        analysis_height, analysis_width = map(int, shape)
    except (TypeError, ValueError):
        return False
    if (width <= 0 or height <= 0 or x < 0 or y < 0
            or box_width <= 0 or box_height <= 0
            or x + box_width > width or y + box_height > height
            or (box_height, box_width) != (analysis_height, analysis_width)):
        return False
    rows = [manifest.get("reference") or {}] if reference else (manifest.get("frame_sources") or [])
    receipt_prefix = "v5_baseline" if reference else "v5_frame"
    return bool(rows) and all(
        isinstance(row, Mapping)
        and (reference or row.get("frame_index") == index)
        and isinstance(row.get("source_path"), str)
        and bool(row["source_path"])
        and isinstance(row.get("source_sha256"), str)
        and len(row["source_sha256"]) == 64
        and isinstance(row.get(f"{receipt_prefix}_receipt_path"), str)
        and bool(row[f"{receipt_prefix}_receipt_path"])
        and isinstance(row.get(f"{receipt_prefix}_receipt_sha256"), str)
        and len(row[f"{receipt_prefix}_receipt_sha256"]) == 64
        for index, row in enumerate(rows)
    )


def _change_manifest_path(review_root, run) -> Path:
    return (Path(review_root).resolve() / "native_frames" / str(run).casefold()
            / "change_display_manifest.json")


def change_display_available(review_root, run, *, frame_count: int | None = None) -> bool:
    """Check the optional addendum without hashing large JPEGs on the GUI thread.

    The original V5 manifest is sealed, so this display-only mode has its own
    source-bound frame records. The resolver performs full SHA verification.
    """
    if not review_root or not run:
        return False
    try:
        raw = _raw_review(review_root, run)
    except (OSError, ValueError):
        return False
    if raw is not None:
        return (_raw_modes(raw["frames"], ("change",)) and _raw_modes([raw["reference"]], ("change",))
                and (frame_count is None or len(raw["frames"]) == frame_count))
    native_path=Path(review_root)/"native_frames"/str(run).casefold()/"manifest.json"
    try:
        native=json.loads(native_path.read_text(encoding="utf-8-sig"))
        if SELECTED_SOURCE_KEY in native:
            from .phase_selected_policy_review import load_native_source
            state=load_native_source(review_root,run,native,verify_geometry=False)
            return (frame_count is None or len(state["inputs"]["frames"])==frame_count)
        if "ffac_rig_camera_sources" in native:
            from .phase_saved_review_metadata import load_native_source
            state=load_native_source(review_root,run,native,verify_geometry=False)
            return (frame_count is None or len(state["inputs"]["frames"])==frame_count)
    except (OSError,ValueError,KeyError,TypeError):
        return False
    try:
        original_root = _photographic_root(review_root, run)
        row = json.loads(_change_manifest_path(original_root, run).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    frames = row.get("frames")
    def declared(asset):
        return (isinstance(asset, Mapping)
                and isinstance(asset.get("jpeg_path"), str)
                and isinstance(asset.get("jpeg_sha256"), str)
                and len(asset["jpeg_sha256"]) == 64)
    return (row.get("contract") == CHANGE_DISPLAY_CONTRACT
            and str(row.get("run") or "").casefold() == str(run).casefold()
            and isinstance(row.get("source_manifest_sha256"), str)
            and declared(row.get("reference"))
            and isinstance(frames, list)
            and (frame_count is None or len(frames) == frame_count)
            and all(isinstance(frame, Mapping)
                    and frame.get("frame_index") == index
                    and declared(frame)
                    for index, frame in enumerate(frames)))


def resolve_change_display(review_root, run, frame_index, *, reference=False) -> Path:
    """Verify a separately recorded change photograph against the sealed run."""
    if _raw_review(review_root, run) is not None:
        from .raw_photographic_review import resolve_raw_photograph
        return resolve_raw_photograph(review_root, run, frame_index, mode="change", reference=reference)
    if _selected_manifest(review_root,run) is not None:
        from .phase_selected_policy_review import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference,mode="change")
    manifest, _, directory, stem = _display_record(review_root, run, frame_index, reference)
    if "ffac_rig_camera_sources" in manifest:
        from .phase_saved_review_metadata import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference,mode="change")
    original_root = _photographic_root(review_root, run, manifest)
    directory = _photographic_directory(original_root, run, reference)
    source_path = original_root / "native_frames" / str(run).casefold() / "manifest.json"
    addendum = json.loads(_change_manifest_path(original_root, run).read_text(encoding="utf-8"))
    if (addendum.get("contract") != CHANGE_DISPLAY_CONTRACT
            or str(addendum.get("run") or "").casefold() != str(run).casefold()
            or addendum.get("source_manifest_sha256") != _sha256(source_path)):
        raise ValueError("Change display is not bound to this sealed native source manifest")
    frames = addendum.get("frames")
    if not isinstance(frames, list) or len(frames) != len(manifest.get("frame_sources") or []):
        raise ValueError("Change display does not cover the native timeline")
    if reference:
        record = addendum.get("reference") or {}
    else:
        record = frames[frame_index]
        if int(record.get("frame_index", -1)) != frame_index:
            raise ValueError("Change display frame disagrees with its source index")
    return _verified_display_path(record, directory / f"{stem}.change.jpg",
                                  "jpeg_path", "jpeg_sha256")


def _display_record(review_root, run, frame_index, reference):
    root = Path(review_root).resolve()
    run = run.casefold()
    manifest = json.loads((root / "native_frames" / run / "manifest.json").read_text(encoding="utf-8-sig"))
    if (manifest.get("v5_photometry") or {}).get("format") != V5_SOURCE_FORMAT:
        raise ValueError("Display modes require a source-bound V5 manifest")
    if str(manifest.get("run") or "").casefold() != run:
        raise ValueError("Display manifest identifies another experiment")
    if reference:
        record = manifest.get("reference") or {}
        directory, stem = root / "native_references" / run, "pre_injection"
    else:
        rows = manifest.get("frame_sources") or []
        if frame_index is None or not 0 <= frame_index < len(rows):
            raise ValueError("Display frame is outside the source-bound timeline")
        record = rows[frame_index]
        if int(record.get("frame_index", frame_index)) != frame_index:
            raise ValueError("Display frame disagrees with its source record")
        directory, stem = root / "native_frames" / run, f"frame_{frame_index:03d}"
    return manifest, record, directory, stem


def _verified_display_path(record, expected, path_key, sha_key):
    path_text, digest = record.get(path_key), record.get(sha_key)
    if not isinstance(path_text, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Display asset lacks an exact path and SHA-256")
    path = Path(path_text).resolve()
    if path != expected.resolve():
        raise ValueError("Display asset points outside its fixed native path")
    if _sha256(path) != digest.lower():
        raise ValueError("Display asset changed after its native manifest was saved")
    return path


def _photographic_root(review_root, run, manifest=None):
    root = Path(review_root).resolve()
    if manifest is None:
        manifest_path = root / "native_frames" / str(run).casefold() / "manifest.json"
        if not manifest_path.is_file():
            return root
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if "v5_native_source_alias" not in manifest:
        return root
    from .phase_source_alias import resolve_original_native_source
    original_root, _path, _original = resolve_original_native_source(root, run, manifest)
    return original_root


def _photographic_directory(root, run, reference):
    return Path(root) / ("native_references" if reference else "native_frames") / str(run).casefold()


def resolve_display_mode(review_root, run, frame_index, mode="natural", *, reference=False) -> Path:
    """Resolve the exact pre-rendered photographic asset on an image worker."""
    if mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown photographic display mode: {mode}")
    if mode == "change":
        return resolve_change_display(review_root, run, frame_index, reference=reference)
    if _raw_review(review_root, run) is not None:
        from .raw_photographic_review import resolve_raw_photograph
        return resolve_raw_photograph(review_root, run, frame_index, mode=mode, reference=reference)
    if _selected_manifest(review_root,run) is not None:
        from .phase_selected_policy_review import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference,mode=mode)
    manifest, record, directory, stem = _display_record(review_root, run, frame_index, reference)
    if "ffac_rig_camera_sources" in manifest:
        from .phase_saved_review_metadata import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference,mode=mode)
    directory = _photographic_directory(_photographic_root(review_root, run, manifest), run, reference)
    modes = record.get("v5_display_modes") or {}
    declaration = modes.get(mode) or {}
    suffix = "" if mode == "natural" else f".{mode}"
    return _verified_display_path(declaration, directory / f"{stem}{suffix}.jpg", "jpeg_path", "jpeg_sha256")


def load_display_image(review_root, run, frame_index=None, *, mode="natural", reference=False):
    """Decode a selected photograph on an image worker, including rig RAW.

    The Path resolvers retain their archival-JPEG contract. This Qt adapter
    renders source-bound extended rig RAW/Change in memory, including the
    exact baseline without a legacy reference-offset list. Other photographic
    transfers remain unchanged; declared bottom padding preserves coordinates.
    """
    from PySide6.QtGui import QImage
    from .phase_rig_raw_display import rig_raw_display_available, load_rig_raw_display_image
    from .phase_rig_display_canvas import pad_rig_photograph

    if _raw_review(review_root, run) is not None:
        # RAW review assets are lossless PNGs already on the phase canvas.
        if mode not in (*DISPLAY_MODES, "raw_detail", "original"):
            raise ValueError(f"Unknown photographic display mode: {mode}")
        from .raw_photographic_review import resolve_raw_photograph
        image = QImage(str(resolve_raw_photograph(review_root, run, frame_index, mode=mode, reference=reference)))
        if image.isNull():
            raise ValueError("RAW review photograph could not be decoded")
        return image
    if _selected_manifest(review_root,run) is not None:
        from .phase_selected_policy_review import resolve_photograph
        image=QImage(str(resolve_photograph(review_root,run,frame_index,mode=mode,reference=reference)))
        if image.isNull():
            raise ValueError("Selected full-rig photographic file could not be decoded")
        return image
    if mode not in (*DISPLAY_MODES, "raw_detail", "original"):
        raise ValueError(f"Unknown photographic display mode: {mode}")
    if mode in ("raw_detail", "change"):
        # Rig RAW uses its display-only, stamp-bound manifest/receipt cache.
        # Dispatch before the generic 8-MB JSON read on every frame. Legacy
        # candidates retain their existing archived-photograph route.
        from .phase_rig_raw_display import _native_frame_assets_cached
        try:
            state, _, _ = _native_frame_assets_cached(review_root, run, frame_index, reference=reference)
        except (OSError, ValueError, KeyError, TypeError):
            state = None
        if state is not None and rig_raw_display_available(state["native"], frame_index=frame_index, reference=reference):
            return load_rig_raw_display_image(review_root, run, frame_index, mode=mode, reference=reference)
    manifest, record, _directory, _stem = _display_record(review_root, run, frame_index, reference)
    if (mode in ("raw_detail", "change")
            and rig_raw_display_available(manifest, frame_index=frame_index, reference=reference)):
        return load_rig_raw_display_image(review_root, run, frame_index, mode=mode, reference=reference)
    if mode == "original":
        rgb = load_original_display_rgb(review_root, run, frame_index, reference=reference)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format.Format_RGB888).copy()
    else:
        path = (resolve_raw_detail(review_root, run, frame_index, reference=reference)
                if mode == "raw_detail" else
                resolve_display_mode(review_root, run, frame_index, mode, reference=reference))
        image = QImage(str(path))
    if image.isNull():
        raise ValueError("The selected source-bound photograph could not be decoded")
    return pad_rig_photograph(image, record.get("rig_domain_revision"))


def load_original_display_rgb(review_root, run, frame_index, *, reference=False) -> np.ndarray:
    """Rectify the genuine camera JPEG onto the native review canvas.

    This display-only worker operation does not alter the sealed source or the
    model input. The per-frame V5 receipt supplies the independently measured
    review-content-to-camera transform, so existing boxes and overlays retain
    their native coordinates.
    """
    import cv2
    from PIL import Image

    if _raw_review(review_root, run) is not None:
        # The camera RAW, already pulled onto the canvas by the review writer.
        from .raw_photographic_review import resolve_raw_photograph
        with Image.open(resolve_raw_photograph(review_root, run, frame_index, mode="original",
                                               reference=reference)) as image:
            return np.asarray(image.convert("RGB"))
    if _selected_manifest(review_root,run) is not None:
        from .phase_selected_policy_review import resolve_photograph
        with Image.open(resolve_photograph(review_root,run,frame_index,mode="original",reference=reference)) as image:
            return np.asarray(image.convert("RGB"))
    manifest, record, directory, stem = _display_record(
        review_root, run, frame_index, reference)
    directory = _photographic_directory(_photographic_root(review_root, run, manifest), run, reference)
    if not original_display_available(manifest, reference=reference):
        raise ValueError("Original JPEG lacks complete source-bound metadata")
    receipt_prefix = "v5_baseline" if reference else "v5_frame"
    receipt_frame_index = frame_index
    if "ffac_rig_camera_sources" in manifest:
        from .phase_saved_review_metadata import bound_path
        from .phase_rig_raw_display import _native_frame_assets_cached
        _state, _record, rig_binding = _native_frame_assets_cached(review_root,run,frame_index,reference=reference)
        if not reference and not rig_binding.get("source_changed", False):
            receipt_frame_index = rig_binding.get("parent_source_frame_index")
            if type(receipt_frame_index) is not int or receipt_frame_index < 0:
                raise ValueError("Retained rig Original source lacks its exact parent frame index")
        receipt_path=bound_path(dict(path=record[f"{receipt_prefix}_receipt_path"],
                                    sha256=record[f"{receipt_prefix}_receipt_sha256"]))
    else:
        receipt_path = _verified_display_path(
            record, directory / f"{stem}.v5.json",
            f"{receipt_prefix}_receipt_path", f"{receipt_prefix}_receipt_sha256")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    source = receipt.get("source") or {}
    certificate = receipt.get("certificate") or {}
    renderer = certificate.get("renderer_certificate") or {}
    camera_path = Path(record["source_path"]).resolve()
    camera_sha = record["source_sha256"].lower()
    if (receipt.get("format") != V5_SOURCE_FORMAT
            or str(receipt.get("run") or "").casefold() != str(run).casefold()
            or str(renderer.get("run") or "").casefold() != str(run).casefold()
            or (not reference and int(receipt.get("frame_index", -1)) != receipt_frame_index)
            or Path(source.get("camera_jpeg") or "").resolve() != camera_path
            or source.get("camera_jpeg_sha256") != camera_sha
            or certificate.get("camera_jpeg_sha256") != camera_sha
            or (renderer.get("sources") or {}).get("camera_jpeg_sha256") != camera_sha):
        raise ValueError("Original JPEG receipt does not match its native source record")
    if _sha256(camera_path) != camera_sha:
        raise ValueError("Original camera JPEG changed after source binding")

    try:
        canvas_wh = tuple(map(int, manifest["canvas_wh"]))
        box = tuple(map(int, manifest["content_box_xywh"]))
        analysis_hw = tuple(map(int, manifest["analysis_image_shape_hw"]))
        camera_wh = tuple(map(int, renderer["camera_jpeg_wh"]))
        receipt_shape = tuple(map(int, certificate["native_shape_hw"]))
        homography = np.asarray(renderer["review_to_camera_h"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Original JPEG receipt has incomplete native geometry") from exc
    x, y, width, height = box
    if (tuple(map(int, renderer.get("review_canvas_wh") or ())) != canvas_wh
            or tuple(map(int, renderer.get("content_box_xywh") or ())) != box
            or analysis_hw != (height, width)
            or receipt_shape != analysis_hw
            or homography.shape != (3, 3)
            or not np.isfinite(homography).all()
            or abs(np.linalg.det(homography)) < 1e-12):
        raise ValueError("Original JPEG receipt disagrees with native review geometry")
    with Image.open(camera_path) as image:
        if image.size != camera_wh:
            raise ValueError("Original camera JPEG dimensions disagree with its receipt")
        camera_rgb = np.asarray(image.convert("RGB"))
    content = cv2.warpPerspective(
        camera_rgb, homography, (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(19, 22, 27))
    canvas = np.empty((canvas_wh[1], canvas_wh[0], 3), dtype=np.uint8)
    canvas[:] = (19, 22, 27)
    canvas[y:y + height, x:x + width] = content
    return canvas


def resolve_photo_sand_domain(review_root, run, frame_index, *, reference=False, canvas=True) -> np.ndarray:
    """Load the immutable photo-sand mask, independent of model feature support."""
    if _raw_review(review_root, run) is not None:
        from .raw_photographic_review import load_raw_photo_support
        return load_raw_photo_support(review_root, run, frame_index, reference=reference, canvas=canvas)
    if _selected_manifest(review_root,run) is not None:
        from .phase_selected_policy_review import resolve_photo_sand_domain as selected_domain
        return selected_domain(review_root,run,frame_index,reference=reference,canvas=canvas)
    manifest, record, directory, stem = _display_record(review_root, run, frame_index, reference)
    if "ffac_rig_camera_sources" in manifest:
        from .phase_saved_review_metadata import native_frame_assets
        from .phase_saved_review_metadata import load_rig_classes
        native_frame_assets(review_root,run,frame_index,reference=reference)
        revision=record["rig_domain_revision"]
        fields,_=load_rig_classes(revision,base_dir=directory,
            shape_hw=None if canvas else tuple(manifest["analysis_image_shape_hw"]))
        return fields["rig_domain"]
    path = _verified_display_path(record, directory / f"{stem}.photo_sand_mask.npy",
                                  "v5_photo_sand_mask_path", "v5_photo_sand_mask_sha256")
    domain = np.load(path, allow_pickle=False)
    shape = tuple(map(int, manifest.get("analysis_image_shape_hw") or []))
    if domain.dtype != np.bool_ or domain.ndim != 2 or domain.shape != shape:
        raise ValueError("Photo-sand display mask disagrees with the native analysis grid")
    if not canvas:
        return domain
    canvas_wh, box = manifest.get("canvas_wh"), manifest.get("content_box_xywh")
    if not canvas_wh or not box:
        return domain
    width, height = map(int, canvas_wh)
    x, y, box_w, box_h = map(int, box)
    if (box_h, box_w) != domain.shape or x < 0 or y < 0 or x + box_w > width or y + box_h > height:
        raise ValueError("Photo-sand display mask has an invalid native canvas placement")
    result = np.zeros((height, width), dtype=bool)
    result[y:y + box_h, x:x + box_w] = domain
    return result


def resolve_raw_detail(
    review_root: str | Path,
    run: str,
    frame_index: int | None,
    *,
    reference: bool = False,
) -> Path:
    """Return a source-bound display JPEG or fail closed.

    This runs on the image-loading worker.  The path, immutable manifest row,
    and file digest are checked on every resolution.  No JPEG fallback occurs
    after the user selects RAW detail.
    """
    if _raw_review(review_root, run) is not None:
        from .raw_photographic_review import resolve_raw_photograph
        return resolve_raw_photograph(review_root, run, frame_index, mode="raw_detail", reference=reference)
    if _selected_manifest(review_root,run) is not None:
        from .phase_selected_policy_review import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference,mode="raw_detail")
    root = Path(review_root).resolve()
    native_dir = root / "native_frames" / run.casefold()
    manifest_path = native_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (manifest.get("v5_photometry") or {}).get("format") != V5_SOURCE_FORMAT:
        raise ValueError("RAW-detail display is available only for a source-bound V5 model")
    if str(manifest.get("run") or "").casefold() != run.casefold():
        raise ValueError("RAW-detail manifest identifies another experiment")
    if "ffac_rig_camera_sources" in manifest:
        from .phase_saved_review_metadata import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference,mode="raw_detail")
    original_root = _photographic_root(root, run, manifest)
    if reference:
        record = manifest.get("reference") or {}
        expected = original_root / "native_references" / run.casefold() / "pre_injection.raw_detail.jpg"
    else:
        rows = manifest.get("frame_sources") or []
        if frame_index is None or not 0 <= frame_index < len(rows):
            raise ValueError("RAW-detail frame is outside the source-bound timeline")
        record = rows[frame_index]
        if int(record.get("frame_index", frame_index)) != frame_index:
            raise ValueError("RAW-detail frame index disagrees with its source record")
        expected = original_root / "native_frames" / run.casefold() / f"frame_{frame_index:03d}.raw_detail.jpg"
    path_text = record.get("v5_raw_detail_jpeg_path")
    digest = record.get("v5_raw_detail_jpeg_sha256")
    if not isinstance(path_text, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("RAW-detail display lacks an exact path and SHA-256")
    path = Path(path_text).resolve()
    if path != expected.resolve():
        raise ValueError("RAW-detail display points outside its fixed native frame path")
    if _sha256(path) != digest.lower():
        raise ValueError("RAW-detail display changed after the native manifest was saved")
    return path
