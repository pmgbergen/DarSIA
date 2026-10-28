"""SHA-bound aliases for a top-domain revision's unchanged photographs/RAW.

Only photo-sand mask declarations may differ from the original native source.
The original scientific loader remains unchanged and still enforces its exact
fixed paths, source hashes and float32 sidecar representation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .project import sha256_file


from .phase_saved_review_metadata import configured_parameters

def _parameters():
    return configured_parameters("source_alias")

def __getattr__(name):
    if name == "KEY": return _parameters()["key"]
    if name == "CONTRACT": return _parameters()["contract"]
    raise AttributeError(name)


def resolve_original_native_source(review_root, run, manifest=None):
    root = Path(review_root).resolve()
    run = str(run).casefold()
    native_path = root / "native_frames" / run / "manifest.json"
    current = (json.loads(native_path.read_text(encoding="utf-8-sig"))
               if manifest is None else manifest)
    marker = current.get(_parameters()["key"])
    if marker is None:
        return root, native_path, current
    if (not isinstance(marker, dict) or set(marker) != {
            "contract", "original_review_root", "original_native_manifest", "geometry_manifest"}
            or marker["contract"] != _parameters()["contract"]):
        raise ValueError("Invalid top-domain native source alias")
    original_root = Path(marker["original_review_root"]).resolve()
    bound = marker["original_native_manifest"]
    if (not isinstance(bound, dict) or set(bound) != {"path", "sha256"}
            or Path(bound["path"]).resolve() != original_root / "native_frames" / run / "manifest.json"
            or sha256_file(Path(bound["path"])) != bound["sha256"]):
        raise ValueError("Top-domain alias original native source changed or escaped its fixed path")
    geometry = marker["geometry_manifest"]
    if (not isinstance(geometry, dict) or set(geometry) != {"path", "sha256"}
            or sha256_file(Path(geometry["path"])) != geometry["sha256"]):
        raise ValueError("Top-domain alias photographic geometry binding changed")
    original_path = Path(bound["path"]).resolve()
    original = json.loads(original_path.read_text(encoding="utf-8-sig"))
    if _parameters()["key"] in original or str(original.get("run", "")).casefold() != run:
        raise ValueError("Top-domain alias must point directly to the original run")
    before, after = deepcopy(original), deepcopy(current)
    after.pop(_parameters()["key"])
    # Everything except these exact declarations, including the complete
    # scientific rows, source timestamps and geometry, must remain identical.
    for document in (before, after):
        rows = document.get("frame_sources") or []
        if not isinstance(rows, list):
            raise ValueError("Top-domain alias source timeline is invalid")
        for row in [document.get("reference") or {}, *rows]:
            row.pop("v5_photo_sand_mask_path", None)
            row.pop("v5_photo_sand_mask_sha256", None)
    if before != after:
        raise ValueError("Top-domain alias changed original scientific, photographic or geometry fields")
    return original_root, original_path, original


def resolve_native_photo_path(review_root, run, frame_index=None, *, reference=False):
    """Return the same fixed original path; callers retain normal file checks."""
    native_path=Path(review_root)/"native_frames"/str(run).casefold()/"manifest.json"
    native=json.loads(native_path.read_text(encoding="utf-8-sig"))
    if "ffac_rig_camera_sources" in native:
        from .phase_saved_review_metadata import resolve_photograph
        return resolve_photograph(review_root,run,frame_index,reference=reference)
    root, _native, _manifest = resolve_original_native_source(review_root, run)
    path = (root / "native_references" / str(run).casefold() / "pre_injection.jpg"
            if reference else root / "native_frames" / str(run).casefold() / f"frame_{frame_index:03d}.jpg")
    return path.with_suffix(".png") if path.with_suffix(".png").is_file() else path


from .phase_training.log_linear_input import RunV5LinearInput


class RunV5AliasedLinearInput(RunV5LinearInput):
    """Reuse the original strict float reader; expose the new source identity."""
    def __init__(self, *, native_manifest_path, run_id, **kwargs):
        shadow = Path(native_manifest_path).resolve()
        _root, original, _manifest = resolve_original_native_source(
            shadow.parent.parent.parent, run_id)
        super().__init__(native_manifest_path=original, run_id=run_id, **kwargs)
        self.original_native_manifest_path = original
        self.original_native_manifest_sha256 = self.manifest_sha256
        self.manifest_path = shadow
        self.manifest_sha256 = sha256_file(shadow)

    def report(self):
        return dict(super().report(), original_native_manifest_path=str(self.original_native_manifest_path),
                    original_native_manifest_sha256=self.original_native_manifest_sha256,
                    source_alias_contract=_parameters()["contract"])


__all__ = ["KEY", "CONTRACT", "resolve_original_native_source", "resolve_native_photo_path",
           "RunV5AliasedLinearInput"]
