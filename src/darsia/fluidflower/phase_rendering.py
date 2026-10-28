"""Two-color phase display, independent of training and concentration models.

Keep the established gas threshold: CO2-positive pixels below it are displayed
as aqueous. Scores and region-label classes (including mixtures) stay intact.
Legacy amber review maps can be rendered without rerunning inference or writing
to their source folders.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image


PHASE_DISPLAY_POLICY = "two-phase-gas060-v1"
CO2_DISPLAY_THRESHOLD = np.float32(0.5)
GAS_DISPLAY_THRESHOLD = np.float32(0.6)
AQUEOUS_RGBA = (49, 217, 160, 255)
GAS_RGBA = (242, 76, 157, 255)
LEGACY_AMBER_RGB = (255, 191, 71)


def two_phase_masks(predictions):
    """Partition CO2 support, retaining the old gas decision boundary."""
    co2 = np.asarray(predictions["co2_presence_probability"], dtype=np.float32)
    gas_score = np.asarray(predictions["gas_probability_conditional_on_co2"], dtype=np.float32)
    if co2.shape != gas_score.shape:
        raise ValueError("CO2 and gas scores must share the same grid")
    active = co2 >= CO2_DISPLAY_THRESHOLD
    gas = active & (gas_score >= GAS_DISPLAY_THRESHOLD)
    return {"aqueous": active & ~gas, "gas": gas}


def two_phase_rgba(image: np.ndarray, phase: str = "all") -> np.ndarray:
    """Render an existing categorical RGBA map, never a camera image.

Amber formerly meant gas score [0.4, 0.6); it therefore belongs below the
unchanged gas threshold. Preserve every pink pixel and every alpha value.
"""
    rgba = np.array(image, dtype=np.uint8, copy=True)
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError("Expected an RGBA phase map")
    amber = (rgba[..., 3] > 0) & np.all(rgba[..., :3] == LEGACY_AMBER_RGB, axis=-1)
    rgba[amber, :3] = AQUEOUS_RGBA[:3]
    if phase in {"gas", "aqueous"}:
        color = GAS_RGBA if phase == "gas" else AQUEOUS_RGBA
        rgba[~np.all(rgba[..., :3] == color[:3], axis=-1)] = 0
    elif phase == "mixed":
        # Old external links remain valid, but there is no third display layer.
        rgba[:] = 0
    elif phase != "all":
        raise ValueError(f"Unknown phase display layer: {phase}")
    return rgba


def phase_display_cache_root() -> Path:
    configured = os.environ.get("DARSIA_PHASE_DISPLAY_CACHE")
    if configured:
        return Path(configured).expanduser()
    root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "DarSIA" / "phase-display" / PHASE_DISPLAY_POLICY


def two_phase_overlay_path(source: Path, phase: str = "all", *, cache_root: Path | None = None) -> Path:
    """Return a versioned local display copy; the original remains read-only."""
    if phase not in {"all", "gas", "aqueous", "mixed"}:
        raise ValueError(f"Unknown phase display layer: {phase}")
    from darsia_cache.adapters import optional_import_lease, managed_display_png, reusable_display_png
    from darsia_cache.policy import CachePolicyError
    managed=optional_import_lease()
    if managed is not None and cache_root is not None:
        if Path(cache_root).expanduser().resolve()!=(managed.root/"display").resolve():
            raise CachePolicyError("Explicit display cache is outside the admitted service scope")
    source = Path(source).resolve()
    stat = source.stat()
    signature = f"{PHASE_DISPLAY_POLICY}:{source}:{stat.st_size}:{stat.st_mtime_ns}"
    key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    logical_gate=f"phase-display:{key}:{phase}"
    if managed is not None:
        existing=reusable_display_png(logical_gate=logical_gate)
        if existing is not None:return existing
    if managed is None:
        folder = (cache_root or phase_display_cache_root()) / key
        target = folder / f"{phase}.png"
        if target.is_file():
            return target
    data = source.read_bytes()
    if (source.stat().st_size, source.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"Phase map changed while loading: {source}")
    with Image.open(io.BytesIO(data)) as image:
        rgba = two_phase_rgba(np.asarray(image.convert("RGBA")), phase)
    if managed is not None:
        encoded=io.BytesIO()
        Image.fromarray(rgba).save(encoded,format="PNG",compress_level=1)
        return managed_display_png(encoded.getvalue(),{
            "display_policy": PHASE_DISPLAY_POLICY,
            "source": str(source), "source_sha256": hashlib.sha256(data).hexdigest(),
            "co2_threshold": 0.5, "gas_threshold": 0.6,
            "legacy_amber_display": "aqueous", "model_scores_changed": False,
        },logical_gate=logical_gate)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{phase}.{uuid.uuid4().hex}.png"
    try:
        Image.fromarray(rgba).save(temporary, compress_level=1)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = folder / f".{uuid.uuid4().hex}.json"
    metadata.write_text(json.dumps({
        "display_policy": PHASE_DISPLAY_POLICY,
        "source": str(source), "source_sha256": hashlib.sha256(data).hexdigest(),
        "co2_threshold": 0.5, "gas_threshold": 0.6,
        "legacy_amber_display": "aqueous", "model_scores_changed": False,
    }, indent=2), encoding="utf-8")
    os.replace(metadata, folder / "provenance.json")
    return target
