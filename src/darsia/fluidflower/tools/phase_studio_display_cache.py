"""Bounded, lossless local derivatives for native Studio playback.

Only presentation copies are cached. The normalized full map remains separate
for region scoring; photographs, predictions, annotations and NAS assets are
never written by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

import numpy as np
from PySide6.QtGui import QImage

from .phase_view_overlay import display_phase_overlay
from darsia.fluidflower.phase_filled_display import hatch_filled_overlay
from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_overlay
from darsia.fluidflower.phase_l1_neighbourhood_display import hatch_l1_neighbourhood_overlay

VERSION = "studio-prepared-overlay-v1"


def default_image_cache_bytes(available_bytes=None):
    """Use one eighth of available RAM, capped at sixteen GiB on large hosts."""
    if available_bytes is None:
        try:
            import psutil
            available_bytes = psutil.virtual_memory().available
        except (ImportError, OSError):
            available_bytes = 4 << 30
    return max(256 << 20, min(16 << 30, int(available_bytes) // 8))


def image_memory_bytes(image):
    """Include masks and prepared variants, sharing each allocation once."""
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, QImage):
            return int(value.sizeInBytes()) + sum(size(item) for item in vars(value).values())
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        return 0
    return size(image)


def overlay_variant_key(image, layer="all", *, filled=True, phase_visible=True, floor_contact=True,
                        l1_neighbourhood=True):
    """Describe a display variant using only its loaded masks and controls."""
    contact_mask = getattr(image, "_floor_contact_fill_mask", None)
    l1_mask = getattr(image, "_l1_neighbourhood_gas_mask", None)
    suffix = ((bool(floor_contact),) if contact_mask is not None else ()) + ((bool(l1_neighbourhood),) if l1_mask is not None else ())
    return (str(layer), bool(filled), bool(phase_visible), *suffix)


def prepared_overlay(image, layer="all", *, filled=True, phase_visible=True, floor_contact=True,
                     l1_neighbourhood=True):
    """Reuse an exact display variant; phase scoring still uses ``image``."""
    contact_mask = getattr(image, "_floor_contact_fill_mask", None)
    l1_mask = getattr(image, "_l1_neighbourhood_gas_mask", None)
    key = overlay_variant_key(image, layer, filled=filled, phase_visible=phase_visible,
                              floor_contact=floor_contact, l1_neighbourhood=l1_neighbourhood)
    variants = getattr(image, "_display_variants", None)
    if variants is None:
        variants = image._display_variants = {}
    if key not in variants:
        unhatched = (str(layer), False, True,
                     *((False,) if contact_mask is not None else ()),
                     *((False,) if l1_mask is not None else ()))
        filtered = variants.get(unhatched)
        if filtered is None:
            filtered = (image if layer == "__unfiltered__" else display_phase_overlay(image, str(layer),
                photo_sand=getattr(image, "_photo_sand_domain", None)))
        if phase_visible:
            variants[unhatched] = filtered
        rendered = hatch_filled_overlay(filtered, getattr(image, "_filled_phase_mask", None),
            enabled=filled, phase_visible=phase_visible)
        rendered = hatch_floor_contact_overlay(rendered, contact_mask,
            enabled=floor_contact, phase_visible=phase_visible)
        variants[key] = hatch_l1_neighbourhood_overlay(rendered, l1_mask,
            enabled=l1_neighbourhood, phase_visible=phase_visible)
        # Two independent panels may request different L1 hatch settings on the
        # same full-resolution map. Keep both renderings plus the base variants
        # instead of reprocessing the whole grid each time a panel is painted.
        while len(variants) > 6:
            variants.pop(next(iter(variants)))
    return variants[key]


def binding_token(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class PreparedOverlayStore:
    """Lossless PNGs and packed masks under a bound source/recipe identity."""
    def __init__(self, root, *, max_bytes=2 << 30):
        self.root = Path(root).resolve()
        self.max_bytes = max(64 << 20, int(max_bytes))
        self.lock = threading.RLock()

    def _folder(self, identity, binding, layer):
        key = binding_token([VERSION, identity, binding, layer])
        return self.root / key

    def _remove(self, folder):
        if folder.resolve().parent != self.root:
            raise ValueError("Presentation cache cleanup escaped its own directory")
        shutil.rmtree(folder)

    @staticmethod
    def _stamp(path):
        path = Path(path)
        stat = path.stat()
        return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def _source_stamps(self, sources, expected_stamps=None):
        paths = tuple(dict.fromkeys(str(Path(path)) for path in sources))
        if expected_stamps is None:
            return [self._stamp(path) for path in paths]
        expected = {item["path"]: item for item in expected_stamps}
        stamps = []
        for path in paths:
            if path not in expected or self._stamp(path) != expected[path]:
                raise OSError("Presentation cache source changed after decoding")
            stamps.append(dict(expected[path]))
        return stamps

    def _check_source_stamps(self, stamps):
        for item in stamps:
            if self._stamp(item["path"]) != item:
                raise OSError("Presentation cache source changed during persistence")

    def load(self, identity, binding, layer):
        folder = self._folder(identity, binding, layer)
        try:
            receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
            if receipt.get("binding") != binding or receipt.get("version") != VERSION:
                return None
            for item in receipt["sources"]:
                try:
                    if self._stamp(item["path"]) != item:
                        return None
                except OSError:
                    pass  # Match the existing local cache's offline fallback.
            image, rendered = QImage(str(folder / "map.png")), QImage(str(folder / "display.png"))
            unhatched = QImage(str(folder / "unhatched.png"))
            if image.isNull() or rendered.isNull() or unhatched.isNull() or image.size() != rendered.size() or image.size() != unhatched.size():
                return None
            with np.load(folder / "masks.npz", allow_pickle=False) as masks:
                for name, attr in (("domain", "_photo_sand_domain"), ("filled", "_filled_phase_mask"), ("floor_contact", "_floor_contact_fill_mask"), ("l1_neighbourhood", "_l1_neighbourhood_gas_mask")):
                    if name in masks:
                        shape = tuple(map(int, masks[name + "_shape"]))
                        if shape != (image.height(), image.width()):
                            return None
                        setattr(image, attr, np.unpackbits(masks[name], count=shape[0] * shape[1]).reshape(shape).astype(bool))
            suffix = ((True,) if getattr(image, "_floor_contact_fill_mask", None) is not None else ()) + ((True,) if getattr(image, "_l1_neighbourhood_gas_mask", None) is not None else ())
            image._display_variants = {(str(layer), True, True, *suffix): rendered, (str(layer), False, True, *suffix): unhatched}
            os.utime(folder / "receipt.json", None)
            return image
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, identity, binding, layer, image, sources, *, expected_stamps=None):
        folder = self._folder(identity, binding, layer)
        stamps = self._source_stamps(sources, expected_stamps)
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / ("." + folder.name + "." + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                rendered = prepared_overlay(image, layer)
                unhatched = prepared_overlay(image, layer, filled=False)
                if (not image.save(str(temporary / "map.png"), "PNG")
                        or not rendered.save(str(temporary / "display.png"), "PNG")
                        or not unhatched.save(str(temporary / "unhatched.png"), "PNG")):
                    raise OSError("Could not save a local presentation image")
                masks = {}
                for name, attr in (("domain", "_photo_sand_domain"), ("filled", "_filled_phase_mask"), ("floor_contact", "_floor_contact_fill_mask"), ("l1_neighbourhood", "_l1_neighbourhood_gas_mask")):
                    mask = getattr(image, attr, None)
                    if mask is not None:
                        masks[name] = np.packbits(mask)
                        masks[name + "_shape"] = np.asarray(mask.shape, dtype=np.int64)
                np.savez_compressed(temporary / "masks.npz", **masks)
                receipt = dict(version=VERSION, binding=binding, layer=str(layer),
                    sources=stamps)
                (temporary / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                self._check_source_stamps(stamps)
                if folder.exists():
                    self._remove(folder)
                if temporary.resolve().parent != self.root or folder.resolve().parent != self.root:
                    raise ValueError("Presentation cache publication escaped its own directory")
                temporary.rename(folder)
                self._prune(folder)
            finally:
                if temporary.exists():
                    self._remove(temporary)

    def _prune(self, keep):
        entries = []
        total = 0
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            size = sum(path.stat().st_size for path in folder.iterdir() if path.is_file())
            total += size
            entries.append(((folder / "receipt.json").stat().st_mtime_ns, folder, size))
        for _time, folder, size in sorted(entries):
            if total <= self.max_bytes:
                break
            if folder != keep:
                self._remove(folder)
                total -= size


class PreparedPhotoStore(PreparedOverlayStore):
    """Bounded local decoded pixels avoid repeated full-camera JPEG decoding."""
    def __init__(self, root, *, max_bytes=4 << 30):
        super().__init__(root, max_bytes=max_bytes)

    def load(self, identity, binding):
        folder = self._folder(identity, binding, "photo")
        try:
            receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
            if receipt.get("binding") != binding or receipt.get("version") != VERSION:
                return None
            for item in receipt["sources"]:
                try:
                    if self._stamp(item["path"]) != item:
                        return None
                except OSError:
                    pass
            values = np.load(folder / "pixels.npy", mmap_mode="r", allow_pickle=False)
            width, height = receipt["size_wh"]
            stride = receipt["stride"]
            if values.dtype != np.uint8 or values.shape != (height, stride):
                return None
            image = QImage(values.data, width, height, stride, QImage.Format(receipt["pixel_format"])).copy()
            del values
            if image.isNull():
                return None
            with np.load(folder / "masks.npz", allow_pickle=False) as masks:
                for name in masks.files:
                    if not name.endswith("_shape"):
                        shape = tuple(map(int, masks[name + "_shape"]))
                        if shape != (height, width):
                            return None
                        setattr(image, name, np.unpackbits(masks[name], count=height * width).reshape(shape).astype(bool))
            if receipt.get("support_audit") is not None:
                image._display_support_audit = receipt["support_audit"]
            os.utime(folder / "receipt.json", None)
            return image
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, identity, binding, image, sources, *, expected_stamps=None):
        folder = self._folder(identity, binding, "photo")
        stamps = self._source_stamps(sources, expected_stamps)
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / ("." + folder.name + "." + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                values = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
                np.save(temporary / "pixels.npy", values, allow_pickle=False)
                masks = {}
                for name in ("_photo_sand_domain", "_raw_display_support_mask", "_raw_display_missing_mask"):
                    mask = getattr(image, name, None)
                    if mask is not None:
                        masks[name] = np.packbits(mask)
                        masks[name + "_shape"] = np.asarray(mask.shape, dtype=np.int64)
                np.savez_compressed(temporary / "masks.npz", **masks)
                receipt = dict(version=VERSION, binding=binding, size_wh=[image.width(), image.height()],
                    pixel_format=image.format().value, stride=image.bytesPerLine(),
                    support_audit=getattr(image, "_display_support_audit", None),
                    sources=stamps)
                (temporary / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                self._check_source_stamps(stamps)
                if folder.exists():
                    self._remove(folder)
                if temporary.resolve().parent != self.root or folder.resolve().parent != self.root:
                    raise ValueError("Presentation cache publication escaped its own directory")
                temporary.rename(folder)
                self._prune(folder)
            finally:
                if temporary.exists():
                    self._remove(temporary)
