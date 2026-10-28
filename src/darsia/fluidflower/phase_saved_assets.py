"""Strict SHA-bound saved metadata IO, independent of campaign calculation."""
from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np
from .native.io import sha256_file as sha256

def asset(path: str | Path) -> dict:
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


class VerifiedAssetReader:
    """Verify actual bytes once per unchanged path/size/mtime, then mmap."""
    def __init__(self, *, asset_reader=None):
        self.asset_reader = asset_reader
        self.verified = {}
        self.raw_baselines = {}
        self.physical_references = {}
        self.json_documents = {}

    def path(self, descriptor: Mapping) -> Path:
        if self.asset_reader is not None:
            return self.asset_reader.path(descriptor)
        from .review_storage import resolve_review_path
        path = resolve_review_path(descriptor["path"])
        before = path.stat()
        key = (str(path.resolve()).casefold(), before.st_size, before.st_mtime_ns)
        expected = descriptor["sha256"]
        cached = self.verified.get(key)
        if cached is not None and cached != expected:
            raise ValueError("Asset has conflicting SHA bindings")
        if cached is None:
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Scientific input changed during verification")
            if digest != expected:
                raise ValueError(f"Scientific asset SHA differs: {path}")
            self.verified[key] = digest
        return path

    def array(self, descriptor: Mapping) -> np.ndarray:
        if self.asset_reader is not None:
            return self.asset_reader.array(descriptor)
        return np.load(self.path(descriptor), mmap_mode="r", allow_pickle=False)

    def json(self, descriptor: Mapping) -> dict:
        if self.asset_reader is not None:
            return self.asset_reader.json(descriptor)
        path = self.path(descriptor)
        key = (str(path.resolve()).casefold(), descriptor["sha256"])
        if key not in self.json_documents:
            self.json_documents[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        return self.json_documents[key]
