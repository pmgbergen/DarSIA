"""Opt-in, context-local SHA verification reuse for immutable input files only.

Outputs are never cached. Every source reuse checks its filesystem fingerprint;
every successful scope exit performs another actual SHA read of every cached
source. The scope must seal before its caller accepts aggregate output receipts.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

CONTRACT = "immutable-source-sha-worker-cache-v1"
_ACTIVE: ContextVar[ImmutableSourceHashCache | None] = ContextVar(
    "fluidflower_immutable_source_sha_cache", default=None)


def _key(path: Path) -> tuple[str, Path]:
    absolute = Path(path).expanduser().resolve(strict=True)
    return os.path.normcase(str(absolute)), absolute


def _fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            getattr(stat, "st_dev", None), getattr(stat, "st_ino", None),
            getattr(stat, "st_birthtime_ns", None))


def policy_identity() -> dict:
    """Execution identity; never enters optical features or fitted parameters."""
    return dict(contract=CONTRACT, opt_in=True,
                allowlist="exact-frozen-row-sources-only",
                output_sha_cache_allowed=False,
                first_verification="actual-full-file-sha256",
                every_reuse="absolute-path-size-mtime_ns-ctime_ns-device-fileid-fingerprint",
                seal="actual-full-file-sha256-and-unchanged-fingerprint-before-worker-acceptance",
                scope="materializer-worker-context-only")


@dataclass
class _Verified:
    path: Path
    digest: str
    fingerprint: tuple
    hasher: Callable[[Path], str]


class ImmutableSourceHashCache:
    def __init__(self, allowed_sources: Mapping[Path, str | None], *,
                 excluded_roots: Sequence[Path] = ()):
        self.excluded_roots = tuple(Path(root).expanduser().resolve() for root in excluded_roots)
        self.allowed: dict[str, str | None] = {}
        for path, digest in allowed_sources.items():
            key, absolute = _key(path)
            if any(absolute.is_relative_to(root) for root in self.excluded_roots):
                continue
            if digest is not None and (len(str(digest)) != 64 or
                    any(char not in "0123456789abcdef" for char in str(digest).lower())):
                raise ValueError(f"Invalid frozen source SHA: {absolute}")
            normalized = str(digest).lower() if digest is not None else None
            if key in self.allowed and self.allowed[key] not in (None, normalized) and normalized is not None:
                raise ValueError(f"Contradictory frozen source SHA: {absolute}")
            if key not in self.allowed or normalized is not None:
                self.allowed[key] = normalized
        self.verified: dict[str, _Verified] = {}
        self.cache_hits = 0
        self.avoided_read_bytes = 0
        self.sealed = False

    def hash(self, path: Path, hasher: Callable[[Path], str]) -> str:
        # Unlisted outputs and module artifacts retain the normal full SHA read.
        key, absolute = _key(path)
        if key not in self.allowed:
            return hasher(absolute)
        if self.sealed:
            raise ValueError("Immutable source hash scope has already been sealed")
        before = _fingerprint(absolute)
        saved = self.verified.get(key)
        if saved is not None:
            if before != saved.fingerprint:
                raise ValueError(f"Immutable source changed during worker scope: {absolute}")
            self.cache_hits += 1
            self.avoided_read_bytes += int(before[0])
            return saved.digest
        digest = hasher(absolute)
        if _fingerprint(absolute) != before:
            raise ValueError(f"Immutable source changed while hashing: {absolute}")
        expected = self.allowed[key]
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen source SHA mismatch: {absolute}")
        self.verified[key] = _Verified(absolute, digest, before, hasher)
        return digest

    def seal(self) -> None:
        """A source modified without observable timestamp changes still fails."""
        if self.sealed:
            return
        for saved in self.verified.values():
            if _fingerprint(saved.path) != saved.fingerprint:
                raise ValueError(f"Immutable source changed before SHA seal: {saved.path}")
            actual = saved.hasher(saved.path)
            if (actual != saved.digest or _fingerprint(saved.path) != saved.fingerprint):
                raise ValueError(f"Immutable source SHA seal failed: {saved.path}")
        self.sealed = True

    def receipt(self) -> dict:
        return dict(policy_identity(), allowed_sources=len(self.allowed),
                    cached_sources=len(self.verified), cache_hits=self.cache_hits,
                    avoided_source_read_bytes=self.avoided_read_bytes,
                    sealed_actual_sha_rechecks=self.sealed)


def active_source_hash_cache() -> ImmutableSourceHashCache | None:
    return _ACTIVE.get()


@contextmanager
def immutable_source_hash_cache(allowed_sources: Mapping[Path, str | None], *,
                                excluded_roots: Sequence[Path] = ()):
    if _ACTIVE.get() is not None:
        raise ValueError("Nested immutable source SHA scopes are not supported")
    cache = ImmutableSourceHashCache(allowed_sources, excluded_roots=excluded_roots)
    token = _ACTIVE.set(cache)
    try:
        yield cache
    finally:
        # Reset first so seal reads cannot accidentally reuse cached digests.
        _ACTIVE.reset(token)
        cache.seal()
