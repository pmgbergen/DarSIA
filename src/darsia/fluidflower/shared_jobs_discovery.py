"""Associate authenticated, own-series publications with the local Jobs list."""

from __future__ import annotations

import hashlib
import ntpath
from pathlib import Path

from .shared_job_folder import SharedJobFolder, _bytes, _decode, _JOB_LIMIT


def _root_key(value):
    return ntpath.normcase(ntpath.normpath(str(value)))


def discover_published_jobs(
    folder, series_root, series_id, registry, *, include_archived=False
):
    """Return local Jobs rows after discovering this series' published jobs.

    Only the selected folder's bounded publication index and its authenticated
    metadata are read. A publication must declare the exact selected series ID;
    missing or foreign IDs never establish an association. Existing local rows,
    including archived rows, retain their labels, archive state and selection.
    Scientific job metadata is never written. Invalid publication bindings raise
    rather than establishing an unauthenticated local association.
    """
    if not isinstance(series_id, str) or not series_id.strip():
        return registry.jobs(series_root, series_id, include_archived=include_archived)
    if folder is None or not str(folder).strip():
        return registry.jobs(series_root, series_id, include_archived=include_archived)
    if not isinstance(folder, SharedJobFolder) and not Path(folder).exists():
        return registry.jobs(series_root, series_id, include_archived=include_archived)
    shared = folder if isinstance(folder, SharedJobFolder) else SharedJobFolder(folder)
    publications = shared.jobs()
    state = registry.context(series_root, series_id)
    known_roots = {_root_key(row["root"]) for row in state["jobs"]}
    seen_publications = set()
    for publication in publications:
        root = Path(publication["job_root"])
        key = _root_key(root)
        identifier = publication["job_id"]
        if key in known_roots or identifier in seen_publications:
            continue
        seen_publications.add(identifier)
        # Parse the same bounded bytes whose digest the publication authenticates.
        raw = _bytes(root / "job.json", _JOB_LIMIT)
        binding = dict(path="job.json", sha256=hashlib.sha256(raw).hexdigest(),
                       size_bytes=len(raw))
        if binding != publication["job"]:
            raise ValueError("Published job metadata changed during discovery")
        metadata = _decode(raw)
        if metadata.get("series_id") != series_id:
            continue
        # Keep the established validation refusal visible for an own-series job.
        # Registration writes only the local UI association.
        registry.register_job(series_root, series_id, root, activate=False)
        known_roots.add(key)
    return registry.jobs(series_root, series_id, include_archived=include_archived)
