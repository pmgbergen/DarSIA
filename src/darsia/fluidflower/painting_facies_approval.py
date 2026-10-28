"""Read-only GUI comparison with the selected approved manual session.

This reads metadata only. It neither grants approval nor replaces the existing
full saved-map verifier, changes units, or computes a map.
"""
import hashlib
import json
from pathlib import Path

from . import layer_painting as lp
from .storage_paths import resolve_storage_path


def metadata_key(record, approval):
    """Identify the selected approval and current saved-metadata file stamps."""
    folder = resolve_storage_path(record["folder"])
    files = record.get("files")
    if not isinstance(files, dict):
        raise ValueError("The saved manual version does not list its files")
    stamps = []
    for leaf in ("version.json", "session.json"):
        st = (folder / leaf).stat()
        stamps.append((st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    return (str(folder), record.get("sha256"), record.get("display_name"), tuple(sorted(files.items())),
            json.dumps(approval, sort_keys=True), tuple(stamps))


def _read(path):
    """Read bounded metadata bytes and refuse a change during the read."""
    before = path.stat()
    if not path.is_file() or before.st_size > 16 * 1024**2:
        raise ValueError("The saved facies metadata is unavailable or too large")
    raw = path.read_bytes()
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("The saved facies metadata changed while being read")
    return raw


def approved_reference(record, approval):
    """Read the SHA-bound session of this exact approved manual version."""
    folder = resolve_storage_path(record["folder"])
    version = json.loads(_read(folder / "version.json"))
    if not isinstance(version, dict):
        raise ValueError("The saved version metadata is not an object")
    if (record.get("kind") != "manual map" or version.get("kind") != "manual map"
            or version.get("format") != lp.LAYER_VERSION_FORMAT
            or approval.get("kind") != "manual map"
            or version.get("sha256") != record.get("sha256")
            or version.get("files") != record.get("files")
            or approval.get("sha256") != version.get("sha256")
            or approval.get("version") != folder.name
            or approval.get("files") != version.get("files")
            or approval.get("format") != "darsia-fluidflower-layer-map-approval-v1"):
        raise ValueError("The selected manual version and its approval do not bind the same saved files")
    raw = _read(folder / "session.json")
    if hashlib.sha256(raw).hexdigest() != version["files"].get("session.json"):
        raise ValueError("The approved session metadata has changed")
    content = json.loads(raw)
    if not isinstance(content, dict):
        raise ValueError("The approved session metadata is not an object")
    session = lp.PaintingSession.from_dict(content)
    if session.run_id != version.get("run_id") or session.run_id != approval.get("run_id"):
        raise ValueError("The approved facies metadata belongs to another run")
    return dict(name=str(record.get("display_name") or version.get("name") or "selected version"), run_id=session.run_id,
        sha256=version["sha256"], unit=session.permeability_unit,
        facies={name: (entry.role, entry.porosity, entry.permeability)
                for name, entry in session.facies.items()})


def differences(session, reference):
    """Describe changed, added and removed facies without changing either input."""
    result = {}
    for name, entry in session.facies.items():
        before = reference["facies"].get(name)
        current = (entry.role, entry.porosity, entry.permeability)
        fields = (["facies added"] if before is None else
                  [field for field, old, new in zip(("role", "porosity", "permeability"), before, current)
                   if old != new])
        if session.permeability_unit != reference["unit"]:
            fields.append("unit")
        if fields:
            result[name] = fields
    for name in reference["facies"]:
        if name not in session.facies:
            result[name] = ["facies removed"]
    return result
