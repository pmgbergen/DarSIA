"""Read-only series defaults for painting display, placement and analysis boxes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path, PureWindowsPath
import re

FORMAT = "darsia-series-display-frame-models-v1"
SCOPE = "display-placement-boxes"
MAX_INDEX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class FrameModelSelection:
    index_path: str
    source: str
    project_root: str
    project_id: str
    series_root: str
    series_id: str
    sha256: str = ""
    size_bytes: int = 0


def capture_frame_model_selection(project, *, override_path=""):
    """Capture only in-memory metadata; USER choice is an explicit override.

    A series default remains a series choice. It is never written back as a
    user override, and it grants no adoption or scientific geometry authority.
    """
    manifest = project.manifest
    tools = manifest.get("tools") or {}
    link = tools.get("native_series") or {}
    if not isinstance(link, Mapping):
        raise ValueError("The series link is unavailable. Reopen the series and retry.")
    context = dict(project_root=str(project.root),
        project_id=str(getattr(project, "project_id", manifest.get("project_id", ""))),
        series_root=str(link.get("series_root") or ""),
        series_id=str(link.get("series_id") or manifest.get("series_id") or ""))
    if override_path:
        path = str(override_path)
        if not Path(path).is_absolute():
            raise ValueError("The frame-model override needs an absolute index path. Use Frame models to change it.")
        return FrameModelSelection(path, "user override", **context)
    binding = link.get("display_frame_models")
    if binding is None:
        raise ValueError("The series has no measured frame models. Use Frame models to set an override.")
    if (not isinstance(binding, Mapping) or binding.get("format") != FORMAT
            or binding.get("scope") != SCOPE or binding.get("adopted") is not False):
        raise ValueError("The series frame-model binding is invalid. Retry after its display binding is corrected.")
    descriptor = binding.get("geometry_index")
    if not isinstance(descriptor, Mapping):
        raise ValueError("The series frame-model index binding is missing. Retry after its display binding is corrected.")
    relative, digest, size = (descriptor.get(key) for key in ("path", "sha256", "size_bytes"))
    if (not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive or relative.startswith(("/", "\\"))
            or ".." in PureWindowsPath(relative).parts
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int or not 0 < size <= MAX_INDEX_BYTES
            or not all(context.values())):
        raise ValueError("The series frame-model path, SHA, size or identity is invalid. Retry after its display binding is corrected.")
    return FrameModelSelection(str(Path(project.root) / relative), "series display binding",
        sha256=digest, size_bytes=size, **context)


def verify_frame_model_selection(selection):
    """Metadata-worker check of the immutable captured series descriptor."""
    if selection is None:
        return
    if not isinstance(selection, FrameModelSelection):
        raise ValueError("The captured frame-model selection is invalid. Reload the series and retry.")
    if not selection.sha256:
        return
    path = Path(selection.index_path)
    before = path.stat()
    with path.open("rb") as stream:
        raw = stream.read(MAX_INDEX_BYTES + 1)
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (version(before) != version(after) or len(raw) != selection.size_bytes
            or hashlib.sha256(raw).hexdigest() != selection.sha256):
        raise ValueError("The series frame-model index changed. Retry after its display binding is corrected.")


def read_adoption_cohort(index_path, registered_runs, reference_run):
    """Read only selected index metadata in the GUI worker, never model arrays.

    An explicit complete ordered cohort is authoritative for this GUI action.
    A legacy index without any cohort header retains the all-registered route.
    Partial declarations, intersections and inferred exclusions are refused.
    """
    from .pipeline.job_inputs import _json_bytes
    from .map_placement import GEOMETRY_FORMAT
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(index_path).resolve()
    reason = 'The rigs need their RAW basis before the map can be adopted'
    try:
        before = path.stat()
        with path.open('rb') as stream:
            raw = stream.read(MAX_INDEX_BYTES + 1)
        after = path.stat()
        version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) > MAX_INDEX_BYTES or version(before) != version(after):
            raise ValueError('The selected geometry index changed or exceeds its metadata bound')
        value = _json_bytes(raw)
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError(reason + '. ' + str(error)) from error
    if not isinstance(value, Mapping) or value.get('format') != GEOMETRY_FORMAT:
        raise ValueError(reason + '. Select one measured geometry index for adoption')

    def names(rows, label):
        """Compare explicit safe run identities without silently dropping duplicates."""
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError(label + ' must be an explicit nonempty ordered cohort')
        if any(not isinstance(run, str) or re.fullmatch(r'[a-z0-9][a-z0-9_-]*', run.casefold()) is None for run in rows):
            raise ValueError(label + ' contains an invalid run identity')
        result = [run.casefold() for run in rows]
        if len(result) != len(set(result)):
            raise ValueError(label + ' contains duplicate runs')
        return result

    registered = names(list(registered_runs), 'Registered runs')
    explicit = any(key in value for key in ('run_order', 'run_count'))
    if explicit:
        if not {'run_order', 'run_count', 'runs'} <= value.keys():
            raise ValueError('The selected index cohort header is incomplete')
        selected = names(value['run_order'], 'Selected index run_order')
        entries = value['runs']
        if not isinstance(entries, Mapping):
            raise ValueError('The selected index cohort needs its explicit run mapping')
        declared = names(list(entries), 'Selected index runs')
        if (type(value['run_count']) is not int or value['run_count'] != len(selected)
                or set(declared) != set(selected) or len(declared) != len(selected)
                or value.get('reference_models_complete') is not True
                or value.get('pending_runs')):
            raise ValueError('The selected index cohort header is inconsistent or incomplete')
        if not set(selected) <= set(registered):
            raise ValueError('The selected index cohort contains foreign unregistered runs')
    else:
        selected = list(registered)
    if str(reference_run).casefold() not in selected:
        raise ValueError('The selected index cohort does not include the map reference run')
    remaining = [run for run in registered if run not in set(selected)]
    saved_excluded = value.get('excluded_runs') or {}
    if isinstance(saved_excluded, list):
        saved_excluded = {row.get('run_id', row.get('run')): row
                          for row in saved_excluded if isinstance(row, Mapping)}
    if not isinstance(saved_excluded, Mapping):
        saved_excluded = {}
    excluded = []
    for run in remaining:
        saved = saved_excluded.get(run)
        reason = saved.get('reason') if isinstance(saved, Mapping) else saved
        excluded.append(dict(run_id=run, reason=reason if isinstance(reason, str) and reason.strip()
                             else 'Outside the selected adoption cohort'))
    summary = (f'Adoption scope: {len(selected)} of {len(registered)} registered runs. '
               + (f'{len(remaining)} remain outside this adoption: ' + ', '.join(remaining) + '.'
                  if remaining else 'No registered runs remain outside this adoption.'))
    return dict(runs=selected, registered=registered, remaining=remaining, excluded_runs=excluded, summary=summary,
        explicit=explicit, geometry_index=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))


def verify_adoption_cohort(cohort):
    """Recheck the captured metadata identity before/after the existing preparation."""
    from .project import sha256_file
    bound = cohort['geometry_index']
    path = Path(bound['path'])
    if path.stat().st_size != bound['size_bytes'] or sha256_file(path) != bound['sha256']:
        raise ValueError('The selected adoption cohort index changed; retry the action')
