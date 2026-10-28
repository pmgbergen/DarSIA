"""Explicit source-folder relocation for the selected calibration inputs."""
from copy import deepcopy
from pathlib import Path, PureWindowsPath
import ntpath


def overflow_source_folder(request, *, series_root):
    """Suggest only the common folder actually named by the selected sources."""
    from .mass.overflow_method import checked_metadata
    from .pipeline.current_inputs import CurrentInputReader
    value = checked_metadata(CurrentInputReader(series_roots=(series_root,)), request,
                             limit=1024 * 1024)
    paths = [str(PureWindowsPath(bound['path']).parent)
             for bound in value.get('scale_reviews', {}).values()]
    return ntpath.commonpath(paths) if paths else ''


def relocate_overflow_sources(series_root, request, source_root, destination_root):
    """Verify the unchanged sources, then save one shared locator alias.

    Called only by the explicit GUI action. No source is moved or rewritten;
    all original request, method and review SHA descriptors stay unchanged.
    """
    from .mass.n1_pipeline import DataReader
    from .mass.overflow_method import checked_metadata
    from .pipeline.current_inputs import CurrentInputReader
    from .pipeline.local_overflow_refit import read_request
    from .refit_job_panel import series_refit_roles
    from .series_locations import _read
    from .series_method import _write_manifest
    from .storage_paths import (_aliases, _normalized, _is_absolute, _resolve_with, machine_stored_text,
                                refresh_storage_paths)

    root, destination = Path(series_root), Path(destination_root).expanduser()
    source = _normalized(source_root)
    if not _is_absolute(source) or not destination.is_absolute() or not destination.is_dir():
        raise ValueError('Choose an absolute original source folder and an existing destination folder')
    original, document = _read(root / 'series.toml')
    prior = CurrentInputReader(series_roots=(root,))
    value = checked_metadata(prior, request, limit=1024 * 1024)
    bindings = [value['method'], *value['scale_reviews'].values()]
    # The method can also have been moved. Only explicitly named metadata is read.
    try:
        method = checked_metadata(prior, value['method'], limit=64 * 1024)
    except FileNotFoundError:
        method = None
    if method is not None:
        bindings += [method[key] for key in ('protocol', 'addendum', 'overflow_review')]
    aliases = []
    for bound in bindings:
        logical = _normalized(bound['path'])
        if logical.casefold().startswith(source.casefold().rstrip('/') + '/'):
            relative = logical[len(source.rstrip('/')) + 1:]
            physical = dict(bound, path=str(destination / Path(relative)))
            aliases.append(dict(logical=bound, physical=physical))
    if not aliases:
        raise ValueError('The selected request names no sources inside that original folder')
    reader = CurrentInputReader(DataReader(aliases), series_roots=(root,))
    if method is None:
        method = checked_metadata(reader, value['method'], limit=64 * 1024)
        for key in ('protocol', 'addendum', 'overflow_review'):
            bound = method[key]
            logical = _normalized(bound['path'])
            if logical.casefold().startswith(source.casefold().rstrip('/') + '/'):
                relative = logical[len(source.rstrip('/')) + 1:]
                aliases.append(dict(logical=bound,
                                    physical=dict(bound, path=str(destination / Path(relative)))))
        reader = CurrentInputReader(DataReader(aliases), series_roots=(root,))
    roles, _ = series_refit_roles(root)
    read_request(request, roles=roles, reader=reader)
    updated = deepcopy(document)
    updated.setdefault('storage', {}).setdefault('path_aliases', {})[source] = machine_stored_text(destination)
    rows = sorted(_aliases(updated['storage']['path_aliases'], root).items(),
                  key=lambda row: len(row[0]), reverse=True)
    for key, _ in rows:
        _resolve_with(key, rows)
    _write_manifest(root / 'series.toml', original, updated)
    refresh_storage_paths(root)
    return dict(request=request, verified_sources=len(aliases), source_root=source,
                destination_root=str(destination))
