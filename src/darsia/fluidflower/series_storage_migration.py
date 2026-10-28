"""Reviewed same-filesystem Results relocation through shared series settings.

Only the directory entry and editable location settings change. Saved products,
their immutable metadata and declared content digests keep their original bytes.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import uuid

from .series_locations import read_series_locations, save_series_locations


def _absolute(path):
    path = Path(path).expanduser()
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Choose an absolute storage folder without parent traversal')
    return Path(os.path.abspath(path))


def _directory(path):
    path = _absolute(path)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_reparse_tag', 0)
            or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise ValueError('Storage moves require ordinary directories: ' + str(path))
    return info.st_dev, info.st_ino


def _empty(path):
    with os.scandir(path) as entries:
        return next(entries, None) is None


def preview_results_migration(series_root, project_folder, results_folder, *,
                              expected_bytes=None, replace_custom_results=False):
    """Review locations using only directory and configuration metadata.

    A same-volume rename does not need to enumerate or hash the payload tree.
    Cross-volume copies are deliberately outside this operation.
    """
    root, project, target = map(_absolute, (series_root, project_folder, results_folder))
    current = read_series_locations(root)
    if expected_bytes is not None and current['manifest_bytes'] != expected_bytes:
        raise ValueError('Series locations changed; load and review them again')
    source = _absolute(current['results_folder'])
    if source == target:
        raise ValueError('The selected Results folder is already in use')
    if (source == root or root.is_relative_to(source) or target == root
            or root.is_relative_to(target) or target.is_relative_to(source)):
        raise ValueError('Results relocation must not move or contain the setup folder')
    source_id, parent_id = _directory(source), _directory(target.parent)
    _directory(project)
    if source_id[0] != parent_id[0]:
        raise ValueError('Results must stay on the same filesystem for a rename without copying')
    target_id = None
    if target.exists():
        target_id = _directory(target)
        if not _empty(target):
            raise ValueError('The chosen Results folder contains files; no existing data will be replaced')
    return dict(format='darsia-results-location-review-v1', series_root=str(root),
        series_id=current['series_id'], project_folder=str(project),
        old_results_folder=str(source), new_results_folder=str(target),
        source_identity=source_id, parent_identity=parent_id, target_identity=target_id,
        manifest_bytes=current['manifest_bytes'], replace_custom_results=bool(replace_custom_results),
        payload_reads=0, payload_copies=0)


def apply_results_migration(plan, *, writers_quiescent):
    """Apply a reviewed rename and the normal shared-location writer.

    An editable-settings failure restores the source directory. The settings
    writer is responsible for restoring its own successfully written documents.
    """
    if writers_quiescent is not True:
        raise ValueError('Close jobs and other writers before moving Results')
    if plan.get('format') != 'darsia-results-location-review-v1':
        raise ValueError('Review the Results move first')
    root = _absolute(plan['series_root'])
    source, target = map(_absolute, (plan['old_results_folder'], plan['new_results_folder']))
    reviewed = preview_results_migration(root, plan['project_folder'], target,
        expected_bytes=plan['manifest_bytes'], replace_custom_results=plan['replace_custom_results'])
    if reviewed != plan:
        raise ValueError('Results locations changed after review; preview the move again')
    lock = root / '.results-location.lock'
    token = uuid.uuid4().hex
    try:
        with lock.open('x', encoding='ascii') as stream:
            stream.write(token)
    except FileExistsError as error:
        raise ValueError('Another Results move is in progress') from error
    removed_empty = False
    moved = False
    try:
        if _directory(source) != tuple(plan['source_identity']):
            raise ValueError('The reviewed Results directory changed')
        if _directory(target.parent) != tuple(plan['parent_identity']):
            raise ValueError('The selected storage parent changed')
        if plan['target_identity'] is not None:
            if _directory(target) != tuple(plan['target_identity']) or not _empty(target):
                raise ValueError('The reviewed empty destination changed')
            target.rmdir()
            removed_empty = True
        elif target.exists():
            raise ValueError('The destination was created after the move review')
        # pathlib.rename never falls back to a recursive cross-volume copy.
        source.rename(target)
        moved = True
        if _directory(target) != tuple(plan['source_identity']):
            raise ValueError('The renamed Results directory has another identity')
        saved = save_series_locations(root, plan['project_folder'], target,
            expected_bytes=plan['manifest_bytes'], preserve_inputs=False,
            replace_custom_results=plan['replace_custom_results'],
            path_aliases={str(source): str(target)})
    except BaseException:
        if moved:
            if source.exists() or _directory(target) != tuple(plan['source_identity']):
                raise RuntimeError('Results relocation needs recovery; moved data remain at ' + str(target))
            target.rename(source)
        if removed_empty and not target.exists():
            target.mkdir()
        raise
    finally:
        if lock.read_text(encoding='ascii') != token:
            raise ValueError('The Results move lock changed; it is preserved')
        lock.unlink()
    from .storage_paths import refresh_storage_paths
    refresh_storage_paths(root)
    from .review_storage import review_aliases
    review_aliases(refresh=True)
    return dict(moved=True, series_root=str(root), source=str(source), destination=str(target),
        updated_runs=saved['updated_runs'], retained_runs=saved['retained_runs'],
        payload_reads=0, payload_copies=0)
