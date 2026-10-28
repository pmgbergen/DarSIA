"""Installed-package paths for inherited FluidFlower workflow adapters.

Only source-file references retain the historical ``scripts/`` spelling in
immutable provenance records. Runtime configuration and products belong to a
series project, independently of the checkout and current working directory.
"""
from pathlib import Path
import importlib.util
import re
import sys

from darsia.fluidflower.series_paths import (
    local_cache_root, package_root, series_path, series_root, tool_path,
)


def source_path(relative):
    value = str(relative).replace('\\', '/')
    if value.startswith('scripts/'):
        if Path(value).suffix != '.py':
            asset = package_root() / 'tools' / Path(value).name
            if not asset.is_file():
                raise FileNotFoundError(asset)
            return asset
        return tool_path(Path(value).name)
    prefix = 'external/darsia/src/darsia/'
    if value.startswith(prefix):
        return package_root().parent / value[len(prefix):]
    raise ValueError(f'Unknown inherited source reference: {relative}')


def project_path(relative, dataset='ffac'):
    value = str(relative).replace('\\', '/')
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith(('scripts/', 'external/darsia/src/darsia/')):
        return source_path(value)
    for prefix, key in (('config_seg6/run_ac', 'run_configs'),
                        ('config_seg6', 'config'), ('config', 'config'),
                        ('protocols', 'protocols'), ('data', 'data'),
                        ('logs', 'results')):
        if value == prefix or value.startswith(prefix + '/'):
            return series_path(dataset, key) / value[len(prefix):].lstrip('/')
    return series_root(dataset) / path


def process_directory():
    return local_cache_root()


def canonical_metadata_paths(value):
    """Compare frozen metadata across verified storage relocations in memory.

    Only absolute filesystem strings are resolved. This never edits a receipt,
    changes its digest, or relaxes comparison of non-path scientific metadata.
    """
    from darsia.fluidflower.review_storage import resolve_review_path
    if isinstance(value, dict):
        return {key: canonical_metadata_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical_metadata_paths(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return str(resolve_review_path(value))
    return value


def load_snapshot_module(name, path):
    """Load an already hash-verified legacy snapshot using installed imports.

    The persisted bytes and their provenance stay unchanged. Only the old host
    namespace is rebound in memory; no scientific statements are transformed.
    Callers remain responsible for validating the recorded snapshot digest.
    """
    path = Path(path)
    source = path.read_text(encoding='utf-8-sig')
    source = re.sub(r'\bfrom scripts\b', 'from darsia.fluidflower.tools', source)
    source = re.sub(r'(?m)^(\s*)import scripts\s*$',
                    r'\1import darsia.fluidflower.tools as scripts', source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
