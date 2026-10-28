"""Hash-bound prepared target context validation; never prepare or infer."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import numpy as np
from .phase_saved_review_metadata import configured_parameters, read_json
from .phase_plume import PlumeContext
from .phase_runtime import load_physics_profile, physics_runtime_identity
from .phase_training.scope import series_id
from .project import sha256_file
from .review_storage import resolve_review_path


def _parameters():
    return configured_parameters("saved_transfer_context")


def _typed(item):
    from .phase_saved_review_metadata import typed_saved_data_value
    return typed_saved_data_value(item)


def _literal(key):
    return _typed(_parameters()["literals"][key])


def _global(name):
    return _typed(_parameters()["globals"][name])


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding=_literal('read:32:67')))

def resolve(path, origin):
    result = Path(path)
    return resolve_review_path(result if result.is_absolute() else origin / result)

def _same_historical_localcache_manifest(source: Path, bound: Path) -> bool:
    """Recognize the old packaged-app (MSIX) LocalCache spelling of one source path.

    This is intentionally limited to phase-transfer source manifests. It does
    not make arbitrary same-content files interchangeable as provenance.
    """
    for sandbox, ordinary in ((source, bound), (bound, source)):
        parts = sandbox.parts
        lower = tuple((part.casefold() for part in parts))
        for index in range(len(parts) - _literal('_same_historical_localcache_manifest:49:40')):
            if lower[index:index + _literal('_same_historical_localcache_manifest:50:35')] != (_literal('_same_historical_localcache_manifest:50:42'), _literal('_same_historical_localcache_manifest:50:53'), _literal('_same_historical_localcache_manifest:50:62')):
                continue
            if not lower[index + _literal('_same_historical_localcache_manifest:52:33')].startswith(_literal('_same_historical_localcache_manifest:52:47')):
                continue
            if lower[index + _literal('_same_historical_localcache_manifest:54:29'):index + _literal('_same_historical_localcache_manifest:54:39')] != (_literal('_same_historical_localcache_manifest:54:46'), _literal('_same_historical_localcache_manifest:54:60')):
                continue
            suffix = lower[index + _literal('_same_historical_localcache_manifest:56:35'):]
            if len(suffix) != _literal('_same_historical_localcache_manifest:57:31') or suffix[:_literal('_same_historical_localcache_manifest:57:44')] != (_literal('_same_historical_localcache_manifest:57:51'), _literal('_same_historical_localcache_manifest:57:61'), _literal('_same_historical_localcache_manifest:57:76'), _literal('_same_historical_localcache_manifest:57:89')) or suffix[-_literal('_same_historical_localcache_manifest:58:31')] != _literal('_same_historical_localcache_manifest:58:37'):
                continue
            expected = Path(*parts[:index + _literal('_same_historical_localcache_manifest:60:44')], *parts[index + _literal('_same_historical_localcache_manifest:60:63'):])
            if os.path.normcase(str(expected)) == os.path.normcase(str(ordinary)):
                return _literal('_same_historical_localcache_manifest:62:23')
    return _literal('_same_historical_localcache_manifest:63:11')

def prepared_target_physics(model_root, target, *, expected_source_manifest_sha256=None):
    """Validate registered target inputs without creating files or running inference.

    ``target.job.phase_transfer_context`` must contain the descriptor. Its
    provenance.target_binding repeats dataset/run/native SHA/source manifest,
    making the provenance explicitly belong to the selected experiment.
    """
    root = resolve_review_path(model_root)
    source_manifest = resolve_review_path(target[_literal('prepared_target_physics:74:49')])
    descriptor = copy.deepcopy(target.get(_literal('prepared_target_physics:75:42'), {}).get(_literal('prepared_target_physics:75:57')))
    if not isinstance(descriptor, dict) or descriptor.get(_literal('prepared_target_physics:76:58')) != _global('FORMAT'):
        raise ValueError(_literal('prepared_target_physics:77:25'))
    dataset, run = (str(target[_literal('prepared_target_physics:78:30')]), str(target[_literal('prepared_target_physics:78:54')]))
    if series_id(dataset) != dataset or dataset not in {_literal('prepared_target_physics:79:56'), _literal('prepared_target_physics:79:64'), _literal('prepared_target_physics:79:72')} or (not run) or (run != run.casefold()):
        raise ValueError(_literal('prepared_target_physics:80:25'))
    key = f'{dataset}/{run}'
    native_path = resolve_review_path(target[_literal('prepared_target_physics:82:45')])
    native_hash = sha256_file(native_path)
    expected_binding = dict(dataset=dataset, run=run, native_manifest_sha256=native_hash)
    if native_hash != target[_literal('prepared_target_physics:85:29')] or any((descriptor.get(k) != v for k, v in expected_binding.items())):
        raise ValueError(_literal('prepared_target_physics:86:25'))
    registered = [job for job in read(source_manifest).get(_literal('prepared_target_physics:87:59'), []) if isinstance(job, dict) and str(job.get(_literal('prepared_target_physics:88:34')) or _literal('prepared_target_physics:88:44')).casefold() == run and (job.get(_literal('prepared_target_physics:89:30')) == descriptor)]
    if not registered or any((target.get(_literal('prepared_target_physics:90:40'), {}).get(k) != v for k, v in ((_literal('prepared_target_physics:90:77'), dataset), (_literal('prepared_target_physics:90:99'), run)))):
        raise ValueError(_literal('prepared_target_physics:91:25'))
    if descriptor.get(_literal('prepared_target_physics:92:22')) != _global('FACIES_IDENTITY'):
        raise ValueError(_literal('prepared_target_physics:93:25'))
    provenance = descriptor.get(_literal('prepared_target_physics:94:32'))
    binding = provenance.get(_literal('prepared_target_physics:95:29')) if isinstance(provenance, dict) else _literal('prepared_target_physics:95:84')
    if not isinstance(binding, dict) or any((binding.get(k) != v for k, v in expected_binding.items())):
        raise ValueError(_literal('prepared_target_physics:97:25'))
    bound_value = Path(binding[_literal('prepared_target_physics:98:31')]) if binding.get(_literal('prepared_target_physics:98:66')) else _literal('prepared_target_physics:98:90')
    bound_manifest = resolve_review_path(bound_value if bound_value.is_absolute() else source_manifest.parent / bound_value, context=source_manifest.parent) if bound_value is not _literal('prepared_target_physics:101:44') else _literal('prepared_target_physics:101:54')
    if bound_manifest is _literal('prepared_target_physics:102:25'):
        raise ValueError(_literal('prepared_target_physics:103:25'))
    if expected_source_manifest_sha256 is not _literal('prepared_target_physics:104:47') and sha256_file(source_manifest) != expected_source_manifest_sha256:
        raise ValueError(_literal('prepared_target_physics:106:25'))
    if bound_manifest.resolve() != source_manifest.resolve():
        if not _same_historical_localcache_manifest(source_manifest, bound_manifest):
            raise ValueError(_literal('prepared_target_physics:109:29'))
        try:
            bound_manifest.stat()
        except FileNotFoundError:
            if expected_source_manifest_sha256 is _literal('prepared_target_physics:116:50'):
                raise ValueError(_literal('prepared_target_physics:117:33')) from _literal('prepared_target_physics:117:102')
        except OSError as error:
            raise ValueError(_literal('prepared_target_physics:119:29')) from error
        else:
            if not bound_manifest.is_file() or sha256_file(bound_manifest) != sha256_file(source_manifest):
                raise ValueError(_literal('prepared_target_physics:123:33'))
    context_path = resolve(descriptor[_literal('prepared_target_physics:124:38')], source_manifest.parent)
    if context_path.is_relative_to(root):
        raise ValueError(_literal('prepared_target_physics:126:25'))
    if sha256_file(context_path) != descriptor.get(_literal('prepared_target_physics:127:51')):
        raise ValueError(_literal('prepared_target_physics:128:25'))
    profile = load_physics_profile(root)
    profile_paths = [root / name for name in _global('PROFILE_FILES') if (root / name).is_file()]
    if profile is _literal('prepared_target_physics:131:18') or len(profile_paths) != _literal('prepared_target_physics:131:48'):
        raise ValueError(_literal('prepared_target_physics:132:25'))
    if {str(k).split(_literal('prepared_target_physics:133:21'))[_literal('prepared_target_physics:133:26')] for k in profile[_literal('prepared_target_physics:133:46')]} != {dataset}:
        raise ValueError(_literal('prepared_target_physics:134:25'))
    with np.load(context_path, allow_pickle=_literal('prepared_target_physics:135:44')) as archive:
        context = PlumeContext(domain=archive[_literal('prepared_target_physics:136:46')], facies=archive[_literal('prepared_target_physics:136:72')], x_m=archive[_literal('prepared_target_physics:137:24')], y_m=archive[_literal('prepared_target_physics:137:44')], injection_stages=tuple(copy.deepcopy(descriptor[_literal('prepared_target_physics:138:60')])), permeability=archive[_literal('prepared_target_physics:139:33')] if _literal('prepared_target_physics:139:52') in archive else _literal('prepared_target_physics:139:83'))
    if not np.issubdtype(context.facies.dtype, np.integer):
        raise ValueError(_literal('prepared_target_physics:141:25'))
    native = read(native_path)
    box = native[_literal('prepared_target_physics:143:17')]
    canvas = native[_literal('prepared_target_physics:144:20')]
    if len(box) != _literal('prepared_target_physics:145:20') or len(canvas) != _literal('prepared_target_physics:145:40') or any((type(value) is not int for value in (*box, *canvas))) or (min(box[:_literal('prepared_target_physics:146:24')]) < _literal('prepared_target_physics:146:30')) or (min(box[_literal('prepared_target_physics:146:43'):]) < _literal('prepared_target_physics:146:50')) or (box[_literal('prepared_target_physics:146:59')] + box[_literal('prepared_target_physics:146:68')] > canvas[_literal('prepared_target_physics:146:80')]) or (box[_literal('prepared_target_physics:147:19')] + box[_literal('prepared_target_physics:147:28')] > canvas[_literal('prepared_target_physics:147:40')]) or (context.domain.shape != (box[_literal('prepared_target_physics:147:75')], box[_literal('prepared_target_physics:147:83')])):
        raise ValueError(_literal('prepared_target_physics:148:25'))
    from darsia.fluidflower.phase_physics import PHASE_PHYSICS_VERSION, PhasePhysicsContext
    if profile[_literal('prepared_target_physics:152:15')] == PHASE_PHYSICS_VERSION:
        if context.permeability is _literal('prepared_target_physics:153:35'):
            raise ValueError(_literal('prepared_target_physics:154:29'))
        PhasePhysicsContext(context.domain, context.facies, context.permeability, context.x_m, context.y_m, context.injection_stages)
    source_files = [source_manifest, native_path, context_path, profile_paths[_literal('prepared_target_physics:157:78')]]
    source_files += extension_files(root)
    return dict(profile=profile, profile_path=profile_paths[_literal('prepared_target_physics:159:60')], source_runtime=physics_runtime_identity(profile), context_path=context_path, context=context, provenance=copy.deepcopy(provenance), key=key, source_files=source_files, source_sha256={str(path): sha256_file(path) for path in source_files}, descriptor=descriptor)

def extension_files(root):
    """Result extensions are separate bundle files, never inline base rules."""
    root = Path(root)
    files = [root / name for name in (_literal('extension_files:168:36'), _literal('extension_files:169:8'), _literal('extension_files:169:44')) if (root / name).is_file()]
    completion = root / _literal('extension_files:170:22')
    if completion.is_file():
        checkpoint = (root / read(completion)[_literal('extension_files:172:44')]).resolve()
        if not checkpoint.is_relative_to(root.resolve()):
            raise ValueError(_literal('extension_files:174:29'))
        files.append(checkpoint)
    regions = root / _literal('extension_files:176:21')
    if regions.is_file():
        model = read(regions)[_literal('extension_files:178:30')]
        checkpoint = (root / model[_literal('extension_files:179:35')]).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or sha256_file(checkpoint) != model[_literal('extension_files:180:93')]:
            raise ValueError(_literal('extension_files:181:29'))
        files.append(checkpoint)
    return files
