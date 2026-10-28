"""Explicit saved-review enrollment and scientific-source metadata validation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from .phase_saved_review_metadata import configured_parameters
from .execution import default_execution_cache_root
from .phase_validation import _model_identity
from .phase_runtime import load_physics_profile, physics_runtime_identity
from .phase_workflow import discover_phase_review_sessions
from .project import sha256_file
from .review_storage import resolve_review_path
from . import phase_saved_transfer_review as transfer


def _parameters():
    return configured_parameters("saved_training_promotions")


def _typed(item):
    from .phase_saved_review_metadata import typed_saved_data_value
    return typed_saved_data_value(item)


def _literal(key):
    return _typed(_parameters()["literals"][key])


def _global(name):
    return _typed(_parameters()["globals"][name])


def _read(path):
    return json.loads(Path(path).read_text(encoding=_literal('_read:30:52')))

def promotion_root(series=None):
    """Where training enrollments are kept.

    A series whose series.toml sets [paths] phase_model keeps them in its
    training_enrollments folder; all others use the host's execution cache.
    """
    if series:
        from .series_paths import phase_model_root, series_folder_path, series_root
        try:
            root = series_root(series)
        except (OSError, ValueError):
            root = None
        if root is not None and phase_model_root(root) is not None:
            return series_folder_path(root, "training_enrollments")
    return default_execution_cache_root().resolve() / _literal('promotion_root:34:54')


def _promotion_roots():
    """The host's enrollment folder and those of the PhaseModel series."""
    from .series_paths import configured_series
    roots = [promotion_root()]
    for series, _label, _root in configured_series():
        folder = promotion_root(series)
        if folder not in roots:
            roots.append(folder)
    return roots

def _command_value(command, flag):
    if command.count(flag) != _literal('_command_value:38:30'):
        raise ValueError(f'Completed review command must specify {flag} exactly once')
    index = command.index(flag)
    if index + _literal('_command_value:41:15') >= len(command):
        raise ValueError(f'Missing completed review command value: {flag}')
    return str(command[index + _literal('_command_value:43:31')])

def _selection(evaluation_root, model_root, reviewer=None, save_root=None):
    root, model = (Path(evaluation_root).resolve(), Path(model_root).resolve())
    plan_path, binding_path = (root / _literal('_selection:48:37'), root / _literal('_selection:48:68'))
    plan, result = (_read(plan_path), _read(root / _literal('_selection:49:50')))
    if result.get(_literal('_selection:50:18')) != _literal('_selection:50:30'):
        raise ValueError(_literal('_selection:51:25'))
    identity = _model_identity(model)
    if (plan.get(_literal('_selection:53:17')), plan.get(_literal('_selection:53:46'))) != identity:
        raise ValueError(_literal('_selection:54:25'))
    if (result.get(_literal('_selection:55:19')), result.get(_literal('_selection:55:43'))) != identity:
        raise ValueError(_literal('_selection:56:25'))
    if Path(plan[_literal('_selection:57:17')]).resolve() != root or Path(result[_literal('_selection:57:70')]).resolve() != root:
        raise ValueError(_literal('_selection:58:25'))
    if plan.get(_literal('_selection:59:16')) != _literal('_selection:59:34') or result.get(_literal('_selection:59:58')) != _literal('_selection:59:76'):
        raise ValueError(_literal('_selection:60:25'))
    if plan.get(_literal('_selection:61:16')) not in {_literal('_selection:61:40'), _literal('_selection:61:59')}:
        raise ValueError(_literal('_selection:62:25'))
    source_runtime = physics_runtime_identity(load_physics_profile(model))
    if source_runtime is _literal('_selection:64:25') or source_runtime != plan.get(_literal('_selection:64:60')):
        raise ValueError(_literal('_selection:65:25'))
    command = result.get(_literal('_selection:66:25')) or result.get(_literal('_selection:66:57')) or []
    actor = _command_value(command, _literal('_selection:67:36')).strip()
    saved = Path(_command_value(command, _literal('_selection:68:41'))).resolve()
    session_id = _command_value(command, _literal('_selection:69:41'))
    if not actor or not session_id:
        raise ValueError(_literal('_selection:71:25'))
    if reviewer is not _literal('_selection:72:23') and str(reviewer).strip().casefold() != actor.casefold():
        raise ValueError(f'This saved review belongs to {actor}; select its reviewer before including it')
    if save_root is not _literal('_selection:74:24') and Path(save_root).resolve() != saved:
        raise ValueError(_literal('_selection:75:25'))
    combined = root / _literal('_selection:76:22')
    if Path(_command_value(command, _literal('_selection:77:36'))).resolve() != combined:
        raise ValueError(_literal('_selection:78:25'))
    if _command_value(command, _literal('_selection:79:31')) != plan[_literal('_selection:79:60')]:
        raise ValueError(_literal('_selection:80:25'))
    if Path(_command_value(command, _literal('_selection:81:36'))).resolve() != plan_path:
        raise ValueError(_literal('_selection:82:25'))
    targets = [row for row in transfer.target_sources(plan[_literal('_selection:83:59')], plan[_literal('_selection:83:84')]) if row[_literal('_selection:84:22')] == plan[_literal('_selection:84:37')] and Path(row[_literal('_selection:84:57')]).resolve() == Path(plan[_literal('_selection:85:28')]).resolve()]
    if len(targets) != _literal('_selection:86:23'):
        raise ValueError(_literal('_selection:87:25'))
    target = targets[_literal('_selection:88:21')]
    expected = plan[_literal('_selection:89:20')].get(str(Path(target[_literal('_selection:89:60')]).resolve()))
    if not expected or target[_literal('_selection:90:30')] != expected:
        raise ValueError(_literal('_selection:91:25'))
    source_model_root = Path(plan.get(_literal('_selection:95:38')) or model).resolve()
    record = dict(format=_global('FORMAT'), source_model_id=identity[_literal('_selection:96:58')], source_model_sha256=identity[_literal('_selection:96:91')], source_model_root=str(source_model_root), evaluation_root=str(root), evaluation_model_id=plan[_literal('_selection:98:33')], evaluation_model_sha256=plan[_literal('_selection:99:37')], dataset=plan[_literal('_selection:100:21')], run=plan[_literal('_selection:100:42')], physics_mode=plan[_literal('_selection:100:68')], target_native_manifest=str(Path(target[_literal('_selection:101:47')]).resolve()), target_native_manifest_sha256=expected, source_manifest=str(Path(plan[_literal('_selection:103:38')]).resolve()), source_manifest_sha256=sha256_file(Path(plan[_literal('_selection:104:53')])), plan_path=str(plan_path), plan_sha256=sha256_file(plan_path), binding_path=str(binding_path), binding_sha256=sha256_file(binding_path), combined_root=str(combined), reviewer=actor, save_root=str(saved), review_session_id=session_id)
    return (record, plan, target)

def _key(record):
    body = {key: value for key, value in record.items() if key not in {_literal('_key:115:27'), _literal('_key:115:42'), _literal('_key:115:58'), _literal('_key:115:76'), _literal('_key:115:89')}}
    return hashlib.sha256(json.dumps(body, sort_keys=_literal('_key:116:53'), separators=(_literal('_key:116:71'), _literal('_key:116:76'))).encode()).hexdigest()

def _path(record):
    return promotion_root(record.get('dataset')) / (_key(record) + _literal('_path:120:46'))

def _validate_science(record, plan, target, parent):
    transfer.verify_hashes(plan[_literal('_validate_science:124:32')])
    transfer.validate_review_binding(plan, _read(record[_literal('_validate_science:125:56')]))
    from .phase_saved_transfer_context import prepared_target_physics
    if target.get(_literal('_validate_science:128:18'), {}).get(_literal('_validate_science:128:33')):
        source_digest = plan[_literal('_validate_science:129:29')].get(str(target[_literal('_validate_science:129:64')]))
        prepared_target_physics(Path(parent).resolve(), target, expected_source_manifest_sha256=source_digest)
    else:
        transfer.registered_target_physics(Path(parent).resolve(), target)

def registered_promotion(evaluation_root, model_root, reviewer=None, save_root=None):
    """Read a matching selection; full image validation happens before capture."""
    record, _, _ = _selection(evaluation_root, model_root, reviewer, save_root)
    path = _path(record)
    if not path.is_file():
        return _literal('registered_promotion:141:15')
    stored = _read(path)
    if any((stored.get(key) != value for key, value in record.items())) or _key(stored) != path.stem:
        raise ValueError(_literal('registered_promotion:144:25'))
    return dict(stored, receipt_path=str(path), receipt_sha256=sha256_file(path))

def register_promotion(evaluation_root, model_root, reviewer, save_root):
    """Select one editable visible session for future training, without fitting."""
    record, plan, target = _selection(evaluation_root, model_root, reviewer, save_root)
    _validate_science(record, plan, target, model_root)
    path = _path(record)
    path.parent.mkdir(parents=_literal('register_promotion:153:30'), exist_ok=_literal('register_promotion:153:45'))
    stored = dict(record, created_utc=datetime.now(timezone.utc).isoformat(), selection=_literal('register_promotion:155:18'), training_performed=_literal('register_promotion:156:27'))
    from darsia.fluidflower.tools.calibration_pairwise_review import _exclusive_path_lock
    with _exclusive_path_lock(path.with_suffix(_literal('register_promotion:158:47')), timeout_s=_literal('register_promotion:158:67'), stale_after_s=_literal('register_promotion:158:86')):
        if not path.exists():
            from .phase_saved_transfer_review import write_metadata as write
            write(path, stored)
        elif _key(_read(path)) != _key(record):
            raise ValueError(_literal('register_promotion:165:29'))
    return registered_promotion(evaluation_root, model_root, reviewer, save_root)

def validated_promotions(parent, selected, lineage):
    """Resolve explicitly included, editable reviews for detached input capture.

    Empty enrollments wait for a review to be saved. Unrelated experiments and
    actors remain excluded. A matching stale selection fails instead of quietly
    dropping the user's intended evidence.
    """
    output, seen_sessions = ([], set())
    for path in sorted(path for root in _promotion_roots() for path in root.glob(_literal('validated_promotions:177:45'))):
        stored = _read(path)
        if stored.get(_literal('validated_promotions:179:22')) != selected or stored.get(_literal('validated_promotions:179:59')) not in lineage:
            continue
        ancestor = lineage[stored[_literal('validated_promotions:181:34')]]
        if stored.get(_literal('validated_promotions:182:22')) != _global('FORMAT') or _key(stored) != path.stem:
            raise ValueError(f'Training selection has changed: {path}')
        if stored[_literal('validated_promotions:184:18')] != ancestor[_literal('validated_promotions:184:53')]:
            raise ValueError(_literal('validated_promotions:185:29'))
        record, plan, target = _selection(stored[_literal('validated_promotions:186:49')], ancestor[_literal('validated_promotions:186:78')], stored[_literal('validated_promotions:187:48')], stored[_literal('validated_promotions:187:68')])
        if any((stored.get(key) != value for key, value in record.items())):
            raise ValueError(f'Included experiment changed after selection: {path}')
        _validate_science(record, plan, target, parent)
        sessions = discover_phase_review_sessions([Path(stored[_literal('validated_promotions:191:63')])]) + discover_phase_review_sessions()
        for session in sessions:
            if session.get(_literal('validated_promotions:193:27')) != stored[_literal('validated_promotions:193:58')]:
                continue
            if session.get(_literal('validated_promotions:195:28')) != stored[_literal('validated_promotions:195:57')] or str(session.get(_literal('validated_promotions:196:39')) or _literal('validated_promotions:196:54')).casefold() != stored[_literal('validated_promotions:196:79')].casefold() or Path(session.get(_literal('validated_promotions:197:40')) or _literal('validated_promotions:197:60')).resolve() != Path(stored[_literal('validated_promotions:197:89')]) or (session.get(_literal('validated_promotions:198:35')) != _literal('validated_promotions:198:56')):
                raise ValueError(_literal('validated_promotions:199:33'))
            decisions = resolve_review_path(session[_literal('validated_promotions:200:52')], context=session.get(_literal('validated_promotions:200:90')))
            selected_store = resolve_review_path(stored[_literal('validated_promotions:201:56')], context=session.get(_literal('validated_promotions:201:90')))
            if not decisions.resolve().is_relative_to(selected_store):
                raise ValueError(_literal('validated_promotions:203:33'))
            if not decisions.is_file():
                continue
            identity = str(decisions.resolve()).casefold()
            if identity in seen_sessions:
                continue
            seen_sessions.add(identity)
            output.append(dict(record, receipt_path=str(path), receipt_sha256=sha256_file(path), target=target, session=session, decision_path=str(decisions), combined_root=stored[_literal('validated_promotions:212:37')]))
    return output

def __getattr__(name):
    if name != "FORMAT":
        raise AttributeError(name)
    return _global("FORMAT")
