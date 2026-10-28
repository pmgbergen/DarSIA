"""Read-only saved-registration discovery and original review warnings.

Application DATA supplies every frozen declaration. Historical candidate builders
and command-line calculations are never imported or executed here.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from .native.io import sha256_file
from .review_storage import resolve_review_path
from .phase_saved_review_metadata import (
    configured_parameters, configured_registration_path as registration_path,
    configured_model_identity as _model_identity, bound_path, _SavedRigMetadata,
    _saved_profile_typed, saved_revision_bindings,
)

def _literal(parameters, index):
    return _saved_profile_typed(parameters["literals"][index])

def _global(parameters, name):
    item = parameters["globals"][name]
    value = _saved_profile_typed(item)
    for path in sorted(item.get("nested_tuple_paths", ()), key=len, reverse=True):
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = tuple(parent[path[-1]])
    return value

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def _require(value, message):
    if not value:
        raise ValueError(message)

def __getattr__(name):
    if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
    if name not in configured_parameters("registration")["public_exports"]:
        raise AttributeError(name)
    return _global(configured_parameters("registration")["track1"], name)

def _checked(value):
    path = Path(value['path'])
    if sha256_file(path) != value['sha256']:
        raise ValueError(f'Source artifact changed: {path}')
    return path

def _read_downstream(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected an object: {path}')
    return value

def _warning_seam(root, value):
    _p = configured_parameters('registration')['seam']
    try:
        root = Path(root)
        if value.get(_literal(_p, 0)) != _global(_p, 'FORMAT') or value.get(_literal(_p, 1)) != _global(_p, 'KIND') or [value.get(_literal(_p, 2)), value.get(_literal(_p, 3))] != list(_model_identity(root)) or ((value.get(_literal(_p, 4)), value.get(_literal(_p, 5))) != (_literal(_p, 6), _literal(_p, 7))) or (len(value.get(_literal(_p, 8), {})) != _literal(_p, 9)) or (len(value.get(_literal(_p, 10), {})) != _literal(_p, 11)) or (value.get(_literal(_p, 12)) is not _literal(_p, 13)) or (value.get(_literal(_p, 14)) is not _literal(_p, 15)) or any((value.get(key) is not _literal(_p, 16) for key in (_literal(_p, 17), _literal(_p, 18), _literal(_p, 19), _literal(_p, 20)))):
            raise ValueError(_literal(_p, 21))
        for key, name in _global(_p, 'FILES').items():
            if sha256_file(root / name) != value.get(key):
                raise ValueError(_literal(_p, 22))
        seal = _read(root / _global(_p, 'SEAL_FILE'))
        for key in (_literal(_p, 23), _literal(_p, 24), _literal(_p, 25), _literal(_p, 26), _literal(_p, 27), _literal(_p, 28), _literal(_p, 29)):
            bound_path(seal[key])
        return _global(_p, 'WARNING')
    except (OSError, ValueError, KeyError, TypeError):
        return _global(_p, 'INVALID')

def _warning_area(root, value):
    _p = configured_parameters('registration')['area']
    try:
        root = Path(root)
        if value.get(_literal(_p, 0)) != _global(_p, 'FORMAT') or value.get(_literal(_p, 1)) != _global(_p, 'KIND') or [value.get(_literal(_p, 2)), value.get(_literal(_p, 3))] != list(_model_identity(root)) or ((value.get(_literal(_p, 4)), value.get(_literal(_p, 5))) != (_literal(_p, 6), _literal(_p, 7))) or (len(value.get(_literal(_p, 8), {})) != _literal(_p, 9)) or (len(value.get(_literal(_p, 10), {})) != _literal(_p, 11)) or (value.get(_literal(_p, 12)) is not _literal(_p, 13)) or (value.get(_literal(_p, 14)) is not _literal(_p, 15)) or any((value.get(k) is not _literal(_p, 16) for k in (_literal(_p, 17), _literal(_p, 18), _literal(_p, 19), _literal(_p, 20)))):
            raise ValueError(_literal(_p, 21))
        for key, name in _global(_p, 'FILES').items():
            if sha256_file(root / name) != value.get(key):
                raise ValueError(_literal(_p, 22))
        seal = _read(root / _literal(_p, 23))
        _checked(seal[_literal(_p, 24)])
        _checked(seal[_literal(_p, 25)])
        _checked(seal[_literal(_p, 26)])
        if not seal[_literal(_p, 27)] or (seal[_literal(_p, 28)], seal[_literal(_p, 29)]) != (_literal(_p, 30), _literal(_p, 31)):
            raise ValueError(_literal(_p, 32))
        _checked(seal[_literal(_p, 33)])
        return _global(_p, 'WARNING')
    except (OSError, ValueError, KeyError, TypeError):
        return _literal(_p, 34)

def _validate_rig_qa(report):
    _p = configured_parameters('registration')['rig']
    metadata = _SavedRigMetadata(configured_parameters('rig'))
    metadata.require(report.get(_literal(_p, 0)) == _global(_p, 'metadata.QA_CONTRACT') and report.get(_literal(_p, 1)) is _literal(_p, 2) and (report.get(_literal(_p, 3)) == _global(_p, 'SPEC_SHA256')) and (report.get(_literal(_p, 4)) == _global(_p, 'metadata.CONFIGURATION')) and (report.get(_literal(_p, 5)) == _literal(_p, 6)) and (report.get(_literal(_p, 7)) == _literal(_p, 8)) and (report.get(_literal(_p, 9)) == _global(_p, 'G3_SHA256')) and (report.get(_literal(_p, 10)) == _global(_p, 'REFIT_SHA256')) and (report.get(_literal(_p, 11)) == _global(_p, 'CO2_SHA256')) and (report.get(_literal(_p, 12)) is _literal(_p, 13)), _literal(_p, 14))
    rows = report.get(_literal(_p, 15)) or []
    keys = {(row.get(_literal(_p, 16)), row.get(_literal(_p, 17))) for row in rows}
    grid = metadata.validate_phase_grid(report.get(_literal(_p, 18), metadata.phase_grid()))
    metadata.require(len(rows) == len(keys) == _literal(_p, 19) and len({run for run, _ in keys}) == _literal(_p, 20), _literal(_p, 21))
    for row in rows:
        error = row.get(_literal(_p, 22))
        metadata.require(isinstance(error, (int, float)) and math.isfinite(error) and (abs(error) <= _literal(_p, 23)), _literal(_p, 24))
        metadata.require(type(row.get(_literal(_p, 25))) is bool and (row[_literal(_p, 26)] or (row.get(_literal(_p, 27)) == _literal(_p, 28) and row.get(_literal(_p, 29)) is _literal(_p, 30))), _literal(_p, 31))
        metadata.require(row.get(_literal(_p, 32)) == grid, _literal(_p, 33))
        for key in (_literal(_p, 34), _literal(_p, 35), _literal(_p, 36), _literal(_p, 37), _literal(_p, 38), _literal(_p, 39), _literal(_p, 40)):
            metadata.require(isinstance(row.get(key), str) and metadata._sha.fullmatch(row[key]), _literal(_p, 41))
    return report

def _reports_rig(model):
    _p = configured_parameters('registration')['rig']
    metadata = _SavedRigMetadata(configured_parameters('rig'))
    report = _validate_rig_qa(_read(model / _global(_p, 'FILES')[_literal(_p, 42)]))
    acceptance = _read(bound_path(report[_literal(_p, 43)], base=model))
    metadata.validate_geometry_acceptance(acceptance, geometry_sha256=report[_literal(_p, 44)], spec_sha256=_global(_p, 'SPEC_SHA256'))
    for name in (_literal(_p, 45), _literal(_p, 46)):
        bound_path(acceptance[name], base=model)
    labels = _read(bound_path(report[_literal(_p, 47)], base=model))
    metadata.require(labels.get(_literal(_p, 48)) is _literal(_p, 49) and labels.get(_literal(_p, 50)) == _literal(_p, 51), _literal(_p, 52))
    return report

def _warning_rig(model, value):
    _p = configured_parameters('registration')['rig']
    metadata = _SavedRigMetadata(configured_parameters('rig'))
    try:
        model = resolve_review_path(model).resolve()
        metadata.require(value.get(_literal(_p, 53)) == _global(_p, 'FORMAT') and value.get(_literal(_p, 54)) == _global(_p, 'KIND') and resolve_review_path(value[_literal(_p, 55)]).samefile(model) and ((value.get(_literal(_p, 56)), value.get(_literal(_p, 57))) == _model_identity(model)) and (value.get(_literal(_p, 58)) == _literal(_p, 59)) and (value.get(_literal(_p, 60)) == _literal(_p, 61)) and (value.get(_literal(_p, 62)) == _global(_p, 'SPEC_SHA256')) and (value.get(_literal(_p, 63)) == _literal(_p, 64)) and (value.get(_literal(_p, 65)) == _literal(_p, 66)) and (len(value.get(_literal(_p, 67)) or {}) == _literal(_p, 68)) and (len(value.get(_literal(_p, 69)) or {}) == _literal(_p, 70)) and all((value.get(key) is _literal(_p, 71) for key in (_literal(_p, 72), _literal(_p, 73), _literal(_p, 74), _literal(_p, 75)))), _literal(_p, 76))
        for key, name in _global(_p, 'FILES').items():
            metadata.require(sha256_file(model / name) == value.get(key), _literal(_p, 77))
        _reports_rig(model)
        return _global(_p, 'WARNING')
    except (OSError, ValueError, KeyError, TypeError):
        return _global(_p, 'INVALID')

def _reports_barrier(model: Path) -> dict:
    _p = configured_parameters('registration')['barrier']
    report = _read(model / _global(_p, 'FILES')[_literal(_p, 0)])
    _require(report.get(_literal(_p, 1)) == _global(_p, 'QA_CONTRACT') and report.get(_literal(_p, 2)) is _literal(_p, 3) and (report.get(_literal(_p, 4)) == _global(_p, 'CONFIG')) and (report.get(_literal(_p, 5)) == _global(_p, 'SPEC_SHA')) and (report.get(_literal(_p, 6)) == _global(_p, 'G3_SHA')) and (report.get(_literal(_p, 7)) == _global(_p, 'REGIONAL_SHA')) and (report.get(_literal(_p, 8)) == _global(_p, 'CO2_SHA256')) and (report.get(_literal(_p, 9)) == _literal(_p, 10)) and (report.get(_literal(_p, 11)) == _literal(_p, 12)), _literal(_p, 13))
    rows = report.get(_literal(_p, 14)) or []
    keys = {(row.get(_literal(_p, 15)), row.get(_literal(_p, 16))) for row in rows}
    _require(len(rows) == len(keys) == _literal(_p, 17) and len({run for run, _ in keys}) == _literal(_p, 18), _literal(_p, 19))
    for row in rows:
        _require(row.get(_literal(_p, 20)) is _literal(_p, 21) and row.get(_literal(_p, 22)) is _literal(_p, 23), _literal(_p, 24))
        before, after, removed = (row.get(key) for key in (_literal(_p, 25), _literal(_p, 26), _literal(_p, 27)))
        _require(all((type(value) is int for value in (before, after, removed))) and _literal(_p, 28) <= after <= before and (removed == before - after), _literal(_p, 29))
        _require(isinstance(row.get(_literal(_p, 30)), str) and len(row[_literal(_p, 31)]) == _literal(_p, 32) and (row[_literal(_p, 33)] == row.get(_literal(_p, 34))), _literal(_p, 35))
        area = row.get(_literal(_p, 36))
        _require(isinstance(area, (int, float)) and _literal(_p, 37) < area < float(_literal(_p, 38)), _literal(_p, 39))
        _require(isinstance(row.get(_literal(_p, 40)), list) and isinstance(row.get(_literal(_p, 41)), list), _literal(_p, 42))
    return report

def _warning_barrier(model: Path, value: dict) -> str:
    _p = configured_parameters('registration')['barrier']
    try:
        _require(value.get(_literal(_p, 43)) == _global(_p, 'FORMAT') and value.get(_literal(_p, 44)) == _global(_p, 'KIND') and resolve_review_path(value[_literal(_p, 45)]).samefile(model) and ((value.get(_literal(_p, 46)), value.get(_literal(_p, 47))) == _model_identity(model)) and (value.get(_literal(_p, 48)) == _literal(_p, 49)) and (value.get(_literal(_p, 50)) == _literal(_p, 51)) and (value.get(_literal(_p, 52)) == _global(_p, 'CONFIG')) and (value.get(_literal(_p, 53)) == _global(_p, 'SPEC_SHA')) and (value.get(_literal(_p, 54)) == _literal(_p, 55)) and (value.get(_literal(_p, 56)) == _literal(_p, 57)) and (len(value.get(_literal(_p, 58)) or {}) == _literal(_p, 59)) and (len(value.get(_literal(_p, 60)) or {}) == _literal(_p, 61)) and all((value.get(key) is _literal(_p, 62) for key in (_literal(_p, 63), _literal(_p, 64), _literal(_p, 65), _literal(_p, 66), _literal(_p, 67)))), _literal(_p, 68))
        for key, relative in _global(_p, 'FILES').items():
            _require(sha256_file(model / relative) == value.get(key), _literal(_p, 69))
        _reports_barrier(model)
        return _global(_p, 'WARNING')
    except (OSError, ValueError, TypeError, KeyError):
        return _global(_p, 'INVALID')

def _reports_top(model: Path) -> dict:
    _p = configured_parameters('registration')['top']
    report = _read(model / _global(_p, 'FILES')[_literal(_p, 0)])
    _require(report.get(_literal(_p, 1)) == _global(_p, 'QA_CONTRACT') and report.get(_literal(_p, 2)) is _literal(_p, 3) and (report.get(_literal(_p, 4)) == _global(_p, 'CONFIG')) and (report.get(_literal(_p, 5)) == _global(_p, 'SPEC_SHA')) and (report.get(_literal(_p, 6)) == _global(_p, 'G3_SHA')) and (report.get(_literal(_p, 7)) == _global(_p, 'REGIONAL_SHA')) and (report.get(_literal(_p, 8)) == _global(_p, 'CO2_SHA256')) and (report.get(_literal(_p, 9)) == _literal(_p, 10)) and (report.get(_literal(_p, 11)) == _literal(_p, 12)), _literal(_p, 13))
    rows = report.get(_literal(_p, 14)) or []
    keys = {(row.get(_literal(_p, 15)), row.get(_literal(_p, 16))) for row in rows}
    _require(len(rows) == len(keys) == _literal(_p, 17) and len({run for run, _ in keys}) == _literal(_p, 18), _literal(_p, 19))
    for row in rows:
        _require(all((row.get(key) == _literal(_p, 20) for key in (_literal(_p, 21), _literal(_p, 22), _literal(_p, 23), _literal(_p, 24), _literal(_p, 25)))) and all((row.get(key) is _literal(_p, 26) for key in (_literal(_p, 27), _literal(_p, 28), _literal(_p, 29)))), _literal(_p, 30))
        for phase in (_literal(_p, 31), _literal(_p, 32)):
            before, after = (row.get(f'{_literal(_p, 33)}{phase}{_literal(_p, 34)}'), row.get(f'{_literal(_p, 35)}{phase}{_literal(_p, 36)}'))
            _require(type(before) is int and type(after) is int and (_literal(_p, 37) <= after <= before), _literal(_p, 38))
        _require(row[_literal(_p, 39)] <= row[_literal(_p, 40)], _literal(_p, 41))
        _require(all((row.get(f'{_literal(_p, 42)}{phase}{_literal(_p, 43)}') == row.get(f'{_literal(_p, 44)}{phase}{_literal(_p, 45)}') and isinstance(row.get(f'{_literal(_p, 46)}{phase}{_literal(_p, 47)}'), str) and (len(row[f'{_literal(_p, 48)}{phase}{_literal(_p, 49)}']) == _literal(_p, 50)) for phase in (_literal(_p, 51), _literal(_p, 52)))), _literal(_p, 53))
        area = row.get(_literal(_p, 54))
        _require(isinstance(area, (int, float)) and _literal(_p, 55) < area < float(_literal(_p, 56)), _literal(_p, 57))
    return report

def _warning_top(model: Path, value: dict) -> str:
    _p = configured_parameters('registration')['top']
    try:
        _require(value.get(_literal(_p, 58)) == _global(_p, 'FORMAT') and value.get(_literal(_p, 59)) == _global(_p, 'KIND') and resolve_review_path(value[_literal(_p, 60)]).samefile(model) and ((value.get(_literal(_p, 61)), value.get(_literal(_p, 62))) == _model_identity(model)) and (value.get(_literal(_p, 63)) == _literal(_p, 64)) and (value.get(_literal(_p, 65)) == _literal(_p, 66)) and (value.get(_literal(_p, 67)) == _global(_p, 'CONFIG')) and (value.get(_literal(_p, 68)) == _literal(_p, 69)) and (value.get(_literal(_p, 70)) == _literal(_p, 71)) and (len(value.get(_literal(_p, 72)) or {}) == _literal(_p, 73)) and (len(value.get(_literal(_p, 74)) or {}) == _literal(_p, 75)) and all((value.get(key) is _literal(_p, 76) for key in (_literal(_p, 77), _literal(_p, 78), _literal(_p, 79), _literal(_p, 80)))), _literal(_p, 81))
        for key, relative in _global(_p, 'FILES').items():
            _require(sha256_file(model / relative) == value.get(key), _literal(_p, 82))
        _reports_top(model)
        return _global(_p, 'WARNING')
    except (OSError, ValueError, TypeError, KeyError):
        return _global(_p, 'INVALID')

def _reports_downstream(model: Path) -> tuple[dict, dict]:
    _p = configured_parameters('registration')['downstream']
    reports = [_read_downstream(model / _literal(_p, 0) / name) for name in _global(_p, 'QA_FILES')]
    for report in reports:
        _require(report.get(_literal(_p, 1)) == _global(_p, 'CONFIG') and report.get(_literal(_p, 2)) == _global(_p, 'SPEC_SHA') and (report.get(_literal(_p, 3)) == _global(_p, 'G3_SHA')) and (report.get(_literal(_p, 4)) == _global(_p, 'REGIONAL_SHA')), _literal(_p, 5))
    parity, union = reports
    _require(parity.get(_literal(_p, 6)) == _literal(_p, 7), _literal(_p, 8))
    rows = parity.get(_literal(_p, 9)) or []
    keys = {(row.get(_literal(_p, 10)), row.get(_literal(_p, 11))) for row in rows}
    expected = {(run, index) for run, indices in _global(_p, 'QA_FRAMES').items() for index in indices}
    _require(len(rows) == len(keys) == _literal(_p, 12) and keys == expected and (parity.get(_literal(_p, 13)) is _literal(_p, 14)), _literal(_p, 15))
    for row in rows:
        score = row.get(_literal(_p, 16))
        _require(isinstance(score, (int, float)) and _literal(_p, 17) <= score <= _literal(_p, 18), _literal(_p, 19))
        for key in (_literal(_p, 20), _literal(_p, 21)):
            area = row.get(key)
            _require(isinstance(area, (int, float)) and _literal(_p, 22) <= area < float(_literal(_p, 23)), _literal(_p, 24))
    _require(union.get(_literal(_p, 25)) == _literal(_p, 26), _literal(_p, 27))
    rows = union.get(_literal(_p, 28)) or []
    keys = {(row.get(_literal(_p, 29)), row.get(_literal(_p, 30))) for row in rows}
    _require(len(rows) == len(keys) == _literal(_p, 31) and len({key[_literal(_p, 32)] for key in keys}) == _literal(_p, 33), _literal(_p, 34))
    _require(all((row.get(_literal(_p, 35)) is _literal(_p, 36) and isinstance(row.get(_literal(_p, 37)), str) and (len(row[_literal(_p, 38)]) == _literal(_p, 39)) and (row[_literal(_p, 40)] == row.get(_literal(_p, 41))) for row in rows)), _literal(_p, 42))
    return (parity, union)

def _warning_downstream(model: Path, value: dict) -> str:
    """Check the immutable metadata chain on opening; do not rehash 864 PNGs."""
    _p = configured_parameters('registration')['downstream']
    try:
        _require(value.get(_literal(_p, 43)) == _global(_p, 'FORMAT') and value.get(_literal(_p, 44)) == _global(_p, 'KIND') and (resolve_review_path(value.get(_literal(_p, 45)) or _literal(_p, 46)) == model) and ((value.get(_literal(_p, 47)), value.get(_literal(_p, 48))) == _model_identity(model)) and (value.get(_literal(_p, 49)) == _literal(_p, 50)) and (value.get(_literal(_p, 51)) == _literal(_p, 52)) and (value.get(_literal(_p, 53)) == _global(_p, 'CONFIG')) and (value.get(_literal(_p, 54)) == _global(_p, 'SPEC_SHA')) and (value.get(_literal(_p, 55)) == _global(_p, 'G3_SHA')) and (value.get(_literal(_p, 56)) == _global(_p, 'REGIONAL_SHA')) and (value.get(_literal(_p, 57)) == _literal(_p, 58)) and (value.get(_literal(_p, 59)) == _literal(_p, 60)) and (len(value.get(_literal(_p, 61)) or {}) == _literal(_p, 62)) and (len(value.get(_literal(_p, 63)) or {}) == _literal(_p, 64)) and all((value.get(key) is _literal(_p, 65) for key in (_literal(_p, 66), _literal(_p, 67), _literal(_p, 68), _literal(_p, 69)))), _literal(_p, 70))
        for key, relative in _global(_p, 'FILES').items():
            _require(sha256_file(model / relative) == value.get(key), _literal(_p, 71))
        qa = resolve_review_path(value[_literal(_p, 72)])
        _require(set(value.get(_literal(_p, 73)) or {}) == set(_global(_p, 'QA_FILES')), _literal(_p, 74))
        for name, digest in value[_literal(_p, 75)].items():
            _require(sha256_file(qa / name) == digest, _literal(_p, 76))
        _reports_downstream(model)
        return _global(_p, 'WARNING')
    except (OSError, ValueError, TypeError, KeyError):
        return _global(_p, 'INVALID')

def registered_roots(*, results: Path | None=None) -> list[Path]:
    """Discover all explicitly registered candidates regardless of QA status."""
    _p = configured_parameters('registration')['track1']
    folder = registration_path(_literal(_p, 0), results=results).parent
    try:
        receipts = list(folder.glob(_literal(_p, 1)))
    except OSError:
        return []
    roots = []
    for path in receipts:
        try:
            value = _read(path)
            if isinstance(value.get(_literal(_p, 2)), str):
                roots.append(resolve_review_path(value[_literal(_p, 3)]))
        except (OSError, ValueError, TypeError):
            continue
    return roots

def registration_warning(model: Path, *, results: Path | None=None) -> str:
    """Return the prominent QA warning when a registered pointer is intact."""
    _p = configured_parameters('registration')['track1']
    model = resolve_review_path(model)
    path = registration_path(model.name, results=results)
    if not path.is_file():
        return _literal(_p, 4)
    invalid = _literal(_p, 5)
    try:
        value = _read(path)
        if value.get(_literal(_p, 6)) == _literal(_p, 7):
            from .phase_saved_selected_registration import registered_warning
            return registered_warning(model, results=results) or invalid
        if value.get(_literal(_p, 8)) == _literal(_p, 9):
            from darsia.fluidflower.phase_saved_contact_review import registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 10)) == _literal(_p, 11):
            from darsia.fluidflower.phase_saved_categorical_review import registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 12)) == _literal(_p, 13):
            from darsia.fluidflower.phase_saved_boundary_review import registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 14)) == _literal(_p, 15):
            from darsia.fluidflower.phase_saved_review_metadata import registered_warning
            from darsia.fluidflower.native.config import require_parameters
            parameters = require_parameters(_literal(_p, 16))[_literal(_p, 17)]
            return registered_warning(model, value, parameters=parameters)
        if value.get(_literal(_p, 18)) == _literal(_p, 19):
            from .phase_saved_review_metadata import saved_revision_bindings
            registered_warning = saved_revision_bindings(_literal(_p, 20)).registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 21)) == _literal(_p, 22):
            from .phase_saved_review_metadata import saved_revision_bindings
            registered_warning = saved_revision_bindings(_literal(_p, 23)).registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 24)) == _literal(_p, 25):
            from .phase_saved_review_metadata import saved_revision_bindings
            registered_warning = saved_revision_bindings(_literal(_p, 26)).registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 27)) == _literal(_p, 28):
            from .phase_saved_review_metadata import saved_revision_bindings
            registered_warning = saved_revision_bindings(_literal(_p, 29)).registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 30)) == _literal(_p, 31):
            from .phase_saved_review_metadata import saved_revision_bindings
            registered_warning = saved_revision_bindings(_literal(_p, 32)).registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 33)) == _literal(_p, 34):
            from .phase_saved_review_metadata import saved_revision_bindings
            registered_warning = saved_revision_bindings(_literal(_p, 35)).registered_warning
            return registered_warning(model, value)
        if value.get(_literal(_p, 36)) == _literal(_p, 37):
            registered_warning = _warning_seam
            return registered_warning(model, value)
        if value.get(_literal(_p, 38)) == _literal(_p, 39):
            registered_warning = _warning_area
            return registered_warning(model, value)
        if value.get(_literal(_p, 40)) == _literal(_p, 41):
            registered_warning = _warning_rig
            return registered_warning(model, value)
        if value.get(_literal(_p, 42)) == _literal(_p, 43):
            registered_warning = _warning_barrier
            return registered_warning(model, value)
        if value.get(_literal(_p, 44)) == _literal(_p, 45):
            registered_warning = _warning_top
            return registered_warning(model, value)
        if value.get(_literal(_p, 46)) == _literal(_p, 47):
            registered_warning = _warning_downstream
            return registered_warning(model, value)
        rerun = value.get(_literal(_p, 48)) in _global(_p, 'RERUN_KINDS')
        expected_qa = _global(_p, 'RERUN_QA_FILES') if rerun else _global(_p, 'ORIGINAL_QA_FILES')
        if not (value.get(_literal(_p, 49)) == _global(_p, 'FORMAT') and resolve_review_path(value.get(_literal(_p, 50)) or _literal(_p, 51)) == model and (value.get(_literal(_p, 52)) == model.name) and (value.get(_literal(_p, 53)) == _model_identity(model)[_literal(_p, 54)]) and (value.get(_literal(_p, 55)) == _literal(_p, 56)) and (value.get(_literal(_p, 57)) == _literal(_p, 58)) and (value.get(_literal(_p, 59)) is _literal(_p, 60)) and (value.get(_literal(_p, 61)) is _literal(_p, 62)) and (value.get(_literal(_p, 63)) is _literal(_p, 64)) and (value.get(_literal(_p, 65)) is _literal(_p, 66)) and (value.get(_literal(_p, 67)) == _literal(_p, 68)) and (value.get(_literal(_p, 69)) == _literal(_p, 70)) and (len(value.get(_literal(_p, 71)) or {}) == _literal(_p, 72)) and (len(value.get(_literal(_p, 73)) or {}) == _literal(_p, 74)) and (set(value.get(_literal(_p, 75)) or {}) == expected_qa)):
            return invalid
        if rerun:
            if sha256_file(model / _global(_p, "RECEIPT_FILE")) != value.get(_literal(_p, 76)):
                return invalid
            if value.get(_literal(_p, 77)) == _literal(_p, 78):
                qa_report = _read(resolve_review_path(value[_literal(_p, 79)]) / _literal(_p, 80))
                if qa_report.get(_literal(_p, 81)) != _global(_p, 'DISPLAY_MODES_V2_ACCEPTANCE_CONTRACT') or qa_report.get(_literal(_p, 82)) != model.name or qa_report.get(_literal(_p, 83)) != value[_literal(_p, 84)] or (set(qa_report.get(_literal(_p, 85)) or {}) != _global(_p, 'DISPLAY_MODES_V2_CHECKS')) or (qa_report.get(_literal(_p, 86)) != value.get(_literal(_p, 87))):
                    return invalid
                for evidence in (qa_report.get(_literal(_p, 88)) or {}).values():
                    if not isinstance(evidence, dict) or sha256_file(resolve_review_path(evidence[_literal(_p, 89)])) != evidence[_literal(_p, 90)]:
                        return invalid
        for key, relative in ((_literal(_p, 91), _literal(_p, 92)), (_literal(_p, 93), _literal(_p, 94)), (_literal(_p, 95), _literal(_p, 96)), (_literal(_p, 97), _literal(_p, 98)), (_literal(_p, 99), _literal(_p, 100))):
            if sha256_file(model / relative) != value.get(key):
                return invalid
        qa = resolve_review_path(value[_literal(_p, 101)])
        if any((sha256_file(qa / name) != digest for name, digest in (value.get(_literal(_p, 102)) or {}).items())):
            return invalid
        return str(value.get(_literal(_p, 103)) or invalid)
    except (OSError, TypeError, ValueError, KeyError):
        return invalid
