"""Read-only saved mass freezes, cohorts and their original producer evidence."""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from .native.io import sha256_file
from .native.config import require_parameters

_UNSET = object()
_VERIFIED_FREEZES = {}


def _parameters():
    parameters = require_parameters("saved_review_metadata")["mass_freeze"]
    current = parameters["current_code_source"]
    if not isinstance(current, dict) or Path(current["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("Saved mass metadata implementation is not explicitly bound")
    if sha256_file(Path(__file__)) != current["sha256"] or Path(__file__).stat().st_size != current["size_bytes"]:
        raise ValueError("Saved mass metadata implementation changed")
    return parameters


def _declared(section, name):
    item = _parameters()[section][name]
    value = item["value"]
    if item["python_type"] == "tuple":
        return tuple(value)
    if item["python_type"] == "frozenset":
        return frozenset(value)
    if item["python_type"] == "set":
        return set(value)
    if type(value).__name__ != item["python_type"]:
        raise TypeError("Saved mass metadata DATA changed its original Python type")
    return value


def _original_common_source(engine_sha256):
    """Verify the freeze's original engine bytes without a new freeze-key gate."""
    entry = _parameters()["common"]["producer_source"]
    path = Path(entry["path"]).resolve()
    if (entry["sha256"] != engine_sha256
            or path.stat().st_size != entry["size_bytes"]
            or sha256_file(path) != engine_sha256):
        raise ValueError("Common mass implementation changed after freeze")
    return path


def __getattr__(name):
    if name != "EXCLUDED_RUNS":
        raise AttributeError(name)
    return _declared("common", name)


@dataclass(frozen=True)
class FrozenEvidence:
    path: Path
    sha256: str
    record: dict

    @classmethod
    def verify(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('contract') != _declared('round2', 'FREEZE_CONTRACT') or data.get('frozen') is not True or (not data.get('frozen_utc')):
            raise ValueError('New round-2 freeze required before out-of-fit pixels')
        if set(data.get('calibration_runs', ())) != _declared('round2', 'CALIBRATION') or set(data.get('heldout_runs', ())) != _declared('round2', 'HELDOUT') or set(data.get('descriptive_runs', ())) != _declared('round2', 'DESCRIPTIVE'):
            raise ValueError('Round-2 freeze needs the exact 26/12/AC38 split')
        if data.get('map_mass_rescaling') is not False or data.get('overflow_in_objective') is not False:
            raise ValueError('A freeze cannot permit map rescaling or overflow fitting')
        files = data.get('calibration_files', {})
        if not files:
            raise ValueError('Frozen numerical source inventory required')
        for name, sha in files.items():
            asset = Path(name)
            if not asset.is_absolute():
                asset = path.parent / asset
            if sha256_file(asset) != sha:
                raise ValueError(f'Frozen calibration bytes changed: {asset}')
        return cls(path, sha256_file(path), data)

    def check_frame_run(self, run):
        run = str(run).lower()
        if run not in _declared('round2', 'CALIBRATION') | _declared('round2', 'HELDOUT') | _declared('round2', 'DESCRIPTIVE'):
            raise ValueError('Scientific output outside the frozen 39-run cohort')
        return run

def normalize_run(run: str) -> str:
    value = str(run).lower()
    if value not in _declared('common', 'ALL_RUNS'):
        raise ValueError(f'Unknown FF_AC run: {run}')
    return value

def validate_cohort(calibration_runs=_UNSET, heldout_runs=_UNSET) -> None:
    if calibration_runs is _UNSET:
        calibration_runs = _declared('common', 'CALIBRATION_RUNS')
    if heldout_runs is _UNSET:
        heldout_runs = _declared('common', 'HELDOUT_RUNS')
    calibration = tuple(map(normalize_run, calibration_runs))
    heldout = tuple(map(normalize_run, heldout_runs))
    if len(calibration) != 27 or len(set(calibration)) != 27 or set(calibration) != set(_declared('common', 'CALIBRATION_RUNS')):
        raise ValueError('The common calibration must contain exactly the 27 pre-declared calibration runs')
    if len(heldout) != 12 or len(set(heldout)) != 12 or set(heldout) != set(_declared('common', 'HELDOUT_RUNS')):
        raise ValueError('The twelve held-out runs are fixed before fitting')

def verify_frozen_calibration(freeze_path, *, expected_sha256=None, asset_reader=None):
    path = Path(freeze_path).resolve() if asset_reader is None else asset_reader.resolve_path(freeze_path, expected_sha256)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('Frozen common calibration SHA-256 mismatch')
    freeze = json.loads(path.read_text(encoding='utf-8'))
    if freeze.get('contract') != _declared('common', 'CONTRACT') or freeze.get('frozen') is not True:
        raise ValueError('Held-out and production evaluation require a published common freeze')
    validate_cohort(freeze['calibration_runs'], freeze['heldout_runs'])
    if freeze.get('map_mass_rescaling') is not False or freeze.get('overflow_in_objective') is not False:
        raise ValueError('The frozen common model violates the no-rescaling/overflow-compare-only contract')
    for name, expected in {**freeze['calibration_files'], **freeze.get('native_implementation_files', {})}.items():
        source = Path(name) if asset_reader is None else asset_reader.resolve_path(name, expected)
        stat = source.stat()
        key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns, expected)
        if key not in _VERIFIED_FREEZES:
            if sha256_file(source) != expected:
                raise ValueError(f'Frozen calibration file changed: {source}')
            _VERIFIED_FREEZES[key] = True
    implementation = _original_common_source(freeze['common_engine_sha256']) if asset_reader is None else asset_reader.source_path(_original_common_source(freeze['common_engine_sha256']), freeze['common_engine_sha256'])
    if sha256_file(implementation) != freeze['common_engine_sha256']:
        raise ValueError('Common mass implementation changed after freeze')
    return (freeze, actual)


@dataclass(frozen=True)
class FrozenCalibrationEvidence:
    path: Path
    sha256: str
    record: dict

    @classmethod
    def verify(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("frozen") is not True or not data.get("frozen_utc"):
            raise ValueError("Common calibration must be frozen before any production evaluation")
        files = data.get("calibration_files")
        calibration = {str(x).lower() for x in data.get("calibration_runs", ())}
        heldout = {str(x).lower() for x in data.get("heldout_runs", ())}
        if not files or len(calibration) != _declared('production', 'CALIBRATION_COUNT') or heldout != _declared('production', 'HELDOUT') or calibration & heldout or (calibration | heldout) & _declared('production', 'EXCLUDED'):
            raise ValueError("Frozen calibration lacks the exact 27/12 independent run split")
        for name, expected in files.items():
            asset = Path(name)
            if not asset.is_absolute():
                asset = path.parent / asset
            if sha256_file(asset) != expected:
                raise ValueError(f"Frozen common calibration file changed: {asset}")
        scales = data.get("shared_scales") or {}
        if set(scales) != _declared('production', 'SHARED_SCALE_KEYS') or any(not np.isfinite(x) or x <= _declared('production', 'MINIMUM_SCALE') for x in scales.values()):
            raise ValueError("Exactly one positive common aqueous/gas scale pair is required")
        return cls(path, sha256_file(path), data)

    def check_frame_run(self, run):
        run = str(run).lower()
        allowed = set(self.record["calibration_runs"]) | set(self.record["heldout_runs"])
        allowed = {str(x).lower() for x in allowed}
        if run not in allowed or run in _declared('production', 'EXCLUDED'):
            raise ValueError(f"Run is outside the frozen common-calibration cohort: {run}")
        return run
