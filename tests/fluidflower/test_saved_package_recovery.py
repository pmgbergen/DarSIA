"""Saved package metadata admission never launches historical preparation."""
from pathlib import Path
import hashlib
import pytest
from darsia.fluidflower import phase_saved_package_recovery as recovery


def test_bound_metadata_rejects_wrong_digest_without_numeric_import(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, 'configured_parameters', lambda _: {'rerun': {}, 'literal_values': {'digest_hex_length': 64}})
    path = tmp_path / 'receipt.json'
    path.write_bytes(b'{"complete": true}')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert recovery._bound(str(path), expected, 'receipt') == path.resolve()
    with pytest.raises(ValueError, match='changed or is absent'):
        recovery._bound(str(path), '0' * 64, 'receipt')


def test_saved_aggregate_rejects_empty_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, 'configured_parameters', lambda _: {'rerun': {}, 'literal_values': {}})
    with pytest.raises(ValueError, match='empty validation selection'):
        recovery.assemble_saved_packages(tmp_path, [])


def test_saved_package_run_set_rejects_duplicates():
    recovery.require_run_set([{'run': 'left'}, {'run': 'right'}], ['left', 'right'])
    with pytest.raises(ValueError, match='Incomplete review scope'):
        recovery.require_run_set([{'run': 'left'}, {'run': 'left'}], ['left', 'right'])
