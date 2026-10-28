"""Bind the FF_BM optical inventory to its later source-image label corrections."""
from collections import Counter
import csv
from io import StringIO
import hashlib
import json
from pathlib import Path

from .um_bm_review_regions import load_model_regions


OPTICAL_INVENTORY_SHA256 = '9c4f6649dc335df3a0c1c3dfa34e93431a64ac2b8df14bfd8966af1a1c28512a'
LEGACY_EVIDENCE_SHA256 = 'b5aa4801d98749110dfb0678b408371ee2dec37641db0d3af2cbeaed1728f50d'
ACTIVE_EVIDENCE_SHA256 = '92ee98d724223efaae0e76533711b7effa33307ef75f05a59f28e39c7c4d9a8c'


def reconcile_inventory(optical_inventory, legacy_regions, active_regions):
    """Retain every old ID, explicitly accounting for current reviewer corrections."""
    originals = {r.record['region_id']: r for r in legacy_regions}
    active = {r.record['region_id']: r for r in active_regions}
    if len(originals) != len(legacy_regions) or len(active) != len(active_regions):
        raise ValueError('Duplicate original or current BM annotation ID')
    ids = [r['region_id'] for r in optical_inventory]
    if len(set(ids)) != len(ids) or set(ids) != set(originals) or not set(active) <= set(ids):
        raise ValueError('Original optical inventory and bound review geometries differ')
    provenance = []
    for row in optical_inventory:
        rid = row['region_id']
        original = originals[rid]
        for field in ('dataset', 'run', 'frame_index', 'confidence', 'source', 'reviewer'):
            if str(row[field]) != str(original.record[field]):
                raise ValueError(f'Original optical inventory binding differs: {rid}/{field}')
        current = active.get(rid)
        previous_class = row['phase_class']
        if current is None:
            if previous_class not in {'gas_absent_region', 'aqueous_absent_region'}:
                raise ValueError(f'Unexplained missing original BM region: {rid}')
            reason = 'legacy candidate phase-error box is not a source-image absence label; current reviewer policy excludes it'
            state = 'excluded'
        else:
            if current.record['payload'] != original.record['payload'] or current.frame != original.frame:
                raise ValueError(f'Historical geometry/source changed for BM region: {rid}')
            changed = current.record['phase_class'] != previous_class
            if changed and (previous_class, current.record['phase_class']) != ('aqueous_present_region', 'aqueous'):
                raise ValueError(f'Unexplained BM label change: {rid}')
            reason = ('later reviewer clarification labels the whole marked area aqueous'
                      if changed else 'original optical label retained')
            state = 'active'
        provenance.append(dict(region_id=rid, original_phase_class=previous_class,
                               current_phase_class=current.record['phase_class'] if current else None,
                               status=state, reason=reason,
                               manifest_sha256=original.frame.manifest_sha256))
    return provenance


def load_current_bm_cohort(*, optical_model, legacy_model, current_model, resolve_path=Path):
    """Verify the actual 772-original / 713-current cohort before RAW refitting."""
    paths = ((Path(optical_model)/'region_inventory.csv', OPTICAL_INVENTORY_SHA256),
             (Path(legacy_model)/'evidence.json', LEGACY_EVIDENCE_SHA256),
             (Path(current_model)/'evidence.json', ACTIVE_EVIDENCE_SHA256))
    bindings = []
    for path, digest in paths:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'BM cohort source checksum differs: {path}')
        bindings.append(dict(path=str(path), sha256=digest))
    inventory = list(csv.DictReader(StringIO(paths[0][0].read_text(encoding='utf-8-sig'))))
    legacy = load_model_regions(legacy_model, series_id='FF_BM', resolve_path=resolve_path)
    current = load_model_regions(current_model, series_id='FF_BM', resolve_path=resolve_path)
    provenance = reconcile_inventory(inventory, legacy, current)
    counts = Counter(row['status'] for row in provenance)
    changes = Counter((row['original_phase_class'], row['current_phase_class']) for row in provenance
                      if row['original_phase_class'] != row['current_phase_class'])
    expected = {('gas_absent_region', None): 40, ('aqueous_absent_region', None): 19,
                ('aqueous_present_region', 'aqueous'): 199}
    if counts != {'active': 713, 'excluded': 59} or changes != expected:
        raise ValueError('BM cohort differs from the explicitly approved current label policy')
    return current, dict(series='FF_BM', original_optical_regions=772, active_regions=713,
                         exclusions=59, whole_area_aqueous_clarifications=199,
                         sources=bindings, regions=provenance,
                         fit_heads=['co2_presence', 'gas_within_co2'],
                         inherited_optical_heads=[], training_scope='training-only',
                         generalization_validated=False)
