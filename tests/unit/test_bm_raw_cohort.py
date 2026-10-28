from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia.fluidflower.phase.bm_raw_cohort import reconcile_inventory


def example():
    frame = SimpleNamespace(manifest_sha256='bound-source')
    records = [dict(region_id=str(i), dataset='ffbm', run='c1', frame_index=2,
                    confidence='high', source='manual', reviewer='Reviewer',
                    phase_class=label, payload={'id': str(i), 'x': .1})
               for i, label in enumerate(('water', 'aqueous', 'gas_absent_region'))]
    legacy = [SimpleNamespace(record=r, frame=frame) for r in records]
    active = deepcopy(legacy[:2])
    inventory = [{k: str(v) for k, v in r.items() if k != 'payload'} for r in records]
    inventory[1]['phase_class'] = 'aqueous_present_region'
    return inventory, legacy, active


def test_original_ids_and_later_label_policy_are_both_retained():
    provenance = reconcile_inventory(*example())
    assert [r['region_id'] for r in provenance] == ['0', '1', '2']
    assert provenance[1]['original_phase_class'] == 'aqueous_present_region'
    assert provenance[1]['current_phase_class'] == 'aqueous'
    assert provenance[2]['status'] == 'excluded'
    assert 'candidate phase-error box' in provenance[2]['reason']


def test_unexplained_label_loss_fails():
    inventory, legacy, active = example()
    with pytest.raises(ValueError, match='Unexplained missing'):
        reconcile_inventory(inventory, legacy, active[1:])


def test_changed_geometry_is_not_silently_rebound():
    inventory, legacy, active = example()
    active[0].record['payload']['x'] = .2
    with pytest.raises(ValueError, match='geometry/source'):
        reconcile_inventory(inventory, legacy, active)
