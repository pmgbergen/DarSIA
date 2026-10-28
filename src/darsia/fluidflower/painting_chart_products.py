"""Approved chart inputs on reused maps, before physical fields are derived."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .painting_chart_transfer import KEY, ChartTransferProposal, apply_chart_transfer


def _fields(base, result):
    fields = {key: value.copy() for key, value in base.items()}
    fields['array'] = result['effective_labels'].astype(np.int32)
    changed = result['original_chart_mask'] | result['chart_mask']
    fields['effective_coverage'][changed] = fields['array'][changed] != 0
    fields['gaps'] = fields['inside'] & ~fields['effective_coverage']
    return fields


def prepare_chart_products(preparation, folder, approvals, *, detection_manifest=None):
    """Write changed effective products after every adoption run was approved."""
    from .painting_chart_transfer_reader import ChartTransferSession, proposal_for_approval
    from .painted_adoption_table import _bound
    from .painted_target_integration import _effective_without_chart
    expected_runs = {row['run_id'] for row in preparation.get('runs', [])}
    if not expected_runs or not isinstance(approvals, dict) or set(approvals) != expected_runs:
        raise ValueError('Approve every adoption run before reusing the saved maps.')
    proof = preparation['painting_geometry']
    options = {} if detection_manifest is None else dict(detection_manifest=detection_manifest)
    session = ChartTransferSession(proof['review'], **options)
    records, reviewed = {}, {}
    for entry in preparation['runs']:
        run = entry['run_id']
        base, base_bound = _effective_without_chart(preparation, folder, entry)
        proposal = proposal_for_approval(session, run, approvals[run])
        result = apply_chart_transfer(proposal, approvals[run])
        metadata = proposal.metadata()
        from .painting_chart_transfer import _identity
        if (metadata['base_arrays']['labels'] != _identity(base['array'])
                or metadata['base_arrays']['inside'] != _identity(base['inside'])):
            raise ValueError('The approved ColorChecker belongs to another reused map.')
        reviewed[run] = dict(proposal=metadata, approval=deepcopy(approvals[run]))
        if not metadata['moved']:
            continue
        fields = _fields(base, result)
        target = Path(folder) / run / 'chart_labels.npz'
        with np.load(base_bound['path'], allow_pickle=True) as archive:
            grid = deepcopy(archive['metadata'].item())
        with target.open('xb') as stream:
            np.savez_compressed(stream, **fields, metadata=grid, **result)
        records[run] = dict(artifact=dict(_bound(target), path=f'{run}/chart_labels.npz'),
            proposal=metadata, approval=deepcopy(approvals[run]))
    preparation['chart_review'] = dict(review=deepcopy(proof['review']), runs=reviewed)
    return records


def chart_product(preparation, folder, row, base, base_bound):
    """Verify saved exact approved pixels and their bound original inputs."""
    from .painted_adoption_table import _local, _checked
    from .painted_analysis_exclusions import _arrays
    from .painting_chart_transfer import _identity, _json, _validate_approval
    records = preparation.get(KEY)
    if records is None:
        return base, base_bound
    review = preparation.get('chart_review') or {}
    reviewed = review.get('runs') or {}
    expected_runs = {entry['run_id'] for entry in preparation['runs']}
    if (set(reviewed) != expected_runs or review.get('review') != preparation['painting_geometry']['review']
            or not isinstance(records, dict)
            or set(records) != {run for run, item in reviewed.items() if item['proposal']['moved']}):
        raise ValueError('The saved maps need the exact approval of every adoption run.')
    run = row['run_id']
    metadata = reviewed[run]['proposal']
    approval = reviewed[run]['approval']
    _validate_approval(metadata, _json(metadata), approval)
    if (metadata['run_id'] != run or metadata['base_arrays']['labels'] != _identity(base['array'])
            or metadata['base_arrays']['inside'] != _identity(base['inside'])
            or metadata['source_binding']['review'] != preparation['painting_geometry']['review']
            or metadata['source_binding']['painting_facies'] != preparation['painting_facies']
            or metadata['source_binding']['target_model'] != row['target_model']):
        raise ValueError('The ColorChecker product lost its original run, map or source association.')
    def check_sources(value):
        if isinstance(value, dict):
            if {'path', 'sha256', 'size_bytes'} <= set(value):
                _checked({key: value[key] for key in ('path', 'sha256', 'size_bytes')})
            else:
                for child in value.values():
                    check_sources(child)
        elif isinstance(value, list):
            for child in value:
                check_sources(child)
    check_sources(metadata['source_binding'])
    if run not in records:
        return base, base_bound
    record = records[run]
    if any(record[key] != reviewed[run][key] for key in ('proposal', 'approval')):
        raise ValueError('The changed ColorChecker differs from its approved run review.')
    bound = _local(folder, record['artifact'])
    actual = _arrays(bound)
    with np.load(bound['path'], allow_pickle=False) as archive:
        result = {key: np.array(archive[key], copy=True) for key in metadata['result']}
    proposal = ChartTransferProposal(_json(metadata), result)
    apply_chart_transfer(proposal, record['approval'])
    expected = _fields(base, result)
    if any(not np.array_equal(actual[key], expected[key]) for key in expected):
        raise ValueError('The stored ColorChecker inputs differ from the approved pixels.')
    return actual, bound
