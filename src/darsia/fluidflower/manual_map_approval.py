"""Keep exact saved map approvals authoritative during geometry adoption."""
def has_manual_map_approval(preparation):
    """Cheap eligibility only; the Check still verifies every consumed product."""
    rows = preparation.get('runs') or []
    reviewed = (preparation.get('chart_review') or {}).get('runs') or {}
    return bool(rows and reviewed and set(reviewed) == {row.get('run_id') for row in rows}
                and all(row.get('status') == 'prepared' for row in rows)
                and not preparation.get('analysis_exclusions')
                and not preparation.get('target_corrections'))


def verify_manual_map_approval(preparation):
    """Authenticate saved per-run approvals, version, frames and zero actual gaps.

    This grants no new human decisions. Numeric/product admission remains in
    the ordinary Check. Display warnings stay in that Check's report.
    """
    if not has_manual_map_approval(preparation):
        return False
    from .painted_adoption_table import _checked, _same
    from .painting_chart_transfer import _json, _validate_approval
    from .painting_chart_approval_store import _sources
    from .painted_analysis_exclusions import read_gap_review

    chart = preparation['chart_review']
    if not _same(chart['review'], preparation['painting_geometry']['review']):
        raise ValueError('The manual map approvals belong to another transfer review')
    _checked(chart['review'])
    descriptor = preparation['gap_review']
    review = read_gap_review(descriptor['path'])
    if (not _same(review['review_binding'], descriptor)
            or not _same(review['map_version'], preparation['map_version'])
            or not _same(review['geometry_index'], preparation['geometry_index'])
            or any(row['gap']['pixels'] for row in review['runs'])):
        raise ValueError('The approved map has changed or still contains unreviewed gaps')
    if {row['run_id'] for row in review['runs']} != set(chart['runs']):
        raise ValueError('The approved map review has a different run selection')
    seen = set()
    for row in preparation['runs']:
        run = row['run_id']
        saved = chart['runs'][run]
        proposal, approval = saved['proposal'], saved['approval']
        _validate_approval(proposal, _json(proposal), approval)
        source = proposal['source_binding']
        if (proposal['run_id'] != run
                or not _same(source['review'], chart['review'])
                or not _same(source['painting_facies'], preparation['painting_facies'])
                or not _same(source['target_model'], row['target_model'])):
            raise ValueError('A manual map approval lost its exact run, map or frame')
        _sources(source, seen)
    return True


def approved_candidate(candidate):
    """Use the immutable preparation already bound into the checked candidate."""
    from .painted_adoption_table import _checked
    table = candidate.get('painted_adoption_table') or {}
    descriptor = table.get('preparation')
    return bool(descriptor and verify_manual_map_approval(_checked(descriptor, metadata=True)))


def approved_boundary_authorities(candidate):
    """Return saved visual decisions only for the exact maps being checked.

    A copied/replaced label path, changed registration or changed approval
    cannot suppress a boundary screen. This does not validate array shapes,
    physical properties or source-mode admission; their original checks run.
    """
    from pathlib import Path
    from .painted_adoption_table import _checked, _local, _same
    descriptor = (candidate.get('painted_adoption_table') or {}).get('preparation')
    if not descriptor:
        return {}
    preparation = _checked(descriptor, metadata=True)
    if not verify_manual_map_approval(preparation):
        return {}
    folder = Path(descriptor['path']).parent
    entries = {row['run_id']: row for row in preparation['runs']}
    authorities = {}
    for row in (candidate.get('run_geometry') or {}).get('runs', []):
        run = row.get('run_id')
        if run not in entries:
            continue
        entry = entries[run]
        artifact = (preparation.get('chart_transfers') or {}).get(run, {}).get('artifact', entry['labels'])
        bound = _local(folder, artifact)
        registration = row.get('registration') or {}
        if (not _same(registration.get('measurement'), descriptor)
                or not _same(registration.get('labels'), bound)
                or not _same(registration.get('target_model'), entry['target_model'])
                or Path(str(row.get('labels_path') or '')).resolve() != Path(bound['path']).resolve()):
            continue
        authorities[run] = dict(preparation=descriptor,
            review=preparation['gap_review'],
            map_version=preparation['map_version'],
            approval=preparation['chart_review']['runs'][run]['approval'])
    return authorities
