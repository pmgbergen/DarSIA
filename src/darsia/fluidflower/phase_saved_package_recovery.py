"""Admit and publish only already completed, source-bound saved review packages.

No image decoder, prediction or preparation worker is imported by this reader.
A missing presentation record is reported by its caller as unavailable.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
from .project import sha256_file, utc_now
from .phase_validation import _model_identity
from .phase_saved_review_metadata import configured_parameters, _read as read, _read, _write as write

def require_run_set(jobs, expected):
    actual = [j['run'] for j in jobs]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f'Incomplete review scope: expected {sorted(expected)}, found {sorted(actual)}')


def verify_incremental(root: Path, runs: list[str] | None=None) -> dict:
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['incremental']
    _literal_values = _parameters['literal_values']
    root = Path(root).resolve()
    plan = _read(root / _admission['plan_file'])
    _require(plan.get('contract') == _admission['contract'] and Path(plan['candidate']).resolve() == root and (plan.get('model_identity') == list(_model_identity(root))) and (plan.get('run_count') == _literal_values['run_count']) and (plan.get('frame_count') == _literal_values['frame_count']) and (plan.get('optical_training_performed') is False) and (plan.get('regional_training_performed') is False) and (plan.get('mass_eligible') is False), 'Incremental frozen fit plan is invalid')
    _bound(plan['source_job_path'], plan['source_job_sha256'], 'frozen inference source job')
    for relative, digest in plan['trained_artifacts_preserved'].items():
        _bound(str(root / relative), digest, f'preserved {relative}')
    seal = _read(root / 'inference_incremental_profile_seal.json')
    _bound(str(root / _admission['plan_file']), seal['plan_sha256'], 'incremental plan')
    _bound(str(root / 'phase_optical_regions.json'), seal['profile_sha256'], 'static runtime profile')
    allowed = {row['run'] for row in plan['expected_sources']}
    selected = list(runs or [])
    _require(set(selected) <= allowed, 'Requested run is outside the frozen cohort')
    for run in selected:
        binding = _read(root / 'inference_run_bindings' / f'{run}.json')
        _require(binding.get('contract') == _admission['binding_contract'] and binding.get('run') == run and (binding.get('plan_sha256') == seal['plan_sha256']) and (binding.get('model_identity') == plan['model_identity']), f'{run}: immutable run binding is invalid')
        _bound(binding['source_receipt_path'], binding['source_receipt_sha256'], f'{run} source seal')
        _bound(binding['materialized_manifest_path'], binding['materialized_manifest_sha256'], f'{run} native timeline')
        _bound(binding['review_source_manifest_path'], binding['review_source_manifest_sha256'], f'{run} map source')
    return dict(plan, source_bindings=[_read(root / 'inference_run_bindings' / f'{run}.json') for run in selected])

def verify_sealed_incremental(root: Path) -> dict:
    """Require complete, unchanged43/864 inference provenance before registration.

    The pending incremental plan is insufficient. This branch accepts only the
    real finalized source batch and every already sealed per-run source. It
    does not change masks, coefficients, QA eligibility or runtime identities.
    """
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['incremental']
    _literal_values = _parameters['literal_values']
    root = Path(root).resolve()
    receipt = _read(root / 'inference_rerun_receipt.json')
    plan = verify_incremental(root)
    _require(receipt.get('contract') == _admission['contract'] and receipt.get('stage') == _literal_values['sealed_stage'] and (Path(receipt.get('candidate') or '').resolve() == root) and (receipt.get('model_identity') == plan['model_identity']) and (receipt.get('old_candidate') == plan['old_candidate']) and (receipt.get('old_model_identity') == plan['old_model_identity']) and (receipt.get('run_count') == _literal_values['run_count']) and (receipt.get('frame_count') == _literal_values['frame_count']) and (receipt.get('plan_sha256') == sha256_file(root / _admission['plan_file'])) and (receipt.get('trained_artifacts_preserved') == plan['trained_artifacts_preserved']) and (receipt.get('frozen_training_request_sha256') == plan['trained_artifacts_preserved']['training_request.json']) and (receipt.get('frozen_regional_candidate_sha256') == plan['trained_artifacts_preserved']['regional_model.json']) and (receipt.get('optical_training_performed') is False) and (receipt.get('regional_training_performed') is False) and (receipt.get('mass_eligible') is False) and (receipt.get('scientific_acceptance') is False) and (receipt.get('enclosed_hole_policy') == _literal_values['enclosed_hole_policy']) and (receipt.get('enclosed_hole_max_area_px') == _literal_values['enclosed_hole_max_area_px']) and (receipt.get('vertical_gap_max_length_px') == _literal_values['vertical_gap_max_length_px']) and (receipt.get('vertical_gap_rule') == _literal_values['vertical_gap_rule']) and (receipt.get('display_mode') == plan.get('display_mode', _literal_values['default_display_mode'])) and (plan.get('display_mode') != _literal_values['natural_display_mode'] or receipt.get('isolated_invalid_policy') == _literal_values['isolated_invalid_policy']) and (receipt.get('cpu_path_required') is True), 'Completed incremental inference receipt is invalid')
    old = Path(plan['old_candidate']).resolve()
    _require(list(_model_identity(old)) == plan['old_model_identity'], 'Original fitted model identity changed')
    _bound(plan['old_profile_path'], plan['old_profile_sha256'], 'original fitted runtime profile')
    for relative, digest in plan['trained_artifacts_preserved'].items():
        if relative != 'summary.json':
            _bound(str(old / relative), digest, f'original preserved {relative}')
    _require(receipt.get('user_authorized_spec') == {'path': plan['spec_path'], 'sha256': plan['spec_sha256']}, 'Completed inference receipt differs from the authorized specification')
    _bound(plan['spec_path'], plan['spec_sha256'], 'frozen authorized specification')
    expected = {row['run']: row for row in plan['expected_sources']}
    bindings = verify_incremental(root, list(expected))['source_bindings']
    _require(receipt.get('source_bindings') == bindings and set(receipt.get('run_binding_sha256') or {}) == set(expected), 'Completed inference sources differ from their immutable run bindings')
    for binding in bindings:
        run = binding['run']
        _bound(str(root / 'inference_run_bindings' / (run + '.json')), receipt['run_binding_sha256'][run], f'{run} frozen inference binding')
        _require(_read(old / 'review_source' / run / 'original_job.json') == _read(Path(plan['original_sources_root']) / run / 'original_job.json'), f'{run}: inherited source job changed')
    batch_path = _bound(receipt['materialization_receipt_path'], receipt['materialization_receipt_sha256'], 'completed source batch')
    batch = _read(batch_path)
    source_job = _read(Path(plan['source_job_path']))
    rows = batch.get('canonical_mapping') or []
    by_run = {row['run']: row for row in rows}
    _require(batch_path == Path(source_job['settings']['output_root']).resolve() / 'materialization_receipt.json' and Path(batch.get('output_root') or '').resolve() == batch_path.parent and (batch.get('source_job_sha256') == plan['source_job_sha256']) and (batch.get('inference_only') is True) and (batch.get('retraining') is False) and (batch.get('labels_changed') is False) and (batch.get('source_revisions') == _literal_values['run_count']) and (batch.get('timeline_frames') == _literal_values['frame_count']) and (batch.get('revisions') == rows) and (batch.get('processor') == source_job.get('processor')) and (len(rows) == len(by_run) == _literal_values['run_count']) and (set(by_run) == set(expected)) and (sum((row['frame_count'] for row in rows)) == _literal_values['frame_count']), 'Completed batch differs from the frozen43/864 acquisition plan')
    for binding in bindings:
        row, expected_row = (by_run[binding['run']], expected[binding['run']])
        _require(row['source_manifest_sha256'] == binding['source_manifest_sha256'] == expected_row['source_manifest_sha256'] and row['frame_count'] == binding['frame_count'] == expected_row['frame_count'] and (row['materialized_manifest_path'] == binding['materialized_manifest_path']) and (row['materialized_manifest_sha256'] == binding['materialized_manifest_sha256']) and (row['receipt_path'] == binding['source_receipt_path']) and (row['receipt_sha256'] == binding['source_receipt_sha256']), 'Completed batch differs from an immutable inference run')
    source_path = root / 'review_source/postflight_manifest.json'
    _bound(str(source_path), receipt['review_source_sha256'], 'completed review source campaign')
    _bound(str(root / 'phase_optical_regions.json'), receipt['inference_profile_sha256'], 'static inference profile')
    jobs = _read(source_path).get('jobs') or []
    jobs_by_run = {job.get('run'): job for job in jobs}
    _require(len(jobs) == len(jobs_by_run) == _literal_values['run_count'] and set(jobs_by_run) == set(expected) and all((_read(root / 'review_source' / run / 'postflight_manifest.json') == {'jobs': [jobs_by_run[run]]} for run in expected)), 'Completed review campaign differs from per-run source bindings')
    from darsia.fluidflower.phase_optical_regions_runtime import load_profile
    loaded = load_profile(root)
    _require(set(loaded.get('_incremental_sealed_runs') or []) == set(expected), 'Completed inference runtime lacks sealed sources')
    return receipt

def _require(condition: bool, message: str) -> None:
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['rerun']
    _literal_values = _parameters['literal_values']
    if not condition:
        raise ValueError(message)

def _bound(path: Any, digest: Any, label: str) -> Path:
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['rerun']
    _literal_values = _parameters['literal_values']
    _require(isinstance(path, str) and isinstance(digest, str) and (len(digest) == _literal_values['digest_hex_length']), f'{label} lacks its immutable path and SHA')
    result = Path(path).resolve()
    _require(result.is_file() and sha256_file(result) == digest, f'{label} changed or is absent: {result}')
    return result

def verify_inference_rerun(model: Path) -> dict[str, Any]:
    """Verify immutable fit/source provenance without replaying model fitting."""
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['rerun']
    _literal_values = _parameters['literal_values']
    model = Path(model).resolve()
    receipt = _read(model / _admission['receipt_file'])
    if receipt.get('contract') == 'ffac-track1-incremental-inference-v1':
        return verify_sealed_incremental(model)
    _require(receipt.get('contract') == _admission['contract'] and Path(receipt.get('candidate') or '').resolve() == model and (receipt.get('model_identity') == list(_model_identity(model))) and (receipt.get('run_count') == _admission['run_count']) and (receipt.get('frame_count') == _admission['frame_count']) and (receipt.get('optical_training_performed') is False) and (receipt.get('regional_training_performed') is False) and (receipt.get('vertical_gap_max_length_px') == _literal_values['vertical_gap_max_length_px']) and (receipt.get('vertical_gap_rule') == _literal_values['vertical_gap_rule']) and (receipt.get('mass_eligible') is False), 'Inference rerun receipt is invalid')
    for relative, digest in receipt['trained_artifacts_preserved'].items():
        _bound(str(model / relative), digest, f'preserved {relative}')
    _bound(receipt['materialization_receipt_path'], receipt['materialization_receipt_sha256'], 'inference materialization receipt')
    _require(sha256_file(model / 'review_source/postflight_manifest.json') == receipt['review_source_sha256'] and sha256_file(model / 'phase_optical_regions.json') == receipt['inference_profile_sha256'], 'Inference source or nominal context binding changed')
    for binding in receipt['source_bindings']:
        _bound(binding['materialized_manifest_path'], binding['materialized_manifest_sha256'], f"{binding['run']} inference native manifest")
    return receipt

def _is_detached_review_candidate(root: Path) -> bool:
    """Honor an explicit detached-run receipt before registering a Studio model."""
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['rerun']
    _literal_values = _parameters['literal_values']
    if (root / 'inference_incremental_plan.json').is_file():
        verify_incremental(root)
        return True
    if (root / _admission['receipt_file']).is_file():
        verify_inference_rerun(root)
        return True
    request_path = root.parent.parent / 'regional_run' / 'launch_request.json'
    if not request_path.is_file():
        return False
    request = read(request_path)
    return request.get('review_candidate_only') is True and Path(str(request.get('output', ''))).resolve() == root.resolve()

def assemble_saved_packages(root, results, *, partial=False):
    _parameters = configured_parameters('package_recovery')
    _admission = _parameters['rerun']
    _literal_values = _parameters['literal_values']
    from .phase_saved_review_metadata import _asset
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness, RECEIPT, FORMAT
    packages = [Path(v) for v in results]
    if not packages:
        raise ValueError('Cannot publish an empty validation selection')
    for package in packages:
        require_reference_readiness(package, allow_pending=True)
    manifests = [read(p / 'postflight_manifest.json') for p in packages]
    expected_path = root / 'expected_review_runs.json'
    if expected_path.is_file():
        selected_jobs = [j for manifest in manifests for j in manifest['jobs']]
        expected_runs = read(expected_path)['runs']
        if partial:
            actual = [j['run'] for j in selected_jobs]
            if len(set(actual)) != len(actual) or not set(actual) <= set(expected_runs):
                raise ValueError('Partial validation contains duplicate or unexpected experiments')
        else:
            require_run_set(selected_jobs, expected_runs)
    bindings = [read(p / 'validation_binding.json') for p in packages]
    for key in ('model_sha256', 'pipeline_sha256', 'physics_runtime'):
        if len({b[key] for b in bindings}) != 1:
            raise ValueError('Run packages do not identify one model')
    source_root = root / 'review_source'
    source_path = source_root / 'postflight_manifest.json'
    source_digest = sha256_file(source_path)
    source_jobs = read(source_path)['jobs']
    source_by_run = {job['run']: job for job in source_jobs}
    if len(source_by_run) != len(source_jobs):
        raise ValueError('Frozen source campaign has duplicate experiments')
    for package, manifest, binding in zip(packages, manifests, bindings):
        jobs = manifest['jobs']
        if len(jobs) != 1:
            raise ValueError('A validation run package must contain exactly one experiment')
        run = jobs[0]['run']
        per_run_source = source_root / run / 'postflight_manifest.json'
        per_run_digest = sha256_file(per_run_source)
        if manifest.get('source_manifest_sha256') != per_run_digest or binding.get('source_manifest_sha256') != per_run_digest or read(per_run_source)['jobs'] != [source_by_run.get(run)]:
            raise ValueError(f'Validation package does not bind frozen source for {run}')
    destination = root / 'validation'
    import hashlib
    snapshot_material = 'aggregate-source-v2:' + source_digest + ':' + ''.join((sha256_file(p / name) for p in packages for name in ('postflight_manifest.json', 'validation_binding.json', RECEIPT)))
    snapshot_id = hashlib.sha256(snapshot_material.encode()).hexdigest()[:20]
    destination = destination / 'prepared_sets' / snapshot_id
    manifest = dict(manifests[0], jobs=[j for m in manifests for j in m['jobs']], source_campaign_root=str(source_root), source_manifest_sha256=source_digest)
    binding = dict(bindings[0], ffac_root=str(destination), ffum_root=str(destination), combined_root=str(destination), selected_candidates={k: v for b in bindings for k, v in b['selected_candidates'].items()}, source_manifest_sha256=source_digest, run_count=len(packages), completed_utc=utc_now())
    receipts = [read(p / RECEIPT) for p in packages]
    reference_status = 'ready' if all((r.get('status') == 'ready' for r in receipts)) else 'deferred'
    if (destination / 'validation_binding.json').is_file():
        require_reference_readiness(destination, allow_pending=True)
        existing = read(destination / 'validation_binding.json')
        if read(destination / 'postflight_manifest.json') != manifest or any((existing.get(key) != binding.get(key) for key in ('model_sha256', 'pipeline_sha256', 'physics_runtime', 'selected_candidates', 'combined_root', 'source_manifest_sha256'))):
            raise ValueError('Existing prepared snapshot changed')
        binding = existing
    else:
        write(destination / 'postflight_manifest.json', manifest)
        write(destination / 'validation_binding.json', binding)
        write(destination / RECEIPT, dict(format=FORMAT, status=reference_status, output_mode='review', binding=_asset(destination / 'validation_binding.json'), campaign=_asset(destination / 'postflight_manifest.json'), references=[r for receipt in receipts for r in receipt['references']], unavailable_references=[r for receipt in receipts for r in receipt['unavailable_references']]))
    from darsia.fluidflower.tools.phase_review_collections import refresh_prepared_collection
    collection = refresh_prepared_collection(root, destination)
    if collection is None:
        if (root / 'validation/registered_base_frozen.json').exists():
            raise ValueError('Registered validation base is preserved; restore its collection discovery before publication')
        for filename in ('postflight_manifest.json', 'validation_binding.json'):
            write(root / 'validation' / filename, read(destination / filename))
        receipt = read(destination / RECEIPT)
        receipt.update(binding=_asset(root / 'validation/validation_binding.json'), campaign=_asset(root / 'validation/postflight_manifest.json'))
        write(root / 'validation' / RECEIPT, receipt)
    settings_file = None
    if not _is_detached_review_candidate(root):
        from PySide6.QtCore import QSettings
        settings = QSettings('DarSIA', 'DarSIA Phase Model Studio')
        saved = json.loads(str(settings.value('modelHistory/completedModelRoots', '{}')))
        saved[root.name] = str(root)
        settings.setValue('modelHistory/completedModelRoots', json.dumps(saved))
        settings.sync()
        settings_file = settings.fileName()
    write(root / 'studio_preparation.json', dict(status='preparing' if partial else 'ready', runs=binding['run_count'], runs_done=binding['run_count'], presentation_ready=True, partial=partial, model_identity=list(_model_identity(root)), validation=str(destination), settings_file=settings_file))
