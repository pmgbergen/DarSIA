"""Code update preserves completed Build science and retries only remaining runs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from darsia.fluidflower import job_control, setup_completed_apply
from darsia.fluidflower.pipeline import job_queue, worker
from darsia.fluidflower.job_launch_state import begin_launch_request
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.corrected_setup import EXPORT_FILES, FORMAT
from darsia.fluidflower.pipeline.series_setup_protocol import APPLY_COMPLETE_FORMAT
from .test_series_setup_jobs import setup_job, controls  # noqa: F401
from .test_setup_job_reprepare import _unstarted_case, _doc, _retry, _bytes
from .test_folder_cache_worker import _activate
from ._folder_worker_fixture import work


def test_compact_completion_reference_preserves_exact_original_result(tmp_path):
    import pytest
    from darsia.fluidflower.setup_completed_builds import receipt_row
    _doc(tmp_path/'job.json', dict(job='original'))
    (tmp_path/'results/sample').mkdir(parents=True)
    result = dict(operation='build', input_catalog=['bound-original-data']*100)
    original = _doc(tmp_path/'results/sample/completed.json', result)
    row = dict(run='sample', metadata_root=str(tmp_path), result=result, output='unchanged')
    compact = receipt_row(row)
    assert compact['result_source'] == original and 'result' not in compact
    with pytest.raises(ValueError, match='completion changed'):
        receipt_row(dict(row, result=dict(result, operation='other')))


def _apply(case, monkeypatch):
    root = case.job_root
    base = root / 'exports/native_setup' / ('a'*32)
    rows, pins = [], []
    for i, build in enumerate(case.plan['build_plans']):
        run = build['run_id']
        directory = base if i == 0 else base / 'runs' / run
        directory.mkdir(parents=True)
        files = {}
        for filename in EXPORT_FILES:
            path = directory / filename
            path.write_bytes(b'unchanged synthetic exported map')
            files[filename] = job_queue.descriptor(path)
        export = _doc(directory/'export.json', dict(format=FORMAT, schema=1, frame='rig-corrected',
            revision_id='a'*32, run_id=run, reference_run=case.plan['display_reference_run'],
            current_sha256=case.plan['adopted_revision']['sha256'], files=files))
        before = _doc(directory/'apply-private/before.json', {})
        target = Path(build['configuration_paths'][0]); target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b'# retained original configuration\n')
        current = job_queue.descriptor(target)
        attempt = _doc(directory/'apply-private/receipt.json', dict(complete=True,run_id=run))
        claim = _doc(directory/'apply-claim.json', dict(attempt_directory=str(directory/'apply-private'),export=export,run_id=run))
        applied = _doc(directory/'apply-complete.json', dict(format='darsia-corrected-setup-apply-complete-v1',complete=True,run_id=run,revision_id='a'*32,
            export=export,attempt_receipt=attempt,configurations=[dict(before=before,after=current)]))
        rows.append(dict(run_id=run,export=export,apply_completion=applied,configuration_files=[current]))
        pins.extend([export,before,current,attempt,claim,applied])
    core = dict(format=APPLY_COMPLETE_FORMAT,complete=True,setup_plan=job_queue.descriptor(case.plan_path),
        **{k:deepcopy(case.plan[k]) for k in ('adopted_revision','project_manifest','series_manifest')},
        revision_id='a'*32,current_sha256=case.plan['adopted_revision']['sha256'],runs=rows,
        Build_run=False,optical_calibration_run=False)
    bound = _doc(root/'setup-controls/apply-complete.json',core)
    pins.append(bound)
    def completed(config,plan,revision):
        return dict(coordinator=deepcopy(core),bound=bound,runs=deepcopy(rows),guarded_geometry=[],metadata=pins)
    monkeypatch.setattr(setup_completed_apply,'read_completed_apply',completed)
    config = job_queue.read_json(root/'setup-controls/apply-operation.json')
    children=[]
    for build,row in zip(case.plan['build_plans'],rows):
        directory=root/'setup-controls/build'/build['run_id']
        request=_doc(directory/'raw-build-request.json',dict(deepcopy(build['request_template']),configuration_files=row['configuration_files']))
        operation=dict(format=config['format'],run=build['run_id'],operation='build',
            **{k:deepcopy(config[k]) for k in ('project_root','project_manifest','adopted_revision','series_manifest','setup_plan')},
            **{k:deepcopy(build[k]) for k in ('native_parameters','target_machine','metadata_output','output_scope')},
            configuration_files=row['configuration_files'],raw_build_request=request,apply_completion=row['apply_completion'],
            coordinator_completion=bound)
        op=_doc(directory/'operation.json',operation)
        children.append(dict(run_id=build['run_id'],configuration=op,raw_build_request=request,target_machine=build['target_machine']))
    result=dict(format='darsia-series-setup-result-v1',complete=True,pipeline='series-setup',run='setup-apply',operation='apply',
        backend='cpu',precision='float32',setup_plan=config['setup_plan'],worker_references_closed=True,
        coordinator_completion=bound,guarded_geometry=[],children=children,Build_run=False,optical_calibration_run=False)
    source=case.root/'synthetic-apply-result.json';_doc(source,result)
    monkeypatch.setenv('DARSIA_TEST_FOLDER_COMPLETED_APPLY',str(source))


def _run(case, monkeypatch):
    root=case.job_root
    prior=job_control.read_control(root)
    job_control.publish_control(root,'run',machine='private-host',expected_previous=prior.get('command'))
    begin_launch_request(root,machine='private-host',backend='cpu',workers=1,kind='local')
    _activate(case.stores,root,monkeypatch)
    monkeypatch.setattr(worker,'_work',work)
    args=SimpleNamespace(job=root,machine='private-host',backend='cpu',workers=1,once=False,
        exit_when_idle=True,watchdog=False,watchdog_seconds=30,_alias_stores=case.stores,
        stores=case.stores.source['path'],_cache_terminal_events=[])
    with (root/'logs/private-host-cpu.log').open('a',encoding='utf-8') as log:
        job=job_queue.read_json(root/'job.json')
        worker._parent(args,'private-host-cpu',job,job['commit'],log)


def test_five_completed_builds_and_partial_failure_keep_original_results_on_code_update(setup_job, monkeypatch):
    source=deepcopy(setup_job.plan['build_plans'][0])
    runs=['sample','two','three','four','five','zzz']
    setup_job.plan['runs']=runs
    setup_job.plan['metadata_output']=dict(maximum_output_bytes=20*1024**2,maximum_file_bytes=5*1024**2)
    setup_job.plan['build_plans']=[]
    for run in runs:
        build=deepcopy(source);build['run_id']=run
        build['metadata_output']=deepcopy(setup_job.plan['metadata_output'])
        build['configuration_paths']=[str(setup_job.root/'future'/(run+'.toml'))]
        build['request_template']['run_id']=run
        build['request_template']['output']['destination']=str(setup_job.root/'outputs'/run)
        setup_job.plan['build_plans'].append(build)
    case=_unstarted_case(setup_job,monkeypatch)
    _apply(case,monkeypatch)
    for build in case.plan['build_plans']:
        Path(build['request_template']['output']['intended_root']).mkdir(parents=True,exist_ok=True)
    monkeypatch.setenv('DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS',','.join(runs[:5]))
    monkeypatch.setenv('DARSIA_TEST_FOLDER_PARTIAL_BUILD','1')
    _run(case,monkeypatch)
    assert all((case.job_root/'done'/(run+'.json')).exists() for run in runs[:5])
    assert (case.job_root/'failed/zzz.json').exists()
    originals={run:_bytes(Path(build['request_template']['output']['destination']))
        for run,build in zip(runs,case.plan['build_plans'])}
    monkeypatch.setattr(worker,'_same_process',lambda identity:None)
    update=_retry(case)
    current=JobSnapshotReader()(case.registry.jobs(case.series,'synthetic')[0])
    assert [row['run'] for row in current['runs'] if row['state']=='Complete']==runs[:5]
    for run,build in zip(runs[:5],case.plan['build_plans']):
        assert _bytes(Path(build['request_template']['output']['destination']))==originals[run]
    assert not Path(case.plan['build_plans'][5]['request_template']['output']['destination']).exists()
    assert _bytes(Path(update['history'])/'partial-builds/zzz')==originals['zzz']
    # Another code-only Update while pending must retain the original five
    # publications too, rather than turning them into new completions.
    case.identity['commit']='4'*40
    _retry(case)
    monkeypatch.delenv('DARSIA_TEST_FOLDER_PARTIAL_BUILD')
    monkeypatch.setenv('DARSIA_TEST_FOLDER_BUILD_SUCCEEDS','1')
    _run(case,monkeypatch)
    current=JobSnapshotReader()(case.registry.jobs(case.series,'synthetic')[0])
    assert current['complete'],current['errors']
    assert [row['run'] for row in current['runs'] if row['state']=='Complete']==['setup-apply',*runs]
    assert not any((case.job_root/'results'/run).exists() for run in runs[:5])
    for run,build in zip(runs[:5],case.plan['build_plans']):
        assert _bytes(Path(build['request_template']['output']['destination']))==originals[run]
    from darsia.fluidflower.setup_completed_builds import retained
    import pytest
    changed=Path(case.plan['build_plans'][0]['request_template']['output']['destination'])/'shape_corrected_baseline.npz'
    changed.write_bytes(b'changed after completion')
    with pytest.raises(ValueError,match='retained completed Build output changed'):
        retained(case.job_root)


def test_pending_update_preserves_five_retained_and_six_new_builds_without_failure(setup_job, monkeypatch):
    source = deepcopy(setup_job.plan['build_plans'][0])
    runs = ['sample', *[f'two{i:02}' for i in range(1, 12)]]
    setup_job.plan['runs'] = runs
    setup_job.plan['metadata_output'] = dict(maximum_output_bytes=20*1024**2, maximum_file_bytes=5*1024**2)
    setup_job.plan['build_plans'] = []
    for run in runs:
        build = deepcopy(source)
        build['run_id'] = build['request_template']['run_id'] = run
        build['metadata_output'] = deepcopy(setup_job.plan['metadata_output'])
        build['configuration_paths'] = [str(setup_job.root/'future'/(run+'.toml'))]
        build['request_template']['output']['destination'] = str(setup_job.root/'outputs'/run)
        setup_job.plan['build_plans'].append(build)
    case = _unstarted_case(setup_job, monkeypatch)
    _apply(case, monkeypatch)
    for build in case.plan['build_plans']:
        Path(build['request_template']['output']['intended_root']).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS', ','.join(runs))
    monkeypatch.setenv('DARSIA_TEST_FOLDER_PAUSE_AFTER_COMPLETED_BUILD', runs[4])
    _run(case, monkeypatch)
    assert not list((case.job_root/'failed').glob('*.json'))
    assert all((case.job_root/'done'/(run+'.json')).is_file() for run in runs[:5])
    first = _retry(case)
    from darsia.fluidflower.setup_completed_builds import retained
    assert set(retained(case.job_root)) == set(runs[:5])
    monkeypatch.setenv('DARSIA_TEST_FOLDER_PAUSE_AFTER_COMPLETED_BUILD', runs[10])
    _run(case, monkeypatch)
    assert not list((case.job_root/'failed').glob('*.json'))
    assert all((case.job_root/'done'/(run+'.json')).is_file() for run in runs[5:11])
    originals = {run: _bytes(Path(build['request_template']['output']['destination']))
                 for run, build in zip(runs[:11], case.plan['build_plans'])}
    case.identity['commit'] = '4'*40
    second = _retry(case)
    kept = retained(case.job_root)
    assert set(kept) == set(runs[:11])
    receipt = job_queue.read_json(Path(second['history'])/'receipt.json')
    assert 'completed_builds' not in receipt.get('failed_preparation', {})
    assert all('result' not in row and 'result_source' in row
               for row in receipt['retained_completed_builds'])
    assert (Path(second['history'])/'receipt.json').stat().st_size < 1024**2
    assert first['history'] != second['history']
    for run, build in zip(runs[:11], case.plan['build_plans']):
        assert _bytes(Path(build['request_template']['output']['destination'])) == originals[run]
        assert not (case.job_root/'done'/(run+'.json')).exists()
    assert all(str(kept[run]['metadata_root']).startswith(first['history']) for run in runs[:5])
    assert all(str(kept[run]['metadata_root']).startswith(second['history']) for run in runs[5:11])


