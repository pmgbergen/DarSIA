"""One real Qt/queue/process recovery sequence around tiny synthetic science.

The scientific RAW executor is substituted in the spawned test worker; source
identity and filesystem observations are explicit private test fixtures.
Apply/CAS, metadata admission, retirement, GUI actions, source update, shared
admission and parent/child processes use their production implementations.
"""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import toml
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow

from darsia.fluidflower import job_control, native_jobs_registry, shared_admission_lock
from darsia.fluidflower import job_filesystem
from darsia.fluidflower.native_jobs_close import NativeJobsCloseController
from darsia.fluidflower.native_jobs_workspace import NativeJobsWorkspace
from darsia.fluidflower.pipeline import job_queue, series_setup_jobs, series_setup_operations
from darsia.fluidflower.pipeline.series_setup_sink import SetupMetadataSink
from darsia.fluidflower.shared_job_folder import SharedJobFolder
from ..test_series_setup_operations import (
    applied_plan, named_setup, imported, synthetic_payload_admission,
    prepared_painting, _saved, _document,
)
from ..test_series_setup_admission import _admission
from ..test_folder_cache_worker import _stores, _activate


@pytest.mark.parametrize('imported', ['agree'], indirect=True)
def test_failed_setup_close_reopen_recover_update_and_real_worker_output(applied_plan, tmp_path, monkeypatch, qtbot):
    case = applied_plan
    repo = Path(__file__).resolve().parents[3]
    root = tmp_path/'jobs'/'setup-job'
    root.mkdir(parents=True)
    controls = root/'setup-controls'
    controls.mkdir()
    series = case.data.series
    manifest = toml.loads((series/'series.toml').read_text())
    manifest.update(schema_version=1, series_id='synthetic', roots=dict(
        series='.', private=str(tmp_path), results=str(tmp_path/'results'), jobs=str(root.parent)))
    (series/'series.toml').write_text(toml.dumps(manifest), encoding='utf-8')
    plan = deepcopy(case.plan)
    plan['series_manifest'] = job_queue.descriptor(series/'series.toml')
    plan['control_root'] = str(controls)
    plan['runs'] = ['one']
    plan['build_plans'] = [row for row in plan['build_plans'] if row['run_id']=='one']
    (tmp_path/'results').mkdir()
    plan['output_scope']['metadata_roots'].extend((str(root), str(tmp_path/'results')))
    for row in plan['build_plans']:
        row['request_template']['series_context']['manifest'] = plan['series_manifest']
        row['output_scope'] = deepcopy(plan['output_scope'])
        Path(row['request_template']['output']['intended_root']).mkdir(parents=True, exist_ok=True)
    plan_pin = _document(controls/'plan.json', plan)
    monkeypatch.setattr(job_filesystem,'inspect_shared_folder',lambda path:
        SimpleNamespace(allowed=True, reason='private synthetic transport', evidence='test only'))
    config = dict(case.config, run='setup-apply', setup_plan=plan_pin, **{
        key: deepcopy(plan[key]) for key in ('series_manifest','runs','build_plans','output_scope')})
    series_setup_jobs.prepare_setup_job(plan_pin, root, job_id=root.name, commit='1'*40,
        environment_lock=dict(path='requirements-worker.lock', sha256='2'*64))
    (controls/'series-manifest.toml').write_bytes((series/'series.toml').read_bytes())
    stores = _stores(tmp_path, machine='fixture', roots=dict(
        series=str(series), private=str(tmp_path), results=str(tmp_path/'results'),
        jobs=str(root.parent)))
    _activate(stores, root, monkeypatch)
    from darsia.fluidflower.pipeline.run_io import RunIOBinding
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog, verify_inputs
    task=job_queue.read_json(root/'queue/setup-apply.json')
    task['attempt_id']='e'*32
    output=tmp_path/'results'/root.name/'setup-apply'/'.attempts'/task['attempt_id']
    output.mkdir(parents=True)
    local=job_queue.LocalOutputBinding('fixture',tmp_path/'results',output,
        job_filesystem.LocalFileAdmission(True,'fixture-local','synthetic filesystem observation',
            'private test only',str(tmp_path/'results')))
    monkeypatch.setattr(RunIOBinding,'_admit_native_code',lambda *a,**k:None)
    binding=RunIOBinding.replay(task,verify_inputs(task,resolve_catalog(task,stores,'synthetic')),
        local,repository=repo,admitted_commit=task['commit'])
    admission = _admission(tmp_path/'apply-capacity', monkeypatch,
        metadata_cap=8*1024**2, metadata_roots=plan['output_scope']['metadata_roots'],
        task=binding.task, configuration=binding.configuration)
    with binding.activate(), SetupMetadataSink(admission).activate():
        core = series_setup_operations.run_apply(config, plan, case.revision, case.data.project)
    core.update(format='darsia-series-setup-result-v1', complete=True,
        pipeline='series-setup', run='setup-apply', operation='apply', backend='cpu',
        precision='float32', setup_plan=plan_pin, worker_references_closed=True)
    output.rmdir()  # run_apply writes controls only; no pretend worker attempt.
    core_file = tmp_path/'actual-apply.json'
    _document(core_file, core)
    # This is the real transaction reader, not the simplified recovery seam.
    from darsia.fluidflower.setup_completed_apply import read_completed_apply
    assert read_completed_apply(config, plan, case.revision)['runs'][0]['run_id']=='one'
    configs_before = {path: path.read_bytes() for path in (series/'config').rglob('*.toml')}
    apply_before = (controls/'apply-complete.json').read_bytes()
    monkeypatch.setattr(native_jobs_registry, 'require_unc', lambda value: None)
    registry = native_jobs_registry.JobsUIRegistry(tmp_path/'ui-registry')
    registry.register_job(series, 'synthetic', root)
    folder = SharedJobFolder(root.parent)
    folder.publish(root, master_machine='fixture', cpu_workers=1, cuda_workers=0)
    # The installed source is an actual clean, committed tiny Git checkout.
    # Development edits in the test runner's checkout are not job identity.
    installed = tmp_path/'installed-source'
    installed.mkdir()
    (installed/'requirements-worker.lock').write_bytes((repo/'requirements-worker.lock').read_bytes())
    subprocess.run(['git','init','-q',str(installed)],check=True)
    subprocess.run(['git','-C',str(installed),'add','requirements-worker.lock'],check=True)
    subprocess.run(['git','-C',str(installed),'-c','user.name=Fixture',
        '-c','user.email=fixture@example.invalid','commit','-qm','Installed fixture'],check=True)
    from darsia.fluidflower import setup_job_reprepare
    from darsia.fluidflower.analysis_preparation import job_identity
    monkeypatch.setattr(setup_job_reprepare,'current_setup_identity',lambda:job_identity(installed))
    processes = []

    def launch(job, **options):
        env = dict(os.environ, DARSIA_TEST_FOLDER_COMPLETED_APPLY=str(core_file),
                   DARSIA_TEST_FOLDER_PARTIAL_BUILD='1')
        env['PYTHONPATH'] = os.pathsep.join((str(repo), str(repo/'src'), env.get('PYTHONPATH','')))
        command = [sys.executable, '-B', '-m', 'tests.unit._folder_worker_fixture',
            '--job', str(job), '--stores', str(stores.source['path']), '--machine', 'fixture',
            '--workers', str(options.get('workers', 1))]
        if options.get('once'):
            command.append('--once')
        if options.get('success'):
            env['DARSIA_TEST_FOLDER_BUILD_SUCCEEDS']='1'
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        processes.append(process)
        return dict(started=True, launcher_pid=process.pid)

    from darsia.fluidflower.job_launch_state import begin_launch_request
    begin_launch_request(root,machine='fixture',backend='cpu',workers=1,kind='local')
    state=job_control.read_control(root)
    if state['action']!='run':
        job_control.publish_control(root,'run',machine='fixture',expected_previous=state.get('command'))
    launch(root)
    initial = processes[-1]
    try:
        qtbot.waitUntil(lambda: initial.poll() is not None, timeout=60000)
        out, err = initial.communicate()
        assert initial.returncode == 0, out+'\n'+err
        assert (root/'done/setup-apply.json').is_file()
        assert (root/'failed/one.json').is_file()
        partial = Path(plan['build_plans'][0]['request_template']['output']['destination'])
        partial_before = {p.relative_to(partial):p.read_bytes() for p in partial.rglob('*') if p.is_file()}
        assert partial_before
        for claimed in (root/'claimed/fixture-cpu').glob('*.json'):
            task=job_queue.read_json(claimed)
            _document(root/'claim_observations'/(task['run']+'.'+task['attempt_id']+'.json'),
                dict(state='ownerless_claim_first_observed',run=task['run'],attempt_id=task['attempt_id'],
                     claimed_relative_path=claimed.relative_to(root).as_posix()))
        marker, _ = shared_admission_lock._paths(folder, 'fixture')
        marker.write_bytes(b'a'*32)

        class Window(QMainWindow):
            def closeEvent(self,event):
                self.jobs.prepare_close()
                if not self.jobs.can_close() or not self.controller.allow_close():
                    event.ignore()
                else:
                    super().closeEvent(event)

        def open_window():
            window=Window()
            window.jobs=NativeJobsWorkspace(window,registry=registry,local_machine='fixture',
                local_launcher=lambda job,**options:launch(job,success=True,**options))
            window.jobs.job_management._source_commit=job_identity(installed)['commit']
            window.setCentralWidget(window.jobs)
            window.jobs.stores.path=Path(stores.source['path'])
            window.jobs.shared_jobs._settings=QSettings(str(tmp_path/'settings.ini'),QSettings.Format.IniFormat)
            window.jobs.set_series('synthetic',series)
            window.controller=NativeJobsCloseController(window,window.jobs)
            qtbot.addWidget(window)
            window.show()
            qtbot.waitUntil(window.jobs.can_close,timeout=15000)
            return window

        window=open_window()
        window.close()
        qtbot.waitUntil(lambda:window.controller.thread is None)
        assert window.controller.stop_button.isEnabled(), window.controller.status.text()
        window.controller.stop_button.click()
        qtbot.waitUntil(lambda:window.controller.thread is None)
        assert job_control.read_control(root)['action']=='stop'
        from darsia.fluidflower.job_management import _retirement_binding
        for claimed in (root/'claimed/fixture-cpu').glob('*.json'):
            if job_queue.read_json(claimed)['operation']=='apply':
                continue  # Completed publication closes its own references.
            assert job_queue.retired_claim_evidence(root,claimed,
                local_output=_retirement_binding(root,claimed,stores.source['path']))
        window.controller.keep_button.click()
        qtbot.waitUntil(lambda:not window.isVisible())
        window=open_window()
        shared=window.jobs.shared_jobs
        shared.folder.setText(str(root.parent))
        shared.recover_admission.click()
        shared.admission_confirmation.setChecked(True)
        # Only the OS inventory is supplied in the synthetic host. The real
        # kernel lock, legacy marker and recovery action are left unchanged.
        monkeypatch.setattr(shared_admission_lock,'other_local_sessions',lambda:[])
        shared.admission_recover_now.click()
        qtbot.waitUntil(window.jobs.can_close,timeout=15000)
        assert job_queue.read_json(marker)['recovered_legacy']
        window.jobs.job_management.actor.setText('fixture-user')
        window.jobs.job_management.reprepare.click()
        qtbot.waitUntil(window.jobs.can_close,timeout=15000)
        assert not partial.exists(), window.jobs.job_management.note.text()
        assert (controls/'apply-complete.json').read_bytes()==apply_before
        assert all(path.read_bytes()==raw for path,raw in configs_before.items())
        assert len(registry.jobs(series,'synthetic'))==1
        assert len(list(root.glob('history/*/partial-builds/one/raw_build_owner.json')))==1
        # A code deployment while Pending must keep the same completed Apply.
        history_before=set((root/'history').iterdir())
        window.jobs.job_management.reprepare.click()
        qtbot.waitUntil(window.jobs.can_close,timeout=15000)
        assert window.jobs.job_management.note.text().startswith('This job already uses the current code')
        assert set((root/'history').iterdir())==history_before
        assert (controls/'apply-complete.json').read_bytes()==apply_before
        assert len(registry.jobs(series,'synthetic'))==1
        shared.automatic.setChecked(False)
        shared.cpu.setValue(1)
        shared.gpu.setValue(0)
        shared.timer.setInterval(100)
        shared.publish.click()
        qtbot.waitUntil(lambda:(root/'done/one.json').is_file(),timeout=60000)
        qtbot.waitUntil(lambda:all(p.poll() is not None for p in processes),timeout=15000)
        assert len(processes)>1
        result=job_queue.read_json(root/'results/one/completed.json')
        operation=job_queue.read_json(Path(result['files']['operation_result.json']['path']))
        assert operation['fixture_worker_pid']>0
        assert (partial/'figures/corrected_baseline.png').is_file()
        assert (controls/'apply-complete.json').read_bytes()==apply_before
        window.close()
        qtbot.waitUntil(lambda:window.controller.thread is None)
        window.controller.keep_button.click()
        qtbot.waitUntil(lambda:not window.isVisible())
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)
