"""Independent run fits use original statistics, optimizer and frozen writer."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from darsia.fluidflower.pipeline import refit_jobs, refit_runner, local_refit, local_overflow_refit
from test_refit_runner import write_plan
from test_local_refit import local_metadata
from test_refit_jobs import series
from test_n1_refit import TRUE_GAIN, TRUE_BETA


def test_individual_plans_fit_only_their_run_and_write_separate_parameters(tmp_path):
    source, pin = write_plan(tmp_path / 'source')
    plans = refit_jobs.build_plans(statistics=source['statistics'], roles=source['runs'],
        rules=source['rules'], baseline_calibration=source['baseline']['calibration'],
        calibration_scope='individual')
    assert len(plans) == len(source['runs']['fit'])
    for target, plan in plans.items():
        run=plan['calibrated_run']
        assert plan['runs']['fit']==[run] and set(plan['statistics'])=={run}
        assert plan['source_roles']==source['runs']
        path=tmp_path/(target+'.json');path.write_text(json.dumps(plan))
        result=refit_runner.run_local_refit(dict(pipeline=refit_runner.PIPELINE,run=target,
            configuration=refit_jobs._descriptor(path),backend='cpu',precision='float32',threads=1,commit='a'*40),tmp_path/target)
        assert not result['stop_required']
        frozen=json.loads(Path(result['frozen_calibration']['path']).read_text())
        assert frozen['calibration_runs']==[run] and frozen['calibrated_run']==run
        assert frozen['parameters']['gain']==pytest.approx(TRUE_GAIN,rel=1e-4)
        assert frozen['parameters']['beta']==pytest.approx(TRUE_BETA,rel=1e-3)


def test_local_individual_job_uses_same_executor_and_exposes_all_results(local_metadata):
    case=local_metadata
    local_refit.create_local_refit_job(**case.arguments,calibration_scope='individual')
    result=local_refit.run_local_refit_job(case.root)
    assert result['status']['complete'] and result['frozen'] is None
    assert set(result['individual_calibrations'])==set(case.arguments['roles']['fit'])
    for run,(frozen,pin) in result['individual_calibrations'].items():
        assert frozen['calibration_runs']==[run]
        assert Path(pin['path']).is_file()


def test_individual_overflow_accepts_only_its_subset_of_the_shared_reviewed_preparation(tmp_path):
    from test_local_overflow_refit import source
    from darsia.fluidflower.mass.n1_pipeline import _write_json
    common,original=source(tmp_path)
    for run in common['runs']['fit']:
        plan=deepcopy(common)
        target='individual-'+run
        plan.update(run=target,calibration_scope='individual',calibrated_run=run,
            source_roles=deepcopy(common['runs']),runs=dict(fit=[run],heldout=[],descriptive=[],excluded=[]),
            statistics={run:common['statistics'][run]})
        pin=_write_json(tmp_path/(target+'.json'),plan)
        result=local_overflow_refit.run_fit(dict(original,run=target,configuration=pin),plan,tmp_path/target)
        frozen=json.loads(Path(result['frozen_calibration']['path']).read_text())
        assert frozen['calibrated_run']==run and frozen['calibration_runs']==[run]
        assert frozen['variant_f'] and result['parameters']['gain']>TRUE_GAIN

def test_individual_overflow_job_keeps_shared_level_preparation_and_separate_actual_fits(local_metadata, monkeypatch):
    import gzip
    from darsia.fluidflower.mass import overflow_statistics
    from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json
    case=local_metadata
    case.arguments['roles']['descriptive']=[]
    source=_write_json(case.series['tmp']/'overflow-input.json',dict(test_source=True))
    monkeypatch.setattr(local_overflow_refit,'read_request',lambda *a,**k:{})
    monkeypatch.setattr(local_overflow_refit,'preparation_budget',lambda *a,**k:dict(
        products_estimated_bytes=1024,transient_estimated_bytes=1024,workspace_estimated_bytes=1024))
    def prepare(job,plans,root,**kwargs):
        assert 'refit' in plans
        entries={}
        for run,rows in plans['refit']['statistics'].items():
            entries[run]=[]
            for entry in rows:
                row=refit_runner.read_statistics(DataReader(),entry['statistics'])
                row['contract']=overflow_statistics.STATISTICS_FORMAT
                row['overflow_steering']=dict(format='darsia-overflow-statistics-derivation-v1')
                row['main_gas_mass_g']*=.5
                path=root/(run+'-'+str(row['frame_index'])+'.json.gz')
                path.write_bytes(gzip.compress(json.dumps(row).encode(),mtime=0))
                entries[run].append(dict(entry,statistics=asset(path)))
        evidence=_write_json(root/'preparation.json',dict(format=local_overflow_refit.PREPARATION_FORMAT,
            statistics=entries,production_admitted=False,pending_nonfit_runs=[]))
        return {name:dict(deepcopy(plan),format=local_overflow_refit.PLAN_FORMAT,
            statistics={run:entries[run] for run in plan['statistics']},variant_f={'enabled':True},
            overflow_preparation=evidence) for name,plan in plans.items()}
    monkeypatch.setattr(local_overflow_refit,'prepare',prepare)
    local_refit.create_local_refit_job(**case.arguments,calibration_scope='individual',overflow=source)
    result=local_refit.run_local_refit_job(case.root)
    assert result['status']['complete']
    assert set(result['individual_calibrations'])==set(case.arguments['roles']['fit'])
    for run,(frozen,pin) in result['individual_calibrations'].items():
        assert frozen['variant_f'] and frozen['calibration_runs']==[run]
        original=DataReader().json(frozen['reference_without_overflow'])
        assert original['run']=='individual-'+run
        assert frozen['parameters']['gain']>original['parameters']['gain']
