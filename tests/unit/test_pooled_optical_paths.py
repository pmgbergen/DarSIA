"""Original numerical oracle and typed pooled-recipe boundaries on private data.

Native source/Build collection is not mocked or claimed: the production module
is an explicit numerical core, and these synthetic descriptor files exercise
its declared input contract only. No real RAW, fitted history or project moves.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import darsia
import numpy as np
import pytest

from darsia.fluidflower.mass import pooled_optical_paths as optical
from darsia.fluidflower.mass.n1_models import FrameData


def _bound(path, value):
    """Create real private descriptor bytes, not a claimed completed Build."""
    raw=(json.dumps(value, sort_keys=True)+'\n').encode()
    path.write_bytes(raw)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def recipe(tmp_path):
    """Explicit synthetic source contracts and two distinct equal-weight fit runs."""
    bindings={key:_bound(tmp_path/(key+'.json'), {'synthetic_contract':key})
              for key in ['adopted_revision','build_context','phase_configuration','native_configuration',
                          'baseline','clearwater','profile','source']}
    value=dict(format=optical.FORMAT, method=optical.METHOD, fit_runs=['alpha','beta'], excluded_runs=['heldout'],
        roles=[dict(facies_id=label,name=name,analysis_role=role) for label,name,role in
               [(0,'water','free_water'),(1,'sand-a','porous_medium'),(2,'sand-b','porous_medium'),
                (5,'sand-c','porous_medium'),(9,'chart','colorchecker'),(10,'seal','excluded'),(255,'outside','frame')]],
        runs={run:{**{k:deepcopy(bindings[k]) for k in ['baseline','clearwater','profile']},
                   'frames':[dict(case_id=run+'-1',frame_index=1,elapsed_h=1.0,target_h=1.0,source=deepcopy(bindings['source']))]}
              for run in ['alpha','beta']},
        **{k:deepcopy(bindings[k]) for k in ['adopted_revision','build_context','phase_configuration','native_configuration']},
        sampling=dict(stride_px=1,interior_radius_px=0,min_samples=16,min_runs=2),
        flash_bounds=dict(min_value_aq=0.,max_value_aq=1.,min_value_g=0.,max_value_g=1.),
        path_name_prefix='FF_AC common RAW facies ')
    return optical.recipe_from_document(value)


def _frame(recipe, run):
    """Actual FrameData validation with finite float32 native test buffers."""
    value=recipe.document
    facies=np.repeat(np.array([1,2,5],np.uint8),4)[None,:].repeat(12,axis=0)
    facies[0,:4]=[0,9,10,255]
    y,x=np.mgrid[:12,:12]
    base=np.full((12,12,3),.2,np.float32)
    scale=np.float32(1. if run=='alpha' else 1.3)
    change=np.stack([x/50+y/100,x/70+y/90,y/80-x/150],axis=-1).astype(np.float32)*scale
    current=base+change
    clear=np.ones_like(current)
    domain=np.ones((12,12),bool)
    eligible=np.isin(facies,[1,2,5])
    phase=np.ones((12,12),np.uint8)
    row=value['runs'][run]
    contract={k:value[k] for k in ['adopted_revision','build_context','phase_configuration','native_configuration']}
    contract.update({k:row[k] for k in ['baseline','clearwater','profile']})
    contract['source']=row['frames'][0]['source']
    return FrameData(run_id=run,frame_index=1,current_rgb=current,baseline_rgb=base,
        clear_water_rgb=clear,clear_water_supported_mask=domain.copy(),phase_labels=phase,facies=facies,
        domain=domain,supported_mask=domain.copy(),mass_eligible_mask=eligible,
        depth_m=np.full((12,12),.01),porosity=np.where(eligible,.4,0.),pixel_area_m2=np.full((12,12),1e-6),
        dimensions_m=(.012,.012),pressure_bar=1.,temperature_c=20.,elapsed_h=1.0,
        nominal_injected_mass_g=None,recipe={},provenance={'physical_context_verified':True,
            'facies_interior_radius_px':0,'optical_source_contract':contract})


def _samples(recipe):
    """Collect declared frames without changing their arrays or recipe."""
    return [optical.sample_frame_for_common_paths(_frame(recipe,run),recipe,case_id=run+'-1')
            for run in recipe.document['fit_runs']]


@pytest.fixture
def original():
    """Execute ONLY four frozen original function definitions with test DATA globals."""
    scope=dict(np=np,darsia=darsia,Mapping=Mapping,Path=Path,json=json,
        normalize_run=lambda run:run,CALIBRATION_RUNS=('alpha','beta'),HELDOUT_RUNS=('heldout',),
        SUPPORTED_FACIES=(1,2,5),UNQUANTIFIED_FACIES=(0,9,10,255),CONTRACT='original-numerical-oracle',
        DEFAULT_FLASH_BOUNDS=dict(min_value_aq=0.,max_value_aq=1.,min_value_g=0.,max_value_g=1.))
    path=Path(__file__).parent/'fixtures/original_pooled_optical_method.py.fixture'
    exec(compile(path.read_bytes(),str(path),'exec'),scope)
    return SimpleNamespace(**scope)


def test_sampler_and_fit_are_array_exact_original_method(recipe,original):
    """Compare original source on real FrameData, including every supported facies."""
    batches=[]
    pooled={label:{} for label in [1,2,5]}
    for run in recipe.document['fit_runs']:
        frame=_frame(recipe,run)
        before={k:getattr(frame,k).copy() for k in ['current_rgb','baseline_rgb','clear_water_rgb','facies','phase_labels']}
        expected,audit=original.sample_frame_for_common_paths(frame,stride_px=1,interior_radius_px=0)
        batch=optical.sample_frame_for_common_paths(frame,recipe,case_id=run+'-1')
        batches.append(batch)
        assert batch.audit['sample_counts']==audit['sample_counts']
        for label in pooled:
            np.testing.assert_array_equal(batch.samples[label],expected[label])
            pooled[label][run]=expected[label]
        for key,raw in before.items():np.testing.assert_array_equal(getattr(frame,key),raw)
    old,old_audit=original.fit_pooled_color_paths(pooled,min_samples=16,min_runs=2)
    actual,audit,_=optical.fit_pooled_color_paths(recipe,batches)
    assert audit==old_audit
    for label in old:
        np.testing.assert_array_equal(actual[label].color_path.colors,old[label].color_path.colors)
        np.testing.assert_array_equal(actual[label].values,old[label].values)
        np.testing.assert_array_equal(actual[label].color_path.equidistant_distances,old[label].color_path.equidistant_distances)


def test_new_package_matches_original_factory_for_active_and_ignored_roles(recipe,original,tmp_path):
    """New9/10 have actual old-factory files, never a borrowed porous response."""
    batches=_samples(recipe)
    paths,audit,sources=optical.fit_pooled_color_paths(recipe,batches)
    old=tmp_path/'original-package'
    original.save_common_path_package(old,paths,path_audit=audit,source_bindings=sources,
                                     flash_bounds=recipe.document['flash_bounds'])
    target=tmp_path/'new-package'
    bound=optical.save_common_path_package(target,recipe,batches)
    metadata=json.loads(Path(bound['metadata']['path']).read_bytes())
    assert metadata['label_ids']==[0,1,2,5,9,10,255]
    assert metadata['ignore_labels']==[0,9,10,255]
    assert metadata['calibration_runs']==['alpha','beta']
    assert metadata['recipe_sha256']==recipe.sha256
    for label in metadata['label_ids']:
        for folder,filename in [('color_path_interpretation',f'color_path_interpretation_{label}.json'),
                                ('signal_model',f'signal_model_{label}.csv')]:
            assert (target/folder/filename).read_bytes()==(old/folder/filename).read_bytes()
    assert (target/'flash/flash.json').read_bytes()==(old/'flash/flash.json').read_bytes()
    before={str(p.relative_to(target)):p.read_bytes() for p in target.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):optical.save_common_path_package(target,recipe,batches)
    assert before=={str(p.relative_to(target)):p.read_bytes() for p in target.rglob('*') if p.is_file()}


def test_changed_sample_bytes_refuse_instead_of_refitting_silently(recipe,tmp_path):
    batches=_samples(recipe)
    batches[0].samples[1][0,0]+=np.float32(.1)
    with pytest.raises(ValueError,match='sample bytes'):
        optical.save_common_path_package(tmp_path/'unwritten',recipe,batches)
    assert not (tmp_path/'unwritten').exists()


@pytest.mark.parametrize('damage',['source','roles','time'])
def test_changed_recipe_cannot_reuse_old_samples(recipe,damage):
    batches=_samples(recipe)
    value=recipe.document
    if damage=='source':value['runs']['alpha']['frames'][0]['source']['sha256']='0'*64
    elif damage=='roles':value['roles'][1]['name']='different actual identity'
    else:value['runs']['alpha']['frames'][0]['elapsed_h']=1.1
    changed=optical.recipe_from_document(value)
    with pytest.raises(ValueError,match='exact recipe'):
        optical.fit_pooled_color_paths(changed,batches)


def test_literal_source_contract_mismatch_refuses_before_sampling(recipe):
    frame=_frame(recipe,'alpha')
    frame.provenance['optical_source_contract']['build_context']={'path':'not-the-build'}
    with pytest.raises(ValueError,match='source contract'):
        optical.sample_frame_for_common_paths(frame,recipe,case_id='alpha-1')


def test_original_independent_physical_context_requirement_is_preserved(recipe,original):
    frame=_frame(recipe,'alpha')
    frame.provenance['physical_context_verified']=False
    for sample in [lambda:original.sample_frame_for_common_paths(frame,stride_px=1,interior_radius_px=0),
                   lambda:optical.sample_frame_for_common_paths(frame,recipe,case_id='alpha-1')]:
        with pytest.raises(ValueError,match='independently source-verified'):sample()


def test_unresolved_actual_build_is_plain_upstream_dependency(recipe,tmp_path):
    value=recipe.document
    value['build_context']=None
    pending=optical.recipe_from_document(value)
    assert pending.missing_inputs==('build_context',)
    with pytest.raises(ValueError,match='Actual adopted/Build/native source inputs'):
        optical.save_common_path_package(tmp_path/'unwritten',pending,[])
    assert not (tmp_path/'unwritten').exists()


@pytest.mark.parametrize('damage',['heldout','duplicate','missing'])
def test_only_the_exact_declared_frame_cohort_is_fitted(recipe,damage):
    batches=_samples(recipe)
    if damage=='heldout':batches[0]=replace(batches[0],run='heldout')
    elif damage=='duplicate':batches.append(batches[0])
    else:batches.pop()
    with pytest.raises(ValueError,match='exact recipe|incomplete'):
        optical.fit_pooled_color_paths(recipe,batches)


def test_unobserved_porous_role_stays_unquantified_with_original_load_only_factory(recipe,tmp_path):
    value=recipe.document
    value['roles'].append(dict(facies_id=11,name='no optical span',analysis_role='porous_medium'))
    revised=optical.recipe_from_document(value)
    samples=_samples(revised)
    bound=optical.save_common_path_package(tmp_path/'package',revised,samples)
    metadata=json.loads(Path(bound['metadata']['path']).read_bytes())
    assert metadata['path_audit']['11']['status']=='unquantified'
    assert 11 in metadata['ignore_labels']
    assert '11' in bound['color_paths'] and '11' in bound['signals']


def test_explicit_26_run_recipe_has_no_old_27_run_or_facies_id_rule(recipe):
    value=recipe.document
    names=[f'run{i:02d}' for i in range(26)]
    value['fit_runs']=names
    value['runs']={run:deepcopy(value['runs']['alpha']) for run in names}
    value['roles'][1]['facies_id']=101
    prepared=optical.recipe_from_document(value)
    assert len(prepared.document['fit_runs'])==26
    assert prepared.document['roles'][1]['facies_id']==101
    assert not prepared.missing_inputs


def test_sample_provenance_cannot_change_after_collection(recipe,tmp_path):
    batches=_samples(recipe)
    batches[0].audit['source_binding']['source']['sha256']='0'*64
    with pytest.raises(ValueError,match='source contract/audit changed'):
        optical.save_common_path_package(tmp_path/'unwritten',recipe,batches)
    assert not (tmp_path/'unwritten').exists()


def test_full_frame_provenance_survives_package_without_sharing_frame_state(recipe,tmp_path):
    """Keep source receipts and negative scope flags detached through the original package."""
    frames={run:_frame(recipe,run) for run in recipe.document['fit_runs']}
    expected={}
    batches=[]
    for run,frame in frames.items():
        evidence=_bound(tmp_path/(run+'-context.json'),{'synthetic_context':'source identities only'})
        frame.provenance.update(
            build=deepcopy(recipe.document['build_context']),
            current_bridge=deepcopy(evidence),phase=deepcopy(evidence),
            baseline_bridge=deepcopy(evidence),clearwater_bridge=deepcopy(evidence),
            geometry=deepcopy(evidence),context_evidence=deepcopy(evidence),
            physical_transport={'source':deepcopy(evidence),'matrix':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]},
            baseline_transport={'source':deepcopy(evidence)},clearwater_transport={'source':deepcopy(evidence)},
            source_binding_verification_grants_scientific_approval=False,
            source_qa={'physical_approval_claimed':False,'phase_categories_used':False,
                       'components_relabelled':False,'production_qualification_claimed':False},
            eligibility_scope='synthetic checked-frame fixture; no production approval')
        expected[run]=deepcopy(frame.provenance)
        batch=optical.sample_frame_for_common_paths(frame,recipe,case_id=run+'-1')
        assert batch.frame_provenance_json==optical._json(expected[run])
        batches.append(batch)
        frame.provenance['physical_transport']['matrix'][0][2]=7.
        frame.provenance['source_qa']['components_relabelled']=True
        frame.provenance['optical_source_contract']['source']['sha256']='0'*64
    bound=optical.save_common_path_package(tmp_path/'package',recipe,batches)
    metadata=json.loads(Path(bound['metadata']['path']).read_bytes())
    for run in frames:
        audit=metadata['source_bindings'][run][0]
        assert optical._json(audit['frame_provenance'])==optical._json(expected[run])
        assert audit['frame_provenance']['source_binding_verification_grants_scientific_approval'] is False
        assert all(value is False for value in audit['frame_provenance']['source_qa'].values())


def test_sampling_uses_the_same_provenance_snapshot_checked_before_numeric_work(recipe,monkeypatch):
    """A later mutable-frame change cannot replace the already checked source description."""
    frame=_frame(recipe,'alpha')
    expected=optical._json(frame.provenance)
    original=optical._reference_normalized_rgb

    def change_caller_provenance(value):
        """Change caller metadata during the original normalization without changing pixels."""
        value.provenance['physical_context_verified']=False
        value.provenance['optical_source_contract']['source']['sha256']='0'*64
        return original(value)

    monkeypatch.setattr(optical,'_reference_normalized_rgb',change_caller_provenance)
    batch=optical.sample_frame_for_common_paths(frame,recipe,case_id='alpha-1')
    assert batch.frame_provenance_json==expected
    assert optical._json(batch.audit['frame_provenance'])==expected
    assert frame.provenance['physical_context_verified'] is False


@pytest.mark.parametrize('damage',['flag','receipt','boolean_type'])
def test_full_frame_provenance_mutation_refuses_before_package_output(recipe,tmp_path,damage):
    """Do not repair a changed captured QA flag, source receipt or JSON scalar type."""
    batches=_samples(recipe)
    captured=batches[0].audit['frame_provenance']
    if damage=='flag':captured['physical_context_verified']=False
    elif damage=='receipt':captured['optical_source_contract']['source']['sha256']='0'*64
    else:captured['physical_context_verified']=1
    with pytest.raises(ValueError,match='source contract/audit changed'):
        optical.save_common_path_package(tmp_path/'unwritten',recipe,batches)
    assert not (tmp_path/'unwritten').exists()


def test_negative_physical_context_stays_unmodified_and_cannot_be_sampled(recipe):
    """Retaining provenance cannot upgrade a numeric-only physical or phase context."""
    frame=_frame(recipe,'alpha')
    frame.provenance.update(physical_context_verified=False,
        geometry={'physical_scale_approved':False,'edge_gate':False,'unobserved_wall_uncertainty':True},
        phase={'compatible_phase_approved':False},
        source_binding_verification_grants_scientific_approval=False)
    before=optical._json(frame.provenance)
    with pytest.raises(ValueError,match='independently source-verified'):
        optical.sample_frame_for_common_paths(frame,recipe,case_id='alpha-1')
    assert optical._json(frame.provenance)==before
