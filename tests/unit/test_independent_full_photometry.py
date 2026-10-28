"""Current optical choices use real colour/field maths and immutable small inputs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from darsia.fluidflower.native import full_photometry as subject
from darsia.fluidflower.mass.n1_pipeline import asset


@pytest.fixture
def photo_parameters():
    source = Path(__file__).parents[1]/'fluidflower/fixtures/native_photometry_parameters.json'
    value = json.loads(source.read_bytes())
    value['fit_observed_field_int_1000'] = 20
    value['native_area_log_int_8'] = 2
    value['minimum_sensor_signal'] = 0.
    return value


def test_light_only_uses_stable_green_and_keeps_rgb_ratios(photo_parameters):
    yy,xx=np.mgrid[:32,:48]
    field=.1+.03*xx/48-.02*yy/32
    reference=np.broadcast_to(np.array([.2,.3,.4]),(32,48,3)).copy().astype(np.float32)
    current=reference*np.exp(-field[...,None]).astype(np.float32)
    fitted=subject._light_fit(current,reference,np.ones((32,48),bool),photo_parameters)
    corrected=subject._apply_light(np.moveaxis(current,-1,0),fitted,np.eye(3),(32,48))
    np.testing.assert_allclose(np.moveaxis(corrected,0,-1),reference,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(corrected[0]/corrected[1],current[...,0]/current[...,1],rtol=1e-6)
    assert fitted['producer'].endswith('._fit_observed_field')


def test_clearwater_keeps_chemical_signal_and_uses_only_fixed_baseline_light(photo_parameters, monkeypatch):
    shape = (32, 48)
    reference = np.broadcast_to(np.array([.2, .3, .4]), (*shape, 3)).copy().astype(np.float32)
    baseline = reference * np.float32(.8)
    fixed = subject._light_fit(baseline, reference, np.ones(shape, bool), photo_parameters)
    # Before indicator addition this source has a different colour/intensity.
    # Re-fitting green to the injection baseline would destroy that signal.
    clear = np.moveaxis(baseline * np.array([.4, .5, .6], np.float32), -1, 0)
    sensor = SimpleNamespace(source=dict(sha256='c' * 64), metadata=dict(white_level=1, black_level=0))
    processor = SimpleNamespace(binding=dict(contract='darsia-raw-rig-source-v1', native_shape_hw=list(shape)),
        processor=SimpleNamespace(independent=dict(record=dict(baseline_chart_roi_rc=None,
            baseline_light_fit=fixed, choice={}), corrections=dict(lighting=True, color_method='off'),
            stable_mask=np.ones(shape, bool), reference_swatches=np.empty((0,), np.float32))),
        parameters=dict(registration={}, photometry=photo_parameters,
            support=dict(coverage_tolerance=1e-6), source=dict(symbols=dict(reference_to_frame_h_key='h'))),
        baseline_sensor=SimpleNamespace(source=dict(sha256='b' * 64)), identity={},
        _render=lambda *a, **kw:(None, dict(native_to_camera_u_h=np.eye(3).tolist())))
    monkeypatch.setattr(subject.raw_positioning, 'camera_u', lambda *a, **kw:clear)
    monkeypatch.setattr(subject, '_light_fit', lambda *a, **kw:pytest.fail('refitted clear-water chemistry'))
    actual, audit = subject.correct_independent(processor, sensor, np.eye(3), np.eye(3), np.eye(3),
                                                source_purpose='clearwater')
    expected = np.moveaxis(reference * np.array([.4, .5, .6], np.float32), -1, 0)
    np.testing.assert_allclose(np.exp(actual), expected, rtol=1e-6, atol=1e-7)
    assert audit['photometry']['lighting_application'] == 'fixed-baseline-field'
    assert audit['photometry']['source_purpose'] == 'clearwater'


@pytest.fixture
def chart_image():
    import darsia
    from darsia.fluidflower.image_correction import srgb_to_linear
    swatches=darsia.ColorCheckerAfter2014().swatches_rgb
    srgb=np.repeat(np.repeat(swatches,80,axis=0),84,axis=1)
    unit=np.moveaxis(srgb_to_linear(srgb,dtype=np.float32),-1,0)
    roi=[[0,0],[0,srgb.shape[1]],[srgb.shape[0],srgb.shape[1]],[srgb.shape[0],0]]
    return unit,swatches,roi


@pytest.mark.parametrize('method',['darsia_original','uniform24'])
def test_current_colour_method_uses_its_real_existing_producer(chart_image,method):
    original,swatches,roi=chart_image
    shifted=original*np.array([.8,1.,.9],np.float32)[:,None,None]
    actual,info=subject._colour_full(shifted,method,swatches,roi)
    assert info['method']==method
    assert info['producer'] in {'darsia.ColorCorrection','fit_uniform24_to_reference_swatches'}
    assert np.mean(np.abs(actual-original)) < np.mean(np.abs(shifted-original))
    assert actual.dtype==np.float32 and actual.shape==original.shape


def test_off_colour_keeps_exact_source_and_nan_without_calling_colour(chart_image,monkeypatch):
    import darsia
    monkeypatch.setattr(darsia,'ColorCorrection',lambda **_:pytest.fail('colour disabled'))
    source=chart_image[0].copy()
    source[:,0,0]=np.nan
    actual,info=subject._colour_full(source,'off',None,None)
    np.testing.assert_array_equal(actual,source)
    assert info=={'method':'off'}


def test_chart_detection_uses_existing_four_corner_detector_and_keeps_invalid_pixels(monkeypatch):
    from darsia.fluidflower.phase_training import radiometry
    from darsia.fluidflower import image_correction
    image=np.full((3,6,8),.2,np.float32);image[:,0,0]=np.nan
    roi=np.array([[1,1],[1,7],[5,7],[5,1]])
    def detect(array):
        assert np.isfinite(array).all()
        return np.ones((4,6,3)),roi,'lower_left'
    monkeypatch.setattr(radiometry,'detect_chart',detect)
    def sample(array,**kw):
        assert np.isnan(array[0,0]).all()
        assert np.array_equal(kw['roi_rc'],roi)
        return np.ones((4,6,3),np.float32),roi
    monkeypatch.setattr(image_correction,'_reference_colorchecker',sample)
    _,actual=subject._chart(image)
    np.testing.assert_array_equal(actual,roi)


@pytest.mark.parametrize("materialize", [False, True])
def test_prepared_light_only_roundtrip_and_lazy_cohort(tmp_path,photo_parameters,monkeypatch,materialize):
    yy,xx=np.mgrid[:32,:48]
    base=np.broadcast_to(np.array([.2,.3,.4]),(32,48,3)).copy().astype(np.float32)
    frames={'one':base,'two':base*np.exp(-(.15+.03*xx/48))[...,None].astype(np.float32)}
    called=[]
    def load(run):
        called.append(run)
        pinfile=tmp_path/(run+'.json')
        if not pinfile.exists():pinfile.write_text(json.dumps({'run':run}))
        pin=asset(pinfile)
        return dict(run=run,rgb_full=np.moveaxis(frames[run],-1,0)*100,
            rgb_native=frames[run],white_range_dn=100.,native_to_camera_u_h=np.eye(3),
            chart_rgb=None,chart_roi_rc=None,shape_assets=[pin],image_metadata={},
            source_pins=dict(raw_acquisition=pin,camera_binding=dict(profile_id='camera'),
                             native_configuration=pin,baseline_renderer=pin),
            parameters=dict(photometry=photo_parameters, support=dict(coverage_tolerance=1e-6)))
    choice=dict(method='match_reference_lighting',color_method='off',reference_run='one')
    choicefile=tmp_path/'series.json';choicefile.write_text('{}')
    result=subject.prepare_independent_photometry(
        {run:lambda run=run:load(run) for run in frames},choice,
        {run:dict(mask=np.ones((32,48),bool),width_mm=10) for run in frames},
        tmp_path/'prepared',choice_authority=asset(choicefile), materialize_baselines=materialize)
    assert called==['one','two']
    assert result['reference_proposal']['selected'] is False
    binding=json.loads(Path(result['runs']['two']['source_binding']['path']).read_bytes())
    from darsia.fluidflower.native.raw_rig_source import RawRigSourceProcessor
    parameters=dict(source=dict(asset_aliases={},profiles={'two':'camera'},native_shape_hw=[32,48],
        source_id_pattern='[a-z]+',homography_min_scale=1e-12,homography_match_atol=1e-10,
        symbols={'reference_to_frame_h_key':'reference_h'}),photometry=photo_parameters,
        registration={},support=dict(coverage_tolerance=1e-6))
    # Only physical RAW decode/geometry are replaced; the actual source
    # constructor, descriptors, model loading and dispatch remain in use.
    def sensor(path,**kw):
        return SimpleNamespace(source=binding['baseline_raw'],profile_id='camera',
            metadata=dict(white_level=100,black_level=0,raw_visible_wh=[48,32]),
            distortion=(),ca_red=(),ca_blue=())
    monkeypatch.setattr(subject.raw_positioning,'read_sensor',sensor)
    monkeypatch.setattr(subject.raw_positioning,'camera_u',lambda *a,**k:np.moveaxis(frames['two'],-1,0)*100)
    monkeypatch.setattr(subject.raw_positioning,'load_shape_chain',lambda *a,**k:[])
    monkeypatch.setattr(subject.raw_positioning,'render_native',lambda *a,**k:(None,dict(
        native_to_camera_u_h=np.eye(3).tolist(),camera_u_wh=[48,32])))
    processor=RawRigSourceProcessor(result['runs']['two']['source_binding']['path'],parameters=parameters)
    assert processor.processor.independent['record']['old_photometry_used'] is False
    actual,info=processor.correct(dict(contract='darsia-raw-native-frame-v1',run='two',
        acquisition=binding['baseline_raw'],reference_to_native_h=np.eye(3).tolist(),
        baseline_to_native_h=np.eye(3).tolist()))
    np.testing.assert_allclose(np.exp(actual)/100,np.moveaxis(base,-1,0),rtol=.003,atol=1e-4)
    assert info['photometry']['color']=={'method':'off'}
    assert info['photometry']['light']['sample_count']>20
    assert info['photometry']['old_photometry_used'] is False
    if materialize:
        cached = json.loads(Path(result['runs']['two']['baseline']['path']).read_bytes())
        with np.load(cached['products']['path'], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['log_linear_rgb'], actual)
        assert cached['photometry_source'] == result['runs']['two']['source_binding']


def test_new_colour_choice_cannot_fall_through_to_saved_native_fit():
    processor=SimpleNamespace(binding=dict(normalisation=dict(choice=dict(
        method='off',color_method='uniform24'))))
    with pytest.raises(ValueError,match='newly prepared'):
        subject.correct(processor,None,None,None,None)


def test_runtime_configuration_changes_only_executor_identity(tmp_path):
    from darsia.fluidflower.pipeline.current_phase_recipe import require_runtime_configuration
    from darsia.fluidflower.native.raw_rig_source import execution_identity
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    original=dict(source=dict(asset_aliases={},raw_execution_identity={'historical':'a'*64}),
                  geometry=dict(height=.49,width=.899),photometry=dict(threshold=.1))
    old=tmp_path/'old.json';old.write_text(json.dumps(original))
    fresh=deepcopy(original);fresh['source']['raw_execution_identity']=execution_identity()
    path=tmp_path/'fresh.json';path.write_text(json.dumps(fresh))
    require_runtime_configuration(asset(old),asset(path),reader=DataReader())
    fresh['geometry']['height']=.5;path.write_text(json.dumps(fresh))
    with pytest.raises(ValueError,match='numerical or geometry'):
        require_runtime_configuration(asset(old),asset(path),reader=DataReader())


def test_runtime_configuration_accepts_added_storage_relocations_only(tmp_path):
    from darsia.fluidflower.pipeline.current_phase_recipe import require_runtime_configuration
    from darsia.fluidflower.native.raw_rig_source import execution_identity
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    target=tmp_path/'archive';target.mkdir()
    original=dict(source=dict(asset_aliases={str(tmp_path/'kept'):str(tmp_path/'kept')},raw_execution_identity={}),
                  geometry=dict(height=.49))
    old=tmp_path/'old.json';old.write_text(json.dumps(original))
    fresh=deepcopy(original);fresh['source']['raw_execution_identity']=execution_identity()
    fresh['source']['asset_aliases'][str(tmp_path/'moved')]=str(target)
    path=tmp_path/'fresh.json';path.write_text(json.dumps(fresh))
    require_runtime_configuration(asset(old),asset(path),reader=DataReader())
    fresh['source']['asset_aliases'][str(tmp_path/'kept')]=str(target);path.write_text(json.dumps(fresh))
    with pytest.raises(ValueError,match='relocate a saved Build asset'):
        require_runtime_configuration(asset(old),asset(path),reader=DataReader())


@pytest.mark.parametrize('method',['darsia_original','uniform24'])
def test_colour_only_uses_outside_crop_chart_and_never_fits_light(tmp_path,chart_image,photo_parameters,monkeypatch,method):
    chart,swatches,_=chart_image
    full=np.full((3,320,800),.3,np.float32)
    full[:,:,250:250+chart.shape[2]]=chart
    target=full*np.array([.8,1.,.9],np.float32)[:,None,None]
    roi=[[0,250],[0,754],[320,754],[320,250]]
    def baseline(run):
        path=tmp_path/(run+'.json');path.write_text(json.dumps({'run':run}))
        pin=asset(path)
        image=full if run=='reference' else target
        return dict(run=run,rgb_full=image,rgb_native=np.moveaxis(image[:,:,:200],0,-1),
            white_range_dn=1.,native_to_camera_u_h=np.eye(3),chart_rgb=subject._chart(image,roi)[0],
            chart_roi_rc=roi,shape_assets=[pin],image_metadata={},
            source_pins=dict(raw_acquisition=pin,camera_binding=dict(profile_id='camera')),
            parameters=dict(photometry=photo_parameters))
    one,two=baseline('reference'),baseline('target')
    # collect_baseline's public chart samples are linear, not transfer encoded.
    from darsia.fluidflower.image_correction import srgb_to_linear
    for row in (one,two):row['chart_rgb']=srgb_to_linear(row['chart_rgb'],dtype=np.float32)
    monkeypatch.setattr(subject,'_light_fit',lambda *a,**k:pytest.fail('lighting disabled'))
    choice=dict(method='off',color_method=method,reference_run='reference')
    result=subject.prepare_independent_photometry({'reference':one,'target':two},choice,{},tmp_path/'output',
        choice_authority=one['source_pins']['raw_acquisition'])
    binding=json.loads(Path(result['runs']['target']['source_binding']['path']).read_bytes())
    from darsia.fluidflower.native.raw_rig_source import RawRigSourceProcessor
    owner=RawRigSourceProcessor.__new__(RawRigSourceProcessor)
    owner.binding=binding
    owner.parameters=dict(source=dict(asset_aliases={},symbols={'reference_to_frame_h_key':'h'}),
                          photometry=photo_parameters,support=dict(coverage_tolerance=1e-6))
    owner.processor=SimpleNamespace(independent=subject.load_independent_model(owner))
    sensor=SimpleNamespace(source=binding['baseline_raw'],metadata=dict(white_level=1,black_level=0))
    owner.baseline_sensor=sensor;owner.baseline_camera_u=target;owner.identity={}
    owner._render=lambda *a,**k:(None,dict(native_to_camera_u_h=np.eye(3).tolist()))
    actual,info=subject.correct_independent(owner,sensor,np.eye(3),np.eye(3),np.eye(3))
    before=np.mean(np.abs(target[:,:,:200]-full[:,:,:200]))
    assert np.mean(np.abs(np.exp(actual)-full[:,:,:200])) < before
    assert info['chart']['roi_rc'][0][1] >= 250
    assert actual.shape==(3,320,200) and info['photometry']['light'] is None
