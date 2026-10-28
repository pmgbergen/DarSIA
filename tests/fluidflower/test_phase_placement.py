"""Small physical and call-order checks for explicit lamp-context assembly."""
import copy
import numpy as np
import pytest
from darsia.fluidflower.phase import features, heads, placement, regional, rules
from test_phase_features_heads_regional import PARAMETERS, models


def data():
    p=copy.deepcopy(PARAMETERS)
    p['regional']['pooling']['review_denominator']=255
    return p


def arguments():
    p=data();model,_,_=models(p)
    shape=(9,11);log=np.log(np.full((3,*shape),300.,np.float32)).astype(np.float32)
    pixel=dict(fit=dict(mean=[0.]*4,scale=[1.]*4,coefficients=[.1,.2,.3,.4],intercept=.25))
    regional_model={'settings':p['regional']['settings']}
    return dict(original_model=model,base_pixel_model=pixel,base_regional_model=regional_model,current_log=log,aligned_baseline_log=log,context_domain=np.ones(shape,bool),light_epoch_id='new',spacing_m=(.001,.001),feature_shape_hw=(2979,5472),metadata={'frozen_input_marker':'unchanged'},parameters={'light_epoch_values':['old','new'],'identity_epoch':'old','base_model_key':'base','projected_model_key':'explicit_projection','regional_feature_count':20,'projected_applied_audit_key':'projected_indicator_applied'},feature_parameters=p['features'],heads_parameters=p['heads'],regional_parameters=p['regional'],rules_parameters={})


def test_identity_epoch_does_not_inspect_models_or_recompute(monkeypatch):
    args=arguments();args.update(light_epoch_id='old',original_model=None,base_pixel_model=None,base_regional_model=None,current_log=None,aligned_baseline_log=None)
    def forbidden(*a,**k):raise AssertionError('Identity epoch reevaluated optical evidence')
    monkeypatch.setattr(features,'log_to_unit_rgb',forbidden)
    monkeypatch.setattr(heads,'epoch_pixel_model',forbidden)
    monkeypatch.setattr(regional,'build_shared_regional_context',forbidden)
    got=placement.prepare_shift_context(**args)
    assert got.flat_valid is None and got.base_pixel is None and not got.logits
    assert got.metadata==args['metadata'] and got.light_epoch_id=='old'


def test_active_base_uses_single_presence_and_zeroes_invalid_before_logits(monkeypatch):
    args=arguments();events=[];shape=args['context_domain'].shape;mask=np.ones(shape,bool);mask[4,5]=False
    original_presence=heads.score_packed_presence;original_logits=heads.pixel_logits;original_shift=heads.shifted_probability;original_prepare=features.FeatureBuilder.prepare_rgb;sigmas=[]
    def prepare(self,image):
        sigmas.append(self.spatial_sigma)
        return original_prepare(self,image)
    def no_packed(*a,**k):raise AssertionError('Context used two-head presence scoring')
    def presence(*a,**k):events.append('single_presence');return original_presence(*a,**k)
    def flat(*a,**k):events.append('flatfield');value=np.full((*shape,4),.03,np.float32);value[~mask]=np.nan;return value,mask
    def logits(model,values,**k):
        assert (values[~mask]==0).all();events.append('logits');return original_logits(model,values,**k)
    def shifted(values,delta,valid=None):assert delta==0.;events.append('delta_zero');return original_shift(values,delta,valid)
    monkeypatch.setattr(features.FeatureBuilder,'prepare_rgb',prepare);monkeypatch.setattr(heads,'score_packed_heads',no_packed);monkeypatch.setattr(heads,'score_packed_presence',presence);monkeypatch.setattr(features,'masked_flatfield_features',flat);monkeypatch.setattr(heads,'pixel_logits',logits);monkeypatch.setattr(heads,'shifted_probability',shifted)
    got=placement.prepare_shift_context(**args)
    assert list(got.logits)==['base'] and got.metadata['projected_indicator_applied'] is False
    assert np.isnan(got.base_pixel[~mask]).all()
    assert events==['single_presence','flatfield','logits','delta_zero']
    assert got.metadata['optical_spatial_sigma_px']==max(args['feature_parameters']['optical']['texture_sigma'],args['feature_parameters']['optical']['spatial_context_fraction']*2979.)
    assert sigmas[0]==2. and sigmas[1]>2.
    assert got.metadata['gas_dependent_columns']==[1,11]
    assert got.metadata['frozen_input_marker']=='unchanged'


def test_optional_projection_requires_explicit_pair_and_preserves_model(monkeypatch):
    args=arguments();args['heads_parameters']['epoch']={'values':['old','new'],'positive_value':'new'}
    joint={'feature_names':[*args['heads_parameters']['flat_feature_names'],'indicator'],'fit':{'mean':[0.]*5,'scale':[1.]*5,'coefficients':[.1,.2,.3,.4,.5],'intercept':.25}}
    args['joint_pixel_model']=joint
    with pytest.raises(ValueError,match='both explicit'):placement.prepare_shift_context(**args)
    before=copy.deepcopy(joint);args['projected_regional_model']={'settings':dict(args['base_regional_model']['settings'])}
    got=placement.prepare_shift_context(**args)
    assert list(got.logits)==['base','explicit_projection'] and got.metadata['projected_indicator_applied'] is True
    assert joint==before
    args['projected_regional_model']['settings']['fine_sigma_m']=.0005
    with pytest.raises(ValueError,match='context settings'):placement.prepare_shift_context(**args)


def test_cached_base_fraction_uses_original_rule_implementation(monkeypatch):
    args=arguments();shape=args['context_domain'].shape;labels=np.ones(shape,np.uint8);labels[4:6,5:7]=2;args['base_labels']=labels;calls=[]
    def validated(value,**k):return value
    def cached(context,base,**k):
        assert context.base_pixel is not None and np.array_equal(base,labels)
        context.base_fraction=np.ones(shape,np.float32);calls.append('original_cached_rule')
    monkeypatch.setattr(placement,'_placement_labels',validated);monkeypatch.setattr(placement,'_cached_fraction',cached)
    got=placement.prepare_shift_context(**args)
    assert calls==['original_cached_rule'] and got.base_fraction is not None


def test_dtype_and_epoch_binding_fail_closed():
    args=arguments();args['current_log']=args['current_log'].astype(np.float64)
    with pytest.raises(ValueError,match='float32'):placement.prepare_shift_context(**args)
    args=arguments();args['light_epoch_id']='undeclared'
    with pytest.raises(ValueError,match='bound light'):placement.prepare_shift_context(**args)
    args=arguments();args['context_domain']=args['context_domain'].astype(np.uint8)
    with pytest.raises(ValueError,match='boolean'):placement.prepare_shift_context(**args)
