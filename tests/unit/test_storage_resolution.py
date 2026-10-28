"""Synthetic storage-only invariants; no production image or classifier needed."""
import numpy as np
import pytest
from darsia.fluidflower.storage_resolution import (
    resolution_mm,make_grid,metric_grid_shape,aggregate_static,aggregate_frame,PROBABILITY_FIELDS)


def grid(*,area=None,include=None,step=1.,precision="float32"):
    x=np.array([[.00025,.00075,.00125],[.00025,.00075,.00125]])
    y=np.array([[.00025]*3,[.00075]*3])
    return make_grid(x,y,np.ones((2,3)) if area is None else area,
                     include=np.ones((2,3),bool) if include is None else include,
                     extent_m=(0.,.002,0.,.001),storage_resolution_mm=step,precision=precision)


def fields(g):
    shape=g.native_shape
    mass=np.arange(1,7,dtype=np.float32).reshape(shape)/10
    return dict(concentration_aqueous=np.array([[1.,2.,3.],[4.,5.,6.]],np.float32),
        saturation_free_gas=np.full(shape,.5,np.float32),
        phase_labels=np.array([[0,1,2],[255,1,2]],np.uint8),
        masks={'main':np.array([[True,False,True],[True,True,False]])},
        probabilities={PROBABILITY_FIELDS[0]:np.full(shape,.25,np.float32),
                       PROBABILITY_FIELDS[1]:np.array([[False,False,True],[False,False,True]])},
        native_pixel_masses_g={name:mass.copy() for name in ('total','aqueous','gas')},
        native_mass_totals_g={name:float(mass.sum(dtype=np.float64)) for name in ('total','aqueous','gas')})


@pytest.mark.parametrize('value',[.25,.5,1.,2.])
def test_allowed_resolution(value):
    assert resolution_mm(value)==value


@pytest.mark.parametrize('value',[True,False,0,.3,-1,float('nan'),'1','coarse'])
def test_invalid_resolution(value):
    with pytest.raises(ValueError): resolution_mm(value)


def test_native_is_explicit_bypass():
    assert resolution_mm('native') is None
    assert make_grid(np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),include=np.ones((1,1),bool),
                     extent_m=(0,2,0,2),storage_resolution_mm='native') is None
    with pytest.raises(ValueError,match='bypass'):
        aggregate_static(None,depth_m=None,porosity=None,facies=None)


def test_area_weighted_means_and_mass_conservation():
    g=grid(area=np.array([[1,3,1],[1,1,1]],np.float32)); f=fields(g)
    out=aggregate_frame(g,**f)
    assert out['arrays']['concentration_aqueous'][0,0]==np.float32((1+6+4+5)/6)
    for name in ('total','aqueous','gas'):
        np.testing.assert_allclose(out['arrays']['cell_mass_'+name+'_g'].sum(dtype=np.float64),
                                   f['native_mass_totals_g'][name],rtol=1e-6,atol=0)
    assert all(value.dtype==np.float32 for value in out['arrays'].values())
    assert out['metadata']['computing_resolution']=='native'


def test_fractions_preserve_labels_and_masks():
    g=grid(); out=aggregate_frame(g,**fields(g))['arrays']
    total=sum(out['phase_'+n+'_area_fraction'] for n in ('water','dissolved','gas','excluded'))
    np.testing.assert_array_equal(total,np.ones(g.shape,np.float32))
    assert out['mask_main_area_fraction'][0,0]==np.float32(.75)
    assert out['phase_gas_area_fraction'][0,1]==1.


def test_q23_before_score_and_exact_final_gas_fraction():
    g=grid(); out=aggregate_frame(g,**fields(g))['arrays']
    np.testing.assert_array_equal(out[PROBABILITY_FIELDS[0]],np.full(g.shape,.25,np.float32))
    np.testing.assert_array_equal(out[PROBABILITY_FIELDS[1]],out["phase_gas_area_fraction"])


def test_missing_probability_is_a_hard_error():
    g=grid(); f=fields(g); del f['probabilities'][PROBABILITY_FIELDS[1]]
    with pytest.raises(ValueError,match='Final gas indicator'): aggregate_frame(g,**f)


def test_invalid_probability_not_replaced_or_clipped():
    g=grid(); f=fields(g); f['probabilities'][PROBABILITY_FIELDS[0]][0,0]=1.1
    with pytest.raises(ValueError,match='outside'): aggregate_frame(g,**f)


def test_nan_mean_retains_validity_fraction():
    g=grid(); f=fields(g); f['concentration_aqueous'][:,2]=np.nan
    out=aggregate_frame(g,**f)['arrays']
    assert np.isnan(out['concentration_aqueous'][0,1])
    assert out['concentration_valid_area_fraction'][0,1]==0.


def test_nonzero_mass_outside_coverage_is_not_dropped():
    include=np.ones((2,3),bool); include[0,0]=False; g=grid(include=include)
    with pytest.raises(ValueError,match='outside declared'): aggregate_frame(g,**fields(g))


def test_scalar_mass_mismatch_cannot_pass():
    g=grid(); f=fields(g); f['native_mass_totals_g']['total']*=1.01
    with pytest.raises(ValueError,match='within 1e-6'): aggregate_frame(g,**f)


def test_negative_mass_rejected():
    g=grid(); f=fields(g); f['native_pixel_masses_g']['total'][0,0]=-1
    with pytest.raises(ValueError,match='nonnegative'): aggregate_frame(g,**f)


def test_static_facies_area_majority_and_lowest_id_tie():
    g=grid(); f=np.array([[2,1,7],[1,2,7]],np.uint8)
    out=aggregate_static(g,depth_m=np.ones((2,3),np.float32),
                         porosity=np.full((2,3),.3,np.float32),facies=f)
    assert out['facies_majority'][0,0]==1
    assert out['facies_majority_area_fraction'][0,0]==.5
    assert out['pixel_area_m2_sum'][0,0]==4.
    assert out['depth_m_mean'].dtype==np.float32


def test_outside_metric_extent_rejected():
    with pytest.raises(ValueError,match='outside declared half-open'):
        make_grid(np.array([[2.]]),np.array([[0.]]),np.ones((1,1)),include=np.ones((1,1),bool),
                  extent_m=(0,1,0,1),storage_resolution_mm=1.)


@pytest.mark.parametrize('axis',['x','y'])
def test_exact_outer_boundary_stops_without_clamp(axis):
    x=np.array([[.0005]]); y=np.array([[.0005]])
    (x if axis=='x' else y)[0,0]=.001
    with pytest.raises(ValueError,match='Included native pixel centre lies outside declared half-open'):
        make_grid(x,y,np.ones((1,1)),include=np.ones((1,1),bool),
                  extent_m=(0,.001,0,.001),storage_resolution_mm=1.)


def test_wrong_native_shape_rejected():
    g=grid(); f=fields(g); f['phase_labels']=np.zeros((1,1),np.uint8)
    with pytest.raises(ValueError,match='native grid'): aggregate_frame(g,**f)


def test_non_boolean_mask_rejected():
    g=grid(); f=fields(g); f['masks']['main']=f['masks']['main'].astype(np.uint8)
    with pytest.raises(ValueError,match='boolean'): aggregate_frame(g,**f)


def test_float64_control_uses_one_storage_precision_setting():
    g=grid(precision="float64"); out=aggregate_frame(g,**fields(g))
    assert all(a.dtype==np.float64 for a in out['arrays'].values())
    assert out['metadata']['precision']=='float64'
    assert g.metadata()['storage_precision']=='float64'


def test_mixed_or_unknown_storage_precision_is_rejected():
    with pytest.raises(ValueError,match='declared storage precision'):
        grid(precision='mixed')


def test_q23_before_pending_does_not_suppress_final_fraction():
    g=grid(); f=fields(g); del f['probabilities'][PROBABILITY_FIELDS[0]]
    out=aggregate_frame(g,**f,before_status={'state':'pending','reason':'native centre absent'})
    assert PROBABILITY_FIELDS[0] not in out['arrays']
    np.testing.assert_array_equal(out['arrays'][PROBABILITY_FIELDS[1]],out['arrays']['phase_gas_area_fraction'])
    assert out['metadata']['before_rules_status']['state']=='pending'


def test_q23_missing_before_without_reason_is_not_silently_accepted():
    g=grid(); f=fields(g); del f['probabilities'][PROBABILITY_FIELDS[0]]
    with pytest.raises(ValueError,match='pending reason'): aggregate_frame(g,**f)


def test_q23_after_arbitrary_probability_is_rejected():
    g=grid(); f=fields(g); f['probabilities'][PROBABILITY_FIELDS[1]]=np.full(g.native_shape,.5,np.float32)
    with pytest.raises(ValueError,match='exact final gas indicator'): aggregate_frame(g,**f)


def test_q23_wrong_binary_after_mask_is_rejected():
    g=grid(); f=fields(g); f['probabilities'][PROBABILITY_FIELDS[1]][0,0]=True
    with pytest.raises(ValueError,match='exact final gas indicator'): aggregate_frame(g,**f)


def test_q28_count_probabilities_include_excluded_outside_domain_with_unequal_areas():
    include=np.ones((2,3),bool); include[0,1]=False
    g=grid(area=np.array([[1.,9.,1.],[2.,3.,1.]]),include=include)
    f=fields(g)
    f['phase_labels']=np.array([[0,255,2],[2,1,2]],np.uint8)
    f['probabilities']={PROBABILITY_FIELDS[0]:np.array([[0.,1.,.8],[np.nan,.5,.4]],np.float32),
                        PROBABILITY_FIELDS[1]:f['phase_labels']==2}
    for name,values in f['native_pixel_masses_g'].items():
        values[~include]=0.
        f['native_mass_totals_g'][name]=float(values.sum(dtype=np.float64))
    result=aggregate_frame(g,**f); out=result['arrays']; metadata=result['metadata']
    assert g.cell_pixel_count[0]==3 and g.cell_probability_pixel_count[0]==4
    assert out[PROBABILITY_FIELDS[0]][0,0]==np.float32(.5)
    assert out[PROBABILITY_FIELDS[0]+'_valid_pixel_fraction'][0,0]==np.float32(.75)
    assert out[PROBABILITY_FIELDS[1]][0,0]==np.float32(.25)
    assert out['phase_gas_area_fraction'][0,0]==np.float32(2/6)
    assert out[PROBABILITY_FIELDS[1]][0,0]!=out['phase_gas_area_fraction'][0,0]
    assert PROBABILITY_FIELDS[0]+'_valid_area_fraction' not in out
    assert PROBABILITY_FIELDS[1]+'_valid_pixel_fraction' not in out  # no label support mask supplied
    assert metadata['label_legend']=={'0':'water','1':'dissolved','2':'gas','255':'excluded'}
    assert metadata['probability_weighting']=='arithmetic pixel count, not pixel area'
    assert g.metadata()['probability_coordinate_counts']==dict(native_pixels=6,inside_extent=6,
                                                               outside_extent=0,invalid_coordinates=0)


def test_q28_zero_valid_and_empty_cell_remain_nan_with_literal_count_fractions():
    g=make_grid(np.array([[.00025,.00125]]),np.array([[.00025,.00025]]),
                np.array([[1.,9.]]),include=np.array([[True,False]]),extent_m=(0.,.003,0.,.001),
                storage_resolution_mm=1.,precision='float32')
    labels=np.array([[2,255]],np.uint8); mass=np.array([[1.,0.]],np.float32)
    result=aggregate_frame(g,concentration_aqueous=np.ones((1,2),np.float32),
        saturation_free_gas=np.zeros((1,2),np.float32),phase_labels=labels,masks={},
        probabilities={PROBABILITY_FIELDS[0]:np.array([[np.nan,.5]],np.float32),
                       PROBABILITY_FIELDS[1]:labels==2},
        native_pixel_masses_g={name:mass.copy() for name in ('total','aqueous','gas')},
        native_mass_totals_g={name:1. for name in ('total','aqueous','gas')})
    out=result['arrays']; before=out[PROBABILITY_FIELDS[0]]; valid=out[PROBABILITY_FIELDS[0]+'_valid_pixel_fraction']
    assert np.isnan(before[0,0]) and valid[0,0]==0.
    assert before[0,1]==.5 and valid[0,1]==1.
    assert np.isnan(before[0,2]) and np.isnan(valid[0,2])
    assert out[PROBABILITY_FIELDS[1]][0,0]==1. and out[PROBABILITY_FIELDS[1]][0,1]==0.
    assert np.isnan(out[PROBABILITY_FIELDS[1]][0,2])
    assert np.isnan(out['phase_gas_area_fraction'][0,1])  # unchanged area-domain product


def test_q28_literal_outside_extent_count_and_unassignable_coordinate_stop():
    x=np.array([[.00025,.00425]]); y=np.array([[.00025,.00025]])
    include=np.array([[True,False]])
    with pytest.raises(ValueError,match='Full-canvas native pixel centre lies outside declared half-open'):
        make_grid(x,y,np.ones((1,2)),include=include,extent_m=(0.,.003,0.,.001),
                  storage_resolution_mm=1.)
    x[0,1]=np.nan
    with pytest.raises(ValueError,match='Unassignable native probability coordinates: invalid=1'):
        make_grid(x,y,np.ones((1,2)),include=include,extent_m=(0.,.003,0.,.001),
                  storage_resolution_mm=1.)


@pytest.mark.parametrize('axis',['x','y'])
@pytest.mark.parametrize('value',[-.0001,.003,.00425])
def test_q34_outside_nonarea_canvas_centre_is_never_dropped(axis,value):
    x=np.array([[.00025,.00025]]); y=np.array([[.00025,.00025]])
    (x if axis=='x' else y)[0,1]=value
    with pytest.raises(ValueError,match='Full-canvas native pixel centre lies outside declared half-open'):
        make_grid(x,y,np.ones((1,2)),include=np.array([[True,False]]),
                  extent_m=(0.,.003,0.,.003),storage_resolution_mm=1.)


def test_q34_lower_and_internal_edges_follow_literal_floor_on_origin_grid():
    x=np.array([[-.001,-.0001,0.,.0009]]); y=np.zeros((1,4))
    g=make_grid(x,y,np.ones((1,4)),include=np.ones((1,4),bool),
                extent_m=(-.001,.001,0.,.001),storage_resolution_mm=1.)
    np.testing.assert_array_equal(g.cell_index,np.array([[0,0,1,1]],np.int64))
    np.testing.assert_array_equal(g.probability_cell_index,g.cell_index)
    assert 'no clamping' in g.metadata()['assignment']


def test_q34_serializes_existing_distinct_counts_and_partial_cell_area():
    included=np.ones((2,3),bool); included[0,1]=False
    g=grid(area=np.array([[1.,9.,1.],[2.,3.,1.]]),include=included)
    out=aggregate_static(g,depth_m=np.ones((2,3),np.float32),
                         porosity=np.full((2,3),.3,np.float32),facies=np.ones((2,3),np.uint8))
    np.testing.assert_array_equal(out['included_area_native_pixel_count'],np.array([[3,2]],np.int64))
    np.testing.assert_array_equal(out['full_canvas_probability_pixel_count'],np.array([[4,2]],np.int64))
    np.testing.assert_array_equal(out['pixel_area_m2_sum'],np.array([[6.,2.]],np.float32))
    assert out['included_area_native_pixel_count'].dtype==np.int64
    assert out['full_canvas_probability_pixel_count'].dtype==np.int64
    assert set(g.metadata()['cell_count_fields'])=={
        'static_included_area_native_pixel_count','static_full_canvas_probability_pixel_count'}


@pytest.mark.parametrize('extent',[(.0001,.0021,0.,.002),(0.,.0025,0.,.002),
                                   (0.,.002,.0001,.0021),(0.,.002,0.,.0025)])
def test_q34_off_lattice_origin_or_width_is_rejected(extent):
    with pytest.raises(ValueError,match='exact origin-zero metric lattice'):
        metric_grid_shape(extent,1.)
    with pytest.raises(ValueError,match='exact origin-zero metric lattice'):
        make_grid(np.array([[.0005]]),np.array([[.0005]]),np.ones((1,1)),
                  include=np.ones((1,1),bool),extent_m=extent,storage_resolution_mm=1.)


def test_q34_negative_origin_fractional_h_retains_integer_lattice_cells():
    g=make_grid(np.array([[-.000875,-.000125,.000125]]),np.full((1,3),.000125),
                np.ones((1,3)),include=np.ones((1,3),bool),
                extent_m=(-.001,.00025,0.,.00025),storage_resolution_mm=.25)
    assert g.shape==(1,5)
    np.testing.assert_array_equal(g.cell_index,np.array([[0,3,4]],np.int64))
    assert metric_grid_shape((-.001,.00025,0.,.00025),.25)==(1,5)


def test_q34_decimal_integer_edges_do_not_create_float_ceil_extra_cell():
    g=make_grid(np.array([[.3005,.3995]]),np.full((1,2),.0015),np.ones((1,2)),
                include=np.ones((1,2),bool),extent_m=(.3,.4,.001,.002),
                storage_resolution_mm=1.)
    assert g.shape==(1,100)
    np.testing.assert_array_equal(g.cell_index,np.array([[0,99]],np.int64))
