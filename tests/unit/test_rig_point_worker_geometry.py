"""The existing transfer workers consume the exact reviewed point selection."""
from copy import deepcopy

import pytest

from darsia.fluidflower import rig_point_alignment as alignment
from darsia.fluidflower.map_placement import read_run_geometry
from darsia.fluidflower.series_layer_view import _reviewed_geometry_options
from .test_rig_point_alignment import case, context  # noqa: F401


def options(case):
    return dict(project=case.project,geometry_selection=case.selection,geometry_index=str(case.index),
        runs=[row['run_id'] for row in case.context.runs],reference_point_review=deepcopy(case.document))


def test_worker_handoff_freezes_the_reviewed_selection_before_transport(case):
    values=options(case)
    _reviewed_geometry_options(case.context.frame,values)
    assert values['geometry_selection'].source=='reviewed reference points'
    assert values['geometry_index']==values['geometry_selection'].index_path
    assert values['geometry_index']!=str(case.index)
    models=read_run_geometry(values['geometry_index'])
    assert models['target'].entry['_reference_point_review']
    assert models['target'].model['H_metric_mm_to_camera_u']==case.context.models['target'].metric_to_grid.tolist()


def test_changed_point_geometry_cannot_reuse_an_old_transfer(case):
    values=options(case)
    values['local_reuse']=dict(review=dict(geometry_index=dict(path=str(case.index),sha256='0'*64,size_bytes=1)))
    with pytest.raises(ValueError,match='Transfer the map again'):
        _reviewed_geometry_options(case.context.frame,values)


def test_same_reviewed_geometry_can_reuse_its_exact_transfer(case):
    selected=alignment.capture_reviewed_selection(case.project,case.selection,case.document,
        case.context.frame,[row['run_id'] for row in case.context.runs])
    values=options(case)
    values['local_reuse']=dict(review=dict(geometry_index=dict(path=selected.index_path,
        sha256=selected.sha256,size_bytes=selected.size_bytes)))
    _reviewed_geometry_options(case.context.frame,values)
    assert values['geometry_selection']==selected


def test_legacy_worker_options_remain_unchanged_without_a_point_review():
    values=dict(geometry_index='existing index')
    _reviewed_geometry_options(None,values)
    assert values==dict(geometry_index='existing index')
