"""Shared edits are atomic; a snapped closing edge keeps its neighbour's curve."""
from copy import deepcopy
import pytest
from darsia.fluidflower import layer_painting as lp
from .test_painting_follow_edges_and_shared_points import settings, make_view, view  # noqa: F401

LEFT=[(5.,5.),(5.,30.),(40.,30.),(40.,5.)]
RIGHT=[(5.,30.),(5.,55.),(40.,55.),(40.,30.)]
CURVED=[(5.,30.),(15.,32.),(25.,30.),(45.,32.),(55.,30.),(55.,5.),(5.,5.)]

def pair(view):
    view.add_polygon(lp.Polygon(2,list(LEFT)))
    view.add_polygon(lp.Polygon(3,list(RIGHT)))
    return deepcopy(view.session.to_dict()),len(view._undo_stack),len(view._redo_stack)

@pytest.mark.parametrize('invalid', ['collapsed','too_few','stale_index'])
def test_refused_shared_neighbour_keeps_entire_geometry_and_undo_history(view,invalid):
    original,undos,redos=pair(view)
    valid=[(5.,5.),(5.,32.),(40.,32.),(40.,5.)]
    changes={0:valid}
    changes[999 if invalid=='stale_index' else 1] = (
        [(5.,30.),(5.,55.)] if invalid=='too_few' else [(5.,30.)]*4)
    assert view._polygons_edited(changes)==[]
    assert view.session.to_dict()==original
    assert (len(view._undo_stack),len(view._redo_stack))==(undos,redos)
    assert [p.points for p in view.canvas._polygons]==[LEFT,RIGHT]

def test_valid_shared_edge_moves_both_members_with_one_undo_and_redo(view):
    original,undos,_redos=pair(view)
    changed={0:[(5.,5.),(5.,32.),(40.,32.),(40.,5.)],
             1:[(5.,32.),(5.,55.),(40.,55.),(40.,32.)]}
    assert view._polygons_edited(changed)==[0,1]
    assert len(view._undo_stack)==undos+1
    assert view.session.polygons[0].points[1:3]==[changed[1][0],changed[1][-1]]
    after=deepcopy(view.session.to_dict())
    view.undo();assert view.session.to_dict()==original
    view.redo();assert view.session.to_dict()==after

@pytest.mark.parametrize('free_first,free_last,follow',[(False,False,True),(True,False,True),(False,True,True),(False,False,False)])
def test_closing_edge_follows_existing_curve_only_with_two_snapped_ends(view,free_first,free_last,follow):
    view.add_polygon(lp.Polygon(2,list(CURVED)))
    original=deepcopy(view.session.to_dict());undos=len(view._undo_stack)
    view.select_layer_row(2)
    canvas=view.canvas
    for point,free in [((5.,30.),free_first),((5.,65.),True),((55.,65.),True),((55.,30.),free_last)]:
        canvas._add_draft_point(canvas.raw_to_scene(*point),free=free)
    assert len(canvas.draft_points)==4
    canvas.set_follow_edges(follow)
    assert canvas.close_polygon()
    points=view.session.polygons[-1].points
    if follow and not free_first and not free_last:
        assert points[-3:]==[(45.,32.),(25.,30.),(15.,32.)]
        assert len(points)==7
    else:
        assert len(points)==4
    assert view.session.polygons[0].points==CURVED
    assert len(view._undo_stack)==undos+1
    view.undo();assert view.session.to_dict()==original
