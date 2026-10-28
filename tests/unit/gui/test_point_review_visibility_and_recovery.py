"""Visibility switches do not alter data; relocated proposals resume in place."""
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
import toml

from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem

from darsia.fluidflower import rig_point_review as review
from .test_rig_point_review_ui import context  # noqa: F401
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401


def test_resume_after_storage_move_preserves_approval_without_rehashing_photos(context, qtbot, tmp_path, monkeypatch):
    view, panel, _ = context
    panel.auto_next.setChecked(False)
    panel.checked.setChecked(True)
    panel.approve.click()
    approved = deepcopy(panel.document['runs']['ac52'])
    view.close_reference_point_review()
    originals = view.static_reference_runs()
    aliases, moved = {}, []
    for reference in originals:
        old = reference.baseline_image
        new = tmp_path/'moved-photos'/old.name
        new.parent.mkdir(exist_ok=True)
        old.rename(new)
        aliases[str(old)] = str(new)
        moved.append(replace(reference, baseline_image=new))
    registry = tmp_path/'private-registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,storage=dict(path_aliases=aliases))),encoding='utf-8')
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(registry))
    from darsia.fluidflower.storage_paths import refresh_storage_paths
    refresh_storage_paths()
    view._static_reference_runs = lambda:moved
    def redundant_hash(*_args, **_kwargs):
        raise AssertionError('A storage alias must not hash every approved RAW again')
    monkeypatch.setattr(review,'verify_photo_relocation',redundant_hash)
    view.open_reference_point_review()
    qtbot.waitUntil(lambda:panel._thread is None and panel._viewed_identity is not None)
    assert panel.document['runs']['ac52'] == approved
    assert panel._viewed_identity == approved['photo']['identity']
    assert panel.canvas.preview['path'] == approved['photo']['path']


def test_points_and_lines_can_be_hidden_independently_without_changing_review(context):
    _, panel, _ = context
    before = deepcopy(panel.document)
    assert any(isinstance(item, QGraphicsEllipseItem) for item in panel._items)
    assert any(isinstance(item, QGraphicsPolygonItem) for item in panel._items)
    panel.show_points.setChecked(False)
    assert not any(isinstance(item, QGraphicsEllipseItem) for item in panel._items)
    assert any(isinstance(item, QGraphicsPolygonItem) for item in panel._items)
    panel.show_lines.setChecked(False)
    assert not panel._items
    panel.show_points.setChecked(True)
    assert any(isinstance(item, QGraphicsEllipseItem) for item in panel._items)
    assert not any(isinstance(item, QGraphicsPolygonItem) for item in panel._items)
    assert panel.document == before


def test_renamed_target_regains_proposal_on_resume_without_losing_source_approval(context, qtbot):
    view, panel, other = context
    panel.auto_next.setChecked(False)
    panel.checked.setChecked(True)
    panel.approve.click()
    approved = deepcopy(panel.document['runs']['ac52'])
    view.close_reference_point_review()
    current = other.with_name('corrected-folder-photo.ARW')
    other.rename(current)
    from dataclasses import replace
    inputs = view.static_reference_runs()
    target = replace(inputs[1], baseline_image=current)
    view._static_reference_runs = lambda: [inputs[0], target]
    # Saved draft predates the model recovery, as happens after a folder correction.
    frame = review.RigFrame.from_dict(panel.document['reference']['frame'])
    document = review.prepare_review(frame, [
        dict(run_id=inputs[0].run_id, path=inputs[0].baseline_image, image_size=(80,60)),
        dict(run_id=target.run_id, path=current),
    ])
    document['runs']['ac52'] = approved
    document['required_runs'] = ['ac52','ac53']
    review.save_review(view.project, document)
    view.open_reference_point_review()
    qtbot.waitUntil(lambda: panel._thread is None and panel.document['runs']['ac53']['proposed_rc'] is not None)
    assert panel.document['runs']['ac52'] == approved
    assert panel.document['runs']['ac53']['proposal_binding']['photograph_relocation']
    assert panel.document['runs']['ac53']['status'] == 'pending'
    assert panel.document['runs']['ac53']['approval'] is None
    panel.runs.selectRow(1)
    qtbot.waitUntil(lambda: panel.current_run()=='ac53' and panel._viewed_identity is not None)
    assert panel._items
    assert Path(panel.canvas.preview['path']) == current.resolve()
