"""Current approved pixels precede their fresh non-mutating findings check."""
from types import SimpleNamespace

from darsia.fluidflower.series_geometry import RigFrame
from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual


def test_current_chart_prepares_before_old_findings_but_never_ignores_pending_edits(view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    selection = SimpleNamespace(index_path='selected-geometry.json')
    monkeypatch.setattr(view, 'current_frame', lambda: RigFrame({}))
    controller = view.adoption_controller
    controller.available = True
    monkeypatch.setattr(controller, 'findings_block_reason', lambda: 'The old mask has old findings.')
    prepared = []
    monkeypatch.setattr(view, 'prepare_adoption_of',
        lambda _record, _index, **options: prepared.append(options) or True)
    request = dict(approvals={'sample': 'exact existing approval'})
    assert controller.start(record, selection, local_reuse=request)
    assert prepared[0]['local_reuse'] == request
    controller.context = None
    assert not controller.start(record, selection)
    controller._findings_draft_contexts = {'current': 'unconfirmed edit'}
    assert not controller.start(record, selection, local_reuse=request)
    assert len(prepared) == 1
    controller._findings_draft_contexts = {}
