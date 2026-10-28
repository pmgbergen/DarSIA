"""Visible review delivery and transient discovery, without experiment DATA."""
from copy import deepcopy
import hashlib
import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from darsia.fluidflower import painting_series_use_ui as status
from darsia.fluidflower import painted_review_comparison as comparison

from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual
from .test_painting_gap_review_ui import _dialog


def test_review_window_is_resizable_screen_fit_with_uncapped_splitter(qtbot):
    widget, backend = _dialog(qtbot)
    flags = widget.windowFlags()
    assert flags & Qt.WindowType.WindowMaximizeButtonHint
    assert flags & Qt.WindowType.WindowMinimizeButtonHint
    assert widget.isSizeGripEnabled()
    assert widget.runs.maximumHeight() == 16777215
    assert widget.content_splitter.widget(0) is widget.runs
    assert widget.content_splitter.widget(1).isAncestorOf(widget.photo)
    assert widget.windowTitle() == 'Findings and decisions'
    assert widget.classification.itemText(1) == 'Water above the sand'
    assert widget.classification.itemData(1) == 'water_above_sand'
    assert widget.reason.text() == 'water and air above the sand'
    assert widget.approval is None and backend.approvals == []
    widget.showMaximized()
    qtbot.waitUntil(widget.isMaximized, timeout=2000)


def test_review_button_uses_registered_descriptor_without_a_picker(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    path = tmp_path / 'review.json'
    raw = b'{}'
    path.write_bytes(raw)
    binding = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    view.register_gap_review(binding)
    calls = []
    monkeypatch.setattr(view.adoption_controller, 'open_gap_review',
        lambda source, **options: calls.append((source, options)))
    def forbidden(*_args, **_options):
        raise AssertionError('Review must resolve the known descriptor without a picker.')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', forbidden)
    review = dict(map_version=deepcopy(record), review_binding=deepcopy(binding))
    reads = []
    monkeypatch.setattr(comparison, 'read_review', lambda source: reads.append(source) or deepcopy(review))
    view._open_gap_review_clicked()
    assert len(calls) == 1
    reader = calls[0][1]['review_reader']
    assert reader(None) == review and reads == [str(path)]
    review['review_binding']['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='exact path, SHA-256 or size'):
        reader(None)
    assert view._adoption_thread is None and view._placement_thread is None
    assert path.read_bytes() == raw


def test_series_review_discovery_keeps_literal_selected_version(tmp_path):
    root = tmp_path / 'series'
    selected = dict(folder=str(tmp_path / 'selected'), sha256='a' * 64)
    other = dict(selected, folder=str(tmp_path / 'other'))
    expected = dict(path=str(tmp_path / 'known-review.json'), sha256='b' * 64, size_bytes=2)
    for name, version, bound in (('own', selected, expected), ('other', other, dict(expected, sha256='c' * 64))):
        folder = root / 'map_placement' / name
        folder.mkdir(parents=True)
        (folder / 'index.json').write_text(json.dumps(dict(format=sorted(status._PLACEMENT)[0],
            created_utc='2026-10-22T11:00:00Z', map_version=version, runs=[], gap_review=bound)), encoding='utf8')
    result = status.existing_progress(root, tmp_path / 'no-current.json', selected)
    assert result['gap_reviews'] == [expected]
