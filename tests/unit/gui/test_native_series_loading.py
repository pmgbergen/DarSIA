"""Slow metadata never blocks Qt or applies a superseded series context."""
import hashlib
import threading
import pytest

from PySide6.QtCore import QTimer, Qt

from .test_native_series_method import method_window  # noqa: F401


def ready(qtbot, window):
    qtbot.waitUntil(lambda: not window._series_loading and not window._series_loader.busy(), timeout=10000)


def configure_run(root):
    native = b'{"tag":"second-series"}'
    (root / 'native.json').write_bytes(native)
    (root / 'series.toml').write_text(
        'schema_version=1\nseries_id="series-b"\n[native]\nconfiguration="native.json"\n'
        f'sha256="{hashlib.sha256(native).hexdigest()}"\n[runs]\nids=["one"]\n', encoding='utf-8')
    path = root / 'config/merged/one.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('source_marker="second-series"\n[rig]\nwidth=0.9\nheight=0.49\n', encoding='utf-8')
    return path


def test_slow_series_read_keeps_qt_alive_then_opens_exact_prepared_reference(method_window, qtbot, monkeypatch):
    window = method_window
    ready(qtbot, window)
    root = window.phase_model_workspace.series_roots['series-b']
    path = configure_run(root)
    entered, release = threading.Event(), threading.Event()
    reader = window._series_loader.reader
    thread_ids, applications, ticks = [], [], []
    def slow(*args):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return reader(*args)
    window._series_loader.reader = slow
    apply = window._series_native_context.apply
    def applied(binding):
        applications.append(threading.get_ident())
        return apply(binding)
    monkeypatch.setattr(window._series_native_context, 'apply', applied)
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    try:
        window._open_series_from_start(str(root))
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: len(ticks) >= 5)
        assert window._series_loading and not window.config_file
        assert window.content_stack.currentWidget() is window._series_loading_page
        assert not window.toolbar_builder.play_action.isEnabled()
        release.set()
        ready(qtbot, window)
        assert not window._series_load_error
        assert window.config_file == str(path)
        assert window.config_dict['source_marker'] == 'second-series'
        assert window._config_series == 'series-b'
        assert window.reference_configuration_choice.run_id == 'one'
        assert window.sidebar.get_selected() == ('preprocessing', 'protocols')
        assert window._series_native_context.binding.document['tag'] == 'second-series'
        assert applications == [threading.get_ident()]
        assert thread_ids and all(value != threading.get_ident() for value in thread_ids)
    finally:
        release.set()
        timer.stop()
        ready(qtbot, window)


def test_new_selection_overtakes_slow_old_error_without_stale_context(method_window, qtbot):
    window = method_window
    ready(qtbot, window)
    reader = window._series_loader.reader
    entered, release = threading.Event(), threading.Event()
    def slow(series, *args):
        if series == 'series-b':
            entered.set()
            assert release.wait(5)
            raise OSError('old unavailable source')
        return reader(series, *args)
    window._series_loader.reader = slow
    try:
        window._select_series('series-b')
        qtbot.waitUntil(entered.is_set)
        window._select_series('series-a')
        qtbot.waitUntil(lambda: not window._series_loading)
        assert window._series_loader.busy()  # The old NAS request is still waiting.
        assert window._series_snapshot[0] == 'series-a'
        release.set()
        ready(qtbot, window)
        assert window.phase_model_workspace.series_id == 'series-a'
        assert window._series_snapshot[0] == 'series-a'
        assert not window._series_load_error
    finally:
        release.set()
        ready(qtbot, window)


def test_current_metadata_failure_finishes_loading_without_rebinding(method_window, qtbot):
    window = method_window
    ready(qtbot, window)
    previous = window._series_native_context.binding
    def unavailable(*args):
        raise OSError('selected source unavailable')
    window._series_loader.reader = unavailable
    window._select_series('series-b')
    ready(qtbot, window)
    assert 'selected source unavailable' in window._series_load_error
    assert window._series_snapshot is None
    assert window._series_native_context.binding is previous
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.home_page.new_button.isEnabled()


@pytest.mark.parametrize('change', ['unsaved_view', 'changed_disk'])
def test_reference_guard_uses_background_disk_snapshot_and_keeps_current_source(
        method_window, qtbot, monkeypatch, change):
    window = method_window
    ready(qtbot, window)
    roots = window.phase_model_workspace.series_roots
    path = configure_run(roots['series-b'])
    window._open_series_from_start(str(roots['series-b']))
    ready(qtbot, window)
    assert window.config_file == str(path)
    accepted = window._series_native_context.binding
    warnings = []
    from darsia.fluidflower import native_darsia_app as native
    monkeypatch.setattr(native.QMessageBox, 'warning', lambda *args: warnings.append(args))
    if change == 'unsaved_view':
        window.config_dict['source_marker'] = 'unsaved-view'
    else:
        path.write_text('source_marker="changed-disk"\n[rig]\nwidth=0.9\nheight=0.49\n', encoding='utf-8')
    # The original guard's synchronous disk reader must not be used in this flow.
    monkeypatch.setattr(window.reference_open_guard, '_read',
        lambda *_: pytest.fail('Reference guard performed a GUI-thread file read'))
    window._open_series_from_start(str(roots['series-a']))
    ready(qtbot, window)
    assert warnings
    assert window.phase_model_workspace.series_id == 'series-b'
    assert window._series_native_context.binding is accepted
    assert window.config_file == str(path)
    assert window.config_dict['source_marker'] == ('unsaved-view' if change == 'unsaved_view' else 'second-series')
    assert not window._series_loading


def test_later_open_and_status_refresh_use_changed_sources(method_window, qtbot, monkeypatch):
    window = method_window
    ready(qtbot, window)
    root = window.phase_model_workspace.series_roots['series-b']
    path = configure_run(root)
    window._open_series_from_start(str(root))
    ready(qtbot, window)
    assert window._current_series_snapshot() is None  # Initial-use packet has expired.
    path.write_text('source_marker="new-current-source"\n[rig]\nwidth=0.7\nheight=0.49\n', encoding='utf-8')
    # A fresh opening must not borrow old config values; the unchanged source
    # guard first requires reloading a file changed on disk.
    window._load_run_config(str(path))
    window._open_series_from_start(str(root))
    ready(qtbot, window)
    assert window.config_dict['source_marker'] == 'new-current-source'
    assert window.config_dict['rig']['width'] == .7
    assert '0.70' in window.rig_label.text()
    manifest = root / 'series.toml'
    manifest.write_text('metadata_revision=2\n' + manifest.read_text(encoding='utf-8'), encoding='utf-8')
    from darsia.fluidflower import workflow_status_ui
    from darsia.fluidflower.workflow_row_status import RowStatus
    monkeypatch.setattr(workflow_status_ui, 'row_status',
        lambda context, *args: RowStatus('missing', f"Source revision {context.manifest['metadata_revision']}"))
    window.workflow_status.refresh()
    qtbot.waitUntil(lambda: not window.workflow_status.checking, timeout=10000)
    assert window.workflow_status.states
    assert all(row.summary == 'Source revision 2' for row in window.workflow_status.states.values())


def test_active_workflow_keeps_accepted_series_and_context(method_window, qtbot, monkeypatch):
    window = method_window
    ready(qtbot, window)
    accepted = window._series_snapshot
    monkeypatch.setattr(type(window.process_runner), 'active', property(lambda self: True), raising=False)
    window._select_series('series-b')
    ready(qtbot, window)
    assert window.phase_model_workspace.series_id == accepted[0]
    assert window._series_snapshot[0] == accepted[0]
    assert window._series_native_context.binding is None


def test_loading_back_button_opens_real_series_list_and_new_selection_wins(method_window, qtbot):
    window = method_window
    ready(qtbot, window)
    roots = window.phase_model_workspace.series_roots
    configure_run(roots['series-b'])
    (roots['series-a'] / 'series.toml').write_text(
        'schema_version=1\nseries_id="series-a"\n[runs]\nids=["new"]\n', encoding='utf-8')
    path = roots['series-a'] / 'config/merged/new.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('source_marker="new-selection"\n', encoding='utf-8')
    entered, release = threading.Event(), threading.Event()
    reader = window._series_loader.reader
    def slow(series, *args):
        if series == 'series-b':
            entered.set()
            assert release.wait(5)
        return reader(series, *args)
    window._series_loader.reader = slow
    try:
        window._open_series_from_start(str(roots['series-b']))
        qtbot.waitUntil(entered.is_set)
        assert window._series_loading_text.text() == 'Loading series metadata…'
        assert window._series_loading_back.isVisible() and window._series_loading_back.isEnabled()
        qtbot.mouseClick(window._series_loading_back, Qt.MouseButton.LeftButton)
        assert window.content_stack.currentWidget() is window.home_page
        assert window.home_page.table.isVisible() and window.home_page.table.isEnabled()
        assert window.sidebar.get_selected() == ('start', 'series')
        assert window._pending_open_series is None and window._pending_series_navigation is None
        assert window._series_loading and not release.is_set()
        row = next(index for index in range(window.home_page.table.rowCount())
            if window.home_page.table.item(index, 1).text() == 'series-a')
        rectangle = window.home_page.table.visualItemRect(window.home_page.table.item(row, 0))
        qtbot.mouseClick(window.home_page.table.viewport(), Qt.MouseButton.LeftButton, pos=rectangle.center())
        qtbot.waitUntil(lambda: not window._series_loading and window.config_file == str(path))
        assert window._series_loader.busy()  # The old read is still held.
        assert window.phase_model_workspace.series_id == 'series-a'
        assert window.config_dict['source_marker'] == 'new-selection'
        generation = window._series_generation
        release.set()
        ready(qtbot, window)
        assert window._series_generation == generation
        assert window.phase_model_workspace.series_id == 'series-a'
        assert window.config_file == str(path)
        assert window._series_native_context.binding is None
    finally:
        release.set()
        ready(qtbot, window)
