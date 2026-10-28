"""The geometry page reads its series' files off the GUI thread.

On a series share the run files, series.toml and the geometry project take
seconds to read. A quick reading opens the page at once; a slow one shows a
loading note and opens the page when it answers. A synthetic series folder
in a temporary folder; no real series is used.
"""
import threading

import pytest

from darsia.fluidflower import background_read
from darsia.fluidflower import series_geometry_workspace as workspace_module
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_geometry_workspace import SeriesGeometryWorkspace


@pytest.fixture
def workspace(qtbot, tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    widget = SeriesGeometryWorkspace(series=[("syn", "Synthetic", root), ("two", "Second", tmp_path)],
                                     initial_series="syn")
    qtbot.addWidget(widget)
    widget.root = root
    return widget


def _inputs(root, *problems):
    return dict(rig_class="", run_inputs=[SeriesRunInputs("ac52", root / "ac52.toml")], project=None,
                checked={}, package=root / "geometry", problems=list(problems))


def _slow(monkeypatch, answers):
    """series_page_inputs waits for the returned event, and the window does not wait for it."""
    release, calls = threading.Event(), []

    def read(root):
        calls.append(root)
        assert release.wait(20)
        return answers(root)

    monkeypatch.setattr(workspace_module, "series_page_inputs", read)
    monkeypatch.setattr(background_read, "QUICK_ANSWER_S", 0.0)
    return release, calls


def test_a_slow_series_reading_shows_a_loading_note_then_the_page(workspace, qtbot, monkeypatch):
    release, calls = _slow(monkeypatch, lambda root: _inputs(root, "No geometry workspace yet."))
    assert workspace.ensure_page() is workspace.loading_panel
    assert workspace.loading_text.text() == "Reading the runs and geometry of Synthetic…"
    assert workspace._loaded_series is None and workspace.project is None
    # Asking again while it reads starts no other reading.
    assert workspace.ensure_page() is workspace.loading_panel
    release.set()
    qtbot.waitUntil(lambda: workspace._loaded_series == "syn", timeout=10000)
    assert calls == [workspace.root]
    assert workspace.pages.currentWidget() is workspace.create_panel
    assert workspace.reference_run.currentData() == "ac52"
    text = workspace.create_text.text()
    assert "No geometry workspace yet." in text and str(workspace.root / "geometry") in text
    # The package folder read with the page is shown without reading series.toml again.
    monkeypatch.setattr(workspace_module, "geometry_package_root", lambda root: pytest.fail("read again"))
    workspace._update_status()
    assert str(workspace.root / "geometry") in workspace.status.text()


def test_a_reading_of_a_series_no_longer_chosen_opens_nothing(workspace, qtbot, monkeypatch):
    release, calls = _slow(monkeypatch, lambda root: _inputs(root))
    assert workspace.ensure_page() is workspace.loading_panel
    workspace.series_id = "two"
    release.set()
    qtbot.wait(200)
    assert workspace._loaded_series is None and workspace.pages.currentWidget() is workspace.loading_panel


def test_a_quick_series_reading_opens_the_page_at_once(workspace, monkeypatch):
    monkeypatch.setattr(workspace_module, "series_page_inputs", lambda root: _inputs(root, "Quick."))
    assert workspace.ensure_page() is workspace.create_panel
    assert workspace._loaded_series == "syn" and "Quick." in workspace.create_text.text()


def test_the_page_inputs_are_read_from_the_series_files(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "syn"\nrig_class = "medium"\n[runs]\nids = ["ac52"]\n', encoding="utf-8")
    (root / "config" / "runs").mkdir(parents=True)
    (root / "config" / "runs" / "ac52.toml").write_text('[data]\nfolder = "."\nbaseline = "raw.png"\n',
                                                        encoding="utf-8")
    inputs = workspace_module.series_page_inputs(root)
    assert inputs["rig_class"] == "medium" and [row.run_id for row in inputs["run_inputs"]] == ["ac52"]
    assert inputs["project"] is None and inputs["checked"] == {}
    assert inputs["package"] == workspace_module.geometry_package_root(root)
    assert workspace_module.series_page_inputs(None) == dict(
        rig_class="", run_inputs=[], project=None, checked={}, package=None, problems=[])
