"""A saved depth map follows the numerical reader's location and stays inline."""

import os
import tomllib

import numpy as np
import pytest
import toml

import darsia
from darsia.fluidflower import depth_setup_ui as depth_ui, storage_paths
from darsia.presets.workflows.config.depth import DepthConfig


@pytest.fixture
def moved_map(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT"):
            monkeypatch.delenv(name)
    root = tmp_path / "project"
    root.mkdir()
    old, actual = tmp_path / "old-results", tmp_path / "archive"
    actual.mkdir()
    registry = tmp_path / "registry.toml"
    registry.write_text(toml.dumps({"schema_version": 1, "series": [
        {"id": "study", "root": str(root)}
    ]}))
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    (root / "series.toml").write_text(toml.dumps({
        "schema_version": 1, "series_id": "study",
        "storage": {"path_aliases": {str(old): str(actual)}},
    }))
    storage_paths.refresh_storage_paths()
    path = actual / "depth_map.npz"
    darsia.ScalarImage(
        np.array([[0.010, 0.011, 0.012], [0.013, 0.014, 0.015]]),
        height=0.5, width=0.9, origin=[0.2, 0.8],
    ).save(path)
    csv = actual / "measurements.csv"
    csv.write_text("x,y,depth\n0,0,0.01\n")
    config = root / "config.toml"
    config.write_text(toml.dumps({"depth": {
        "measurements": str(old / csv.name), "depth_map": str(old / path.name),
    }}))
    return config, path, csv


def test_moved_map_uses_numerical_paths_and_one_read_for_inline_map_and_range(
    moved_map, qtbot
):
    config, actual, csv = moved_map
    original = config.read_bytes()
    contents = tomllib.loads(original.decode())
    numerical = DepthConfig().load(config)
    effective = depth_ui.effective_depth_settings(contents, config_path=config)
    assert effective["measurements"] == numerical.measurements == csv
    assert effective["depth_map"] == numerical.depth_map == actual
    reads = []

    def reader(path):
        reads.append(path)
        return darsia.imread(path)

    saved = depth_ui.load_saved_depth_map(contents, config_path=config, reader=reader)
    panel = depth_ui.DepthSetupPanel()
    qtbot.addWidget(panel)
    panel.set_configuration(effective)
    panel.set_saved_depth_map(saved)
    assert reads == [actual]
    assert "0.01–0.015 m" in panel.status.text()
    assert str(actual) in panel.status.text()
    assert panel.measurements.text() == str(csv)
    assert panel.map_canvas.parentWidget() is panel
    assert not panel.map_canvas.isWindow()
    assert not panel.map_canvas.isHidden()
    np.testing.assert_allclose(panel.map_artist.get_array(), saved.image.img * 1000)
    np.testing.assert_allclose(panel.map_artist.get_extent(), [0.2, 1.1, 0.3, 0.8])
    assert panel.map_artist.axes.get_xlabel() == "x [m]"
    assert panel.map_artist.axes.get_ylabel() == "y [m]"
    assert panel.map_figure.axes[1].get_ylabel() == "Depth [mm]"
    assert config.read_bytes() == original
    assert contents == tomllib.loads(original.decode())


def test_explicit_missing_map_does_not_use_another_map_or_keep_previous_plot(
    moved_map, qtbot
):
    config, actual, _csv = moved_map
    saved = depth_ui.load_saved_depth_map(tomllib.loads(config.read_text()), config_path=config)
    panel = depth_ui.DepthSetupPanel()
    qtbot.addWidget(panel)
    panel.set_saved_depth_map(saved)
    missing = actual.parent / "different-map.npz"
    calls = []
    state = depth_ui.load_saved_depth_map({
        "depth": {"depth_map": str(missing)},
        "data": {"results": str(actual.parent)},
    }, config_path=config, reader=lambda path: calls.append(path))
    panel.set_saved_depth_map(state)
    assert state.path == missing and not calls
    assert "not saved yet" in panel.status.text()
    assert panel.map_artist is None and panel.map_canvas.isHidden()
    assert not panel.map_figure.axes


def test_series_relative_map_and_measurements_match_numerical_reader(tmp_path):
    (tmp_path / "series.toml").write_text('series_id = "study"\n')
    folder = tmp_path / "config"
    folder.mkdir()
    path = folder / "run.toml"
    contents = {"depth": {
        "measurements": "inputs/depth.csv", "depth_map": "maps/depth.npz",
    }}
    path.write_text(toml.dumps(contents))
    numerical = DepthConfig().load(path)
    effective = depth_ui.effective_depth_settings(contents, config_path=path)
    assert effective["measurements"] == numerical.measurements == tmp_path / "inputs/depth.csv"
    assert effective["depth_map"] == numerical.depth_map == tmp_path / "maps/depth.npz"
