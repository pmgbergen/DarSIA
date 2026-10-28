"""Passive manifest links only; no products, jobs or CSVs are read."""
from pathlib import Path

import pytest
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QSignalBlocker

from darsia.fluidflower.frozen_reference_line import FrozenReferenceLine, manifest_references
from .test_native_sidebar import native_window


def raw_reference(description="round 4, 39 runs, 1,564 images, calibration bce49292"):
    import toml
    return toml.dumps({"results": {"reference": {"description": description,
        "report": {"path": "results/final_report.txt", "sha256": "a" * 64},
        "table": {"path": "results/all39_mass.csv", "sha256": "b" * 64}}}}).encode()


def test_reference_is_manifest_data_only_without_following_files(tmp_path, monkeypatch):
    def forbidden(*_a, **_kw):
        pytest.fail("A frozen-reference line must not read/stat a product or write metadata")
    for method in ("open", "read_bytes", "write_bytes", "stat"):
        monkeypatch.setattr(Path, method, forbidden)
    saved = manifest_references(tmp_path, raw_reference())
    assert saved["links"]["report"] == {"path": str(tmp_path / "results" / "final_report.txt"), "sha256": "a" * 64}
    assert saved["description"].startswith("round 4")
    assert manifest_references(tmp_path, b'[results]\n') is None


@pytest.mark.parametrize("value", ["../outside.txt", "https://example.org/x", "C:relative.txt", "\\rooted-without-drive.txt"])
def test_reference_refuses_unsafe_or_nonfile_paths(tmp_path, value):
    import toml
    document = {"results": {"reference": {"description": "reference",
                "report": {"path": value, "sha256": "a" * 64},
                "table": {"path": "table.csv", "sha256": "b" * 64}}}}
    with pytest.raises(ValueError):
        manifest_references(tmp_path, toml.dumps(document).encode())


def test_widget_switch_reset_and_only_explicit_links_open(qtbot, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    line = FrozenReferenceLine()
    qtbot.addWidget(line)
    line.show_for_analysis(True)
    line.set_reference(tmp_path, raw_reference('<b>literal description</b>'))
    assert not line.isHidden()
    assert "&lt;b&gt;literal description&lt;/b&gt;" in line.text()
    assert "a" * 64 in line.toolTip() and "Saved SHA-256" in line.toolTip()
    assert not opened
    line.linkActivated.emit("report")
    assert [Path(item) for item in opened] == [tmp_path / "results" / "final_report.txt"]
    line.set_reference(tmp_path, b'[results]\n')
    assert line.isHidden() and line.text() == ""
    line.linkActivated.emit("table")
    assert len(opened) == 1
    line.set_reference(tmp_path, raw_reference())
    assert not line.isHidden()
    line.set_reference(tmp_path, b'[results.reference]\ndescription="broken"\n')
    assert "unavailable" in line.text() and line._links == {}
    line.linkActivated.emit("report")
    assert len(opened) == 1
    line.show_for_analysis(False)
    line.set_reference(tmp_path, raw_reference())
    assert line.isHidden()


@pytest.mark.parametrize("native_window", [("ffac", "other")], indirect=True)
def test_native_single_reference_line_tracks_series_and_analysis_only(native_window):
    window = native_window
    workspace = window.phase_model_workspace
    root = workspace.series_roots["ffac"]
    manifest = root / "series.toml"
    manifest.write_bytes(manifest.read_bytes() + raw_reference())
    before = manifest.read_bytes()
    with QSignalBlocker(workspace.series_selector):
        workspace.series_selector.setCurrentIndex(workspace.series_selector.findData("ffac"))
    window._refresh_method_state(force=True)
    window._set_method_category_visibility("jobs", "overview")
    assert not window.frozen_reference_line.isHidden()
    assert "39 runs" in window.frozen_reference_line.text()
    window._set_method_category_visibility("phase_model", "training")
    assert window.frozen_reference_line.isHidden()
    window._set_method_category_visibility("jobs", "overview")
    with QSignalBlocker(workspace.series_selector):
        workspace.series_selector.setCurrentIndex(workspace.series_selector.findData("other"))
    window._refresh_method_state(force=True)
    assert window.frozen_reference_line.isHidden() and window.frozen_reference_line.text() == ""
    assert manifest.read_bytes() == before
