"""The optical sampling choice is independent, explicit and lossless."""
from copy import deepcopy
from types import SimpleNamespace
import tomllib

import pytest

from darsia.fluidflower.colour_path_selection import save_colour_path_selection
from darsia.fluidflower.series_method import ManifestChangedError
from darsia.fluidflower.calibration_preparation import CalibrationPreparation
from darsia.fluidflower.refit_job_panel import RefitJobPanel


def manifest(tmp_path):
    raw = (b'\xef\xbb\xbfschema_version=1\r\nseries_id="sample"\r\n'
        b'[runs]\r\nids=["one"]\r\n[future]\r\nopaque=[1,2,3]\r\n'
        b'[image_selection]\r\nformat="saved"\r\n'
        b'[image_selection.runs.one]\r\nselected=[{frame_index=0,elapsed_h=-1},'
        b'{frame_index=1,elapsed_h=1}]\r\n'
        b'[method]\r\nactive="phase_model"\r\n'
        b'[method.phase_model]\r\nmodel_version="keep-model"\r\n')
    (tmp_path / "series.toml").write_bytes(raw)
    return raw


def test_save_preserves_master_images_method_unknown_values_and_line_convention(tmp_path):
    raw = manifest(tmp_path)
    before = tomllib.loads(raw.decode("utf-8-sig"))
    written = save_colour_path_selection(tmp_path, 15, expected_bytes=raw)
    after = tomllib.loads(written.decode("utf-8-sig"))
    expected = deepcopy(before)
    expected["image_selection"]["color_paths"] = dict(max_images_per_run=15)
    assert after == expected
    assert (tmp_path / "series.toml").read_bytes() == written
    assert written.startswith(b"\xef\xbb\xbf") and b"\r\n" in written


def test_stale_or_changed_context_rejects_save_without_overwriting(tmp_path):
    raw = manifest(tmp_path)
    changed = raw + b'\r\n[concurrent]\r\npreserve=true\r\n'
    (tmp_path / "series.toml").write_bytes(changed)
    with pytest.raises(ManifestChangedError, match="reload"):
        save_colour_path_selection(tmp_path, 10, expected_bytes=raw)
    assert (tmp_path / "series.toml").read_bytes() == changed
    with pytest.raises(ManifestChangedError, match="context changed"):
        save_colour_path_selection(tmp_path, 10, expected_bytes=changed, context_current=lambda: False)
    assert (tmp_path / "series.toml").read_bytes() == changed


@pytest.mark.parametrize("count", [0, 1001, True, 2.5])
def test_bad_count_leaves_manifest_unchanged(tmp_path, count):
    raw = manifest(tmp_path)
    with pytest.raises(ValueError):
        save_colour_path_selection(tmp_path, count, expected_bytes=raw)
    assert (tmp_path / "series.toml").read_bytes() == raw


def test_visible_choice_save_uses_real_writer_and_does_not_start_preparation(qtbot, tmp_path, monkeypatch):
    raw = manifest(tmp_path)
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    window = SimpleNamespace(open_image_selection_plan=lambda: None, _select_row=lambda *a: None,
        show_phase_model_workflow=lambda: None,
        ensure_jobs=lambda: pytest.fail("Saving sampling choices must not prepare/start jobs"))
    widget = CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.set_preparation_only_view(True)
    widget.show()
    widget.set_series(tmp_path)
    qtbot.waitUntil(lambda: not widget._preview.busy)
    assert widget.colour_count.value() == 15
    assert widget.save_colour_count.isEnabled()
    widget.colour_count.setValue(12)
    assert not widget.start.isEnabled()
    assert "Unsaved" in widget.colour_status.text()
    widget.save_colour_count.click()
    qtbot.waitUntil(lambda: widget._thread is None)
    saved = tomllib.loads((tmp_path / "series.toml").read_text(encoding="utf-8-sig"))
    assert saved["image_selection"]["color_paths"]["max_images_per_run"] == 12
    assert saved["image_selection"]["runs"] == tomllib.loads(raw.decode("utf-8-sig"))["image_selection"]["runs"]
    assert "✓" in widget.colour_status.text() and "No job started" in widget.colour_status.text()
    assert not widget.save_colour_count.isEnabled()
    assert widget.selection["preview_only"]
    widget.set_preparation_only_view(False)
    assert widget.colour_choices.isHidden()
