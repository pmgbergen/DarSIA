"""Source binding and publication across the isolated replay pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.normalisation_replay import execute_replay, _original_parameters, FITTER_SHA, RESIDUAL_SHA, AREA_SHA
from darsia.fluidflower.native.io import sha256_file


def test_literal_replay_parameters_require_the_original_source_identities():
    fit = dict(fitter_sha256=FITTER_SHA, local_residual_module_sha256=RESIDUAL_SHA)
    source = dict(source_renderer_sha256=AREA_SHA)
    data = _original_parameters(fit, source)
    assert data["area"]["cell_scale"] == 8
    assert type(data["residual"]["colour_sigma"]) is int
    assert type(data["residual"]["sigma_reference_cells"]) is float
    source["source_renderer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact recorded"):
        _original_parameters(fit, source)


def test_changed_manifest_prevents_replay_before_creating_output(tmp_path):
    (tmp_path / "series.toml").write_bytes(b"newer series")
    request = dict(series_root=str(tmp_path), manifest_bytes=b"prepared series")
    with pytest.raises(ValueError, match="series changed"):
        execute_replay(request, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_changed_bound_source_prevents_replay_before_raw_or_output(tmp_path):
    (tmp_path / "series.toml").write_bytes(b"series")
    source = tmp_path / "source.json"
    source.write_bytes(b"changed")
    request = dict(series_root=str(tmp_path), manifest_bytes=b"series",
                   sources=[dict(path=str(source), sha256="0" * 64)])
    with pytest.raises(ValueError, match="replay source changed"):
        execute_replay(request, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_source_reports_exact_restore_identity_before_raw_or_output(tmp_path):
    (tmp_path / "series.toml").write_bytes(b"series")
    missing = tmp_path / "original source" / "frame_000.jpg"
    digest = "1a43ebefbed71e791e27a89b030c3a28f952ffed314ed9ebd0151246acc82520"
    request = dict(series_root=str(tmp_path), manifest_bytes=b"series",
                   sources=[dict(path=str(missing), sha256=digest)])
    with pytest.raises(ValueError, match="Restore this exact file and prepare again") as error:
        execute_replay(request, tmp_path / "output")
    assert str(missing) in str(error.value) and digest in str(error.value)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("wrong_output", [False, True])
def test_raw_fit_apply_pipeline_publishes_only_exact_output(tmp_path, monkeypatch, wrong_output):
    from darsia.fluidflower.native import photometry, normalisation_raw, normalisation_installation, normalisation_reproduction, normalisation_apply, source

    sources = []
    def archive(name, **arrays):
        path = tmp_path / (name + ".npz")
        np.savez(path, **arrays)
        return bound(path)
    def array(name, data):
        path = tmp_path / (name + ".npy")
        np.save(path, data)
        return bound(path)
    def bound(path):
        result = dict(path=str(path), sha256=sha256_file(path))
        sources.append(result)
        return result

    rgb = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
    valid = np.ones((3, 4), bool)
    field = np.zeros((3, 4), np.float32)
    rows = []
    for identity in ("ac28", "ac50"):
        rows.append(dict(run=identity,
            baseline=dict(camera_arw=identity + ".arw", camera_jpeg=identity + ".jpg", original_review_jpeg=identity + "-review.jpg",
                camera_arw_sha256="raw", camera_jpeg_sha256="camera", original_review_jpeg_sha256="review"),
            source=dict(expanded_npz_key="log_rgb"), homography=np.eye(3).tolist(),
            source_certificate=archive(identity + "-certificate"),
            expected_grid=archive(identity + "-grid", log_rgb=rgb),
            mask=archive(identity + "-mask", sand=valid),
            chart=archive(identity + "-chart"), clipped=archive(identity + "-clipped", chart_clip=np.zeros((3, 24), bool)),
            saved_fit=archive(identity)))
    frame = tmp_path / "frame.json"
    frame.write_text(json.dumps(dict(source=dict(camera_arw="selected.arw", camera_jpeg="selected.jpg", original_review_jpeg="selected-review.jpg"))))
    frame_bound = bound(frame)
    (tmp_path / "series.toml").write_bytes(b"unchanged")
    expected_fit = {"ac28": object(), "ac50": object(), "reference": object()}
    parameters = _original_parameters(dict(fitter_sha256=FITTER_SHA, local_residual_module_sha256=RESIDUAL_SHA), dict(source_renderer_sha256=AREA_SHA))
    plan = dict(run="ac28", frame_index=7, verified_metadata=[{}, {}, dict(path="fit.json"), frame_bound],
        expected_reference=archive("reference"), reference_assets=archive("assets", consensus=rgb, flat=rgb),
        expected_log_rgb=array("rgb", rgb), expected_valid_mask=array("valid", valid))
    request = dict(series_root=str(tmp_path), manifest_bytes=b"unchanged", sources=sources,
        parameters=parameters, plan=plan, configuration=dict(photometry=dict(grid_shape_hw=[3, 4]), registration={},
            source=dict(asset_aliases={}, profiles=dict(ac28="old", ac50="new"), content_box_xywh=[0, 0, 4, 3], native_shape_hw=[3, 4],
                policy_ids=dict(keep="keep", raw_display="raw"))),
        fit=dict(grid_shape_hw=[3, 4], grid_origin_cell_xy=[0, 0], source_interior_erosion_cells=0,
            old_donors=["ac28"], new_donors=["ac50"], source_sand_old_donor_minimum=1, source_sand_new_donor_minimum=1, spatial_cv=False),
        rows=rows, support=dict(common_old_cells=12, common_new_cells=12), matrix=archive("matrix", M=np.eye(3)),
        convergence={}, old_fields=dict(ac28=archive("field", field=field)), processor_row=dict(source_manifest_sha256="source"),
        photo_aggregate=archive("photos"))
    calls = []
    def render(arw, camera, review, **kwargs):
        calls.append(("raw", str(arw), kwargs))
        return rgb.copy(), {}
    def fit(**kwargs):
        calls.append(("fit", kwargs))
        return dict(fits={key: expected_fit[key] for key in ("ac28", "ac50")}, reference=expected_fit["reference"],
            consensus=rgb.copy(), flat=rgb.copy(), old_fields=dict(ac28=field.copy()), iteration_history=[])
    def compare(actual, expected):
        assert actual is expected
    class Processor:
        def __init__(self, row, **kwargs):
            import cv2
            assert not cv2.ipp.useIPP()
            assert row == request["processor_row"]
            assert kwargs["display_mode"] == "raw" and kwargs["enclosed_hole_policy"] == "keep"
        def correct(self, **kwargs):
            raise AssertionError("Historical reproduction must not call the current scientific processor")
    def historical_apply(processor, **kwargs):
        assert processor.fit is expected_fit["ac28"] and processor.reference is expected_fit["reference"]
        np.testing.assert_array_equal(processor.old_field, field)
        calls.append(("apply", kwargs))
        return SimpleNamespace(log_linear_rgb=rgb + np.float32(wrong_output), required_valid_mask=valid.copy())
    monkeypatch.setattr(photometry, "ReferenceGrid", lambda shape, origin, content, **_: SimpleNamespace(shape_hw=shape, origin_cell_xy=origin))
    monkeypatch.setattr(photometry, "load_grid_baseline", lambda *_, **__: SimpleNamespace(xy=np.zeros((24, 2)), CB=np.zeros((3, 24))))
    monkeypatch.setattr(photometry, "load_baseline", lambda path, **_: (expected_fit[path.stem], {}))
    monkeypatch.setattr(normalisation_raw, "registered_edge_aware_raw_area_grid", render)
    monkeypatch.setattr(normalisation_installation, "fit_installation", fit)
    monkeypatch.setattr(normalisation_reproduction, "require_exact_fit", compare)
    monkeypatch.setattr(source, "SourceProcessor", Processor)
    monkeypatch.setattr(normalisation_apply, "apply_historical_frame", historical_apply)
    output = tmp_path / "output"
    output.mkdir()
    if wrong_output:
        with pytest.raises(ValueError, match="parity failed"):
            execute_replay(request, output)
        assert not list(output.iterdir())
        return
    result = Path(execute_replay(request, output))
    assert [call[0] for call in calls] == ["raw", "raw", "fit", "apply"]
    assert calls[2][1]["reference_run"] == "ac28" and calls[2][1]["new_light_runs"] == ["ac50"]
    assert calls[3][1]["frame_id"] == "frame_7" and str(calls[3][1]["camera_arw"]) == "selected.arw"
    np.testing.assert_array_equal(np.load(result / "log_linear_rgb.npy"), rgb)
    np.testing.assert_array_equal(np.load(result / "valid_mask.npy"), valid)
    receipt = json.loads((result / "reproduction.json").read_text())
    assert receipt["exact_saved_output_parity"] is True and receipt["frame_index"] == 7
