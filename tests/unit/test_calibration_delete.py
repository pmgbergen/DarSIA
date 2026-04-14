from types import SimpleNamespace

from darsia.presets.workflows.calibration import calibration_color_paths as module


def _mock_config(
    calibration_file,
    baseline_color_spectrum_folder,
    color_range_file,
    cache,
):
    return SimpleNamespace(
        color_paths=SimpleNamespace(
            calibration_file=calibration_file,
            baseline_color_spectrum_folder=baseline_color_spectrum_folder,
            color_range_file=color_range_file,
        ),
        data=SimpleNamespace(cache=cache),
    )


def test_collect_existing_calibration_paths_to_delete_filters_non_existing_and_deduplicates(
    tmp_path, monkeypatch
):
    color_paths_dir = tmp_path / "color_paths"
    baseline_spectrum_dir = tmp_path / "baseline_color_spectrum"
    cache_dir = tmp_path / "cache"
    color_paths_dir.mkdir()
    baseline_spectrum_dir.mkdir()
    cache_dir.mkdir()

    duplicate_target = color_paths_dir
    missing_target = tmp_path / "missing_color_range"

    config = _mock_config(
        calibration_file=duplicate_target,
        baseline_color_spectrum_folder=baseline_spectrum_dir,
        color_range_file=missing_target,
        cache=duplicate_target,
    )
    monkeypatch.setattr(module, "FluidFlowerConfig", lambda *_args, **_kwargs: config)

    paths = module.collect_existing_calibration_paths_to_delete(tmp_path / "config.toml")

    assert paths == [duplicate_target, baseline_spectrum_dir]


def test_delete_calibration_without_confirmation_deletes_existing_targets(
    tmp_path, monkeypatch
):
    color_paths_dir = tmp_path / "color_paths"
    baseline_spectrum_dir = tmp_path / "baseline_color_spectrum"
    color_range_file = tmp_path / "color_range.npy"
    cache_dir = tmp_path / "cache"
    color_paths_dir.mkdir()
    baseline_spectrum_dir.mkdir()
    cache_dir.mkdir()
    color_range_file.write_text("x")

    config = _mock_config(
        calibration_file=color_paths_dir,
        baseline_color_spectrum_folder=baseline_spectrum_dir,
        color_range_file=color_range_file,
        cache=cache_dir,
    )
    monkeypatch.setattr(module, "FluidFlowerConfig", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("input() should not be called")
        ),
    )

    module.delete_calibration(tmp_path / "config.toml", require_confirmation=False)

    assert not color_paths_dir.exists()
    assert not baseline_spectrum_dir.exists()
    assert not color_range_file.exists()
    assert not cache_dir.exists()
