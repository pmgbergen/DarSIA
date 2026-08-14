from types import SimpleNamespace

from darsia.presets.workflows.calibration import calibration_color_paths as module


def _mock_config(calibration_root_1, calibration_root_2, cache):
    return SimpleNamespace(
        color=SimpleNamespace(
            embeddings={
                "a": SimpleNamespace(calibration_root=calibration_root_1),
                "b": SimpleNamespace(calibration_root=calibration_root_2),
            }
        ),
        data=SimpleNamespace(cache=cache),
    )


def test_collect_calibration_paths_filters_and_deduplicates(tmp_path, monkeypatch):
    color_embedding_1 = tmp_path / "color" / "embedding_a"
    color_embedding_2 = tmp_path / "color" / "embedding_b"
    cache_dir = tmp_path / "cache"
    color_embedding_1.mkdir(parents=True)
    color_embedding_2.mkdir(parents=True)
    cache_dir.mkdir()

    duplicate_target = color_embedding_1

    config = _mock_config(
        calibration_root_1=duplicate_target,
        calibration_root_2=color_embedding_2,
        cache=duplicate_target,
    )
    monkeypatch.setattr(module, "FluidFlowerConfig", lambda *args, **kwargs: config)

    paths = module.collect_existing_calibration_paths_to_delete(
        tmp_path / "config.toml"
    )

    assert paths == [duplicate_target, color_embedding_2]


def test_delete_calibration_without_confirmation_deletes_existing_targets(
    tmp_path, monkeypatch
):
    color_embedding_1 = tmp_path / "color" / "embedding_a"
    color_embedding_2 = tmp_path / "color" / "embedding_b"
    cache_dir = tmp_path / "cache"
    color_embedding_1.mkdir(parents=True)
    color_embedding_2.mkdir(parents=True)
    cache_dir.mkdir()

    config = _mock_config(
        calibration_root_1=color_embedding_1,
        calibration_root_2=color_embedding_2,
        cache=cache_dir,
    )
    monkeypatch.setattr(module, "FluidFlowerConfig", lambda *args, **kwargs: config)

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("input() should not be called")

    monkeypatch.setattr("builtins.input", _fail_if_called)

    module.delete_calibration(tmp_path / "config.toml", require_confirmation=False)

    assert not color_embedding_1.exists()
    assert not color_embedding_2.exists()
    assert not cache_dir.exists()
