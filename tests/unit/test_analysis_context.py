from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.config.data_registry import DataRegistry


class FakeExperiment:
    def __init__(self, return_path: Path):
        self.return_path = return_path
        self.paths_calls = []
        self.times_calls = []

    def find_images_for_paths(self, paths):
        self.paths_calls.append(paths)
        return [self.return_path]

    def find_images_for_times(self, times, data=None):
        self.times_calls.append((times, data))
        return [self.return_path]


def create_config_stub():
    return SimpleNamespace(data=SimpleNamespace(data=[]))


@pytest.fixture
def experiment_factory():
    return FakeExperiment


@pytest.fixture
def config_stub():
    return create_config_stub()


def test_select_image_paths_resolves_registry_reference_to_paths(
    tmp_path, experiment_factory, config_stub
):
    image_path = tmp_path / "img.jpg"
    image_path.touch()
    registry = DataRegistry().load(
        {"path": {"imgs": {"paths": ["img.jpg"]}}}, data_folder=tmp_path
    )
    experiment = experiment_factory(image_path)
    sub_config = SimpleNamespace(data="imgs")

    resolved = select_image_paths(
        config=config_stub,
        experiment=experiment,
        sub_config=sub_config,
        data_registry=registry,
    )

    assert resolved == [image_path]
    assert experiment.paths_calls == [[image_path]]
    assert experiment.times_calls == []


def test_select_image_paths_resolves_registry_reference_to_times(
    tmp_path, experiment_factory, config_stub
):
    output_path = tmp_path / "selected.jpg"
    output_path.touch()
    registry = DataRegistry().load({"time": {"snap": {"times": ["01:00:00"]}}})
    experiment = experiment_factory(output_path)
    sub_config = SimpleNamespace(data="snap")

    resolved = select_image_paths(
        config=config_stub,
        experiment=experiment,
        sub_config=sub_config,
        data_registry=registry,
    )

    assert resolved == [output_path]
    assert experiment.paths_calls == []
    assert experiment.times_calls == [([1.0], None)]
