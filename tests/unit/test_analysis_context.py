from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.time_data import TimeData


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


def test_select_image_paths_supports_mixed_timedata_paths_and_times(
    tmp_path, config_stub
):
    selected_from_path = tmp_path / "selected_from_path.jpg"
    selected_from_time = tmp_path / "selected_from_time.jpg"
    selected_from_path.touch()
    selected_from_time.touch()
    (tmp_path / "configured_path.jpg").touch()

    class MixedExperiment:
        def __init__(self):
            self.paths_calls = []
            self.times_calls = []

        def find_images_for_paths(self, paths):
            self.paths_calls.append(paths)
            return [selected_from_path]

        def find_images_for_times(self, times, data=None):
            self.times_calls.append((times, data))
            return [selected_from_time]

    experiment = MixedExperiment()
    data = TimeData().load(
        {
            "path": {"picked": {"paths": ["configured_path.jpg"]}},
            "time": {"snap": {"times": ["01:00:00"]}},
            "interval": {"window": {"start": "02:00:00", "end": "03:00:00", "num": 2}},
        },
        data_folder=tmp_path,
    )
    source = tmp_path / "source"
    sub_config = SimpleNamespace(data=data)

    resolved = select_image_paths(
        config=config_stub,
        experiment=experiment,
        sub_config=sub_config,
        source=source,
    )

    assert resolved == [selected_from_path, selected_from_time]
    assert experiment.paths_calls == [[tmp_path / "configured_path.jpg"]]
    assert experiment.times_calls == [(data.image_times, source)]


def test_select_image_paths_supports_mixed_legacy_paths_and_times(
    tmp_path, config_stub
):
    selected_from_path = tmp_path / "legacy_selected_from_path.jpg"
    selected_from_time = tmp_path / "legacy_selected_from_time.jpg"
    selected_from_path.touch()
    selected_from_time.touch()
    configured_path = tmp_path / "legacy_configured_path.jpg"
    configured_path.touch()

    class LegacyMixedExperiment:
        def __init__(self):
            self.paths_calls = []
            self.times_calls = []

        def find_images_for_paths(self, paths):
            self.paths_calls.append(paths)
            return [selected_from_path]

        def find_images_for_times(self, times, data=None):
            self.times_calls.append((times, data))
            return [selected_from_time]

    experiment = LegacyMixedExperiment()
    source = tmp_path / "source"
    sub_config = SimpleNamespace(image_paths=[configured_path], image_times=[1.0, 2.0])

    resolved = select_image_paths(
        config=config_stub,
        experiment=experiment,
        sub_config=sub_config,
        source=source,
    )

    assert resolved == [selected_from_path, selected_from_time]
    assert experiment.paths_calls == [[configured_path]]
    assert experiment.times_calls == [([1.0, 2.0], source)]
