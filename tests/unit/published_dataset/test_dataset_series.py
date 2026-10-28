"""Published datasets opened as series: the per-user results locations and fresh analysis folders."""

import os
from pathlib import Path

import pytest

from darsia.fluidflower.published_dataset import dataset_series, read_only
from darsia.fluidflower.published_dataset.layout import MANIFEST


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "settings" / "series_registry.toml"))
    return tmp_path / "settings"


def make_dataset(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / MANIFEST).write_text("{}", encoding="utf-8")
    return folder


def test_the_file_sits_beside_the_series_registry(settings):
    assert dataset_series.datasets_path() == (settings / "published_datasets.toml").resolve()
    assert dataset_series.registered_datasets() == []


def test_register_records_updates_and_finds_a_results_location(settings, tmp_path):
    dataset = make_dataset(tmp_path / "data sets" / "første")
    first, second = tmp_path / "results å", tmp_path / "other results"
    dataset_series.register_dataset(dataset, first)
    assert dataset_series.dataset_results(dataset) == first.resolve()
    dataset_series.register_dataset(dataset, second)
    assert dataset_series.registered_datasets() == [{"root": str(dataset.resolve()), "results": str(second.resolve())}]
    assert dataset_series.dataset_results(tmp_path / "unknown") is None
    assert not any(dataset.rglob("*.toml"))  # Nothing is written into the dataset.


def test_results_locations_must_lie_outside_the_dataset(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    assert dataset_series.results_location_problems(dataset, "") == ["Choose a results location outside the dataset"]
    inside = ["The results location must lie outside the dataset folder"]
    assert dataset_series.results_location_problems(dataset, dataset / "results") == inside
    assert dataset_series.results_location_problems(dataset, dataset) == inside
    assert dataset_series.results_location_problems(dataset, tmp_path) == inside  # It would hold the dataset.
    assert dataset_series.results_location_problems(dataset, tmp_path / "results") == []


def test_register_refuses_a_folder_without_manifest_and_a_results_location_inside(settings, tmp_path):
    with pytest.raises(ValueError, match=MANIFEST):
        dataset_series.register_dataset(tmp_path, tmp_path.parent / "results")
    dataset = make_dataset(tmp_path / "dataset")
    with pytest.raises(ValueError, match="outside"):
        dataset_series.register_dataset(dataset, dataset / "results")
    assert dataset_series.registered_datasets() == []


def test_an_unknown_settings_version_is_refused(settings):
    settings.mkdir()
    (settings / "published_datasets.toml").write_text("schema_version = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        dataset_series.registered_datasets()


def test_only_dataset_folders_are_protected(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    series = tmp_path / "series"
    series.mkdir()
    try:
        assert dataset_series.protect_datasets([dataset, series, tmp_path / "missing"]) == [dataset]
        assert os.path.normcase(str(dataset.resolve())) in read_only.protected_roots()
        assert os.path.normcase(str(series.resolve())) not in read_only.protected_roots()
        (series / "notes.txt").write_text("a series stays writable", encoding="utf-8")
    finally:
        read_only.unprotect(dataset)


def test_each_new_analysis_gets_a_fresh_folder(tmp_path):
    results = tmp_path / "results"
    assert dataset_series.fresh_results_folder(results) == results / "analysis-1"
    (results / "analysis-1").mkdir(parents=True)
    assert dataset_series.fresh_results_folder(results) == results / "analysis-1"  # Empty: still fresh.
    (results / "analysis-1" / "report").mkdir()
    assert dataset_series.fresh_results_folder(results) == results / "analysis-2"
