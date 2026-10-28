"""Exact grid comparisons consume metadata, not the saved pixel arrays."""
from copy import deepcopy
from pathlib import Path

import darsia
import numpy as np
import pytest

from darsia.fluidflower.archive_grid import _archive_grid, require_archive_grids
from darsia.fluidflower.setup_rig import _require_exact_grid


def _write(path, shape, metadata, compressed=False):
    writer = np.savez_compressed if compressed else np.savez
    writer(path, array=np.zeros(shape, dtype=np.float32), metadata=metadata)
    return dict(path=str(path))


@pytest.fixture
def grids(tmp_path):
    metadata = dict(space_dim=2, indexing="ij", dimensions=[.5, .9],
                    origin=[0., .5], series=False, scalar=False,
                    type=darsia.OpticalImage, color_space="RGB")
    baseline = _write(tmp_path / "photo.npz", (4, 6, 3), metadata)
    scalar = dict(metadata, type=darsia.ScalarImage, scalar=True)
    sources = dict(shape_corrected_baseline=baseline)
    for name in ("labels", "facies", "depth"):
        sources[name] = _write(tmp_path / (name + ".npz"), (4, 6), scalar, True)
    return sources, scalar


def test_grid_comparison_never_reads_array_payload_or_invokes_pixel_decoder(grids, monkeypatch):
    sources, _ = grids
    read = np.lib.format.read_array
    members = []

    def metadata_only(stream, *args, **kwargs):
        members.append(stream.name)
        assert stream.name == "metadata.npy"
        return read(stream, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("The grid check decoded a pixel array")

    monkeypatch.setattr(np.lib.format, "read_array", metadata_only)
    monkeypatch.setattr(darsia, "imread", forbidden)
    assert require_archive_grids(sources)
    assert members == ["metadata.npy"] * 4


@pytest.mark.parametrize("change", ["dimensions", "origin", "shape", "series", "space_dim"])
def test_fast_comparison_rejects_the_same_actual_grid_mismatches_as_original_reader(grids, change):
    sources, metadata = grids
    metadata = deepcopy(metadata)
    shape = (4, 6)
    if change == "dimensions":
        metadata["dimensions"] = [.6, .9]
    elif change == "origin":
        metadata["origin"] = [.1, .5]
    elif change == "shape":
        shape = (4, 5)
    elif change == "series":
        metadata.update(series=True, time=[0.])
        shape = (4, 6, 1)
    else:
        metadata.update(space_dim=3, indexing="ijk", dimensions=[.5, .9, .1], origin=[0., .5, 0.])
        shape = (4, 6, 1)
    _write(Path(sources["labels"]["path"]), shape, metadata, True)
    baseline = darsia.imread(sources["shape_corrected_baseline"]["path"])
    with pytest.raises(ValueError) as original:
        _require_exact_grid(Path(sources["labels"]["path"]), baseline, "labels")
    with pytest.raises(ValueError) as fast:
        require_archive_grids(sources)
    assert str(fast.value) == str(original.value)


@pytest.mark.parametrize("kind", [darsia.Image, darsia.ScalarImage, "darsia.ScalarImage"])
def test_original_metadata_defaults_and_scalar_image_semantics_are_preserved(tmp_path, kind):
    path = tmp_path / "grid.npz"
    _write(path, (4, 6), dict(type=kind, dimensions=[.5, .9], scalar=True))
    original, fast = darsia.imread(path), _archive_grid(path)
    for name in ("space_dim", "indexing", "series", "shape"):
        assert getattr(fast, name) == getattr(original, name)
    for name in ("dimensions", "origin", "voxel_size"):
        assert np.array_equal(getattr(fast, name), getattr(original, name))
    assert fast.img.strides == (0, 0)


def test_older_formats_retain_original_reader_route(tmp_path, grids):
    sources, _ = grids
    from PIL import Image
    photo = tmp_path / "reference.jpg"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(photo)
    sources["shape_corrected_baseline"] = dict(path=str(photo))
    assert not require_archive_grids(sources)
    assert _archive_grid(photo) is None


def test_pixel_canvas_metadata_uses_its_explicit_approved_native_grid(grids):
    sources, _ = grids
    for name in ("labels", "facies"):
        _write(Path(sources[name]["path"]), (4, 6), {}, True)
    with pytest.raises(ValueError, match="dimensions"):
        require_archive_grids(sources)
    grid = dict(shape_hw=[4, 6], legacy_dimensions_m=[.5, .9], physical_metric_is_separate=True)
    assert require_archive_grids(sources, native_grid=grid)
    with pytest.raises(ValueError, match="approved native canvas"):
        require_archive_grids(sources, native_grid=dict(grid, shape_hw=[4, 5]))
    _write(Path(sources["facies"]["path"]), (4, 6), dict(dimensions=[.5, .8]), True)
    with pytest.raises(ValueError, match="dimensions"):
        require_archive_grids(sources, native_grid=grid)
