"""Stage-2 adapter for the stored stage-1 colour-path coordinate."""

import hashlib
import json

import numpy as np
import pytest

from darsia.fluidflower import stage2_postprocessing as s2
from darsia.fluidflower.stage2_postprocessing import (
    COLOUR_COORDINATE_ARCHIVE as ARCHIVE,
    COLOUR_COORDINATE_ARRAY as ARRAY,
    COLOUR_COORDINATE_FORMAT as FORMAT,
    COLOUR_COORDINATE_RECORD as RECORD,
    StoredField,
    StoredFrameMapping,
    blue_fraction_from_stored_coordinate,
    load_npz_fields,
    read_colour_coordinate,
    read_stage1_frame,
)

COORDINATE = np.array([[0.1, 0.4, np.nan], [0.8, np.nan, 0.0]], np.float32)


def _identity(value):
    """The array identity stage 1 records (C-order bytes, dtype, shape)."""
    digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return dict(sha256=digest, dtype=value.dtype.str, shape=list(value.shape))


def _frame(tmp_path, coordinate=COORDINATE, *, record=True, name="sample_1"):
    """A stage-1 frame directory in the layout of the store_colour_coordinate option."""
    root = tmp_path / name
    root.mkdir()
    np.savez(root / ARCHIVE, **{ARRAY: coordinate})
    data = (root / ARCHIVE).read_bytes()
    result = dict(case_id=name)
    if record:
        result[RECORD] = dict(
            format=FORMAT,
            path=ARCHIVE,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            array=ARRAY,
            identity=_identity(coordinate),
        )
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return root


def adapter(coordinate):  # a test fixture, not a calibration
    return np.where(np.isnan(coordinate), np.nan, 1.0 - coordinate)


def _same(first, second):
    return np.array_equal(first, second, equal_nan=True)


def test_reads_the_coordinate_and_passes_it_through_the_adapter(tmp_path):
    root = _frame(tmp_path)
    value = read_colour_coordinate(root)
    assert value.dtype == np.float32 and value.tobytes() == COORDINATE.tobytes()
    f = blue_fraction_from_stored_coordinate(root, adapter)
    assert f.dtype == np.float64 and _same(f, adapter(COORDINATE))
    with pytest.raises(ValueError, match="outside"):  # the adapter's contract holds
        blue_fraction_from_stored_coordinate(root, lambda c: c * 10.0)
    with pytest.raises(ValueError, match="adapter"):
        blue_fraction_from_stored_coordinate(root, None)


def test_refuses_a_frame_without_the_coordinate(tmp_path):
    root = _frame(tmp_path, record=False)
    for call in (
        lambda: read_colour_coordinate(root),
        lambda: blue_fraction_from_stored_coordinate(root, adapter),
    ):
        with pytest.raises(ValueError, match="store_colour_coordinate"):
            call()
    lost = _frame(tmp_path, name="sample_3")
    (lost / ARCHIVE).unlink()
    with pytest.raises(ValueError, match="missing"):
        read_colour_coordinate(lost)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="result.json"):
        read_colour_coordinate(empty)
    # Through the generic reader, an absent archive is refused as well.
    with pytest.raises(ValueError, match="does not exist"):
        load_npz_fields(lost / ARCHIVE, {"colour": ARRAY})


def test_refuses_a_changed_or_foreign_archive(tmp_path):
    root = _frame(tmp_path)
    path = root / ARCHIVE
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        read_colour_coordinate(root)
    foreign = _frame(tmp_path, name="sample_3")
    result = json.loads((foreign / "result.json").read_bytes())
    result[RECORD]["format"] = "another-format"
    (foreign / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        read_colour_coordinate(foreign)
    wide = _frame(tmp_path, COORDINATE.astype(np.float64), name="sample_5")
    with pytest.raises(ValueError, match="float32"):
        read_colour_coordinate(wide)
    flat = _frame(tmp_path, COORDINATE.ravel(), name="sample_7")
    with pytest.raises(ValueError, match="2D"):
        read_colour_coordinate(flat)


def test_the_coordinate_feeds_the_stage1_frame_reader(tmp_path):
    root = _frame(tmp_path)
    np.savez(
        root / "native_geometry.npz",
        facies=np.full(COORDINATE.shape, 2, np.uint8),
        transport_weight_l=np.full(COORDINATE.shape, 1e-3),
    )
    mapping = StoredFrameMapping(
        colour=StoredField("coordinate", ARRAY),
        labels=StoredField("geometry", "facies"),
        pore_volume=StoredField("geometry", "transport_weight_l"),
        pore_volume_unit="L",
        valid_colour=None,
        gas=None,
        outside=None,
        pixel_area_m2=None,
    )
    archives = dict(coordinate=root / ARCHIVE, geometry=root / "native_geometry.npz")
    inputs = read_stage1_frame(archives, mapping, frame_id="t1", adapter=adapter)
    assert _same(inputs.blue_fraction, adapter(COORDINATE))
    (root / ARCHIVE).unlink()
    with pytest.raises(ValueError, match="does not exist"):
        read_stage1_frame(archives, mapping, frame_id="t1", adapter=adapter)


def test_names_equal_the_stage1_constants():
    storage_profile = pytest.importorskip("darsia.fluidflower.storage_profile")
    if not hasattr(storage_profile, "COLOUR_COORDINATE_ARCHIVE"):
        pytest.skip("this stage 1 has no colour-coordinate storage option")
    for name in ("FORMAT", "ARCHIVE", "ARRAY", "RECORD"):
        key = "COLOUR_COORDINATE_" + name
        assert getattr(s2, key) == getattr(storage_profile, key)
