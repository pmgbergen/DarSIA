"""Saved scale measurements consolidate without new OCR or gas assumptions."""
import hashlib
import json

import pytest

from darsia.fluidflower.overflow_protocol_import import (
    load_overflow_sources, read_overflow_measurements,
)


def test_explicit_moved_source_preserves_original_reading_identity(tmp_path):
    source = tmp_path / "moved/r1/review.json"
    source.parent.mkdir(parents=True)
    raw = json.dumps(dict(metadata=dict(run="r1"), records=[])).encode()
    source.write_bytes(raw)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(dict(scale_reviews={
        "r1": dict(path="D:/old/r1/review.json", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)),
        "excluded": dict(path="missing", sha256="0" * 64, size_bytes=0),
    })))
    assert load_overflow_sources(inputs, tmp_path / "moved", runs=["r1"]) == {"r1": source}
    source.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="differs"):
        load_overflow_sources(inputs, tmp_path / "moved", runs=["r1"])


def test_actual_readings_keep_status_time_origin_and_source_not_predictions(tmp_path):
    source = tmp_path / "review.json"
    source.write_text(json.dumps(dict(
        metadata=dict(run="r1", injection_start="2023-01-02T11:00:00+01:00",
                      weight_data_valid=True, weight_data_status="needs-review"),
        records=[dict(image_time="2023-01-02T10:59:00+01:00", time_from_injection_h=-1/60,
                      weight_g=104.3, suggested_g=134.3, status="Corrected", notes="manual"),
                 dict(weight_g=None, suggested_g=135, status="Pending")],
    )))
    before = source.read_bytes()
    names, values = read_overflow_measurements(source, "r1")
    row = dict(zip(names, values[0]))
    assert len(values) == 1
    assert row["weight_g"] == "104.3" and row["status"] == "Corrected"
    assert row["time_from_injection_h"] == str(-1/60)
    assert row["weight_data_status"] == "needs-review"
    assert row["injection_start"] == "2023-01-02T11:00:00+01:00"
    assert "suggested_g" not in row and "gas_volume_ml" not in row
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="selected run"):
        read_overflow_measurements(source, "different")
