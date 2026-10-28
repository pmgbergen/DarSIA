"""Studio photometry preview cache keys of RAW runs (package D2b).

A RAW preview is keyed on its camera RAW, native record, RAW audit and camera
profile instead of a serialized rig; any change gives a cache miss.
"""
import json

import numpy as np
import pytest

from darsia.fluidflower.tools import phase_studio_photometry_preview_cache as cache
from .test_raw_photographic_review import _write, inputs  # noqa: F401


def _key(tmp_path, frame_index=1):
    return cache.raw_review_key_info(tmp_path / "review", "ac52", dataset="ffac", native_manifest_sha256="a" * 64,
                                     display_shape_hw=(4, 6), frame_index=frame_index)


def test_raw_key_names_the_camera_raw_profile_and_native_identities(tmp_path, inputs):
    frame = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))["frames"][1]
    payload, digest = cache._identity(_key(tmp_path))
    assert payload["source_kind"] == "raw" and payload["rig"] is None and payload["category"] == "frame"
    assert payload["profile"] == {"profile_id": "late", "sha256": inputs["configuration"]["sha256"]}
    assert [row["sha256"] for row in payload["native_records"]] == [frame["native_record"]["sha256"],
                                                                     frame["audit"]["sha256"]]
    assert [row["sha256"] for row in payload["raw_sources"]] == [frame["acquisition"]["sha256"]]
    baseline = cache._identity(_key(tmp_path, None))
    assert baseline[0]["category"] == "baseline" and baseline[1] != digest
    other = dict(_key(tmp_path), profile=dict(_key(tmp_path)["profile"], profile_id="early"))
    assert cache._identity(other)[1] != digest
    with pytest.raises(ValueError, match="source_kind"):
        cache._identity(dict(_key(tmp_path), source_kind="jpeg"))
    with pytest.raises(ValueError, match="profile"):
        cache._identity(dict(_key(tmp_path), profile={"profile_id": "", "sha256": "a" * 64}))


def test_raw_preview_round_trips_and_misses_after_a_record_changes(tmp_path, inputs, monkeypatch):
    _write(tmp_path, inputs)
    monkeypatch.setattr(cache, "series_path", lambda dataset, kind: tmp_path / "results")
    key = _key(tmp_path)
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    cache.store(key, rgb)
    assert np.array_equal(cache.load(key, (4, 6)), rgb)
    audit = key["native_records"][1]["path"]
    with open(audit, "ab") as stream:
        stream.write(b" ")
    assert cache.load(key, (4, 6)) is None
