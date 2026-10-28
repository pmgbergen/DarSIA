import numpy as np
import pytest

from darsia.fluidflower.phase.history import (
    advance_previous,
    chronological_history,
    envelope_ordinal,
)


def test_parent_envelopes_use_strict_camera_times():
    assert [envelope_ordinal(t, 1.0, 2.0) for t in (0.0, 1.0, 1.5, 2.0, 2.5)] == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError):
        envelope_ordinal(1.0, 2.0, 1.0)
    with pytest.raises(ValueError):
        envelope_ordinal(np.nan, 1.0, 2.0)


def _row(digest, elapsed, index):
    return dict(run="sample", source_sha256=digest, elapsed_h=elapsed,
                camera_datetime=str(elapsed), frame_index=index)


def test_history_uses_digest_and_time_without_densifying_indices():
    original = [_row("c", 2.0, 20), _row("a", 0.0, 7), _row("b", 1.0, 9)]
    rows = chronological_history(original, policy="selected acquisitions")
    assert [row["source_sha256"] for row in rows] == ["a", "b", "c"]
    assert [row["previous_source_sha256"] for row in rows] == [None, "a", "b"]
    assert [row["frame_index"] for row in rows] == [7, 9, 20]
    assert all(row["history_policy"] == "selected acquisitions" for row in rows)
    assert "previous_source_sha256" not in original[0]
    with pytest.raises(ValueError):
        chronological_history([_row("a", 0.0, 7), _row("a", 1.0, 9)], policy="selected")
    with pytest.raises(ValueError):
        chronological_history([_row("a", 0.0, 7), _row("b", 0.0, 9)], policy="selected")


def test_review_probe_does_not_advance_history():
    previous, probe, selected = object(), object(), object()
    assert advance_previous(probe, previous, selected=False, baseline=False) is previous
    assert advance_previous(selected, previous, selected=True, baseline=False) is selected
    assert advance_previous(selected, None, selected=False, baseline=True) is selected
