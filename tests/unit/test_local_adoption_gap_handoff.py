"""Local reuse must hand real unresolved pixels to the ordinary GUI review."""
from pathlib import Path

import pytest

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower.local_map_adoption import prepare_local_adoption
from .test_painted_analysis_exclusions import (
    gap_case, registered_setup, imported, synthetic_payload_admission, _approve,
)  # noqa: F401 -- original finite fixture chain
from .test_painted_transfer_replay import _context, _retained, _unchanged


@pytest.mark.parametrize("stale_approval", [False, True])
def test_local_reuse_retains_real_gaps_for_review_before_outside_proof(gap_case, stale_approval):
    retained = _retained(gap_case)
    approval = _approve(gap_case.review) if stale_approval else None
    if approval is not None:
        approval["review"]["sha256"] = "0" * 64
    destination = gap_case.root / "unresolved_local_reuse"
    with pytest.raises(gaps.GapReviewRequired) as stopped:
        prepare_local_adoption(gap_case.review["review_binding"],
            source_context=_context(gap_case), destination=destination,
            analysis_exclusions=approval)
    assert stopped.value.review["review_binding"] == gap_case.review["review_binding"]
    assert sum(row["gap"]["pixels"] for row in stopped.value.review["runs"]) > 0
    assert not (Path(destination) / "index.json").exists()
    assert not (Path(destination) / "source_frame_inside.npz").exists()
    _unchanged(retained)
