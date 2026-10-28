"""Recorded decisions are checked to exist; who decided is recorded data, not code."""
from darsia.fluidflower.geometry_raster_sources import registered_run_status
from darsia.fluidflower.phase_blind_review_policy import explicit_authorization


def test_registered_status_names_its_reviewer_as_recorded_data():
    assert registered_run_status("registered")
    assert registered_run_status("provisional - Test Reviewer to review")
    for status in ("provisional", "provisional -  to review", "provisional - x", None, "unknown"):
        assert not registered_run_status(status)


def test_display_release_authorization_names_who_released_it_and_when():
    assert explicit_authorization("Test Reviewer", "2026-10-06", True)
    for value in ((" ", "2026-10-06", True), ("Test Reviewer", "2026-13-06", True),
                  ("Test Reviewer", "6 Oct", True), ("Test Reviewer", "2026-10-06", False), (None, None, True)):
        assert not explicit_authorization(*value)
