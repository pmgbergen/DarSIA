"""Progress observes the genuine Check without changing its result or scope."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower.adoption_check_progress import observe_check_progress, report_check_progress
from darsia.fluidflower.bound_verification_cache import BoundVerificationCache

from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401


def test_progress_is_nested_operation_local_and_restored_after_failure():
    outer, inner = [], []
    with ThreadPoolExecutor(max_workers=1) as executor:
        with observe_check_progress(outer.append):
            report_check_progress("outer")
            executor.submit(report_check_progress, "unrelated operation").result()
            with pytest.raises(ValueError, match="original validation failure"):
                with observe_check_progress(inner.append):
                    report_check_progress("inner")
                    raise ValueError("original validation failure")
            report_check_progress("restored")
    report_check_progress("finished operation")
    assert outer == ["outer", "restored"]
    assert inner == ["inner"]


def test_a_failed_progress_observer_does_not_change_validation(monkeypatch):
    def failed_observer(message):
        raise RuntimeError("unavailable progress display")

    monkeypatch.setattr(geometry, "_utc_now", lambda: "2020-01-01T00:00:00Z")
    ordinary_report = geometry.validate_geometry_config({})
    with observe_check_progress(failed_observer):
        observed_report = geometry.validate_geometry_config({})
    assert observed_report == ordinary_report
    assert observed_report["summary"]["error_count"] > 0


@pytest.mark.parametrize("continuation", [False, True])
def test_real_automatic_check_reports_runs_and_returns_the_same_candidate_and_report(
        registered_setup, tmp_path, monkeypatch, continuation):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, continuation=continuation)
    monkeypatch.setattr(table, "_FILES", BoundVerificationCache())
    monkeypatch.setattr(table, "_DERIVED", OrderedDict())
    monkeypatch.setattr(geometry, "_utc_now", lambda: "2020-01-01T00:00:00Z")
    messages = []
    with observe_check_progress(messages.append):
        candidate, summary = adoption.adoption_candidate(data.revision, prepared["folder"])
        report = geometry.validate_geometry_config(candidate)
    ordinary_candidate, ordinary_summary = adoption.adoption_candidate(data.revision, prepared["folder"])
    assert candidate == ordinary_candidate and summary == ordinary_summary
    assert report == geometry.validate_geometry_config(ordinary_candidate)
    assert not report["summary"]["error_count"]
    assert messages[0] == "Reading and verifying the prepared map…"
    for number, run in enumerate(summary["runs"], 1):
        if continuation:
            assert f"Verifying preparation for {run.upper()} ({number} of {len(summary['runs'])} runs)…" in messages
        assert f"Verifying native labels for {run.upper()} ({number} of {len(summary['runs'])} runs)…" in messages
        assert f"Building registration for {run.upper()} ({number} of {len(summary['runs'])} runs)…" in messages
        assert f"Validating geometry inputs for {run.upper()} ({number} of {len(summary['runs'])} runs)…" in messages
        assert f"Validating registration for {run.upper()} ({number} of {len(summary['runs'])} runs)…" in messages
    assert messages.index("Validating the geometry input arrays and physical properties…") < messages.index(
        "Validating the geometry source and registration bindings…")
