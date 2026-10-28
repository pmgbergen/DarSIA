"""Shared finding collection preserves the original authenticated Check route."""
from copy import deepcopy
import sys
from types import ModuleType

import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as core
from darsia.fluidflower.painted_findings_catalogue import review_findings_data

from .test_painted_display_findings_backend import display_provider, displayed_case  # noqa: F401
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401


@pytest.mark.parametrize("custom_preferences", [False, True])
def test_shared_data_matches_authenticated_check_order_attachments_and_digest(
        displayed_case, monkeypatch, custom_preferences):
    case = displayed_case
    review_before, candidate_before = deepcopy(case.review), deepcopy(case.candidate)
    attachments = None
    if custom_preferences:
        attachments = deepcopy(case.report["display_attachments"])
        attachments["position"]["shift_tolerance_mm"] = 99.
    attachments_before = deepcopy(attachments)
    expected_display = case.provider.module.display_findings(case.review, attachments)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Findings DATA decoded an array"))

    shared = review_findings_data(case.review, display_attachments=attachments)
    checked = core.check_review_findings(case.candidate, display_attachments=attachments)

    assert shared == checked
    assert case.review["issues"]
    assert shared["issues"] == [*review_before["issues"], *expected_display["issues"]]
    assert shared["display_attachments"] == expected_display["display_attachments"]
    assert shared["display_findings_sha256"] == expected_display["display_findings_sha256"]
    assert case.review == review_before and case.candidate == candidate_before
    assert attachments == attachments_before
    shared["issues"][0]["message"] = "Changed returned DATA"
    shared["display_attachments"]["position"]["shift_tolerance_mm"] = 111.
    assert case.review == review_before and attachments == attachments_before
    assert checked["issues"][0] == review_before["issues"][0]


@pytest.mark.parametrize("fault", ["digest", "attachment_type", "issue_type"])
def test_invalid_provider_preserves_original_issues_and_wraps_failure_only_once(
        displayed_case, monkeypatch, fault):
    case = displayed_case
    original = deepcopy(case.review["issues"])
    provider = case.provider.module.display_findings

    def invalid(review, attachments=None):
        value = provider(review, attachments)
        if fault == "digest":
            value["display_findings_sha256"] = "f" * 64
        elif fault == "attachment_type":
            value["display_attachments"] = None
        else:
            value["issues"] = "not an issue list"
        return value

    monkeypatch.setattr(case.provider.module, "display_findings", invalid)
    failures = []
    for collect, argument in ((review_findings_data, case.review), (core.check_review_findings, case.candidate)):
        with pytest.raises(core.DisplayFindingsChanged, match="provider returned invalid findings") as caught:
            collect(argument)
        failures.append(caught.value)
        assert caught.value.original_issues == original
        assert type(caught.value.__cause__) is ValueError
        assert not isinstance(caught.value.__cause__, core.DisplayFindingsChanged)
    assert str(failures[0]) == str(failures[1])
    failures[0].original_issues[0]["message"] = "Changed exception DATA"
    assert case.review["issues"] == failures[1].original_issues == original


def test_no_display_context_preserves_none_semantics_without_importing_provider(monkeypatch):
    provider = ModuleType("darsia.fluidflower.painted_review_comparison")

    def unavailable(name):
        if name in {"display_findings", "display_context_digest", "validate_display_context", "context_source_sha256"}:
            pytest.fail("The no-context path imported a display provider API")
        raise AttributeError(name)

    provider.__getattr__ = unavailable
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    review = dict(issues=[dict(severity="warning", code="original", message="Original issue")])
    original = deepcopy(review)
    result = review_findings_data(review)
    assert result == dict(issues=original["issues"])
    assert core.check_review_findings({}) == dict(issues=[])
    result["issues"][0]["message"] = "Changed returned DATA"
    assert review == original
    with pytest.raises(core.DisplayFindingsChanged, match="require their genuine bound source context") as caught:
        review_findings_data(review, display_attachments={})
    assert caught.value.original_issues == original["issues"]
    with pytest.raises(ValueError, match="require their genuine bound source context") as caught:
        core.check_review_findings({}, display_attachments={})
    assert not isinstance(caught.value, core.DisplayFindingsChanged)
