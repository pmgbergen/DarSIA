"""Shared ordered findings DATA from an already authenticated native review."""
from copy import deepcopy


def review_findings_data(review, *, display_attachments=None):
    """Collect original review and display issues without creating a candidate."""
    from .painted_analysis_exclusions import DisplayFindingsChanged, _json

    try:
        if review.get("display_context") is None:
            if display_attachments is not None:
                raise ValueError("Display actions/preferences require their genuine bound source context")
            return dict(issues=deepcopy(review["issues"]))
        from .painted_review_comparison import display_findings, display_context_digest
        display = _json(display_findings(review, None if display_attachments is None else _json(display_attachments)))
        if (not isinstance(display, dict) or set(display) != {"display_attachments", "issues", "display_findings_sha256"}
                or not isinstance(display["display_attachments"], dict) or not isinstance(display["issues"], list)
                or any(not isinstance(issue, dict) or issue.get("severity") not in {"error", "warning", "info"} for issue in display["issues"])
                or display["display_findings_sha256"] != display_context_digest(dict(
                    display_attachments=display["display_attachments"], issues=display["issues"]))):
            raise ValueError("The source-bound display provider returned invalid findings")
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        raise DisplayFindingsChanged(review["issues"], str(error)) from error
    return dict(display, issues=[*deepcopy(review["issues"]), *display["issues"]])
