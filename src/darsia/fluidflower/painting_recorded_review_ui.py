"""Inspect exact saved findings and human decisions without an action route.

The existing guided dialog owns its original photo readers and retirement
barrier. This separate window has no adoption controller callbacks or editor.
A recorded snapshot grants no current validation or execution authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .painting_guided_review_ui import GuidedGapReviewDialog

FORMAT = "darsia-recorded-guided-review-v1"
MAX_METADATA_BYTES = 128 * 1024 * 1024
NOTICE = ("Recorded decisions — read only. Opening or navigating this view records "
          "no answer and starts no Check. Current adoption state is not checked here.")


def _read_bound(source):
    expected = source if isinstance(source, dict) else None
    path = Path(expected["path"] if expected else source)
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Recorded review metadata exceeds the supported size")
    bound = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                 size_bytes=len(raw))
    if expected is not None and bound != expected:
        raise ValueError("Recorded review metadata differs from its saved binding")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Recorded review metadata must be an object")
    return value, bound


def read_recorded_review(source):
    """Only bounded JSON/original display metadata; never source replay or Check."""
    from .painted_analysis_exclusions import approval_records, _action, issue_identity
    from .painted_review_comparison import read_review

    document, snapshot = _read_bound(source)
    if set(document) != {"format", "review", "report", "decisions"} or document["format"] != FORMAT:
        raise ValueError("Choose an exact recorded guided-review snapshot")
    review_bound = document["review"]
    report, _report_bound = _read_bound(document["report"])
    review = read_review(review_bound["path"])
    if review["review_binding"] != review_bound:
        raise ValueError("The original review differs from the recorded snapshot")
    issues = report.get("issues")
    if not isinstance(issues, list) or any(
        not isinstance(row, dict) or row.get("severity") not in {"error", "warning", "info"}
        for row in issues
    ):
        raise ValueError("Recorded findings require every original issue and severity")
    choices = document["decisions"]
    if not isinstance(choices, dict) or set(choices) != {"analysis_exclusions", "source_choices", "warnings"}:
        raise ValueError("Recorded decisions require their original ordinary action records")
    approval = choices["analysis_exclusions"]
    if approval is not None:
        approval = approval_records(approval, review)
    sources = choices["source_choices"]
    if not isinstance(sources, list):
        raise ValueError("Recorded source decisions must be a list")
    for action in sources:
        _action(action, state=True)
        if action["acquisition_state"] not in {"clearwater", "indicator", "unknown"}:
            raise ValueError("A recorded source state is not supported")
    warnings = choices["warnings"]
    if not isinstance(warnings, list):
        raise ValueError("Recorded warning decisions must be a list")
    seen = set()
    for action in warnings:
        if not isinstance(action, dict) or set(action) != {
            "issue_index", "issue_sha256", "actor", "approved_utc", "reason", "decision_ref"
        }:
            raise ValueError("A recorded warning needs its original exact action")
        index = action["issue_index"]
        if type(index) is not int or not 0 <= index < len(issues):
            raise ValueError("A recorded warning is outside its report")
        issue = issues[index]
        if (issue["severity"] != "warning" or issue_identity(issue) != action["issue_sha256"]
                or index in seen):
            raise ValueError("A warning decision differs from its exact original finding")
        _action({key: action[key] for key in ("actor", "approved_utc", "reason", "decision_ref")})
        seen.add(index)
    return dict(snapshot=snapshot, review=review, report=report,
                decisions=deepcopy(choices), approval=deepcopy(approval))


class RecordedGuidedReviewDialog(GuidedGapReviewDialog):
    """Original guided presentation with inspection-only reads and navigation."""

    def __init__(self, source, *, parent=None, **display_readers):
        forbidden = set(display_readers) - {"preview_reader", "comparison_reader"}
        if forbidden:
            raise TypeError("A recorded review accepts only original display readers")
        self._recorded_source = deepcopy(source)
        super().__init__(None, actor="", parent=parent, **display_readers)
        self.setWindowTitle("Recorded map decisions — read only")
        self._start(("recorded-review", None), lambda: read_recorded_review(source))
        self._buttons()

    def _start(self, key, function):
        if key[0] not in {"recorded-review", "preview", "comparison"}:
            raise ValueError("This view permits only recorded metadata and photograph reads")
        return super()._start(key, function)

    def _consume_additional_result(self, key, result):
        if key != ("recorded-review", None):
            return False
        if self._closing:
            return True
        self.recorded_snapshot = deepcopy(result["snapshot"])
        self._catalogue(result["review"], validate_context=False, refresh=False)
        self.approval = deepcopy(result["approval"])
        self._acquisition_actions = deepcopy(result["decisions"]["source_choices"])
        self._warning_actions = {
            (a["issue_index"], a["issue_sha256"]): deepcopy(a)
            for a in result["decisions"]["warnings"]
        }
        self.set_catalogue_result(result["report"])
        return True

    def _select(self, *args):
        super()._select(*args)
        row = self._current_row()
        if row is None:
            return
        if row["kind"] == "gap":
            actions = [a for a in (self.approval or {}).get("runs", [])
                       if (a["run_id"], a["component_id"]) == (row["run_id"], row["component_id"])]
        elif row["kind"] == "source_state":
            actions = self._acquisition_actions
        else:
            action = self._warning_actions.get((row.get("issue_index"), row.get("issue_sha256")))
            actions = [action] if action else []
        self.details.setPlainText(json.dumps(
            dict(finding=row.get("issue") or row, recorded_decisions=actions),
            indent=2, ensure_ascii=False))

    def _navigate(self, direction):
        if not self.busy() and not self._closing and not self.operation_waiting:
            self._show_question(self._question_index + direction)

    def _buttons(self, *_args):
        super()._buttons(*_args)
        self._lock_controls()

    def _update_guided_controls(self):
        super()._update_guided_controls()
        self._lock_controls()

    def _lock_controls(self):
        for name in (
            "approve_one", "approve_all", "paint", "refresh_findings", "recheck",
            "approve_warning", "fix", "adopt", "set_state", "classification",
            "source_state", "reason", "other_text", "apply_tolerances",
            "shift_tolerance", "angle_tolerance", "clear_water_confirmations",
            "water_surface", "region_preview_reset", "water_source_button",
        ):
            control = getattr(self, name, None)
            if control is not None:
                control.setEnabled(False)
        if getattr(self, "_guided_ready", False):
            self.guided_status.setText(NOTICE)
            self.check_explanation.setText("This view displays saved findings and decisions. Check is unavailable here.")
            self.adopt_explanation.setText("This view grants no adoption or execution authority.")
            self.adopt.setVisible(False)

    def _set_question_text(self, stage):
        super()._set_question_text(stage)
        self.question_title.setText("Recorded: " + self.question_title.text())
        self.question_explanation.setText(NOTICE)

    def can_recheck(self):
        return False

    def can_adopt(self):
        return False

    def _read_only(self, *_args, **_kwargs):
        return False

    # Guard handlers as well as controls: direct/queued callbacks cannot write.
    _approve = _approve_warning = _set_state = _fix = _adopt = _paint = _read_only
    _recheck = _tolerances = _refresh_findings = _water_tolerances = _read_only
    _water_pick_requested = _water_point_clicked = _water_confirmed = _read_only
    _water_line_edited = _clear_water_confirmations = _point_selected = _read_only
    _region_requested = _sand_requested = _reset_region_preview = _read_only
    _region_draft_changed = _prepare_water_source_review = _editor_photo_requested = _read_only
