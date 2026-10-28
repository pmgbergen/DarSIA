"""Step confirmations kept in a series folder: events, digests and safe saving."""

import json

import pytest

from darsia.fluidflower import step_confirmations as subject
from darsia.fluidflower.workflow_row_status import Artifact, RowStatus


def _state(summary="Saved choice A.", files=()):
    return RowStatus("done", summary, "2026-10-27 12:00 CET", tuple(files))


def test_confirm_and_reopen_keep_every_event_and_the_latest_per_step(tmp_path):
    path = tmp_path / "config" / subject.FILE_NAME
    latest = subject.record_step(path, "method:use_for_series", "confirm", "basis-a")
    assert subject.confirmation_state(latest["method:use_for_series"], "basis-a") == "confirmed"
    latest = subject.record_step(path, "method:use_for_series", "reopen")
    assert subject.confirmation_state(latest["method:use_for_series"], "basis-a") == "open"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == subject.FORMAT
    assert [event["action"] for event in document["events"]] == ["confirm", "reopen"]
    assert all(event["utc"].endswith("Z") for event in document["events"])
    assert subject.read_confirmations(tmp_path / "absent.json") == {}
    with pytest.raises(ValueError, match="Unknown confirmation action"):
        subject.record_step(path, "method:use_for_series", "approve")


def test_a_confirmation_for_other_saved_output_is_changed():
    event = dict(step="geometry_facies:layers", action="confirm", basis="old")
    assert subject.confirmation_state(event, "new") == "changed"
    assert subject.confirmation_state(None, "new") == "open"


def test_basis_follows_summary_and_files_but_not_the_shared_series_file(tmp_path):
    series = Artifact(str(tmp_path / "series.toml"), "series", 10, "2026-10-27 12:00 CET")
    saved = Artifact(str(tmp_path / "labels.npz"), "labels", 100, "2026-10-27 12:00 CET")
    basis = subject.step_basis(_state(files=(series, saved)))
    assert basis == subject.step_basis(_state(files=(saved, Artifact(series.path, "series", 99, "later"))))
    assert basis != subject.step_basis(_state("Saved choice B.", files=(series, saved)))
    assert basis != subject.step_basis(_state(files=(series, Artifact(saved.path, "labels", 101, saved.modified))))
    assert subject.step_basis(None) == ""


def test_a_change_saved_meanwhile_elsewhere_is_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / subject.FILE_NAME
    subject.record_step(path, "a:b", "confirm", "x")
    original = subject._read_bytes
    calls = []

    def reread(target):
        calls.append(target)
        data = original(target)
        return data if len(calls) == 1 else data + b" "
    monkeypatch.setattr(subject, "_read_bytes", reread)
    with pytest.raises(subject.ConfirmationsChangedError):
        subject.record_step(path, "a:b", "reopen")
    monkeypatch.undo()
    assert [event["action"] for event in json.loads(path.read_text(encoding="utf-8"))["events"]] == ["confirm"]
    assert not list(tmp_path.glob("*.tmp"))


def test_confirmations_live_in_the_series_configuration_folder(tmp_path):
    (tmp_path / "series.toml").write_text('schema_version = 1\n[paths]\nconfig = "settings"\n', encoding="utf-8")
    assert subject.confirmations_path(tmp_path) == tmp_path / "settings" / subject.FILE_NAME
    assert subject.step_key(("method", "use_for_series")) == "method:use_for_series"


def test_a_file_entering_by_its_content_ignores_its_size_and_time(tmp_path):
    shared = Artifact(str(tmp_path / "imaging.xlsx"), "saved file", 50, "2026-10-27 12:00 CET")
    injection = Artifact(str(tmp_path / "injection.xlsx"), "injection protocol", 100, "2026-10-25 22:42 CET",
                         content="injection-schedule-v1:" + "a" * 64)
    basis = subject.step_basis(_state(files=(shared, injection)))
    assert basis.startswith(subject.CONTENT_BASIS)
    corrected = Artifact(injection.path, injection.kind, 101, "2026-10-28 09:00 CET", content=injection.content)
    assert subject.step_basis(_state(files=(shared, corrected))) == basis
    rescheduled = Artifact(injection.path, injection.kind, 100, injection.modified,
                           content="injection-schedule-v1:" + "b" * 64)
    assert subject.step_basis(_state(files=(shared, rescheduled))) != basis
    # The other files keep their size and time.
    touched = Artifact(shared.path, shared.kind, 51, shared.modified)
    assert subject.step_basis(_state(files=(touched, injection))) != basis
    assert not subject.step_basis(_state(files=(shared,))).startswith(subject.CONTENT_BASIS)


def test_a_confirmation_from_before_the_content_basis_is_named_as_earlier():
    old = dict(step="preprocessing:protocols", action="confirm", basis="c" * 64)
    current = subject.CONTENT_BASIS + "d" * 64
    assert subject.confirmation_state(old, current) == "earlier"
    assert subject.confirmation_state(dict(old, basis=current), current) == "confirmed"
    assert subject.confirmation_state(dict(old, basis=subject.CONTENT_BASIS + "e" * 64), current) == "changed"
    # A step whose files have no content basis compares as before.
    assert subject.confirmation_state(old, "f" * 64) == "changed"
