"""Append-only approval withdrawal on private packages and the real painting GUI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import layer_painting_version_archive as versions
from ..test_layer_painting_version_archive import make_version as _make_version, payloads, state
from .test_painting_corrections import view, _show, _proposal  # noqa: F401



def make_version(tmp_path, kind="proposal"):
    """Supply the minimal genuine project manifest needed by the original Store."""
    project, record = _make_version(tmp_path, kind)
    project.manifest = {}
    return project, record


def _approve(record):
    return (manual.approve_manual_version(record) if manual.is_manual_version(record)
            else lp.approve_version(record))


def _placement(project, record, approval):
    path = project.root / "annotations/map_placement/retained/index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(map_version=dict(record, approval=approval), runs=[], created_utc="2026-10-22T02:00:00Z")
    path.write_text(json.dumps(value), encoding="utf8")
    return path


@pytest.mark.parametrize("kind", ["proposal", "manual"])
def test_withdrawal_appends_and_preserves_original_saved_content(tmp_path, kind):
    project, record = make_version(tmp_path, kind)
    approval = _approve(record)
    original = payloads(record["folder"])
    log = lp.approvals_path(project, "sample")
    prefix = log.read_bytes()
    event = versions.withdraw_approval(project, "sample", record)
    assert log.read_bytes().startswith(prefix)
    assert payloads(record["folder"]) == original
    events = lp.read_map_approvals(project, "sample")
    assert events == [approval, event]
    assert event["event"] == "approval_withdrawn" and "sha256" not in event
    assert event["withdrawn_approval"] == approval
    assert lp.map_approval(events, record["sha256"]) is None
    row = versions.list_version_metadata(project, "sample")[0]
    assert not row["approved"] and row["approval_withdrawn_utc"] == event["recorded_utc"]
    assert versions.archive_blockers(project, "sample", record) == []


def test_withdrawal_is_literal_version_scoped_and_later_approval_is_active(tmp_path):
    project, first = make_version(tmp_path)
    session, proposal = state()
    second = versions.save_map_version(project, session, proposal, [], name="Second")
    assert first["sha256"] == second["sha256"] and first["folder"] != second["folder"]
    first_approval, second_approval = _approve(first), _approve(second)
    versions.withdraw_approval(project, "sample", first)
    events = lp.read_map_approvals(project, "sample")
    assert lp.map_approval(events, first["sha256"], version=Path(first["folder"]).name) is None
    assert lp.map_approval(events, second["sha256"]) == second_approval
    renewed = _approve(first)
    events = lp.read_map_approvals(project, "sample")
    assert lp.map_approval(events, first["sha256"], version=Path(first["folder"]).name) == renewed
    assert lp.approval_withdrawal(events, first_approval) is not None
    assert lp.approval_withdrawal(events, renewed) is None
    assert lp.map_approval(events, second["sha256"], version=Path(second["folder"]).name) == second_approval


def test_placed_version_can_be_withdrawn_archived_and_restored_without_placement_edits(tmp_path):
    project, record = make_version(tmp_path)
    approval = _approve(record)
    placement = _placement(project, record, approval)
    original = placement.read_bytes()
    assert versions.withdrawal_context(project, "sample", record)["placements"] == [str(placement)]
    versions.withdraw_approval(project, "sample", record)
    row = versions.list_version_metadata(project, "sample")[0]
    assert row["placed"] and row["placement_withdrawn"]
    archived = versions.archive_version(project, "sample", record)
    assert placement.read_bytes() == original
    restored = versions.restore_deleted_version(project, "sample", archived)
    assert restored["folder"] == record["folder"] and placement.read_bytes() == original


@pytest.mark.parametrize("location", ["current", "history"])
def test_current_adoption_refuses_withdrawal_before_log_write(tmp_path, location):
    project, record = make_version(tmp_path)
    _approve(record)
    paths = dict(current=tmp_path/"overrides/geometry_facies/current.json",
                 history=tmp_path/"overrides/geometry_facies/revisions.jsonl",
                 revision=tmp_path/"results/geometry/old/revision.json")
    path = paths[location]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(configuration=dict(provenance=dict(map_version=record))))+"\n")
    log = lp.approvals_path(project, "sample")
    before = log.read_bytes()
    with pytest.raises(ValueError, match="adopt another version first"):
        versions.withdraw_approval(project, "sample", record)
    assert log.read_bytes() == before and Path(record["folder"]).is_dir()


def test_adopted_preparation_is_joined_by_actual_sha_and_exact_version(tmp_path):
    project, first = make_version(tmp_path)
    session, proposal = state()
    second = versions.save_map_version(project, session, proposal, [], name="Second")
    _approve(first)
    _approve(second)
    preparation = tmp_path/"annotations/map_adoption/old/index.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_text(json.dumps(dict(map_version=second)))
    raw = preparation.read_bytes()
    bound = dict(path=str(preparation), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    current = tmp_path/"overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(configuration=dict(provenance=dict(
        map_version=dict(name=second["name"],sha256=second["sha256"]), preparation=bound)))))
    # Same map bytes in a different saved folder cannot inherit its adoption.
    versions.withdraw_approval(project, "sample", first)
    before = lp.approvals_path(project, "sample").read_bytes()
    with pytest.raises(ValueError, match="adopt another version first"):
        versions.withdraw_approval(project, "sample", second)
    assert lp.approvals_path(project, "sample").read_bytes() == before


def test_changed_adopted_preparation_fails_closed_without_withdrawal(tmp_path):
    project, record = make_version(tmp_path)
    _approve(record)
    preparation = tmp_path/"annotations/map_adoption/old/index.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_text(json.dumps(dict(map_version=record)))
    bound = dict(path=str(preparation), sha256="0"*64)
    current = tmp_path/"overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(configuration=dict(provenance=dict(preparation=bound)))))
    before = lp.approvals_path(project,"sample").read_bytes()
    with pytest.raises(ValueError, match="preparation metadata changed"):
        versions.withdraw_approval(project,"sample",record)
    assert lp.approvals_path(project,"sample").read_bytes() == before


def test_gui_withdraw_asks_once_names_identity_and_marks_placement(view, qtbot, monkeypatch):
    _show(view, _proposal())
    record = view.save_version("Chosen map")
    view.versions_table.selectRow(0)
    approval = view.approve_version(record)
    placement = _placement(view.project,record,approval)
    placement_before = placement.read_bytes()
    view.version_actions.reload()
    view.versions_table.selectRow(0)
    qtbot.waitUntil(lambda: not view.busy(), timeout=10000)
    assert view.version_actions.buttons["withdraw"].isEnabled()
    questions = []
    def no(_parent,title,text,*_args):
        questions.append((title,text))
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox,"question",no)
    before = lp.approvals_path(view.project,view.session.run_id).read_bytes()
    assert not view.version_actions.withdraw()
    assert len(questions)==1 and lp.approvals_path(view.project,view.session.run_id).read_bytes()==before
    assert record["name"] in questions[0][1] and record["kind"] in questions[0][1]
    assert record["sha256"] in questions[0][1] and "placement" in questions[0][1]
    questions.clear()
    def yes(_parent,title,text,*_args):
        questions.append((title,text))
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox,"question",yes)
    assert view.version_actions.withdraw() and len(questions)==1
    assert placement.read_bytes()==placement_before and Path(record["folder"]).is_dir()
    assert "approval withdrawn" in view.versions_table.item(0,5).text()
    assert "Placement approval withdrawn" in view.versions_table.item(0,5).text()
    assert view._approval_for_version(view._selected_version()) is None
    assert not view.place_version_button.isEnabled() and not view.adopt_button.isEnabled()
    assert not view.version_actions.buttons["withdraw"].isEnabled()
    assert versions.archive_blockers(view.project,view.session.run_id,record)==[]


def test_gui_adopted_version_is_refused_without_confirmation(view, qtbot, monkeypatch):
    _show(view,_proposal())
    record=view.save_version("Already adopted")
    view.versions_table.selectRow(0)
    view.approve_version(record)
    qtbot.waitUntil(lambda: not view.busy(), timeout=10000)
    current=view.project.root/"overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True,exist_ok=True)
    current.write_text(json.dumps(dict(configuration=dict(provenance=dict(map_version=record)))))
    questions=[]
    monkeypatch.setattr(QMessageBox,"question",lambda *args:questions.append(args))
    before=lp.approvals_path(view.project,view.session.run_id).read_bytes()
    assert not view.version_actions.withdraw() and not questions
    assert "adopt another version first" in view.versions_note.text()
    assert lp.approvals_path(view.project,view.session.run_id).read_bytes()==before


def test_placement_summary_marks_old_withdrawal_after_same_version_reapproval(view, qtbot):
    from darsia.fluidflower.map_placement_view import MapPlacementDialog
    _show(view,_proposal())
    record=view.save_version("Retained placement")
    approval=view.approve_version(record)
    versions.withdraw_approval(view.project,view.session.run_id,record)
    _approve(record)
    events=lp.read_map_approvals(view.project,view.session.run_id)
    placement=dict(folder=str(view.project.root/"annotations/map_placement/retained"),
        map_version=dict(record,approval=approval),reference_run=view.session.run_id,runs=[])
    dialog=MapPlacementDialog(placement,view)
    qtbot.addWidget(dialog)
    dialog.refresh_approval_state(events)
    assert "Approval withdrawn" in dialog.lead.text()
    assert lp.map_approval(events,record["sha256"],version=Path(record["folder"]).name) is not None


def test_adopting_another_version_allows_withdrawal_but_preserves_archive_history(tmp_path):
    project, first = make_version(tmp_path)
    session, proposal = state()
    session.edge_fill_mm = 1.25
    second = versions.save_map_version(project, session, proposal, [], name="Replacement")
    assert first["sha256"] != second["sha256"]
    _approve(first)
    _approve(second)
    override = tmp_path / "overrides/geometry_facies"
    override.mkdir(parents=True)
    current = override / "current.json"
    history = override / "revisions.jsonl"
    old = dict(revision_id="old-A", configuration=dict(provenance=dict(map_version=first)))
    new = dict(revision_id="new-B", configuration=dict(provenance=dict(map_version=second)))
    old_artifact = tmp_path / "results/geometry/old-A/revision.json"
    old_artifact.parent.mkdir(parents=True)
    old_artifact.write_text(json.dumps(old))
    history.write_text(json.dumps(old) + "\n")
    current.write_text(json.dumps(old))
    log = lp.approvals_path(project, "sample")
    before = log.read_bytes()
    with pytest.raises(ValueError, match="adopt another version first"):
        versions.withdraw_approval(project, "sample", first)
    assert log.read_bytes() == before
    # A real current pointer supersedes the old historical entry; retaining
    # that entry must not make the instructed remedy impossible forever.
    current.write_text(json.dumps(new))
    saved_history, saved_artifact = history.read_bytes(), old_artifact.read_bytes()
    event = versions.withdraw_approval(project, "sample", first)
    assert event["withdrawn_approval"]["version"] == Path(first["folder"]).name
    assert log.read_bytes().startswith(before)
    assert history.read_bytes() == saved_history and old_artifact.read_bytes() == saved_artifact
    # Withdrawal is now allowed, but archive still respects adopted history.
    assert any("adopted" in reason for reason in versions.archive_blockers(project, "sample", first))
    with pytest.raises(ValueError, match="adopted"):
        versions.archive_version(project, "sample", first)
    assert Path(first["folder"]).is_dir()


def test_unwithdrawn_same_content_alias_keeps_original_archive_protection(tmp_path):
    project, first = make_version(tmp_path)
    session, proposal = state()
    alias = versions.save_map_version(project, session, proposal, [], name="Unapproved alias")
    assert alias["sha256"] == first["sha256"]
    _approve(first)
    with pytest.raises(ValueError, match="approved"):
        versions.archive_version(project, "sample", alias)
    assert Path(alias["folder"]).is_dir()
