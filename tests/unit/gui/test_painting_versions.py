"""Layer painting tab: saved versions of the map, the proposal with its corrections.

A synthetic 80 x 60 photograph and a project in a temporary folder; no real
series is used.
"""

import re

from PySide6.QtWidgets import QInputDialog, QMessageBox

from darsia.fluidflower import layer_painting as lp

from .test_painting_corrections import (  # noqa: F401
    COARSE,
    _corrected,
    _proposal,
    _show,
    view,
)


def _brush(view):
    view.select_layer_row(2)  # Coarse sand
    view.add_correction(lp.Correction("brush", COARSE, [(30.0, 10.0)], radius_px=2.0))


def test_versions_wait_for_a_proposal(view):
    view._refresh_status()
    assert not view.save_version_button.isEnabled()
    assert view.save_version("A") is None
    assert "Propose the boundaries first" in view.versions_note.text()
    _show(view, _proposal())
    assert view.save_version_button.isEnabled()
    assert not view.restore_version_button.isEnabled()


def test_versions_are_saved_listed_and_restored(view):
    _show(view, _proposal())
    first = view.save_version("Proposal only")
    assert first["kind"] == "proposal"
    _brush(view)
    second = view.save_version("Edited")
    assert second["kind"] == "edited map"
    table = view.versions_table
    assert table.rowCount() == 2
    names = [table.item(row, 0).text() for row in range(2)]
    assert names == ["Edited", "Proposal only"]  # Newest first.
    assert table.item(0, 1).text() == "edited (1)"
    assert table.item(1, 1).text() == "proposal"
    assert table.item(0, 4).text() == second["sha256"][:12]
    # The current map equals "Edited", so restoring keeps no extra version.
    assert view.restore_version(first)
    assert view.corrections == [] and len(view.versions) == 2
    assert view.proposal is not None
    edited = next(row for row in view.versions if row["name"] == "Edited")
    assert view.restore_version(edited)
    assert len(view.corrections) == 1 and _corrected(view)[30, 10] == COARSE


def test_restoring_keeps_an_unsaved_map_as_a_version_first(view):
    _show(view, _proposal())
    saved = view.save_version("Saved")
    _brush(view)
    assert view.restore_version(saved)
    assert [row["name"] for row in view.versions] == ["Saved", "Before restoring Saved"]
    kept = view.versions[1]
    assert kept["kind"] == "edited map" and kept["correction_count"] == 1
    assert view.corrections == []


def test_the_buttons_ask_first(view, monkeypatch):
    _show(view, _proposal())
    monkeypatch.setattr(
        QInputDialog, "getText", lambda *args, **kwargs: ("Named", True)
    )
    view.save_version_button.click()
    assert [row["name"] for row in view.versions] == ["Named"]
    view.versions_table.selectRow(0)
    assert view.restore_version_button.isEnabled()
    assert "SHA-256" in view.versions_note.text()
    asked = []

    def no(*args, **kwargs):
        asked.append(args)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", no)
    view.restore_version_button.click()
    assert len(asked) == 1 and len(view.versions) == 1


def test_an_empty_name_gives_the_kind_of_map_and_the_time(view, monkeypatch):
    _show(view, _proposal())
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("  ", True))
    view.save_version_button.click()
    _brush(view)
    assert view.default_version_name().startswith("Edited map ")
    record = view.save_version("")
    names = [row["name"] for row in view.versions]
    stamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"
    assert re.fullmatch(f"Proposal {stamp}", names[0])
    assert re.fullmatch(f"Edited map {stamp}", names[1]) and record["name"] == names[1]
