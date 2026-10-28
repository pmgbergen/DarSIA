"""Moved source storage retains its original revision and byte identities."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower import storage_paths
from .test_corrected_setup_canonical_layout import export, inventory, layout


def _move_original_sources(data, monkeypatch):
    original = data["series"] / "original"
    archive = data["series"] / "archive/original"
    archive.parent.mkdir()
    original.rename(archive)
    source = str(original).replace("\\", "/").casefold()
    target = str(archive).replace("\\", "/")
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [(source, target)])
    return original, archive


def test_exported_resolved_sources_apply_and_build_against_unchanged_declared_revision(layout, monkeypatch):
    old, archive = _move_original_sources(layout, monkeypatch)
    current_bytes = layout["current"].read_bytes()
    output = export(layout)
    original_receipt = (output.destination / "export.json").read_bytes()
    binding = output.receipt["binding"]
    # The exporter has already verified the relocated bytes. The saved revision
    # deliberately keeps the authored path and original content digest.
    assert Path(binding["configured_depth_source"]["path"]).is_relative_to(archive)
    assert Path(layout["runs"]["ac17"]["configured_depth_map"]["path"]).is_relative_to(old)
    assert not old.exists()
    reused = export(layout)
    assert reused.reused and reused.receipt == output.receipt
    applied = subject.apply_corrected_selected_run(layout["project"], reused)
    merged = layout["series"] / "config/merged/ac17.toml"
    assert subject.require_corrected_export_for_build(merged) == output.receipt
    before = inventory(layout["series"])
    assert subject.apply_corrected_selected_run(layout["project"], reused) == applied
    assert inventory(layout["series"]) == before
    assert layout["current"].read_bytes() == current_bytes
    assert (output.destination / "export.json").read_bytes() == original_receipt


def test_same_bytes_at_an_undeclared_location_do_not_become_the_revision_source(layout, monkeypatch):
    _old, archive = _move_original_sources(layout, monkeypatch)
    output = export(layout)
    foreign = layout["series"] / "unrelated-depth.npz"
    foreign.write_bytes((archive / "ac17/configured-depth.npz").read_bytes())
    receipt = deepcopy(output.receipt)
    receipt["binding"]["configured_depth_source"]["path"] = str(foreign)
    (output.destination / "export.json").write_text(json.dumps(receipt))
    changed = replace(output, receipt=receipt)
    before = inventory(layout["series"])
    with pytest.raises(ValueError, match="input identities or target paths"):
        subject.apply_corrected_selected_run(layout["project"], changed)
    assert inventory(layout["series"]) == before


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("size_bytes", 123),
                                         ("size_bytes", True)])
def test_storage_alias_preserves_exact_descriptor_content_and_types(layout, monkeypatch, field, value):
    _old, _archive = _move_original_sources(layout, monkeypatch)
    declared = subject._identity(layout["runs"]["ac17"]["configured_depth_map"])
    resolved = subject._verified(declared)
    assert subject._same_source(declared, resolved)
    wrong = dict(resolved, **{field: value})
    assert not subject._same_source(declared, wrong)


def test_read_binding_rechecks_alias_bytes_without_rewriting_the_declared_path(layout, monkeypatch):
    from darsia.fluidflower.setup_geometry_apply import ReadBinding
    declared = Path(layout["runs"]["ac17"]["configured_depth_map"]["path"])
    before = declared.read_bytes()
    bound = ReadBinding(declared, before)
    _old, archive = _move_original_sources(layout, monkeypatch)
    bound.require_unchanged()
    assert bound.path == declared and bound.data == before
    (archive / "ac17/configured-depth.npz").write_bytes(b"concurrent source edit")
    with pytest.raises(RuntimeError, match="File changed after geometry was prepared"):
        bound.require_unchanged()
    assert bound.path == declared and bound.data == before
