"""Actual tiny Store/export/Apply transactions, private files only.

The shared owned RAW fixture has real NPZ/source/request/parent-grid evidence;
its decoder/registration/runtime/release are explicitly synthetic. These tests
exercise transactional configuration changes, not native scientific accuracy.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
import toml

from darsia.fluidflower import corrected_setup as setup
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import painted_setup_apply as rebuilt
from . import test_geometry_registered_export_apply as registered
from .test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission, _saved  # noqa: F401


@pytest.fixture
def named_setup(imported, monkeypatch, synthetic_payload_admission):
    """Put real old id-keyed recipes in the original pre-Store fixture inputs."""
    original = toml.dumps

    def historical(document, *args, **kwargs):
        if isinstance(document, dict) and document.get("calibration") == {"marker": "unchanged"}:
            document = deepcopy(document)
            document["facies"].update({"1": {"labels": [101]}, "7": {"labels": [107]}, "2": {"labels": [102]}})
            document["n1"] = dict(porosity_by_facies={"1": .44, "7": .44, "2": .22},
                                  swi_by_facies={"1": .12, "7": .12, "2": .13},
                                  unquantified_facies=[1, 7])
            document["model"] = dict(ignore_labels=[1, 7], artifact="unchanged-old-family")
        return original(document, *args, **kwargs)

    monkeypatch.setattr(toml, "dumps", historical)
    data = registered.registered_setup.__wrapped__(imported, monkeypatch)
    common = data.series / "config/common.toml"
    common.write_bytes(b"# shared before bytes stay retained\r\n[n1.porosity_by_facies]\r\n"
                       b"1 = 0.44\r\n7 = 0.44\r\n2 = 0.22\r\n[model]\r\nignore_labels = [1, 7]\r\n")
    return data


def _painting(data, tmp_path):
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    _saved(data, prepared)
    return setup.export_corrected_setup(data.project, reference=SimpleNamespace(run_id="one"), actor="unit")


def _files(data):
    return [data.series / "config/common.toml", *sorted((data.series / "config/runs").glob("*.toml")),
            *sorted((data.series / "config/merged").glob("*.toml"))]


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_default_apply_keeps_stale_id_refusal_and_all_original_bytes(named_setup, tmp_path):
    data = named_setup
    export = _painting(data, tmp_path)
    before = {path: path.read_bytes() for path in _files(data)}
    with pytest.raises(ValueError, match="7"):
        setup.apply_corrected_selected_run(data.project, export)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (export.destination / "apply-claim.json").exists()


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_by_name_common_and_run_changes_share_original_apply_replay(named_setup, tmp_path):
    data = named_setup
    export = _painting(data, tmp_path)
    common = data.series / "config/common.toml"
    common_before = common.read_bytes()
    current = geometry.GeometryRevisionStore(data.project).current_path
    current_before = current.read_bytes()
    first = setup.apply_corrected_selected_run(data.project, export, rebuild_facies_by_name=True)
    record = json.loads(first.read_bytes())
    assert record["state"] == "complete" and record["facies_rebuild"]["format"] == rebuilt.FORMAT
    complete = json.loads((export.destination / "apply-complete.json").read_bytes())
    common_row = next(row for row in complete["configurations"] if Path(row["after"]["path"]) == common)
    common_after = common.read_bytes()
    assert Path(common_row["before"]["path"]).read_bytes() == common_before
    assert tomllib.loads(common.read_text())["n1"]["porosity_by_facies"] == {"1": .44, "2": .22}
    for kind in ("runs", "merged"):
        path = data.series / "config" / kind / "one.toml"
        document = tomllib.loads(path.read_text())
        assert document["facies"]["1"]["labels"] == [5]
        assert document["facies"]["2"]["labels"] == [6] and "7" not in document["facies"]
        assert document["n1"]["porosity_by_facies"] == {"1": .44, "2": .22}
        assert document["n1"]["swi_by_facies"] == {"1": .12, "2": .13}
        assert document["model"] == {"ignore_labels": [1], "artifact": "unchanged-old-family"}
        assert document["labeling"]["frame"] == "rig-corrected"
        assert document["depth"] == tomllib.loads(data.before[(kind, "one")].decode())["depth"]
    assert setup.require_corrected_export_for_build([common, data.series / "config/merged/one.toml"]) == export.receipt
    assert setup.apply_corrected_selected_run(data.project, export, rebuild_facies_by_name=True) == first
    assert current.read_bytes() == current_before
    # Child export consumes stable common postimage, while the display export
    # retains its exact common before image through its immutable completion.
    child = setup.export_corrected_setup(data.project, reference=SimpleNamespace(run_id="two"), actor="unit")
    setup.apply_corrected_selected_run(data.project, child, rebuild_facies_by_name=True)
    assert setup.require_corrected_export_for_build([common, data.series / "config/merged/two.toml"]) == child.receipt
    assert common.read_bytes() == common_after
    assert setup.export_corrected_setup(data.project, reference=SimpleNamespace(run_id="one"), actor="unit").reused


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_child_cannot_rewrite_common_before_display_apply(named_setup, tmp_path):
    data = named_setup
    display = _painting(data, tmp_path)
    child = setup.export_corrected_setup(data.project, reference=SimpleNamespace(run_id="two"), actor="unit")
    before = {path: path.read_bytes() for path in _files(data)}
    with pytest.raises(ValueError, match="display-reference"):
        setup.apply_corrected_selected_run(data.project, child, rebuild_facies_by_name=True)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (child.destination / "apply-claim.json").exists()
    assert not (display.destination / "apply-claim.json").exists()


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_rehashed_after_receipts_cannot_change_the_named_transformation(named_setup, tmp_path):
    data = named_setup
    export = _painting(data, tmp_path)
    setup.apply_corrected_selected_run(data.project, export, rebuild_facies_by_name=True)
    complete_path = export.destination / "apply-complete.json"
    complete = json.loads(complete_path.read_bytes())
    row = next(row for row in complete["configurations"] if row["after"]["path"].endswith("one.toml"))
    path = Path(row["after"]["path"])
    raw = path.read_bytes()
    changed = raw.replace(b"0.44", b"0.45", 1)
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    row["after"] = setup._descriptor(path)
    attempt = Path(complete["attempt_receipt"]["path"])
    record = json.loads(attempt.read_bytes())
    next(item for item in record["files"] if item["path"] == str(path))["after_sha256"] = hashlib.sha256(changed).hexdigest()
    attempt.write_text(json.dumps(record), encoding="utf-8")
    complete["attempt_receipt"] = setup._descriptor(attempt)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="exact by-name transformation"):
        setup._apply_completion(export.destination, export.receipt)


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_same_uuid_and_csv_cannot_substitute_another_revision_recipe(named_setup, tmp_path):
    data = named_setup
    export = _painting(data, tmp_path)
    setup.apply_corrected_selected_run(data.project, export, rebuild_facies_by_name=True)
    complete_path = export.destination / "apply-complete.json"
    complete = json.loads(complete_path.read_bytes())
    context = complete["facies_rebuild"]
    revision_path = Path(context["revision"]["path"])
    revision = json.loads(revision_path.read_bytes())
    # Same Store UUID and physical CSV; a different literal recipe is still
    # another revision and cannot be substituted by rehashing all receipts.
    revision["configuration"]["replacement_recipe_note"] = "foreign recipe"
    revision_path.write_text(json.dumps(revision, indent=2) + "\n", encoding="utf-8")
    context["revision"] = setup._descriptor(revision_path)
    claim_path = export.destination / "apply-claim.json"
    claim = json.loads(claim_path.read_bytes())
    claim["facies_rebuild"] = deepcopy(context)
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    record_path = Path(complete["attempt_receipt"]["path"])
    record = json.loads(record_path.read_bytes())
    record["facies_rebuild"] = deepcopy(context)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    complete["attempt_receipt"] = setup._descriptor(record_path)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="exact exported revision"):
        setup._apply_completion(export.destination, export.receipt)
