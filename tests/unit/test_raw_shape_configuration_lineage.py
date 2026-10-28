"""Synthetic historical lineage joins using original metadata loaders and pins.

These files model source migration and protocol revision only. No RAW image,
scientific build, calibrated rig or production qualification is represented.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower.raw_shape_basis_reader import _Pins
from darsia.fluidflower import raw_shape_configuration_lineage as lineage


def _write(path, raw):
    """Create one synthetic fixture file and return its exact descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _json(path, value):
    """Write literal fixture metadata without claiming actual operational proof."""
    return _write(path, (json.dumps(value, indent=2) + "\n").encode())


def _config(*, folder, baseline, results, imaging, injection, pressure, rig,
            raw, blacklist=False, target_type="float32"):
    """Render explicit native TOML values for the original metadata loaders."""
    q = lambda value: json.dumps(str(value))
    return (f"data_path = [{q(baseline)}]\n"
            f"[data]\nfolder = {q(folder)}\nbaseline = {q(baseline)}\n"
            f"format = {q('arw' if raw else 'jpg')}\nresults = {q(results)}\nuse_cache = false\n"
            f"[protocols]\nimaging_mode = \"Load from CSV\"\n"
            f"imaging = {{ {q(folder)} = {q(imaging)} }}\n"
            f"injection = {q(injection)}\npressure_temperature = {q(pressure)}\n"
            + ("blacklist = {}\n" if blacklist else "")
            + f"[corrections.type]\ntarget_type = {q(target_type)}\n"
            f"[rig]\npath = {q(rig)}\n").encode()


@pytest.fixture
def synthetic_lineage(tmp_path):
    """Build a complete tiny original source-lineage chain without RAW payloads."""
    run = "sample"
    series = tmp_path / "series"
    manifest = _write(series / "series.toml", b'name = "synthetic"\n')
    project = series / "geometry_facies"
    revision_id = "a" * 32
    old_rig = tmp_path / "original-rig"
    new_rig = project / "results" / "rig_models" / revision_id / run
    folder = tmp_path / "captures"
    results = tmp_path / "results"
    results.mkdir()
    jpeg, raw = folder / "frame.jpg", folder / "frame.ARW"
    old_csv = series / "protocols" / "old.csv"
    raw_csv = series / "protocols" / "raw.csv"
    old_clock = b"path,source_sha256,data_time,selected\n" + (
        f"{jpeg},{'1' * 64},2020-01-01T00:00:00,true\n").encode()
    new_clock = b"path,source_sha256,data_time,selected,legacy_jpeg_sha256\n" + (
        f"{raw},{'2' * 64},2020-01-01T00:00:00,true,{'1' * 64}\n").encode()
    old_csv_binding = _write(old_csv, old_clock)
    new_csv_binding = _write(raw_csv, new_clock)
    injection = series / "protocols" / "injection.csv"
    old_injection = b"start,location_x,location_y,rate\r\n2020-01-01,0.45,0.03,1\r\n"
    new_injection = old_injection.replace(b",0.03,", b",0.02,")
    injection_binding = _write(injection, new_injection)
    pressure = series / "protocols" / "pressure.csv"
    pressure_binding = _write(pressure, b"time,pressure,temperature\n0,1,20\n")
    base = dict(folder=folder, results=results, injection=injection,
                pressure=pressure)
    original_bytes = _config(**base, baseline=jpeg, imaging=old_csv, rig=old_rig, raw=False)
    applied_bytes = _config(**base, baseline=jpeg, imaging=old_csv, rig=new_rig, raw=False)
    accepted_bytes = _config(**base, baseline=raw, imaging=raw_csv, rig=new_rig, raw=True)
    source = _write(series / "legacy_sources" / "merged" / f"{run}.toml", original_bytes)
    previous, applied, accepted = [], [], []
    originals = {}
    for role in ("runs", "merged"):
        previous.append(_write(series / "config" / "clock" / role / f"{run}.toml", original_bytes))
        row = _write(series / "config" / "geometry" / role / f"{run}.toml", applied_bytes)
        applied.append(row)
        archive = _write(series / "archives" / role / f"{run}.toml", applied_bytes)
        originals[row["path"]] = dict(before=row, archive=archive)
        accepted.append(_write(series / "config" / "raw" / role / f"{run}.toml",
            accepted_bytes + (b"# preserved run bundle\n" if role == "runs" else b"")))
    clock_archive = _write(series / "archives" / "old.csv", old_clock)
    originals[str(old_csv)] = dict(before=old_csv_binding, archive=clock_archive)
    migration = _json(series / "migration.json", dict(
        format="n117-raw-configuration-switch-v1", run=run,
        all43_baselines_ARW=True,
        all39_selected_tables_bound_RAW_exact_original_times_indices=True,
        new_files=accepted, originals=originals))
    prior = dict(format="darsia-corrected-setup-configuration-dual-binding-v2",
                 run_id=run, source_configuration=source,
                 current_configuration=applied[-1], apply_configurations=applied,
                 pre_pointer_configurations=previous,
                 protocol_change_mode="unchanged-zero-leaves", protocol_changes={},
                 output_pointer_change=dict(field="rig.path", before=str(old_rig), after=str(new_rig)),
                 comparison_sha256="3" * 64)
    change = dict(configuration_leaves={
        "data.baseline": dict(before=dict(present=True, value=str(jpeg)),
                              after=dict(present=True, value=str(raw))),
        "data.format": dict(before=dict(present=True, value="jpg"),
                            after=dict(present=True, value="arw")),
        "data_path": dict(before=dict(present=True, value=[str(jpeg)]),
                          after=dict(present=True, value=[str(raw)]))},
        imaging_protocols=[dict(folder=str(folder), before=clock_archive,
                                after=new_csv_binding, rows=1)])
    source_lineage = _json(series / "lineage.json", dict(
        format="darsia-corrected-setup-source-lineage-v1", run_id=run,
        prior_binding=prior, legacy_series_root=str(series), legacy_project_root=str(project),
        legacy_revision_id=revision_id, legacy_export=None,
        source_migration_receipt=migration, accepted_configurations=accepted,
        raw_changes=[deepcopy(change), deepcopy(change)]))
    actual_path = series / "config" / "current" / "merged" / f"{run}.toml"
    actual = _write(actual_path, _config(**base, baseline=raw, imaging=raw_csv,
        rig=old_rig, raw=True, blacklist=True) + b"\n[unused]\nvalue = 7\n")
    changeset_path = series / "protocol-revision" / "changeset.json"
    before_local = _write(changeset_path.parent / "before" / "injection.csv", old_injection)
    before = dict(injection_binding, sha256=before_local["sha256"], size_bytes=before_local["size_bytes"])
    replacement = dict(role="injection_protocol", before=before, after=injection_binding,
                       before_local="before/injection.csv", changed_cells=1)
    changeset = _json(changeset_path, dict(replacements=[replacement], source_transform=dict(
        path=str(tmp_path / "existing_transform.py"), sha256=lineage._INJECTION_TRANSFORM_SHA256,
        size_bytes=1967)))
    complete_value = dict(format="ffac-n396-completed-v1", complete=True,
        changeset=changeset, current_series_identity=deepcopy(manifest),
        final_original_states=[dict(path=str(injection), state="after", observed=injection_binding)])
    complete = _json(series / "protocol-revision" / "complete.json", complete_value)
    request = dict(run_id=run, configuration_files=[actual],
        series_context=dict(root=str(series), manifest=manifest),
        baseline=dict(path=str(raw), sha256="2" * 64, size_bytes=1),
        protocol_inputs=[dict(role="imaging", folder=str(folder), sheet=None, source=new_csv_binding),
                         dict(role="injection", folder=None, sheet=None, source=injection_binding),
                         dict(role="pressure_temperature", folder=None, sheet=None, source=pressure_binding)])
    proof = dict(original_configuration_copy=source, source_lineage=source_lineage,
                 accepted_configurations=accepted, protocol_revision=complete,
                 protocol_changeset=changeset)
    parent = dict(source, path=str(tmp_path / "historical" / f"{run}.toml"))
    return dict(run=run, request=request, proof=proof, parent=parent,
                actual_path=actual_path, changeset_value=dict(replacements=[replacement],
                source_transform=dict(path=str(tmp_path / "existing_transform.py"),
                sha256=lineage._INJECTION_TRANSFORM_SHA256, size_bytes=1967)),
                complete_value=complete_value, source_lineage_path=Path(source_lineage["path"]),
                old_injection=old_injection)


def _check(case, pins=None):
    """Invoke the actual opt-in helper with original reader byte transports."""
    return lineage._require_shape_configuration_lineage(_Pins() if pins is None else pins,
        run=case["run"], parent_configuration=case["parent"],
        producer_configuration_files=case["request"]["configuration_files"],
        request=case["request"], proof=case["proof"])


def _replace_current(case, raw):
    """Rebind a synthetic actual request after an intentional metadata change."""
    case["request"]["configuration_files"] = [_write(case["actual_path"], raw)]


def _replace_revision(case):
    """Keep a malicious fixture internally pinned so semantic checks are tested."""
    proof = case["proof"]
    proof["protocol_changeset"] = _json(Path(proof["protocol_changeset"]["path"]), case["changeset_value"])
    case["complete_value"]["changeset"] = proof["protocol_changeset"]
    proof["protocol_revision"] = _json(Path(proof["protocol_revision"]["path"]), case["complete_value"])


def test_complete_original_lineage_and_typed_consumption(synthetic_lineage):
    """The genuine original validator and loaders admit one synthetic complete chain."""
    case = synthetic_lineage
    pins = _Pins()
    before = deepcopy(case["request"])
    audit = _check(case, pins)
    pins.final()
    assert audit["parent_configuration"] == case["parent"]
    assert audit["consumed_configuration_files"] == case["request"]["configuration_files"]
    assert audit["parent_configuration"] != audit["consumed_configuration_files"][0]
    assert audit["typed_consumption"]["typed_protocol_differences"] == [dict(
        field="blacklist", before=None, after={}, consumption="None for each original imaging folder")]
    assert audit["injection_revision"][0]["changed_cells"] == 1
    assert len(audit["typed_consumption"]["data_fields"]) > 0
    assert case["request"] == before
    assert not Path(case["request"]["baseline"]["path"]).exists()
    assert "no complete configuration" in audit["scope"]


@pytest.mark.parametrize("key", sorted(lineage._PROOF_KEYS))
def test_missing_explicit_proof_input_refuses(synthetic_lineage, key):
    """Every historical dependency remains explicit rather than inferred."""
    del synthetic_lineage["proof"][key]
    with pytest.raises(ValueError, match="five explicit"):
        _check(synthetic_lineage)


def test_duplicate_actual_configuration_refuses(synthetic_lineage):
    """Repeated producer locator identities cannot establish a new config chain."""
    rows = synthetic_lineage["request"]["configuration_files"]
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        _check(synthetic_lineage)


def test_existing_exact_membership_is_not_a_lineage_fallback(synthetic_lineage):
    """The optional helper cannot replace the original exact-membership route."""
    synthetic_lineage["parent"] = deepcopy(synthetic_lineage["request"]["configuration_files"][0])
    with pytest.raises(ValueError, match="unmatched"):
        _check(synthetic_lineage)


def test_reversed_accepted_bundle_refuses(synthetic_lineage):
    """Original run/merged order and accepted source identities remain required."""
    synthetic_lineage["proof"]["accepted_configurations"].reverse()
    with pytest.raises(ValueError, match="accepted RAW configuration"):
        _check(synthetic_lineage)


def test_changed_parent_copy_refuses(synthetic_lineage):
    """A retained source must keep the exact historical parent's bytes."""
    synthetic_lineage["parent"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent bytes"):
        _check(synthetic_lineage)


def test_missing_old_apply_target_does_not_use_migration_archive_as_waiver(synthetic_lineage):
    """A migration before archive cannot replace the original independent Apply proof."""
    binding = synthetic_lineage["proof"]["source_lineage"]
    value = json.loads(Path(binding["path"]).read_bytes())
    target = Path(value["prior_binding"]["apply_configurations"][0]["path"])
    target.write_bytes(b"changed target without an original completed Apply backup")
    with pytest.raises(ValueError, match="source identity mismatch"):
        _check(synthetic_lineage)


def test_original_typed_correction_change_refuses(synthetic_lineage):
    """A real native float32-to-float64 setup change fails the shape-input proof."""
    case = synthetic_lineage
    _replace_current(case, case["actual_path"].read_bytes().replace(b'"float32"', b'"float64"'))
    with pytest.raises(ValueError, match="typed data or correction"):
        _check(case)


def test_nonempty_blacklist_is_not_normalized(synthetic_lineage):
    """Only the original absent/empty consumer equivalence is allowed."""
    case = synthetic_lineage
    raw = case["actual_path"].read_bytes().replace(b"blacklist = {}", b'blacklist = "blocked.csv"')
    _replace_current(case, raw)
    with pytest.raises(ValueError, match="typed protocol"):
        _check(case)


def test_clock_row_change_fails_original_migration(synthetic_lineage):
    """Changing an accepted imaging timestamp fails the original raw/clock proof."""
    case = synthetic_lineage
    clock = Path(case["request"]["protocol_inputs"][0]["source"]["path"])
    clock.write_bytes(clock.read_bytes().replace(b"2020-01-01", b"2020-01-02"))
    with pytest.raises(ValueError, match="capture, timestamp, flag or JPEG hash lineage"):
        _check(case)


def test_rebound_injection_other_cell_change_refuses(synthetic_lineage):
    """Even coherently repinned receipts cannot authorize a changed clock cell."""
    case = synthetic_lineage
    item = case["request"]["protocol_inputs"][1]
    changed = _write(Path(item["source"]["path"]), case["old_injection"].replace(
        b",0.03,", b",0.02,").replace(b"2020-01-01", b"2020-01-02"))
    item["source"] = changed
    case["changeset_value"]["replacements"][0]["after"] = changed
    case["complete_value"]["final_original_states"][0]["observed"] = changed
    _replace_revision(case)
    with pytest.raises(ValueError, match="exact existing byte transform"):
        _check(case)


def test_protocol_backup_cannot_leave_preparation(synthetic_lineage):
    """A revision's relative backup cannot select an unrelated historical file."""
    case = synthetic_lineage
    case["changeset_value"]["replacements"][0]["before_local"] = "../injection.csv"
    _replace_revision(case)
    with pytest.raises(ValueError, match="preparation-relative"):
        _check(case)


def test_completed_protocol_series_identity_must_match(synthetic_lineage):
    """The protocol revision cannot be borrowed from a different current series."""
    case = synthetic_lineage
    case["complete_value"]["current_series_identity"]["sha256"] = "9" * 64
    _replace_revision(case)
    with pytest.raises(ValueError, match="exact completed protocol revision"):
        _check(case)


def test_original_source_remains_in_final_rehash(synthetic_lineage):
    """The reader still detects a consumed retained source changing after admission."""
    pins = _Pins()
    _check(synthetic_lineage, pins)
    source = Path(synthetic_lineage["proof"]["original_configuration_copy"]["path"])
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="size changed"):
        pins.final()


def test_typed_loader_alias_cannot_hide_changed_logical_config(synthetic_lineage, tmp_path):
    """Original loaders may not consume an unbound logical file behind a valid copy."""
    case = synthetic_lineage
    original = deepcopy(case["request"]["configuration_files"][0])
    folder = tmp_path / "preparation"
    copied = _write(folder / "input.toml", case["actual_path"].read_bytes())
    alias = dict(original=original, copy=dict(copied, path="input.toml"))
    case["actual_path"].write_bytes(case["actual_path"].read_bytes().replace(
        b'"float32"', b'"float64"'))
    pins = _Pins([alias], folder=folder)
    with pytest.raises(ValueError, match="unaliased consumed TOML"):
        _check(case, pins)


def test_byte_transform_preserves_bom_newlines_and_all_other_cells():
    """The original protocol transformation changes only the intended coordinate."""
    raw = b"\xef\xbb\xbfstart,location_x,location_y,rate\r\n2020-01-01,0.45,0.03,1\r\n2020-01-02,0.72,0.18,2\r\n"
    actual, changed = lineage._transform_injection_csv(raw)
    assert actual == raw.replace(b",0.03,", b",0.02,")
    assert changed == 1
