"""Explicit historical configuration lineage for a RAW shape-only reader.

This unused opt-in helper preserves the original source-migration validator.
It compares the original typed metadata consumed by shape setup, without
claiming full rig, optical calibration, physical-model or pixel equivalence.
It never rewrites a parent revision, configuration, certificate or output.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib


_ORIGINAL_VALIDATOR_SHA256 = "30019a393e1fd5874b68740d17909c26bd8a0ee901f98a811c42cc8ebd586569"
_INJECTION_TRANSFORM_SHA256 = "615ee1ceb8452852d522dd598855a73681d87b4329005613b8b642ac2e280400"
_IDENTITY_KEYS = {"path", "sha256", "size_bytes"}
_PROOF_KEYS = {"original_configuration_copy", "source_lineage",
               "accepted_configurations", "protocol_revision", "protocol_changeset"}


def _identity(value):
    """Validate one existing absolute file descriptor without recapturing it."""
    if (not isinstance(value, dict) or set(value) != _IDENTITY_KEYS
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
        raise ValueError("Shape configuration lineage requires exact file descriptors")
    return deepcopy(value)


def _key(path):
    """Use the reader's locator identity without resolving historical aliases."""
    return str(Path(path)).replace("\\", "/").casefold()


def _read_bound(pins, binding, *, limit):
    """Read bounded bytes through the original reader's verified physical pin."""
    binding = _identity(binding)
    if binding["size_bytes"] > limit:
        raise ValueError("Shape configuration lineage input exceeds its existing bound")
    pins.add(binding, copyable=True)
    physical = pins.rows[_key(binding["path"])]["physical"]
    with Path(physical["path"]).open("rb") as stream:
        raw = stream.read(limit + 1)
    if (len(raw) > limit or len(raw) != binding["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != binding["sha256"]):
        raise ValueError("Shape configuration lineage bytes changed")
    return raw


def _read_current(pins, path, *, limit):
    """Retain every original validator metadata read for final reader rehash."""
    path = Path(path)
    key = _key(path)
    if key in getattr(pins, "metadata_replay_keys", ()):
        return _read_bound(pins, pins.aliases[key][0], limit=limit)
    before = path.stat()
    if before.st_size > limit:
        raise ValueError("Shape configuration lineage metadata exceeds its existing bound")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    if (len(raw) > limit or (before.st_size, before.st_mtime_ns)
            != (after.st_size, after.st_mtime_ns)):
        raise ValueError("Shape configuration lineage metadata changed while reading")
    row = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if _read_bound(pins, row, limit=limit) != raw:
        raise ValueError("Shape configuration lineage metadata changed between reads")
    return raw


def _json(pins, binding, *, limit=1024 * 1024):
    """Use the existing duplicate/nonfinite-rejecting metadata decoder."""
    from .pipeline.job_inputs import _json_bytes
    return _json_bytes(_read_bound(pins, binding, limit=limit))


def _typed_equal(first, second):
    """Compare original dataclass values without numeric or container coercion."""
    if type(first) is not type(second):
        return False
    if is_dataclass(first) and not isinstance(first, type):
        return all(_typed_equal(getattr(first, f.name), getattr(second, f.name))
                   for f in fields(first))
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _typed_equal(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _typed_equal(a, b) for a, b in zip(first, second))
    return bool(first == second)


def _existing_lineage(pins, lineage, run, configuration, previous):
    """Run the preserved original validator with reader-bound I/O transports.

    Validation algorithms are copied from the existing corrected-setup source.
    Only its three filesystem transports register consumed files in ``pins``;
    the accepted migration and original Apply/archive predicates stay intact.
    """
    from .series_geometry import geometry_package_root
    LIMIT = 1024 * 1024
    NAMES = {"labels": "native_labels.npz", "facies": "native_facies.npz"}
    EXPORT_FILES = {*NAMES.values(), "facies_properties.csv", "native_setup.toml", "colored_labels.png"}
    CONFIGURATION_BINDING_V2_FORMAT = "darsia-corrected-setup-configuration-dual-binding-v2"
    PROTOCOL_LEAVES = ("protocols.imaging", "protocols.injection", "protocols.imaging_mode")

    def _small(path):
        return _read_current(pins, path, limit=LIMIT)

    def _descriptor(path):
        path = Path(path)
        key = _key(path)
        if key in getattr(pins, "metadata_replay_keys", ()):
            original = pins.aliases[key][0]
            pins.add(original, copyable=True)
            return deepcopy(original)
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                digest.update(data)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during corrected setup verification: {path}")
        row = dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)
        pins.add(row, copyable=True)
        return row

    def _verified(row):
        actual = _descriptor(row["path"])
        if actual["sha256"] != row.get("sha256") or actual["size_bytes"] != row.get("size_bytes"):
            raise ValueError(f"Corrected setup source identity mismatch: {actual['path']}")
        return actual

    def _same(first, second):
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))

    def _run_name(run):
        if not isinstance(run, str) or not run or run in (".", "..") or Path(run).name != run or any(c in run for c in "\\/:\x00"):
            raise ValueError("Corrected setup requires one explicit safe run ID.")
        return run

    def _typed_equal(first, second):
        """TOML/JSON equality without bool/int, int/float or container coercion."""
        if type(first) is not type(second):
            return False
        if isinstance(first, dict):
            return first.keys() == second.keys() and all(_typed_equal(first[key], second[key]) for key in first)
        if isinstance(first, list):
            return len(first) == len(second) and all(_typed_equal(a, b) for a, b in zip(first, second))
        return first == second

    def _protocol_changes(original, current):
        """Exactly three explicit clock-pointer/mode changes, no other difference."""
        restored = deepcopy(current)
        changes = {}
        old_protocols, new_protocols = original.get("protocols"), restored.get("protocols")
        if not isinstance(old_protocols, dict) or not isinstance(new_protocols, dict):
            raise ValueError("Dual configuration binding requires both explicit protocol tables.")
        for leaf in PROTOCOL_LEAVES:
            key = leaf.split(".")[1]
            if key not in old_protocols or key not in new_protocols or _typed_equal(old_protocols[key], new_protocols[key]):
                raise ValueError("Dual configuration binding requires exactly the three protocol-leaf differences.")
            changes[leaf] = dict(before=deepcopy(old_protocols[key]), after=deepcopy(new_protocols[key]))
            new_protocols[key] = deepcopy(old_protocols[key])
        if not _typed_equal(original, restored):
            raise ValueError("Dual configuration binding changes a non-protocol value or TOML type.")
        return changes

    def _configuration_identity(row):
        if (not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}
                or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                or not isinstance(row["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or type(row["size_bytes"]) is not int or row["size_bytes"] < 0):
            raise ValueError("Dual configuration identities require exact absolute path/SHA/byte descriptors.")
        return row

    def _configuration_bytes_v2(row):
        """Bind the exact bytes consumed by the new v2 TOML parser."""
        raw = _small(row["path"])
        if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("V2 consumed configuration bytes differ from the bound SHA/size identity.")
        return raw

    def _dual_configuration_binding_v2(value, run, configuration, selected, *, series,
                                       project_root, revision, destination, receipt):
        """Bind one separately migrated output pointer, never a geometry override."""
        keys = {"format", "run_id", "source_configuration", "current_configuration",
                "apply_configurations", "pre_pointer_configurations", "protocol_change_mode", "protocol_changes",
                "output_pointer_change", "comparison_sha256"}
        if (not isinstance(value, dict) or set(value) != keys
                or value["format"] != CONFIGURATION_BINDING_V2_FORMAT or value["run_id"] != run
                or not isinstance(value["comparison_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["comparison_sha256"]) is None):
            raise ValueError("The v2 selected-run configuration metadata is not explicit and complete.")
        if (project_root is None or not Path(project_root).is_absolute()
                or not _same(project_root, geometry_package_root(Path(series)))
                or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{32}", revision) is None):
            raise ValueError("The v2 output pointer requires the actual series project and revision.")
        original = _configuration_identity(value["source_configuration"])
        current = _configuration_identity(value["current_configuration"])
        targets = value["apply_configurations"]
        if (not _typed_equal(original, configuration) or not isinstance(targets, list)
                or len(selected) != 2 or len(targets) != 2):
            raise ValueError("The v2 binding belongs to another source or selected bundle.")
        for target, declared in zip(targets, selected):
            _configuration_identity(target)
            if not _same(target["path"], declared) or _same(target["path"], original["path"]):
                raise ValueError("V2 Apply targets must be the exact current declared bundle, never the historical source.")
        if not _typed_equal(current, targets[-1]):
            raise ValueError("The v2 current configuration must be the declared merged target.")
        previous = value["pre_pointer_configurations"]
        if not isinstance(previous, list) or len(previous) != 2:
            raise ValueError("V2 requires exact preserved pre-pointer partial and merged identities.")
        for before, role in zip(previous, ("runs", "merged")):
            _configuration_identity(before)
            path = Path(before["path"])
            if (not path.resolve().is_relative_to(Path(series).resolve())
                    or path.parent.name != role or path.name != run + ".toml"
                    or any(_same(path, target["path"]) for target in targets)):
                raise ValueError("V2 pre-pointer identities must be preserved inside this series in exact run partial/merged order.")
        pointer = value["output_pointer_change"]
        expected = Path(project_root) / "results" / "rig_models" / revision / _run_name(run)
        if (not isinstance(pointer, dict) or set(pointer) != {"field", "before", "after"}
                or pointer["field"] != "rig.path"
                or not isinstance(pointer["before"], str) or not pointer["before"]
                or not Path(pointer["before"]).is_absolute()
                or not isinstance(pointer["after"], str) or pointer["after"] != str(expected)):
            raise ValueError("V2 permits only the explicit rig.path change to this project's fixed revision/run destination.")
        old_folder, new_folder = Path(pointer["before"]).resolve(), expected.resolve()
        if old_folder.is_relative_to(new_folder) or new_folder.is_relative_to(old_folder):
            raise ValueError("The v2 rig output must not overlap the preserved original rig folder.")
        source = tomllib.loads(_configuration_bytes_v2(original).decode("utf-8-sig"))
        if (source.get("rig") or {}).get("path") != pointer["before"]:
            raise ValueError("The v2 output pointer does not bind the original rig.path.")
        raw_targets = [_original_configuration_bytes(target, destination, receipt) for target in targets]
        configs = [tomllib.loads(raw.decode("utf-8-sig")) for raw in raw_targets]
        pre_configs = [tomllib.loads(_configuration_bytes_v2(before).decode("utf-8-sig")) for before in previous]
        if any((config.get("rig") or {}).get("path") != pointer["after"] for config in configs):
            raise ValueError("Both v2 declared Apply targets must carry the exact new rig.path.")
        for before, after in zip(pre_configs, configs):
            if (before.get("rig") or {}).get("path") != pointer["before"]:
                raise ValueError("Both preserved v2 pre-pointer configurations must carry the exact old rig.path.")
            restored = deepcopy(after)
            restored["rig"]["path"] = pointer["before"]
            if not _typed_equal(before, restored):
                raise ValueError("V2 target changes a value or TOML type beyond its own preserved rig.path migration.")
        mode = value["protocol_change_mode"]
        if mode == "clock-migrated-three-leaves":
            changes = _protocol_changes(source, pre_configs[-1])
        elif mode == "unchanged-zero-leaves":
            if not _typed_equal(source, pre_configs[-1]):
                raise ValueError("The zero-protocol v2 binding changes another value or TOML type.")
            changes = {}
        else:
            raise ValueError("V2 requires an explicit zero- or three-protocol change mode.")
        if not _typed_equal(value["protocol_changes"], changes):
            raise ValueError("V2 protocol metadata does not record its exact declared changes.")
        return deepcopy(value)

    def _raw_capture_pair(before, after):
        """Change a JPEG suffix only, preserving the exact capture and folder."""
        if not isinstance(before, str) or not isinstance(after, str):
            return False
        a, b = Path(before), Path(after)
        return (a.suffix.lower() in (".jpg", ".jpeg")
                and b.suffix.lower() in (".arw", ".nef", ".cr2", ".cr3", ".dng")
                and os.path.normcase(os.path.normpath(before[:-len(a.suffix)]))
                == os.path.normcase(os.path.normpath(after[:-len(b.suffix)])))

    def _raw_capture_structure(before, after):
        if type(before) is not type(after):
            return False
        if isinstance(before, dict):
            return before.keys() == after.keys() and all(
                _raw_capture_structure(before[k], after[k]) for k in before)
        if isinstance(before, list):
            return len(before) == len(after) and all(
                _raw_capture_structure(a, b) for a, b in zip(before, after))
        return _typed_equal(before, after) or _raw_capture_pair(before, after)

    def _migration_receipt(row):
        _configuration_identity(row)
        if row["size_bytes"] > 64 * 1024 * 1024:
            raise ValueError("The source migration receipt exceeds 64 MiB.")
        raw = _read_bound(pins, row, limit=64 * 1024 * 1024)
        if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("The consumed source migration receipt differs from its identity.")
        value = json.loads(raw)
        if (value.get("format") != "n117-raw-configuration-switch-v1"
                or value.get("all43_baselines_ARW") is not True
                or value.get("all39_selected_tables_bound_RAW_exact_original_times_indices") is not True):
            raise ValueError("A completed, accepted RAW source migration receipt is required.")
        return value

    def _migration_targets(value, run):
        rows = value.get("new_files", []) if run == value.get("run") else (
            value.get("remaining_FF_AC", {}).get("runs", {}).get(run, {}).get("new_files", []))
        result = []
        for role in ("runs", "merged"):
            matches = [x for x in rows if Path(x["path"]).parent.name == role
                       and Path(x["path"]).name == run + ".toml"]
            if len(matches) != 1:
                raise ValueError("The accepted source migration does not uniquely bind this run's two configurations.")
            result.append(_configuration_identity(matches[0]))
        return result

    def _migration_archive(value, original):
        matches = [x for key, x in value.get("originals", {}).items() if _same(key, original)]
        if len(matches) != 1:
            raise ValueError("The migration lacks one exact preserved original configuration or clock file.")
        item = matches[0]
        before, archived = _configuration_identity(item["before"]), _configuration_identity(item["archive"])
        if (not _same(before["path"], original)
                or before["sha256"] != archived["sha256"]
                or before["size_bytes"] != archived["size_bytes"]):
            raise ValueError("The migration archive does not preserve the exact original bytes.")
        return archived

    def _raw_protocol_comparison(before, after, migration):
        """Prove row order, timestamps and flags; RAW hashes keep the JPEG lineage."""
        import csv
        import io
        if not isinstance(before, dict) or not isinstance(after, dict) or before.keys() != after.keys():
            raise ValueError("RAW migration changed the imaging folders.")
        records = []
        for folder in before:
            old = _migration_archive(migration, before[folder])
            old_raw = _configuration_bytes_v2(old)
            new = _descriptor(after[folder])
            new_raw = _configuration_bytes_v2(new)
            left = list(csv.DictReader(io.StringIO(old_raw.decode("utf-8-sig"))))
            right = list(csv.DictReader(io.StringIO(new_raw.decode("utf-8-sig"))))
            if not left or len(left) != len(right):
                raise ValueError("RAW migration changed the imaging protocol length.")
            for a, b in zip(left, right):
                extra = {"legacy_jpeg_sha256"} if "source_sha256" in a else {"source_sha256"}
                if (set(b) != set(a) | extra
                        or not _raw_capture_pair(a.get("path"), b.get("path"))
                        or ("source_sha256" in a and b.get("legacy_jpeg_sha256") != a["source_sha256"])
                        or re.fullmatch(r"[0-9a-f]{64}", b.get("source_sha256", "")) is None
                        or any(a[k] != b[k] for k in a if k not in ("path", "source_sha256"))):
                    raise ValueError("RAW migration changed a capture, timestamp, flag or JPEG hash lineage.")
            records.append(dict(folder=folder, before=old, after=new, rows=len(left)))
        return records

    def _source_migration_changes(before, after, migration):
        restored = deepcopy(after)
        changes = {}
        for section, key in (("data", "baseline"), ("data", "format"), (None, "data_path")):
            old_table = before if section is None else before.get(section, {})
            new_table = restored if section is None else restored.get(section, {})
            old_present, new_present = key in old_table, key in new_table
            old, new = old_table.get(key), new_table.get(key)
            leaf = f"{section}.{key}" if section else key
            if key == "baseline" and not _raw_capture_pair(old, new):
                raise ValueError("RAW baseline migration must preserve the exact original capture.")
            if key == "format":
                baseline = Path(after.get("data", {}).get("baseline", ""))
                if (not new_present or not isinstance(new, str)
                        or new.lower() != baseline.suffix[1:].lower()
                        or (old_present and old.lower() not in ("jpg", "jpeg"))):
                    raise ValueError("RAW format must match the explicitly bound RAW baseline.")
            if key == "data_path" and (not old_present or not new_present or not _raw_capture_structure(old, new)):
                raise ValueError("RAW data_path must preserve every original capture and the typed structure.")
            changes[leaf] = dict(before=dict(present=old_present, value=old) if old_present else dict(present=False),
                                 after=dict(present=new_present, value=new) if new_present else dict(present=False))
            if old_present:
                new_table[key] = deepcopy(old)
            else:
                new_table.pop(key, None)
        old_protocols, new_protocols = before.get("protocols", {}), restored.get("protocols", {})
        protocols = _raw_protocol_comparison(old_protocols.get("imaging"), new_protocols.get("imaging"), migration)
        new_protocols["imaging"] = deepcopy(old_protocols["imaging"])
        if not _typed_equal(before, restored):
            raise ValueError("RAW source migration changed another configuration value or TOML type.")
        return dict(configuration_leaves=changes, imaging_protocols=protocols)

    def _v3_lineage_proof(lineage, run, configuration, previous, *, migration=None):
        keys = {"format", "run_id", "prior_binding", "legacy_series_root", "legacy_project_root",
                "legacy_revision_id", "legacy_export", "source_migration_receipt",
                "accepted_configurations", "raw_changes"}
        if (not isinstance(lineage, dict) or set(lineage) != keys
                or lineage["format"] != "darsia-corrected-setup-source-lineage-v1"
                or lineage["run_id"] != run):
            raise ValueError("The v3 source lineage is incomplete or belongs to another run.")
        prior = lineage["prior_binding"]
        old_series, old_project = Path(lineage["legacy_series_root"]), Path(lineage["legacy_project_root"])
        old_revision = lineage["legacy_revision_id"]
        export_row = lineage["legacy_export"]
        export_receipt = json.loads(_configuration_bytes_v2(export_row)) if export_row else None
        old_destination = Path(export_row["path"]).parent if export_row else None
        if export_receipt is not None:
            saved_source = export_receipt.get("binding", {}).get("configuration", {})
            if (export_receipt.get("run_id") != run or export_receipt.get("revision_id") != old_revision
                    or saved_source.get("sha256") != configuration.get("sha256")
                    or saved_source.get("size_bytes") != configuration.get("size_bytes")):
                raise ValueError("The preserved Apply export belongs to another run, revision or source.")
            _verify_export_files(export_receipt, old_destination)
        _dual_configuration_binding_v2(prior, run, configuration,
            [x["path"] for x in prior["apply_configurations"]], series=old_series,
            project_root=old_project, revision=old_revision,
            destination=old_destination, receipt=export_receipt)
        migration = _migration_receipt(lineage["source_migration_receipt"]) if migration is None else migration
        accepted = _migration_targets(migration, run)
        if not _typed_equal(accepted, lineage["accepted_configurations"]):
            raise ValueError("The v3 accepted configurations differ from the source migration receipt.")
        records = []
        for source, target, local in zip(prior["apply_configurations"], accepted, previous):
            _configuration_identity(local)
            if local["sha256"] != target["sha256"] or local["size_bytes"] != target["size_bytes"]:
                raise ValueError("A v3 preserved bundle differs from the accepted RAW configuration.")
            raw = _configuration_bytes_v2(local)
            archive = _migration_archive(migration, source["path"])
            before = _configuration_bytes_v2(archive)
            bound = _original_configuration_bytes(source, old_destination, export_receipt)
            if before != bound:
                if (export_receipt is None or _apply_completion(old_destination, export_receipt) is None
                        or before != corrected_config_bytes(bound, export_receipt["binding"]["replacements"])):
                    raise ValueError("The source migration's original is neither the bound source nor its proved four-leaf Apply.")
            records.append(_source_migration_changes(tomllib.loads(before.decode("utf-8-sig")),
                                                    tomllib.loads(raw.decode("utf-8-sig")), migration))
        if not _typed_equal(records, lineage["raw_changes"]):
            raise ValueError("The v3 source lineage does not record the exact RAW and clock migration.")
        return lineage

    def _verify_export_files(receipt, destination):
        if set(receipt.get("files") or {}) != EXPORT_FILES:
            raise ValueError("The corrected export must contain exactly its five canonical named files.")
        for name, row in receipt["files"].items():
            if not _same(row["path"], destination / name):
                raise ValueError("A corrected export file belongs to another destination.")
            _verified(row)
        binding = receipt["binding"]
        for name, source in ((NAMES["labels"], binding["sources"]["labels"]),
                             (NAMES["facies"], binding["sources"]["facies"]),
                             ("facies_properties.csv", binding["properties"])):
            row = receipt["files"][name]
            if row["sha256"] != source["sha256"] or row["size_bytes"] != source["size_bytes"]:
                raise ValueError("The corrected export byte copies do not match their revision sources.")

    def _apply_completion(destination, receipt):
        """Read one immutable successful application, refusing any partial claim."""
        complete = destination / "apply-complete.json"
        claim = destination / "apply-claim.json"
        if not complete.exists():
            if claim.exists():
                raise ValueError("An earlier corrected Apply is incomplete; all partial artifacts were preserved.")
            return None
        value = json.loads(_small(complete))
        if (value.get("format") != "darsia-corrected-setup-apply-complete-v1"
                or value.get("complete") is not True
                or value.get("run_id") != receipt["run_id"]
                or value.get("revision_id") != receipt["revision_id"]
                or value.get("export") != _descriptor(destination / "export.json")
                or value.get("input_binding") != receipt["binding"]):
            raise ValueError("An existing corrected Apply completion has a different identity; it was preserved.")
        claim_value = json.loads(_small(claim))
        if (claim_value.get("format") != "darsia-corrected-setup-apply-claim-v1"
                or claim_value.get("run_id") != receipt["run_id"]
                or claim_value.get("revision_id") != receipt["revision_id"]
                or claim_value.get("export") != value["export"]
                or claim_value.get("input_binding") != value["input_binding"]
                or not _same(Path(value["attempt_receipt"]["path"]).parent, claim_value.get("attempt_directory", ""))
                or Path(value["attempt_receipt"]["path"]).name != "receipt.json"
                or not Path(value["attempt_receipt"]["path"]).resolve().is_relative_to(destination.resolve())):
            raise ValueError("The corrected Apply completion does not match its preserved exclusive claim.")
        attempt = _verified(value["attempt_receipt"])
        record = json.loads(_small(attempt["path"]))
        if (record.get("state") != "complete" or record.get("run_id") != receipt["run_id"]
                or record.get("revision_id") != receipt["revision_id"]):
            raise ValueError("The preserved corrected Apply record is not complete for this run.")
        files = value.get("configurations")
        if not isinstance(files, list) or not files or len(files) != len(record.get("files") or []):
            raise ValueError("The corrected Apply completion has no exact configuration bundle.")
        for index, row in enumerate(files):
            old = record["files"][index]
            if (old.get("state") != "applied" or not _same(old["path"], row["after"]["path"])
                    or old.get("before_sha256") != row["before"]["sha256"]
                    or old.get("after_sha256") != row["after"]["sha256"]
                    or not _same(row["before"]["path"], Path(attempt["path"]).parent / f"{index:02d}-before.toml")
                    or not any(_same(row["after"]["path"], p) for p in receipt["binding"]["configuration_bundles"][receipt["run_id"]])):
                raise ValueError("The corrected Apply completion does not bind its actual before/after files.")
            _verified(row["before"])
            _verified(row["after"])
        return value

    def _original_configuration_bytes(configuration, destination, receipt=None):
        """Keep the source SHA; only a proved prior Apply may supply its backup."""
        before = _small(configuration["path"])
        if hashlib.sha256(before).hexdigest() == configuration["sha256"] and len(before) == configuration["size_bytes"]:
            return before
        if receipt is not None:
            completed = _apply_completion(destination, receipt)
            for row in (completed or {}).get("configurations", []):
                if (_same(row["after"]["path"], configuration["path"])
                        and row["before"]["sha256"] == configuration["sha256"]
                        and row["before"]["size_bytes"] == configuration["size_bytes"]):
                    return _small(row["before"]["path"])
        raise ValueError(f"Corrected setup source identity mismatch: {configuration['path']}")

    def corrected_config_bytes(before: bytes, replacements: dict) -> bytes:
        """Change only four physical lines/insertions; preserve all other bytes."""
        original = tomllib.loads(before.decode("utf-8-sig"))
        if set(replacements) != {"labeling", "facies"} or set(replacements["labeling"]) != {"labels", "frame"} or set(replacements["facies"]) != {"path", "props"} or replacements["labeling"]["frame"] != "rig-corrected":
            raise ValueError("Only the four explicit corrected setup leaves may be applied.")
        expected = deepcopy(original)
        bom = b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
        text = before[len(bom):].decode("utf-8")
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines(keepends=True)
        for section, updates in replacements.items():
            expected.setdefault(section, {}).update(updates)
            headers = [(i, re.match(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$", line.rstrip("\r\n"))) for i, line in enumerate(lines)]
            starts = [i for i, m in headers if m and m.group(1).strip() == section]
            if len(starts) > 1:
                raise ValueError("Duplicate setup table.")
            if not starts:
                if lines and not lines[-1].endswith(("\n", "\r")):
                    lines[-1] += newline
                lines += [newline, f"[{section}]{newline}"]
                start = len(lines)-1
            else:
                start = starts[0]
            end = next((i for i in range(start+1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
            for key, value in updates.items():
                found = [i for i in range(start+1, end) if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i])]
                replacement = f"{key} = {json.dumps(value, ensure_ascii=False)}{newline}"
                if found:
                    if len(found) != 1:
                        raise ValueError("Duplicate setup key.")
                    lines[found[0]] = replacement
                else:
                    if end and not lines[end-1].endswith(("\n", "\r")):
                        lines[end-1] += newline
                    lines.insert(end, replacement)
                    end += 1
        after = bom + "".join(lines).encode("utf-8")
        if tomllib.loads(after.decode("utf-8-sig")) != expected:
            raise ValueError("Corrected setup text patch changed unrelated configuration values.")
        return after

    return _v3_lineage_proof(lineage, run, configuration, previous)


def _transform_injection_csv(raw):
    """Replay the existing byte-preserving injection-coordinate transform."""
    import csv
    import io
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    payload = raw[len(bom):]
    lines = payload.splitlines(keepends=True)
    if not lines or b'"' in payload:
        raise ValueError("This adapter admits only the existing unquoted CSV form")
    header = lines[0].rstrip(b"\r\n").split(b",")
    if len(header) != len(set(header)) or header[1:3] != [b"location_x", b"location_y"]:
        raise ValueError("Unexpected coordinate-column schema")
    result, changed, other = [lines[0]], 0, set()
    for line in lines[1:]:
        body = line.rstrip(b"\r\n")
        ending = line[len(body):]
        if not body:
            result.append(line)
            continue
        values = body.split(b",")
        if len(values) != len(header):
            raise ValueError("Unexpected CSV row schema")
        point = (values[1], values[2])
        if point == (b"0.45", b"0.03"):
            values[2] = b"0.02"
            changed += 1
            result.append(b",".join(values) + ending)
        elif point in {(b"0.45", b"0.02"), (b"0.72", b"0.18")}:
            other.add(point)
            result.append(line)
        else:
            raise ValueError("Unexpected injection point; do not guess its identity")
    after = bom + b"".join(result)
    old_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    new_rows = list(csv.DictReader(io.StringIO(after.decode("utf-8-sig"), newline="")))
    if len(old_rows) != len(new_rows):
        raise ValueError("Injection transform changed the row count")
    for before, current in zip(old_rows, new_rows):
        expected = dict(before)
        if (before["location_x"], before["location_y"]) == ("0.45", "0.03"):
            expected["location_y"] = "0.02"
        if current != expected:
            raise ValueError("Injection transform changed another cell")
    return after, changed


def _protocol_revision(pins, request, revision_binding, changeset_binding):
    """Join current injection bytes to the completed existing protocol revision."""
    revision = _json(pins, revision_binding)
    changeset = _json(pins, changeset_binding)
    if (not isinstance(revision, dict) or revision.get("format") != "ffac-n396-completed-v1"
            or revision.get("complete") is not True
            or not _typed_equal(revision.get("changeset"), changeset_binding)
            or not _typed_equal(revision.get("current_series_identity"), request["series_context"]["manifest"])
            or (changeset.get("source_transform") or {}).get("sha256") != _INJECTION_TRANSFORM_SHA256):
        raise ValueError("Shape configuration lineage lacks the exact completed protocol revision")
    records = []
    for item in request["protocol_inputs"]:
        _read_bound(pins, item["source"], limit=16 * 1024 * 1024)
        if item["role"] != "injection":
            continue
        matches = [r for r in changeset.get("replacements", [])
                   if r.get("role") == "injection_protocol"
                   and _typed_equal(r.get("after"), item["source"])]
        states = [r for r in revision.get("final_original_states", [])
                  if r.get("state") == "after"
                  and _typed_equal(r.get("observed"), item["source"])
                  and r.get("path") == item["source"]["path"]]
        if len(matches) != 1 or len(states) != 1:
            raise ValueError("The consumed injection protocol lacks one exact completed revision row")
        row = matches[0]
        relative = row.get("before_local")
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise ValueError("The original protocol revision backup is not preparation-relative")
        before = dict(_identity(row["before"]), path=str(Path(changeset_binding["path"]).parent / relative))
        raw = _read_bound(pins, before, limit=16 * 1024 * 1024)
        after = _read_bound(pins, item["source"], limit=16 * 1024 * 1024)
        transformed, changed = _transform_injection_csv(raw)
        if transformed != after or type(row.get("changed_cells")) is not int or changed != row["changed_cells"]:
            raise ValueError("The current injection protocol differs from its exact existing byte transform")
        records.append(dict(before=deepcopy(row["before"]), verified_backup=before,
                            after=deepcopy(item["source"]), changed_cells=changed))
    if not records:
        raise ValueError("The shape configuration lineage lacks its consumed injection protocol")
    return records


@contextmanager
def _historical_metadata_loader(pins):
    """Route unchanged typed parsers through explicitly selected exact copies.

    Logical source paths stay original. Only authenticated metadata reads are
    replaced, in this context's thread; current acquisition pixels and files
    outside the selected historical metadata retain their physical readers.
    """
    if not getattr(pins, "metadata_replay_keys", ()):
        yield
        return
    from . import preset_wasserstein_inputs as io
    from darsia.presets.workflows.config.utils import _config_path
    if io.current_scope() is not None:
        raise ValueError("Historical metadata replay cannot nest another input scope")

    class MetadataScope:
        # This is replay of original logical metadata, before later host/path
        # relocations. Physical acquisition readers retain their own bindings.
        path_aliases = []
        failed = None

        def read(self, path, **_kwargs):
            try:
                return _read_current(pins, path, limit=16 * 1024 * 1024)
            except Exception as error:
                self.failed = str(error)
                raise

        def descriptor(self, path, expected=None):
            self.read(path)
            result = deepcopy(pins.rows[_key(path)]["original"])
            if expected is not None and any(result[key] != expected[key]
                                            for key in ("sha256", "size_bytes") if key in expected):
                raise ValueError("Historical metadata descriptor differs from its expected source")
            return result

        def exists(self, path, kind):
            if _key(path) in pins.metadata_replay_keys and kind in {"exists", "is_file"}:
                pins.add(pins.aliases[_key(path)][0], copyable=True)
                return True
            return getattr(Path(path), kind)()

        def config_path(self, value, config):
            # Keep the existing logical-series path rules without recursion.
            token = io._ACTIVE.set(None)
            try:
                return _config_path(value, config)
            finally:
                io._ACTIVE.reset(token)

        @contextmanager
        def role(self, _role):
            yield self

    scope = MetadataScope()
    token = io._ACTIVE.set(scope)
    try:
        yield
        if scope.failed is not None:
            raise ValueError("Historical typed metadata read failed: " + scope.failed)
    finally:
        io._ACTIVE.reset(token)


def _typed_consumption(accepted_paths, producer_paths, request, *, pins=None):
    with _historical_metadata_loader(pins):
        return _typed_consumption_original(accepted_paths, producer_paths, request)


def _typed_consumption_original(accepted_paths, producer_paths, request):
    """Compare the original metadata loaders and their exact protocol consumption."""
    from .rig_raw_basis import protocol_sources
    from darsia.presets.workflows.config.data import DataConfig
    from darsia.presets.workflows.config.protocols import ProtocolsConfig
    from darsia.presets.workflows.config.corrections import CorrectionsConfig
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    for paths in (accepted_paths, producer_paths):
        data = _get_section_from_toml(paths, "data")
        if data.get("use_cache", False) is not False or data.get("raw_camera") is not None:
            raise ValueError("Shape lineage requires the original camera-free cache-disabled metadata path")
    old_data = DataConfig().load(accepted_paths, require_data=False, require_results=True)
    new_data = DataConfig().load(producer_paths, require_data=False, require_results=True)
    old_corrections = CorrectionsConfig().load(accepted_paths)
    new_corrections = CorrectionsConfig().load(producer_paths)
    if not _typed_equal(old_data, new_data) or not _typed_equal(old_corrections, new_corrections):
        raise ValueError("The original typed data or correction inputs differ from the accepted RAW lineage")
    old = ProtocolsConfig().load(accepted_paths)
    new = ProtocolsConfig().load(producer_paths)
    difference = []
    for field in fields(old):
        a, b = getattr(old, field.name), getattr(new, field.name)
        if _typed_equal(a, b):
            continue
        if field.name != "blacklist" or not ((a is None and type(b) is dict and not b)
                                             or (b is None and type(a) is dict and not a)):
            raise ValueError("The original typed protocol inputs differ from the accepted RAW lineage")
        difference.append(dict(field="blacklist", before=deepcopy(a), after=deepcopy(b),
                               consumption="None for each original imaging folder"))
    # The original constructor maps absent/empty blacklist dictionaries to the
    # same None argument; no other typed difference is normalized here.
    from .protocol_content import without_content
    old_sources, new_sources = protocol_sources(old), protocol_sources(new)
    if (not _typed_equal(old_sources, new_sources)
            or not _typed_equal(new_sources, without_content(request["protocol_inputs"]))):
        raise ValueError("The exact original consumed protocol files, sheets or folder order differ")
    return dict(data_fields=[f.name for f in fields(old_data)],
                corrections_fields=[f.name for f in fields(old_corrections)],
                protocol_fields=[f.name for f in fields(old)],
                typed_protocol_differences=difference,
                protocol_inputs=deepcopy(new_sources))


def _require_shape_configuration_lineage(pins, *, run, parent_configuration,
                                       producer_configuration_files, request, proof):
    """Prove explicit shape-input lineage only after exact membership is absent.

    The caller keeps its existing refusal when this proof is absent or when
    membership is duplicated. Historical source descriptors remain historical;
    the returned audit identifies the distinct actual consumed configuration.
    """
    if not isinstance(proof, dict) or set(proof) != _PROOF_KEYS:
        raise ValueError("Shape configuration lineage requires its five explicit existing inputs")
    parent = _identity(parent_configuration)
    if not isinstance(producer_configuration_files, list) or not producer_configuration_files:
        raise ValueError("The actual producer configuration chain is missing")
    actual = [_identity(row) for row in producer_configuration_files]
    if len({_key(row["path"]) for row in actual}) != len(actual):
        raise ValueError("Duplicate producer configuration locator identities are forbidden")
    if (not _typed_equal(actual, request.get("configuration_files")) or request.get("run_id") != run
            or any(_typed_equal(row, parent) for row in actual)):
        raise ValueError("Shape lineage requires the exact unmatched producer request chain")
    original_copy = _identity(proof["original_configuration_copy"])
    if any(original_copy[k] != parent[k] for k in ("sha256", "size_bytes")):
        raise ValueError("The preserved original configuration does not keep the parent bytes")
    _read_bound(pins, original_copy, limit=1024 * 1024)
    lineage_binding = _identity(proof["source_lineage"])
    lineage = _json(pins, lineage_binding)
    source = _identity(lineage["prior_binding"]["source_configuration"])
    if any(source[k] != parent[k] for k in ("sha256", "size_bytes")):
        raise ValueError("The existing source lineage belongs to another parent configuration")
    accepted = proof["accepted_configurations"]
    if not isinstance(accepted, list) or len(accepted) != 2:
        raise ValueError("Shape lineage requires the original ordered accepted run and merged files")
    accepted = [_identity(row) for row in accepted]
    for row in [source, *accepted, *actual]:
        _read_bound(pins, row, limit=1024 * 1024)
    series = request["series_context"]
    _read_bound(pins, series["manifest"], limit=1024 * 1024)
    # These original loaders reopen their logical TOML paths. A metadata copy
    # cannot stand in for the distinct physical file they actually consume.
    for row in [*accepted, *actual, series["manifest"]]:
        physical = pins.rows[_key(row["path"])]["physical"]
        if (_key(physical["path"]) != _key(row["path"])
                and _key(row["path"]) not in getattr(pins, "metadata_replay_keys", ())):
            raise ValueError("Original metadata loaders require their unaliased consumed TOML files")
    if (_key(lineage["legacy_series_root"]) != _key(series["root"])
            or _key(series["manifest"]["path"]) != _key(Path(series["root"]) / "series.toml")):
        raise ValueError("The lineage and producer use different actual series roots")
    _existing_lineage(pins, lineage, run, source, accepted)
    consumed = _typed_consumption([Path(r["path"]) for r in accepted],
                                  [Path(r["path"]) for r in actual], request, pins=pins)
    revision = _identity(proof["protocol_revision"])
    changeset = _identity(proof["protocol_changeset"])
    injection = _protocol_revision(pins, request, revision, changeset)
    return dict(parent_configuration=parent, verified_original_configuration_copy=original_copy,
                source_lineage=lineage_binding, accepted_configurations=accepted,
                consumed_configuration_files=actual, protocol_revision=revision,
                protocol_changeset=changeset, typed_consumption=consumed,
                injection_revision=injection, original_validator_source_sha256=_ORIGINAL_VALIDATOR_SHA256,
                scope="RAW shape-input lineage only; no complete configuration or full-rig equivalence claim")
