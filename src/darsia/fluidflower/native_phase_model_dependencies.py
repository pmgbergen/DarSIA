"""Bounded, read-only dependency evidence from explicitly supplied metadata.

This is not a deletion inventory or an admission check. Original Job/InputCatalog,
publication and capacity guards remain authoritative. No directory discovery,
payload hashing, reference following, deletion or relocation is performed here.
An explicit scope can prove a recorded reference; it cannot prove global absence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tomllib


FORMAT = "native-phase-model-dependency-report-v1"
SUPPORTED_SCOPE_KINDS = {
    "training_request": "Saved phase training request and recorded parent evidence",
    "model_ancestry": "Explicit physics inheritance, revision or summary metadata",
    "library_reference": "Original phase-model library pointer",
    "review_session": "Original retained review session",
    "review_source_binding": "Native review source binding receipt",
    "series_manifest": "Saved series.toml model selection",
    "studio_mass_result": "Completed Studio mass result.json",
    "studio_mass_receipt": "Original Studio mass delivery receipt",
    "result_source_manifest": "Original Studio result source path-to-SHA manifest",
    "job_input_catalog": "Original role input catalogue with explicit root aliases",
    "job_registry": "Original local Jobs registry (not a complete results inventory)",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LIMITS = {"max_files": 1024, "max_file_bytes": 8 * 1024 * 1024,
           "max_total_bytes": 64 * 1024 * 1024}
_MAX_ITEMS = 20000


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + " must be an object")
    return value


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(name + " must be a list")
    return value


def _json(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate metadata key: " + key)
            value[key] = item
        return value

    def invalid(value):
        raise ValueError("Nonfinite metadata constant: " + value)

    return _object(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                              parse_constant=invalid), "Metadata")


def _relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("An explicit relative path is required")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value or not path.parts:
        raise ValueError("Relative reference escapes its declared root")
    return path


class _Cancelled(Exception):
    pass


class _Report:
    def __init__(self, entry, series_root, protected_roots, cancelled, limits):
        # The original lexical storage alias mapper reads local configuration;
        # it does not discover models, reviews or results on the configured NAS.
        from .storage_paths import storage_path_remapper

        self.remap = storage_path_remapper()
        self.cancelled, self.limits, self.items = cancelled, limits, 0
        self.root = self.path(entry.get("root"))
        self.series_root = self.path(series_root)
        self.model_id = str(entry.get("model_id") or "").strip()
        self.identity = entry.get("library_source_identity") or entry.get("source_identity") or {}
        self.result = dict(format=FORMAT, model=dict(root=self.root, model_id=self.model_id,
            source_identity=self.identity, identity_verification="recorded identity; not rehashed"),
            refs=[], unknown=[], sources=[], dependency_status="unknown", deletion_allowed=False,
            deletion_reason="Deletion requires a separate explicit approval and an authoritative complete retained-result inventory.",
            coverage=dict(complete=False, scope="explicit metadata only; no references followed",
                path_matching="lexical storage aliases; filesystem links and review relocations not verified",
                requested_files=0, read_files=0, bytes_read=0, cancelled=False, limits=dict(limits)))
        self.source = dict(kind="scope", path="", sha256=None)
        self.unknown("No authoritative complete retained-result or reverse-dependency inventory was supplied.",
                     code="incomplete_scope")
        if (not isinstance(self.identity, dict) or self.identity.get("model_id") != self.model_id
                or not _SHA.fullmatch(str(self.identity.get("version_sha256") or ""))):
            self.identity = {}
            self.unknown("Selected model has no valid cached exact version identity.", code="model_identity_unknown")
        else:
            self.unknown("Cached model identity has not been reverified against current model bytes.",
                         code="model_identity_unverified")
        if not self.model_id:
            self.unknown("Selected model ID is missing.", code="model_id_unknown")
        for value in protected_roots:
            root = self.path(value)
            if self.inside(self.root, root) or self.inside(root, self.root):
                self.ref("protected evidence", "Explicit protected root overlaps the selected model root.",
                         target_path=root, field="protected_roots", match="path")

    def check(self):
        if self.cancelled is not None and self.cancelled():
            raise _Cancelled()

    def tick(self):
        self.items += 1
        if self.items > _MAX_ITEMS:
            raise ValueError("Metadata item limit exceeded")
        if self.items % 64 == 0:
            self.check()

    def path(self, value, *, base=None):
        if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            raise ValueError("An explicit path is required")
        text = self.remap(os.fspath(value))
        if not (Path(text).is_absolute() or PureWindowsPath(text).is_absolute()):
            if base is None:
                raise ValueError("Absolute path required: " + text)
            relative = _relative(text)
            text = str(Path(base).joinpath(*relative.parts))
        # Lexical normalization only: resolve() can probe every component of a
        # remote path. Cross-host Windows spellings still compare correctly.
        if PureWindowsPath(text).is_absolute():
            return ntpath.normpath(text).replace("\\", "/")
        return os.path.normpath(text).replace("\\", "/")

    @staticmethod
    def key(value):
        return value.rstrip("/").casefold() if PureWindowsPath(value).is_absolute() else value.rstrip("/")

    def inside(self, path, root):
        a, b = self.key(path), self.key(root)
        return a == b or a.startswith(b + "/")

    def unknown(self, reason, *, code="unresolved_reference", **extra):
        self.result["unknown"].append(dict(kind=self.source["kind"], path=self.source["path"],
            source_sha256=self.source.get("sha256"), reason=reason, code=code, **extra))

    def ref(self, relation, detail, **extra):
        self.result["refs"].append(dict(kind=self.source["kind"], path=self.source["path"],
            source_sha256=self.source.get("sha256"), relation=relation, detail=detail, **extra))

    def path_ref(self, value, field, relation, *, base=None, sha256=None):
        path = self.path(value, base=base)
        if self.inside(path, self.root):
            self.ref(relation, "Recorded path lies in the selected model root; payload was not opened.",
                     target_path=path, field=field, match="path", recorded_sha256=sha256,
                     self_owned=self.inside(self.source["path"], self.root))
        return path

    def id_ref(self, value, field, relation):
        if self.model_id and value == self.model_id:
            self.ref(relation, "Recorded model ID matches; ID alone does not establish an exact version.",
                     model_id=value, field=field, match="model_id")
            self.unknown("A matching model ID has no exact root/version binding in this field.", field=field)

    def descriptor(self, value, field, relation, *, base=None):
        value = _object(value, field)
        self.path_ref(value["path"], field + ".path", relation, base=base, sha256=value.get("sha256"))

    def ancestry(self, value, prefix=""):
        # Same declared lineage fields read by tools.phase_transfer_provenance.
        for name in ("parent", "parent_model", "calibration_parent", "inherited_from", "model_root",
                     "inherited_training_request", "reference_calibration_evidence"):
            if value.get(name):
                self.path_ref(value[name], prefix + name, "model ancestry", base=str(Path(self.source["path"]).parent))
        for name in ("inherited_optical_model_id", "parent_model_id"):
            self.id_ref(value.get(name), prefix + name, "model ancestry")
        for name in ("settings", "evaluation_parent"):
            child = value.get(name)
            if child is not None:
                child = _object(child, name)
                for key in ("parent", "parent_model", "calibration_parent", "inherited_from", "model_root"):
                    if child.get(key):
                        self.path_ref(child[key], prefix + name + "." + key, "model ancestry",
                                      base=str(Path(self.source["path"]).parent))
        for path in _object(value.get("protected_sources", {}), "protected_sources"):
            self.tick()
            self.path_ref(path, prefix + "protected_sources", "protected source evidence",
                          base=str(Path(self.source["path"]).parent))

    def training(self, value, _record):
        self.require_format(value, "darsia-fluidflower-phase-training-request-v1")
        self.ancestry(value)
        continuation = value.get("continuation")
        if continuation is not None:
            continuation = _object(continuation, "continuation")
            if continuation.get("contract") != "darsia-phase-continuation-v1":
                raise ValueError("Unsupported continuation contract")
            self.id_ref(continuation.get("parent_model_id"), "continuation.parent_model_id", "continued from model")
            self.descriptor(continuation["parent_request"], "continuation.parent_request", "continued from model")
            for head, descriptor in _object(continuation.get("parent_heads"), "parent_heads").items():
                self.tick()
                self.descriptor(descriptor, "continuation.parent_heads." + head, "saved parent head")
        provenance = _object(value.get("provenance", {}), "provenance")
        if provenance.get("parent_request"):
            self.descriptor(provenance["parent_request"], "provenance.parent_request", "training provenance")
        for index, bundle in enumerate(_list(value.get("bundles"), "bundles")):
            self.tick()
            bundle = _object(bundle, "training bundle")
            for role in ("decisions", "combined_manifest"):
                item = _object(bundle[role], role)
                # The original writer may filter/freeze the training view;
                # its snapshot SHA does not describe the retained source bytes.
                for key, digest_key in (("snapshot_path", "sha256"),
                                        ("source_path", "source_sha256"),
                                        ("original_snapshot_path", "original_snapshot_sha256"),
                                        ("path", "sha256")):
                    if item.get(key):
                        self.path_ref(item[key], f"bundles[{index}].{role}.{key}", "retained training evidence",
                                      sha256=item.get(digest_key))

    @staticmethod
    def require_format(value, expected):
        if value.get("format") != expected:
            raise ValueError("Unsupported metadata format; expected " + expected)

    def library(self, value, record):
        self.require_format(value, "phase-model-library-reference-v1")
        identity = _object(value.get("source_identity"), "source_identity")
        self.id_ref(identity.get("model_id"), "source_identity.model_id", "library version reference")
        if (self.identity and identity.get("model_id") == self.model_id
                and identity.get("version_sha256") == self.identity.get("version_sha256")):
            self.ref("library version reference", "Pointer records the selected cached model version digest.",
                     match="recorded_version", field="source_identity", version_sha256=identity["version_sha256"])
        owner = self.path(record.get("series_root") or self.series_root)
        targets = [value["target"], *_list(value.get("previous_targets", []), "previous_targets")]
        for index, target in enumerate(targets):
            self.tick()
            target = _object(target, "target")
            policy = target.get("relative_to")
            if policy not in {"series_root", "absolute"}:
                raise ValueError("Unsupported library target path policy")
            if policy == "series_root":
                _relative(target["path"])
            self.path_ref(target["path"], "target" if index == 0 else f"previous_targets[{index - 1}]",
                          "library target" if index == 0 else "historical library target",
                          base=owner if policy == "series_root" else None)

    def review(self, value, _record):
        self.require_format(value, "fluidflower-local-review-session-v1")
        self.id_ref(value.get("target_model_id"), "target_model_id", "retained review evidence")
        # Discard/inclusion status never removes the historical dependency.
        for index, lock in enumerate(_list(value.get("training_locks", []), "training_locks")):
            self.tick()
            self.id_ref(_object(lock, "training lock").get("model_id"),
                        f"training_locks[{index}].model_id", "review training lock")

    def review_binding(self, value, _record):
        self.require_format(value, "native-phase-review-source-binding-v1")
        self.path_ref(value["model_root"], "model_root", "retained review source binding")
        self.id_ref(value.get("target_model_id"), "target_model_id", "retained review source binding")

    def series(self, value, record):
        method = _object(value.get("method", {}), "method")
        selected = _object(method.get("phase_model", {}), "method.phase_model").get("model_version")
        if selected is None or selected == "":
            return
        if not isinstance(selected, str):
            raise ValueError("Saved phase-model selection must be a string")
        self.id_ref(selected, "method.phase_model.model_version", "saved series selection")
        # The original selection allows ID, folder and library pointer. A bare
        # folder match is positive but ambiguous, never an exact version claim.
        if selected == PurePosixPath(self.root).name and selected != self.model_id:
            self.ref("saved series selection", "Saved folder name matches; exact version is unknown.",
                     field="method.phase_model.model_version", match="folder_name")
            self.unknown("Saved folder-only selection lacks an exact version binding.")
        if ("/" in selected or "\\" in selected or Path(selected).is_absolute()
                or Path(selected).suffix == ".json"):
            path = self.path_ref(selected, "method.phase_model.model_version", "saved series selection",
                                 base=self.path(record.get("series_root") or self.series_root))
            if not self.inside(path, self.root):
                self.unknown("Saved path/pointer target was not followed; supply its library reference explicitly.",
                             target_path=path)

    def source_bindings(self, value, field):
        # Variant receipts retain a producer-defined binding map. Only explicit
        # path descriptors are interpreted; unknown structure stays unknown.
        if not value:
            return
        if not isinstance(value, (dict, list)):
            raise ValueError("Source bindings must be a map or list")
        pending = [(value, field, 0)]
        while pending:
            item, name, depth = pending.pop()
            self.tick()
            if depth > 24:
                raise ValueError("Source bindings nesting limit exceeded")
            if isinstance(item, dict):
                if isinstance(item.get("path"), str):
                    self.descriptor(item, name, "retained result source")
                for key, child in item.items():
                    if isinstance(child, (dict, list)):
                        pending.append((child, name + "." + key, depth + 1))
            elif isinstance(item, list):
                pending.extend((child, f"{name}[{index}]", depth + 1) for index, child in enumerate(item))
        self.unknown("Producer-specific source bindings were inspected for explicit paths only; completeness is unknown.",
                     field=field)

    def mass_result(self, value, _record):
        if value.get("complete") is not True or not _list(value.get("frames"), "frames"):
            raise ValueError("Complete original Studio mass result required")
        selected = _object(value.get("selected_inputs"), "selected_inputs")
        _object(selected.get("candidate"), "candidate")
        for key in ("baseline", "native_manifest"):
            if selected.get(key):
                self.path_ref(selected[key], "selected_inputs." + key, "retained result input",
                              sha256=selected.get(key + "_sha256"))
        for index, frame in enumerate(_list(selected.get("frames", []), "selected_inputs.frames")):
            self.tick()
            frame = _object(frame, "selected input frame")
            for key in ("photo", "phase"):
                if frame.get(key):
                    self.path_ref(frame[key], f"selected_inputs.frames[{index}].{key}", "retained result input",
                                  sha256=frame.get(key + "_sha256"))
        if value.get("result_source_manifest"):
            self.path_ref(value["result_source_manifest"], "result_source_manifest", "retained result source manifest",
                          base=str(Path(self.source["path"]).parent))
            self.unknown("Result source manifest was not followed; supply it explicitly as result_source_manifest.")
        self.source_bindings(value.get("source_bindings"), "source_bindings")
        self.unknown("Candidate/optical digests are not the library version digest; retained inference provenance may be external.",
                     code="result_model_binding_incomplete")

    def mass_receipt(self, value, _record):
        format_ = value.get("format")
        if format_ not in {"studio-mass-delivery-v1", "studio-mass-variant-delivery-v1"}:
            raise ValueError("Unsupported Studio mass receipt format")
        self.path_ref(value["result"], "result", "retained mass result", sha256=value.get("result_sha256"))
        if format_ == "studio-mass-delivery-v1":
            for path, binding in _object(value.get("bindings"), "bindings").items():
                self.tick()
                self.path_ref(path, "bindings", "published result input", sha256=_object(binding, "binding").get("sha256"))
            for path in _object(value.get("physical_directories", {}), "physical_directories"):
                self.tick()
                self.path_ref(path, "physical_directories", "published physical dependency")
        else:
            _object(value.get("candidate"), "candidate")
            self.source_bindings(value.get("source_bindings"), "source_bindings")
        self.unknown("Receipt payload and completion were not verified; the referenced result was not followed.",
                     code="result_not_verified")

    def result_sources(self, value, _record):
        # phase_studio_mass_cache.register_result consumes this exact map.
        if not value:
            raise ValueError("Result source manifest is empty")
        for path, digest in value.items():
            self.tick()
            if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
                raise ValueError("Result source manifest requires path-to-SHA-256 entries")
            self.path_ref(path, "source path", "retained result source", sha256=digest)

    def catalog(self, value, record):
        self.require_format(value, "darsia-role-input-catalog-v1")
        if value.get("complete") is not True:
            raise ValueError("Input catalogue is incomplete")
        # Reuse the original reference validator (pure metadata, no file reads).
        from .store_map import validate_input_reference

        roots = _object(record.get("roots", {}), "explicit catalogue roots")
        rows = _list(value.get("inputs"), "inputs")
        if not rows:
            raise ValueError("Input catalogue is empty")
        validate_input_reference(value["configuration"])
        for index, row in enumerate(rows):
            self.tick()
            reference = _object(row, "catalogue row")["reference"]
            validate_input_reference(reference)
            uses = _list(row.get("uses"), "uses")
            if not uses or any(not isinstance(use, str) or not use for use in uses):
                raise ValueError("Catalogue input needs named original consumers")
            alias = reference["root_alias"]
            if alias not in roots:
                self.unknown("Catalogue root alias was not explicitly supplied: " + alias,
                             code="missing_root_alias", field=f"inputs[{index}]")
                continue
            root = self.path(roots[alias])
            self.path_ref(reference["relative_path"], f"inputs[{index}].reference", "recorded job input: " + ", ".join(uses),
                          base=root, sha256=reference["sha256"])
        self.unknown("Catalogue records input use; job completion, retention and original admission were not reverified.",
                     code="job_retention_unknown")

    def jobs(self, value, _record):
        self.require_format(value, "darsia-native-jobs-registry-v1")
        for group, key in (("jobs", "root"), ("plans", "path")):
            for index, row in enumerate(_list(value.get(group, []), group)):
                self.tick()
                path = self.path_ref(_object(row, group)[key], f"{group}[{index}].{key}", "registered job metadata")
                self.unknown("Registered job/plan was not followed; supply exact input catalogues explicitly.",
                             target_path=path, code="registered_job_not_inspected")
        self.unknown("Local Jobs registry is not an authoritative inventory of all retained results.",
                     code="local_registry_incomplete")

    def read(self, record):
        record = _object(record, "Scope record")
        kind = record.get("kind")
        self.source = dict(kind=str(kind or "unknown"), path=str(record.get("path") or ""), sha256=None)
        if kind not in SUPPORTED_SCOPE_KINDS:
            raise ValueError("Unsupported scope kind: " + str(kind))
        path = self.path(record.get("path"))
        self.source["path"] = path
        coverage = self.result["coverage"]
        remaining = self.limits["max_total_bytes"] - coverage["bytes_read"]
        if remaining <= 0:
            raise ValueError("Total metadata byte limit reached")
        cap = min(self.limits["max_file_bytes"], remaining)
        self.check()
        initial = Path(path).stat()
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > cap:
            raise ValueError("Metadata is not a regular file within the byte limit")
        with Path(path).open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
                raise ValueError("Metadata is not a regular file within the byte limit")
            raw = stream.read(cap + 1)
            after = os.fstat(stream.fileno())
        coverage["read_files"] += 1
        coverage["bytes_read"] += len(raw)
        if len(raw) > cap:
            raise ValueError("Metadata grew beyond the byte limit")
        current = Path(path).stat()
        stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if stamp(initial) != stamp(before) or stamp(before) != stamp(after) or stamp(after) != stamp(current):
            raise ValueError("Metadata changed while being read")
        self.source["sha256"] = hashlib.sha256(raw).hexdigest()
        if record.get("sha256") is not None and record["sha256"] != self.source["sha256"]:
            raise ValueError("Metadata differs from the explicitly supplied SHA-256")
        self.check()
        value = (_object(tomllib.loads(raw.decode("utf-8-sig")), "Series metadata")
                 if kind == "series_manifest" else _json(raw))
        handlers = {"training_request": self.training, "model_ancestry": lambda v, r: self.ancestry(v),
            "library_reference": self.library, "review_session": self.review,
            "review_source_binding": self.review_binding, "series_manifest": self.series,
            "studio_mass_result": self.mass_result, "studio_mass_receipt": self.mass_receipt,
            "result_source_manifest": self.result_sources,
            "job_input_catalog": self.catalog, "job_registry": self.jobs}
        handlers[kind](value, record)
        self.result["sources"].append(dict(self.source, status="read"))


def inspect_model_dependencies(entry, *, scope_records, series_root, protected_roots=(), cancelled=None,
                               max_files=128, max_file_bytes=2 * 1024 * 1024,
                               max_total_bytes=16 * 1024 * 1024):
    """Inspect explicit ``{kind, path, sha256?, series_root?, roots?}`` records.

    ``entry`` is the existing cached model-history entry, including its root and
    optional library_source_identity. ``roots`` maps original InputCatalog root
    aliases to explicitly supplied absolute paths. No referenced file is opened
    unless it is also a scope record. Cancellation returns partial evidence and
    unknown coverage. This function never authorizes deletion, even with no refs.
    """
    limits = dict(max_files=max_files, max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes)
    for name, limit in limits.items():
        if type(limit) is not int or not 0 < limit <= _LIMITS[name]:
            raise ValueError(name + " must be a positive bounded integer")
    if not isinstance(entry, Mapping) or not isinstance(scope_records, Sequence) or isinstance(scope_records, (str, bytes)):
        raise ValueError("Cached model entry and a finite explicit scope list are required")
    if not isinstance(protected_roots, (list, tuple)) or len(protected_roots) > max_files:
        raise ValueError("Protected roots must be a bounded explicit list")
    report = _Report(entry, series_root, protected_roots, cancelled, limits)
    report.result["coverage"]["requested_files"] = len(scope_records)
    for index, record in enumerate(scope_records):
        if index >= max_files:
            report.source = dict(kind="scope", path="", sha256=None)
            report.unknown("File limit reached; remaining explicit records were not read.", code="file_limit")
            break
        report.source = dict(kind="scope", path="", sha256=None)
        try:
            report.check()
            report.read(record)
        except _Cancelled:
            report.result["coverage"]["cancelled"] = True
            report.unknown("Dependency inspection cancelled; coverage remains unknown.", code="cancelled")
            break
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            report.unknown(str(error), code="unreadable_or_unsupported_metadata")
            report.result["sources"].append(dict(report.source, status="unknown"))
    report.result["dependency_status"] = "referenced" if report.result["refs"] else "unknown"
    return report.result
