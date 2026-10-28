"""Explicit per-run host I/O, preserving canonical input bytes and identities.

Only a verified role catalogue and a positive owned local output binding can
activate this context. Unlisted input paths never fall through to historical
locators. No image equation, dtype, model, threshold or frame order lives here.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import ntpath
from pathlib import Path
import re
import subprocess

from .job_inputs import InputCatalog, _json_bytes, validate_catalog, MAX_METADATA_BYTES
from .store_runtime import InputHashMismatch, RunUnavailable
from .io_ops import retry_io

_active = ContextVar("admitted_run_io", default=None)
TASK_FORMAT = "darsia-alias-task-v1"
# Exact source consumers in native.source.SourceProcessor.identity(). These are
# repository identities, never arbitrary data-root mappings.
_CODE_ROLES = {
    "processor_module_sha256": "src/darsia/fluidflower/native/source.py",
    "renderer_module_sha256": "src/darsia/fluidflower/native/registration.py",
    "photometry_module_sha256": "src/darsia/fluidflower/native/photometry.py",
    "pilot_math_module_sha256": "src/darsia/fluidflower/native/photometry.py",
    "photo_sand_mask_module_sha256": "src/darsia/fluidflower/native/support.py",
    "isolated_invalid_module_sha256": "src/darsia/fluidflower/native/support.py",
}

# The RAW-only producer consumes these named current code modules. Data files
# retain their exact byte SHA; source code uses the existing admitted Git-blob
# identity, with only its Windows EOL representation allowed to differ.
RAW_CODE_ROLES = {
    "raw_source": "src/darsia/fluidflower/native/raw_rig_source.py",
    "raw_positioning": "src/darsia/fluidflower/native/raw_positioning.py",
    "raw_port": "src/darsia/fluidflower/native/raw_positioning_port.py",
    "raw_chart": "src/darsia/fluidflower/native/raw_chart.py",
    "full_photometry": "src/darsia/fluidflower/native/full_photometry.py",
    "registration": "src/darsia/fluidflower/native/registration.py",
    "photometry": "src/darsia/fluidflower/native/photometry.py",
    "source_helpers": "src/darsia/fluidflower/native/source.py",
    "extension": "src/darsia/fluidflower/native/extension.py",
    "native_io": "src/darsia/fluidflower/native/io.py",
    "io_ops": "src/darsia/fluidflower/pipeline/io_ops.py",
    "raw_camera": "src/darsia/fluidflower/native/raw_camera.py",
}


# Provenance self-hash in native.display_modes.display_contract(). This reader
# has no field in source.execution_identity: bind this one named repository
# module to the actual job commit, never admit arbitrary tracked source files.
_PROVENANCE_CODE_ROLES = {
    "display_contract_renderer_code_sha256": "src/darsia/fluidflower/native/display_modes.py",
}


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _version(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def current_binding():
    return _active.get()


def for_task(task):
    if task.get("format") != TASK_FORMAT:
        return None
    binding = current_binding()
    if binding is None or binding.task != task_without_resume(task):
        raise InputHashMismatch("Alias execution requires its exact admitted per-run I/O binding")
    return binding


def task_without_resume(task):
    return {key: value for key, value in task.items()
            if key not in {"resume_from", "environment_lock"}}


class _Replay(InputCatalog):
    """Replay the real named consumers against already verified shared refs."""

    def __init__(self, binding, aliases):
        super().__init__({}, aliases, metadata_bytes=binding._metadata_bytes)
        self.binding = binding
        self.used = {}

    def add(self, descriptor, role, use):
        original = self.physical(descriptor)
        row = self.binding.verified.for_use(use)
        ref = row.reference
        if (ref["role"] != role or ref["sha256"] != descriptor["sha256"]
                or ("size_bytes" in descriptor and descriptor["size_bytes"] != ref["bytes"])):
            raise InputHashMismatch("Canonical input descriptor differs for consumer " + use)
        if use in self.used and self.used[use] != ref:
            raise InputHashMismatch("Conflicting replay consumer " + use)
        self.used[use] = deepcopy(ref)
        self.binding._bind_data_path(descriptor["path"], row)
        self.binding._bind_data_path(original["path"], row)
        key = (ref["root_alias"], ref["relative_path"], ref["sha256"], role)
        entry = self.entries.setdefault(key, {"reference":deepcopy(ref),"uses":[]})
        if use not in entry["uses"]:
            entry["uses"].append(use)
        return deepcopy(ref)


    def bind_declared_aliases(self):
        """Admit exact authored spellings only for already consumed inputs.

        The original DATA may name one input both absolutely and relatively.
        These are explicit path/SHA/size aliases, never cwd or suffix searches.
        An alias cannot add a resource to the verified named-consumer closure.
        """
        for logical, physical in self.aliases.items():
            row = self.binding.locations.get(_key(physical["path"]))
            if row is None:
                continue  # An unconsumed alias is not execution admission.
            if (physical["sha256"] != row.reference["sha256"]
                    or ("size_bytes" in physical
                        and physical["size_bytes"] != row.reference["bytes"])):
                raise InputHashMismatch("Declared alias differs from its verified consumed input")
            if (_key(logical) in self.binding.code_locations
                    or self.binding._own_output(logical)):
                raise InputHashMismatch("Declared input alias overlaps admitted code or owned output")
            self.binding._input_digest(row)
            self.binding._bind_data_path(logical, row)

    def doc(self, descriptor, role, use):
        """Decode the same strict JSON after exact once-per-run admission."""
        reference = self.add(descriptor, role, use)
        actual = self.physical(descriptor)
        key = (_key(actual["path"]), descriptor["sha256"])
        expected = self.binding.native_input_digest(actual["path"])
        if expected != descriptor["sha256"]:
            raise ValueError("Consumed metadata SHA/size changed: " + str(actual["path"]))
        if key not in self.documents:
            raw = self.metadata_bytes(actual["path"])
            if (not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES
                    or len(raw) != reference["bytes"]
                    or expected != descriptor["sha256"]):
                raise ValueError("Consumed metadata SHA/size changed: " + str(actual["path"]))
            self.documents[key] = _json_bytes(raw)
        return deepcopy(self.documents[key])


class RunIOBinding:
    def __init__(self, task, verified, local_output, *, repository, admitted_commit):
        from .job_queue import LocalOutputBinding
        from ..job_filesystem import valid_results_admission
        self.task = deepcopy(task_without_resume(task))
        if (task.get("format") != TASK_FORMAT or verified.run != task["run"]
                or admitted_commit != task["commit"]):
            raise InputHashMismatch("Run I/O needs the exact admitted task and checkout commit")
        validate_catalog(task["input_catalog"])
        if (not isinstance(local_output, LocalOutputBinding)
                or not valid_results_admission(local_output.admission, local_output.results_root)):
            raise RunUnavailable("A positive explicit results filesystem binding is required")
        self.verified = verified
        self.local_output = local_output
        from .job_queue import name
        if re.fullmatch(r"[0-9a-f]{32}",task.get("attempt_id","")) is None:
            raise RunUnavailable("A real owned queue attempt is required")
        self.output = Path(local_output.path)
        expected = Path(local_output.results_root) / name(task["job_id"]) / name(task["run"]) / ".attempts" / task["attempt_id"]
        self.results_identity = Path(local_output.results_root).resolve()
        self.output_identity = self.output.resolve()
        if (self.output_identity != expected.resolve()
                or not self.output_identity.is_relative_to(self.results_identity)
                or not self.output_identity.is_relative_to(Path(local_output.admission.checked_container).resolve())):
            raise RunUnavailable("Output does not match its positive owned attempt proof")
        self.machine_id = name(local_output.machine_id)
        self.repository, self.commit = Path(repository).resolve(), admitted_commit
        self.locations, self.code_locations, self.code_audit = {}, {}, []
        self.resume_roots, self.producer_origins = set(), {}
        self.common_publications = {}
        self._prefix_validation_root = None
        self.configuration = None
        self.configuration_descriptor = None
        self._replayed = False

    @classmethod
    def replay(cls, task, verified, local_output, *, repository, admitted_commit):
        self = cls(task, verified, local_output, repository=repository,
                   admitted_commit=admitted_commit)
        row = verified.for_use("run.configuration")
        if row.reference != task["configuration"]:
            raise InputHashMismatch("Canonical configuration reference differs")
        raw = self._verified_bytes(row)
        config = _json_bytes(raw)
        self.configuration = deepcopy(config)
        self.configuration_descriptor = dict(path=str(row.path),
            sha256=row.reference["sha256"], size_bytes=row.reference["bytes"])
        if config.get("format") == "darsia-series-setup-operation-v1":
            from .series_setup_run_io import replay_setup_controls
            return replay_setup_controls(self)
        # Read the original TOML only as raw admitted input. Its historical
        # absolute descriptors are never rewritten or copied into job metadata.
        original_series = verified.for_use("run.execution_series")
        manifest = verified.for_use("run.series_manifest")
        import tomllib
        series = tomllib.loads(self._verified_bytes(original_series).decode("utf-8-sig"))
        aliases = [*series.get("data_aliases", ()), *config.get("data_aliases", ())]
        replay = _Replay(self, aliases)
        replay.add(dict(path=str(original_series.path),sha256=original_series.reference["sha256"],
                        size_bytes=original_series.reference["bytes"]),
                   "configuration","run.execution_series")
        replay.add(dict(path=str(manifest.path),sha256=manifest.reference["sha256"],
                        size_bytes=manifest.reference["bytes"]),
                   "configuration","run.series_manifest")
        # The configuration descriptor's canonical path remains in the original
        # execution series. It must be the exact run declaration, not a guessed
        # or recursively discovered path.
        declarations = [item for item in series["runs"] if item["run"] == task["run"]]
        if len(declarations) != 1:
            raise InputHashMismatch("Exactly one original execution run declaration is required")
        original_bound = declarations[0]["configuration"]
        if original_bound["sha256"] != task["configuration"]["sha256"]:
            raise InputHashMismatch("Execution series/run configuration SHA differs")
        actual = replay.collect(original_bound, config)
        if any(actual[field] != task["input_catalog"][field] for field in
               ("format","complete","run","configuration","inputs",
                "payload_hashes_verified","numerical_calls_started")):
            raise InputHashMismatch("Actual named input consumers differ from the prepared catalogue")
        expected = {use for item in task["input_catalog"]["inputs"] for use in item["uses"]}
        if set(replay.used) != expected:
            raise InputHashMismatch("Prepared input catalogue has unconsumed or missing roles")
        if config.get("format") in {"darsia-native-phase-n1-run-v1", "darsia-native-phase-statistics-run-v1"}:
            native = self._metadata_bytes(self._descriptor_location(config["native"]["configuration"]))
            self._admit_native_code(_json_bytes(native))
        replay.bind_declared_aliases()
        self._replayed = True
        return self

    @retry_io
    def _verified_bytes(self, row):
        self._input_digest(row)
        before = row.path.stat()
        if _version(before) != self.verified.versions[str(row.path)]:
            raise InputHashMismatch("Admitted input changed before read: " + str(row.path))
        with row.path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if (len(raw) > MAX_METADATA_BYTES or len(raw) != row.reference["bytes"]
                or self._input_digest(row) != row.reference["sha256"]
                or _version(row.path.stat()) != _version(before)):
            raise InputHashMismatch("Admitted input metadata SHA/bytes changed: " + str(row.path))
        return raw

    def _bind_data_path(self, source, row):
        for path in (source, row.path):
            key = _key(path)
            old = self.locations.get(key)
            if old is not None and (old.path != row.path or old.reference["sha256"] != row.reference["sha256"]):
                raise InputHashMismatch("Original locator has conflicting verified input identities")
            self.locations[key] = row

    def _descriptor_location(self, descriptor):
        row = self.locations.get(_key(descriptor["path"]))
        if row is None or row.reference["sha256"] != descriptor["sha256"]:
            raise InputHashMismatch("Input descriptor is outside the admitted named role closure")
        return row.path

    def _metadata_bytes(self, path):
        row = self.locations.get(_key(path))
        if row is None:
            raise InputHashMismatch("Metadata is outside the admitted role closure: " + str(path))
        return self._verified_bytes(row)

    def _admit_native_code(self, native, *, setup_raw=False):
        source = native["source"]
        if setup_raw and self.task.get("pipeline") != "series-setup":
            raise InputHashMismatch("Setup RAW code admission requires its original Setup task")
        if setup_raw or "raw_execution_identity" in source:
            expected = source.get("raw_execution_identity")
            if expected is None and setup_raw:
                # Setup consumes the calibrated camera parameters through the
                # RAW route. Historical photograph-processor code identities
                # in that same immutable parameter file belong to another route.
                # Bind every named RAW consumer to this task's exact Git commit.
                expected = {role: hashlib.sha256(subprocess.check_output(
                    ["git", "-C", str(self.repository), "show", self.commit + ":" + relative])).hexdigest()
                    for role, relative in RAW_CODE_ROLES.items()}
            if not isinstance(expected, dict) or set(expected) != set(RAW_CODE_ROLES):
                raise InputHashMismatch("RAW-only execution identity is incomplete")
            for role, relative in RAW_CODE_ROLES.items():
                self._admit_code(relative, expected[role], "raw_only." + role)
        else:
            expected = source["execution_identity"]
            required = source["required_execution_identity_keys"]
            if set(required) != set(expected) or len(required) != len(set(required)):
                raise InputHashMismatch("Original native execution identity is incomplete")
            for role in required:
                relative = _CODE_ROLES.get(role)
                if relative is None:
                    raise InputHashMismatch("Unsupported explicit native code consumer: " + role)
                self._admit_code(relative, expected[role], role)
        for role, relative in _PROVENANCE_CODE_ROLES.items():
            original = subprocess.check_output(
                ["git", "-C", str(self.repository), "show", self.commit + ":" + relative])
            self._admit_code(relative, hashlib.sha256(original).hexdigest(), role)

    @retry_io
    def _admit_code(self, relative, digest, role):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise InputHashMismatch("Explicit original source SHA is required")
        tracked = subprocess.check_output(
            ["git","-C",str(self.repository),"ls-tree","--name-only",
             self.commit,"--",relative],text=True).splitlines()
        if tracked != [relative]:
            raise InputHashMismatch("Current source is not tracked at the admitted Git commit")
        path = (self.repository / relative).resolve(strict=True)
        if not path.is_relative_to(self.repository):
            raise InputHashMismatch("Current source escapes its admitted checkout")
        original = subprocess.check_output(
            ["git", "-C", str(self.repository), "show", self.commit + ":" + relative])
        if hashlib.sha256(original).hexdigest() != digest:
            raise InputHashMismatch("Current source differs from its exact execution binding: " + role)
        before = path.stat()
        raw = path.read_bytes()
        if (_version(before) != _version(path.stat())
                or raw.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n")):
            raise InputHashMismatch("Current source differs from its exact execution binding: " + role)
        raw_digest = hashlib.sha256(raw).hexdigest()
        old = self.code_locations.get(_key(path))
        if old is not None and (old["sha256"] != digest
                or old["working_tree_sha256"] != raw_digest
                or old["working_tree_size_bytes"] != len(raw)):
            raise InputHashMismatch("Conflicting or changed explicit source roles")
        record = dict(path=str(path), sha256=digest, size_bytes=len(original),
            identity_algorithm="sha256-git-blob-content", commit=self.commit,
            repo_relative_path=relative, working_tree_sha256=raw_digest,
            working_tree_size_bytes=len(raw), EOL_only_representation=(raw != original))
        self.code_locations[_key(path)] = record
        self.code_audit.append(dict(role=role, actual_path=str(path),
            **{key: value for key, value in record.items() if key != "path"}))

    @retry_io
    def _checked_code_digest(self, code):
        """Recheck this checkout's raw snapshot; return its canonical Git identity."""
        path = Path(code["path"])
        before = path.stat()
        raw = path.read_bytes()
        if (_version(before) != _version(path.stat())
                or len(raw) != code["working_tree_size_bytes"]
                or hashlib.sha256(raw).hexdigest() != code["working_tree_sha256"]):
            raise InputHashMismatch("Admitted current code changed")
        return code["sha256"]

    @retry_io
    def _input_digest(self, row):
        """Use only actual run-admission evidence, with a fresh file version."""
        key = str(row.path)
        expected_version = self.verified.versions.get(key)
        digest = self.verified.digests.get(key)
        if (expected_version is None or _version(row.path.stat()) != expected_version
                or digest != row.reference["sha256"]
                or expected_version[2] != row.reference["bytes"]):
            raise InputHashMismatch("Verified input changed before native consumption: " + key)
        return digest

    def native_input_digest(self, path):
        """Data stay byte-bound; named code uses Git identity after a fresh raw check."""
        # Code identities and raw input/output hashes are separate contracts.
        code = self.code_locations.get(_key(path))
        if code is not None:
            return self._checked_code_digest(code)
        if self._own_output(path):
            return None
        row = self.locations.get(_key(path))
        if row is not None:
            if _key(row.path) in self.code_locations or self._own_output(row.path):
                raise InputHashMismatch("Verified input overlaps admitted code or owned output: " + str(path))
            return self._input_digest(row)
        raise InputHashMismatch("Native input is outside its admitted role/code closure: " + str(path))

    def verified_source_descriptor(self, path):
        """Reuse a claimed input only; newly produced controls need fresh reads."""
        row = self.locations.get(_key(path))
        if row is None:
            return None
        digest = self._input_digest(row)
        return dict(path=str(row.path), sha256=digest, size_bytes=row.reference["bytes"])

    def native_metadata_bytes(self, path, expected_sha256):
        digest = self.native_input_digest(path)
        if digest is None:
            return None
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or digest != expected_sha256):
            raise ValueError("Source artifact changed: " + str(path))
        return self._verified_bytes(self.locations[_key(path)])

    def reader_verifications(self):
        """Reuse the exact once-per-run SHA admission with original stamp order."""
        return {(_key(row.path),row.reference["sha256"]):
                (self.verified.versions[str(row.path)][2],
                 self.verified.versions[str(row.path)][3],
                 self.verified.versions[str(row.path)][0],
                 self.verified.versions[str(row.path)][1])
                for row in self.verified.rows}


    def asset_aliases(self):
        return {key:str(row.path) for key,row in self.locations.items()}

    def _own_output(self, path):
        actual = Path(path).resolve()
        if self.output.resolve() != self.output_identity:
            raise RunUnavailable("Owned local output ancestor changed")
        temporary = self._prefix_validation_root
        if temporary is not None and temporary.resolve() != temporary:
            raise RunUnavailable("Prefix validation root changed during original artifact validation")
        return (actual.is_relative_to(self.output_identity)
                or any(actual.is_relative_to(root) for root in self.resume_roots)
                or temporary is not None and actual.is_relative_to(temporary))

    def physical_descriptor(self, descriptor):
        if not isinstance(descriptor,dict) or not {"path","sha256"}.issubset(descriptor):
            raise InputHashMismatch("A source-bound I/O descriptor is required")
        row = self.locations.get(_key(descriptor["path"]))
        if row is not None:
            if (descriptor["sha256"] != row.reference["sha256"]
                    or descriptor.get("size_bytes",row.reference["bytes"]) != row.reference["bytes"]):
                raise InputHashMismatch("Consumed input identity differs from its catalogue")
            if _version(row.path.stat()) != self.verified.versions[str(row.path)]:
                raise InputHashMismatch("Verified input changed before scientific consumption")
            return dict(descriptor,path=str(row.path),size_bytes=row.reference["bytes"])
        publication = self.common_publications.get(_key(descriptor["path"]))
        if publication is not None:
            expected, owned, public_version, owned_version = publication
            if (descriptor.get("sha256") != expected["sha256"]
                    or descriptor.get("size_bytes") != expected["size_bytes"]
                    or _version(Path(expected["path"]).stat()) != public_version
                    or _version(Path(owned["path"]).stat()) != owned_version):
                raise InputHashMismatch("The exact own common publication changed after admission")
            return dict(descriptor, path=owned["path"], size_bytes=owned["size_bytes"])
        if self._own_output(descriptor["path"]):
            return dict(descriptor)
        raise InputHashMismatch("Unlisted input cannot fall through to a historical locator: " + descriptor["path"])

    def admit_common_completion(self, published, *, job, index):
        """Bind only an exact own published completion to its indexed owned bytes."""
        from .products_publication import _read_index
        if getattr(self, "common_closure_released", False) is not True:
            raise InputHashMismatch("Own completion admission requires its released common closure")
        if getattr(self, "common_job_binding", None) != job:
            raise InputHashMismatch("Own completion differs from the independently released job binding")
        job_root = Path(job["path"]).parent
        expected_public = job_root / "results" / self.task["run"] / "completed.json"
        if (Path(job["path"]).name != "job.json" or job_root.name != self.task["job_id"]
                or _key(published["path"]) != _key(expected_public)
                or not Path(published["path"]).resolve().is_relative_to(job_root.resolve())):
            raise InputHashMismatch("Own completion belongs to another immutable job")
        _index_root, document, products = _read_index(index)
        original = products.get("completed.json", {})
        same = lambda left, right: (left.get("sha256"), left.get("bytes", left.get("size_bytes"))) == (
            right.get("sha256"), right.get("bytes", right.get("size_bytes")))
        if (document.get("run") != self.task["run"] or document.get("attempt_id") != self.task["attempt_id"]
                or not same(document.get("job", {}), job)
                or _key(document.get("job", {}).get("path", "")) != _key(job["path"])
                or _key(original.get("local_path", "")) != _key(self.output / "candidate_completed.json")
                or not same(original, published)
                or _key(published["path"]) in self.locations or _key(published["path"]) in self.code_locations):
            raise InputHashMismatch("Own completion/index does not match its exact original attempt")
        owned = dict(path=original["local_path"], sha256=original["sha256"], size_bytes=original["bytes"])
        if not self._own_output(owned["path"]):
            raise InputHashMismatch("The original common candidate escaped its positively owned attempt")
        public_version = _version(Path(published["path"]).stat())
        owned_version = _version(Path(owned["path"]).stat())
        public = self._proof_bytes(published, expected_public)
        candidate = self._proof_bytes(owned, self.output / "candidate_completed.json")
        if (public != candidate or candidate.get("task") != self.task
                or candidate.get("attempt_id") != self.task["attempt_id"] or candidate.get("commit") != self.commit
                or _version(Path(published["path"]).stat()) != public_version
                or _version(Path(owned["path"]).stat()) != owned_version):
            raise InputHashMismatch("Published common completion changed or belongs to another actual claim")
        key = _key(published["path"])
        admitted = (deepcopy(published), owned, public_version, owned_version)
        if key in self.common_publications and self.common_publications[key] != admitted:
            raise InputHashMismatch("Conflicting own common completion admission")
        self.common_publications[key] = admitted
        return deepcopy(owned)

    def asset_path(self, path):
        row = self.locations.get(_key(path))
        if row is not None:
            if _version(row.path.stat()) != self.verified.versions[str(row.path)]:
                raise InputHashMismatch("Verified input changed before native consumption")
            return row.path
        code = self.code_locations.get(_key(path))
        if code is not None:
            self._checked_code_digest(code)
            return Path(code["path"])
        if self._own_output(path):
            return Path(path)
        raise InputHashMismatch("Native input is outside its admitted role/code closure: " + str(path))

    def require_output(self, output):
        if (self.output.resolve() != self.output_identity
                or Path(output).resolve() != self.output_identity):
            raise RunUnavailable("Executor output differs from its positively admitted owned attempt")
        if self.output.name != self.task["attempt_id"]:
            raise RunUnavailable("Output and queue attempt identity differ")

    def require_resume_location(self, attempt):
        actual = Path(attempt).resolve()
        if (self.output.resolve() != self.output_identity
                or actual.parent != self.output_identity.parent
                or actual == self.output_identity
                or not actual.is_relative_to(self.results_identity)):
            raise RunUnavailable("Resume source is not another own admitted local attempt")
        return actual

    @retry_io
    def _proof_bytes(self, bound, expected_path):
        path = Path(bound["path"])
        if path.resolve() != Path(expected_path).resolve():
            raise InputHashMismatch("Original resume proof path differs")
        with path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES+1)
        if (len(raw)>MAX_METADATA_BYTES or len(raw)!=bound["size_bytes"]
                or hashlib.sha256(raw).hexdigest()!=bound["sha256"]):
            raise InputHashMismatch("Original resume proof SHA/bytes changed")
        return _json_bytes(raw)

    @contextmanager
    def prefix_validation(self, source):
        """Read one exact original candidate only during its artifact validator.

        This temporary path view grants no executor-output admission and never
        marks a prefix valid. Original SHA/artifact guards remain responsible
        for every read; permanent prefix admission follows full verification.
        """
        if self._prefix_validation_root is not None:
            raise RunUnavailable("Another original prefix artifact validation is active")
        attempt = self.require_resume_location(source["attempt_directory"])
        proof = self._proof_bytes(source["retirement"], attempt / "retirement.json")
        from .job_queue import name
        if (proof.get("complete") is not True or proof.get("all_absent") is not True
                or proof.get("worker_os_references_closed") is not True
                or proof.get("run") != self.task["run"]
                or proof.get("attempt_id") != attempt.name
                or Path(proof["attempt_directory"]).resolve() != attempt
                or not proof.get("identities")
                or not _retired_machine_allowed(self.local_output, self.task, attempt.name, proof)):
            raise RunUnavailable("Original candidate prefix has no admitted retirement binding")
        self._prefix_validation_root = attempt
        try:
            yield
        finally:
            self._prefix_validation_root = None

    def admit_resume_root(self, source):
        """Register only sources already passed by original verify_frame_prefix."""
        if source is None:
            return
        from .job_queue import name
        seen=set()
        def add(original):
            attempt=self.require_resume_location(original["attempt_directory"])
            if attempt in seen:
                raise InputHashMismatch("Original resume proof has a cycle")
            seen.add(attempt)
            proof=self._proof_bytes(original["retirement"],attempt/"retirement.json")
            if (proof.get("complete") is not True or proof.get("all_absent") is not True
                    or proof.get("worker_os_references_closed") is not True
                    or proof.get("run")!=self.task["run"] or proof.get("attempt_id")!=attempt.name
                    or Path(proof["attempt_directory"]).resolve()!=attempt
                    or not _retired_machine_allowed(self.local_output, self.task, attempt.name, proof)):
                raise RunUnavailable("Original prefix belongs to another machine or incomplete retirement")
            self.resume_roots.add(attempt)
            self.producer_origins[attempt]=dict(machine_id=proof["machine_id"],
                proof_descriptor=deepcopy(original["retirement"]),accessible_root=str(attempt))
            if proof.get("manifest") is not None:
                manifest=self._proof_bytes(proof["manifest"],attempt/"manifest.json")
                if manifest.get("resume_checkpoint") is not None:
                    checkpoint=self._proof_bytes(manifest["resume_checkpoint"],attempt/"resume.json")
                    if checkpoint.get("source") is not None:
                        add(checkpoint["source"])
        add(source)

    def producer_origin_for_attempt(self, attempt_directory):
        attempt=Path(attempt_directory).resolve()
        if attempt==self.output_identity:
            return dict(machine_id=self.local_output.machine_id,proof_descriptor=None,
                        accessible_root=str(self.output_identity),origin="owned-current-claim")
        if attempt not in self.producer_origins:
            raise RunUnavailable("No original retired producer proof for this attempt")
        return deepcopy(self.producer_origins[attempt])

    @contextmanager
    def activate(self):
        if not self._replayed:
            raise InputHashMismatch("Incomplete role replay cannot activate local I/O")
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)


def _folder_job_output(local_output, task, token):
    """A foreign prefix is usable only in this exact shared, job-owned output."""
    from darsia_cache.folders import current_folder_context
    from .job_queue import name
    context = current_folder_context()
    if context is None or context.job_root is None:
        return False
    root = Path(context.job_root).resolve()
    expected = Path(local_output.results_root) / name(task["job_id"]) / name(task["run"]) / ".attempts" / token
    return (Path(local_output.results_root).resolve() / name(task["job_id"]) == root
            and expected.resolve().is_relative_to(root))


def alias_resume_output(owner, local_output, task, previous_token):
    """Resolve the exact original attempt, including shared folder-mode Jobs."""
    from .job_queue import LocalOutputBinding, name
    if not isinstance(local_output,LocalOutputBinding):
        raise RunUnavailable("Alias resume needs its positive local output binding")
    reference=owner.get("output")
    if (owner.get("format")!="darsia-alias-lease-owner-v1"
            or not isinstance(reference,dict)
            or set(reference)!={"machine_id","root_alias","relative_path"}
            or (name(reference["machine_id"])!=name(local_output.machine_id)
                and not _folder_job_output(local_output, task, previous_token))
            or owner.get("actual_owner",{}).get("machine")!=owner.get("machine")):
        raise RunUnavailable("Original local prefix belongs to another machine")
    expected="/".join((name(task["job_id"]),name(task["run"]),".attempts",previous_token))
    if reference["root_alias"]!="results" or reference["relative_path"]!=expected:
        raise InputHashMismatch("Original output ref differs from its reclaimed task/token")
    if (name(reference["machine_id"]) != name(local_output.machine_id)
            and owner["machine"] not in {name(reference["machine_id"]) + "-cpu", name(reference["machine_id"]) + "-cuda"}):
        raise InputHashMismatch("Shared prefix producer differs from its original machine")
    output=Path(local_output.results_root)/expected
    root=Path(local_output.results_root).resolve()
    if not output.resolve().is_relative_to(root):
        raise RunUnavailable("Original output escapes its admitted results root")
    return output


def _retired_machine_allowed(local_output, task, token, proof):
    from .job_queue import name
    machine = name(proof.get("machine_id", ""))
    return (machine == name(local_output.machine_id)
            or (_folder_job_output(local_output, task, token)
                and proof.get("supervisor", {}).get("machine") in {machine + "-cpu", machine + "-cuda"}))


def require_retired_source(task, local_output, output, raw, expected_service=None):
    """Metadata/presence admission; original prefix validator remains mandatory."""
    from .job_queue import name
    proof=_json_bytes(raw)
    if (proof.get("complete") is not True or proof.get("all_absent") is not True
            or proof.get("worker_os_references_closed") is not True
            or proof.get("run")!=task["run"] or proof.get("attempt_id")!=Path(output).name
            or Path(proof["attempt_directory"]).resolve()!=Path(output).resolve()
            or not proof.get("identities")
            or not _retired_machine_allowed(local_output, task, Path(output).name, proof)
            or (expected_service is not None and proof["supervisor"]["machine"]!=expected_service)):
        raise RunUnavailable("Original local prefix retirement metadata is unavailable")


def alias_resume_source(shared, claim):
    """Follow exact preserved input-reject hops to genuine original SCI proof."""
    from .job_queue import name, read_json, retired_lease_directory
    root=Path(shared.root)
    run=name(claim.task["run"])
    records=root/"reclaimed"/run
    matches=[]
    for path in records.glob("*/reclaimed.json"):
        value=read_json(path)
        if value.get("next_attempt")==claim.token:
            matches.append((path,value))
    if not matches:
        canonical=read_json(root/"results"/run/"task.json")
        left,right=deepcopy(canonical),deepcopy(claim.task)
        first_token=left.pop("attempt_id",None);right.pop("attempt_id",None)
        if left!=right:
            raise InputHashMismatch("Canonical alias task identity differs")
        if first_token!=claim.token:
            raise RunUnavailable("Retry is visible before its original immutable lineage record")
        return None
    if len(matches)!=1:
        raise InputHashMismatch("Resume has more than one reclaimed predecessor")
    seen=set()
    def same_task(original,expected_token):
        if original.get("attempt_id")!=expected_token:
            raise InputHashMismatch("Archived task generation differs from original lineage")
        left,right=deepcopy(original),deepcopy(claim.task)
        left.pop("attempt_id",None);right.pop("attempt_id",None)
        if left!=right:
            raise InputHashMismatch("Original alias reclaimed task identity differs")
    def read_prior(bound):
        if (not isinstance(bound,dict) or set(bound)!={"relative_path","sha256","bytes"}
                or not isinstance(bound["relative_path"],str)):
            raise InputHashMismatch("Preserved alias lineage needs a job-relative immutable reference")
        relative=bound["relative_path"]
        from pathlib import PurePosixPath
        if (PurePosixPath(relative).is_absolute() or "\\" in relative
                or any(part in {"",".",".."} for part in relative.split("/"))):
            raise InputHashMismatch("Preserved lineage escapes this job")
        path=root/relative
        if (not path.resolve().is_relative_to(records.resolve())
                or path.name!="reclaimed.json" or path.parent.parent.resolve()!=records.resolve()):
            raise InputHashMismatch("Preserved lineage is outside this run's reclaimed records")
        raw=path.read_bytes()
        if len(raw)!=bound["bytes"] or hashlib.sha256(raw).hexdigest()!=bound["sha256"]:
            raise InputHashMismatch("Original preserved lineage SHA/bytes changed")
        return path,_json_bytes(raw)
    def follow(path,record,next_token):
        previous=record.get("previous_attempt")
        if (re.fullmatch(r"[0-9a-f]{32}",previous or "") is None
                or record.get("next_attempt")!=next_token or record.get("run")!=claim.task["run"]
                or path.parent.name!=previous
                or previous in seen):
            raise InputHashMismatch("Original alias resume token chain differs or cycles")
        seen.add(previous)
        if record.get("interrupted_claim_before_science") is True:
            same_task(read_json(path.parent/"task.json"),previous)
            if record.get("preserve_prior_resume_lineage") is not True or "prior_resume_lineage" not in record:
                raise InputHashMismatch("Pre-science retry omitted its explicit original lineage")
            prior=record["prior_resume_lineage"]
            if prior is None:
                if any(read_json(item).get("next_attempt")==previous for item in records.glob("*/reclaimed.json")):
                    raise InputHashMismatch("Pre-science retry dropped a genuine predecessor")
                return None
            earlier,value=read_prior(prior)
            return follow(earlier,value,previous)
        owner=read_json(retired_lease_directory(root,run,previous)/"owner.json")
        same_task(owner["task"],previous)
        if owner.get("token")!=previous:
            raise InputHashMismatch("Original owner/token differs")
        old_output=alias_resume_output(owner,claim.local_output,claim.task,previous)
        if old_output.resolve().parent!=Path(claim.output).resolve().parent:
            raise RunUnavailable("Original prefix is outside this own local run")
        retirement=old_output/"retirement.json"
        if not retirement.is_file():
            raise RunUnavailable("Original local prefix retirement is not available")
        raw=retirement.read_bytes()
        require_retired_source(claim.task,claim.local_output,old_output,raw,owner["machine"])
        return dict(attempt_directory=str(old_output),
                    retirement=dict(path=str(retirement),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw)))
    return follow(*matches[0],claim.token)
