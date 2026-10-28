"""Exact original P1/P2 inputs and publication proofs for full RAW replay.

No metadata or scientific input is rewritten. This is package preparation and
readback only; execution still requires the unchanged original runtime guards.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tomllib

from .archives import Member, file_sha256
from .job_reader import Input
from .layout import inside, record_relative, task_path, MANIFEST, SERIES_ARCHIVE
from ..pipeline.job_inputs import InputCatalog, _key, _json_bytes, MAX_METADATA_BYTES

FORMAT = "darsia-staged-raw-reproduction-v1"
_PROOF_ROLES = ("job", "task", "done", "products_index")


@dataclass(frozen=True)
class StagedInput(Input):
    """Retain original input identity while keeping saved P2 maps out of RAW."""
    camera: bool = False

    @property
    def camera_file(self):
        return self.camera


def _bytes(path, bound, *, limit=MAX_METADATA_BYTES):
    path = Path(path)
    if type(bound.get("size_bytes")) is not int or not 0 <= bound["size_bytes"] <= limit:
        raise ValueError("Bounded original metadata size required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != bound["size_bytes"] or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError("Original staged metadata SHA/size differs")
    return raw


def _bound_reference(row):
    return dict(path=str(row.path), sha256=row.reference["sha256"], size_bytes=row.reference["bytes"])


def _camera_input(reference, uses, phase):
    """Only original image consumers belong to camera archives, never masks."""
    from ..image_formats import image_family
    if phase != "phase1" or reference["role"] != "image":
        return False
    family = image_family(reference["relative_path"])
    return family == "raw" or (family == "processed" and reference["root_alias"] != "series"
        and any(".native." in use or ".camera" in use for use in uses))


def _frame_acquisition(task, case):
    """Keep the exact original RAW consumer, including equal-byte acquisitions."""
    from ..image_formats import image_family
    use = "frame." + case + ".native.acquisition"
    matches = [entry["reference"] for entry in task["input_catalog"]["inputs"] if use in entry["uses"]]
    if (len(matches) != 1 or matches[0]["role"] != "image"
            or image_family(matches[0]["relative_path"]) != "raw"):
        raise ValueError("Full staged reproduction requires the exact original named RAW acquisition: " + case)
    return matches[0]


class _Catalogue(InputCatalog):
    """Walk unchanged named consumers against original fully verified inputs.

    This exporter has no RunIOBinding and grants no execution admission. The
    original consumer names select the recorded references; their SHA/size/role
    must match at every original InputCatalog.add call, as in runtime replay.
    """
    def __init__(self, verified, aliases):
        self.verified, self.locations, self.transport = verified, {}, {}
        super().__init__({}, aliases, metadata_bytes=self.read_bytes)

    def locate(self, descriptor, row):
        pin = dict(path=descriptor["path"], sha256=row.reference["sha256"], size_bytes=row.reference["bytes"])
        key = _key(pin["path"])
        previous = self.locations.get(key)
        if previous is not None and previous.reference != row.reference:
            if (previous.path != row.path or previous.reference["sha256"] != pin["sha256"]
                    or previous.reference["bytes"] != pin["size_bytes"]):
                raise ValueError("One original source locator maps to conflicting catalogue bytes")
        self.locations[key] = row
        self.transport[(key, pin["sha256"], pin["size_bytes"])] = (pin, row)

    def add(self, descriptor, role, use):
        original = super().physical(descriptor)
        row = self.verified.for_use(use)
        ref = row.reference
        if (ref["role"] != role or ref["sha256"] != descriptor["sha256"]
                or ("size_bytes" in descriptor and descriptor["size_bytes"] != ref["bytes"])):
            raise ValueError("Original staged consumer differs from its exact catalogue: " + use)
        if use in self.consumers and self.consumers[use] != ref:
            raise ValueError("A staged consumer has contradictory input references")
        self.consumers[use] = deepcopy(ref)
        self.locate(descriptor, row)
        self.locate(original, row)
        self.locate(_bound_reference(row), row)
        key = (ref["root_alias"], ref["relative_path"], ref["sha256"], role)
        entry = self.entries.setdefault(key, dict(reference=deepcopy(ref), uses=[]))
        if use not in entry["uses"]:
            entry["uses"].append(use)
        return deepcopy(ref)

    def read_bytes(self, path):
        row = self.locations.get(_key(path))
        if row is None:
            raise ValueError("Unlisted staged metadata locator")
        return _bytes(row.path, _bound_reference(row))

    def bind_aliases(self):
        for logical, physical in self.aliases.items():
            row = self.locations.get(_key(physical["path"]))
            if row is None:
                continue
            if (physical["sha256"] != row.reference["sha256"]
                    or ("size_bytes" in physical and physical["size_bytes"] != row.reference["bytes"])):
                raise ValueError("Authored staged alias differs from consumed original bytes")
            self.locate(dict(physical, path=logical), row)


def _verified_catalogue(task, roots, series_id):
    """Original root resolution/full hashing plus the unchanged consumer walk."""
    from ..pipeline.store_runtime import resolve_catalog, verify_inputs
    from ..pipeline.job_queue import validate_task
    validate_task(task)
    stores = SimpleNamespace(roots=lambda selected: {alias: str(path) for alias, path in roots.items()}
                             if selected == series_id else {})
    rows = resolve_catalog(task, stores, series_id)
    verified = verify_inputs(task, rows)
    return _walk_catalogue(task, rows, verified)


def _walk_catalogue(task, rows, verified):
    """Replay the original named metadata consumers with exact resolved rows."""
    configuration = verified.for_use("run.configuration")
    config = _json_bytes(_bytes(configuration.path, _bound_reference(configuration)))
    plan_row, manifest_row = verified.for_use("run.execution_series"), verified.for_use("run.series_manifest")
    plan = tomllib.loads(_bytes(plan_row.path, _bound_reference(plan_row)).decode("utf-8-sig"))
    declarations = [row for row in plan["runs"] if row["run"] == task["run"]]
    if len(declarations) != 1 or declarations[0]["configuration"]["sha256"] != task["configuration"]["sha256"]:
        raise ValueError("Original execution plan does not bind this staged configuration exactly once")
    catalogue = _Catalogue(verified, [*plan.get("data_aliases", ()), *config.get("data_aliases", ())])
    catalogue.add(_bound_reference(plan_row), "configuration", "run.execution_series")
    catalogue.add(_bound_reference(manifest_row), "configuration", "run.series_manifest")
    actual = catalogue.collect(declarations[0]["configuration"], config)
    for field in ("format", "complete", "run", "configuration", "inputs",
                  "payload_hashes_verified", "numerical_calls_started"):
        if actual.get(field) != task["input_catalog"].get(field):
            raise ValueError("Staged package differs from original named input consumers: " + field)
    expected = {use for row in task["input_catalog"]["inputs"] for use in row["uses"]}
    if set(catalogue.consumers) != expected:
        raise ValueError("Staged package contains unconsumed or missing catalogue roles")
    catalogue.bind_aliases()
    return SimpleNamespace(config=config, configuration=declarations[0]["configuration"],
        rows=rows, catalogue=catalogue, verified=verified)


def _p2_publication(publication, run):
    """Original closed N1 publication validation, collecting only shared proof."""
    from ..pipeline.job_queue import _own_asset
    from ..pipeline.reproduction_lineage import _pin
    from ..pipeline.products_publication import _read_index, published_metadata_descriptor
    from ..published_jobs_snapshot import _terminal
    from ..jobs_metadata import JobSnapshotReader
    root = Path(publication["job"]["path"]).parent
    reader, pins = JobSnapshotReader(), {}
    def observe(path, expected=None):
        path = Path(path)
        if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise ValueError("Original N1 publication escapes its explicitly selected job")
        pin = _pin(path, limit=MAX_METADATA_BYTES)
        if expected is not None and pin != expected:
            raise ValueError("Original N1 publication bytes changed")
        old = pins.get(str(path.resolve(strict=True)))
        if old is not None and old != pin:
            raise ValueError("Original N1 publication changed during verification")
        pins[str(path.resolve(strict=True))] = pin
        return pin
    def read(path, expected=None):
        return reader.read(path, observe(path, expected))
    job = read(Path(publication["job"]["path"]), publication["job"])
    for field, path in (("task", root / "results" / run / "task.json"),
                        ("done", root / "done" / (run + ".json"))):
        if Path(publication[field]["path"]) != path:
            raise ValueError("Original N1 publication has a noncanonical task/done path")
    task = read(Path(publication["task"]["path"]), publication["task"])
    done = read(Path(publication["done"]["path"]), publication["done"])
    if task.get("pipeline") != "n1" or task.get("run") != run:
        raise ValueError("The final staged task is not the original N1 run")
    index_path = _own_asset(root, done["products_index"])
    if Path(publication["products_index"]["path"]) != index_path:
        raise ValueError("Original N1 products index differs from its canonical done record")
    read(index_path, publication["products_index"])
    completed, machines, _ = _terminal(root, root / "results" / run, task, job, done, read)
    _, index, products = _read_index(publication["products_index"])
    for product_id, row in products.items():
        if "metadata_copy" in row:
            pin = published_metadata_descriptor(publication["products_index"], product_id)
            observe(Path(pin["path"]), dict(path=pin["path"], sha256=pin["sha256"], size_bytes=pin["bytes"]))
    proof = []
    for pin in pins.values():
        if _pin(Path(pin["path"]), limit=MAX_METADATA_BYTES) != pin:
            raise ValueError("Original N1 proof changed before verification closed")
        proof.append(dict(relative_path=Path(pin["path"]).resolve().relative_to(root.resolve()).as_posix(), **pin))
    return dict(job=job, task=task, completed=completed, products_index=index,
                frame_machines=machines, proof_files=sorted(proof, key=lambda row: row["relative_path"]))


class _Files:
    def __init__(self):
        self.assets, self.members, self.sealed, self.transport, self.sources = {}, {}, set(), {}, {}
        self.priorities = {}
        self.identities = {}

    def add(self, target, source, sha256, size_bytes, *, sealed=False, external=False):
        target = record_relative(target)
        source = Path(source)
        before = source.stat()
        if before.st_size != size_bytes or file_sha256(source) != sha256:
            raise ValueError("Staged source bytes changed: " + target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Staged source changed while hashing: " + target)
        asset = dict(path=target, sha256=sha256, size_bytes=size_bytes)
        if target in self.assets and self.assets[target] != asset:
            raise ValueError("Two staged files conflict at one portable path")
        self.assets[target] = asset
        if not external:
            self.members[target] = Member(target, source, sha256, size_bytes)
        if sealed:
            self.sealed.add(target)
        self.alias(dict(path=str(source), sha256=sha256, size_bytes=size_bytes), asset, source)
        return asset

    def alias(self, original, asset, source=None, *, input_reference=None, priority=0):
        if (original["sha256"], original["size_bytes"]) != (asset["sha256"], asset["size_bytes"]):
            raise ValueError("Staged transport cannot change original bytes")
        key = (_key(original["path"]), original["sha256"], original["size_bytes"])
        if key[0] in self.identities and self.identities[key[0]] != key[1:]:
            raise ValueError("One original locator has conflicting byte identities for the unchanged DataReader")
        self.identities[key[0]] = key[1:]
        previous = self.transport.get(key)
        if previous is not None and priority < self.priorities.get(key, 0):
            return
        if previous is not None and previous["asset"] != asset:
            # A named consumer supersedes a shared proof copy. P2 is visited
            # after P1 so its exact catalogue location wins for coupling.
            if (previous["asset"]["sha256"], previous["asset"]["size_bytes"]) != (asset["sha256"], asset["size_bytes"]):
                raise ValueError("Conflicting staged source transport")
            if priority < self.priorities.get(key, 0) or priority == 0:
                return
            if priority == self.priorities.get(key, 0) and input_reference is not None:
                raise ValueError("One original input locator has conflicting stage catalogue locations")
        row = dict(logical=deepcopy(original), asset=deepcopy(asset))
        if input_reference is not None:
            if (input_reference["sha256"], input_reference["bytes"]) != (original["sha256"], original["size_bytes"]):
                raise ValueError("Staged input transport changes its named catalogue reference")
            row["input_reference"] = {field: input_reference[field]
                for field in ("root_alias", "relative_path", "sha256", "bytes")}
        elif previous is not None and "input_reference" in previous:
            return
        self.transport[key] = row
        self.priorities[key] = priority
        if source is not None:
            self.sources[key] = Path(source)


def _publication_files(files, stage_id, publication, verified):
    prefix = "staged/publications/" + stage_id
    proof, by_path = [], {}
    for row in verified["proof_files"]:
        original = {field: row[field] for field in ("path", "sha256", "size_bytes")}
        asset = files.add(prefix + "/" + row["relative_path"], row["path"], row["sha256"], row["size_bytes"], sealed=True)
        proof.append(dict(relative_path=row["relative_path"], source=original, asset=asset))
        by_path[_key(row["path"])] = asset
    roles = {role: by_path[_key(publication[role]["path"])] for role in _PROOF_ROLES}
    return dict(publication_root=prefix, publication=deepcopy(publication), proof_files=proof, **roles)


def _input_path(reference, camera):
    suffix = record_relative(reference["root_alias"] + "/" + reference["relative_path"])
    return suffix if camera else "staged/roots/" + suffix


def prepare_staged(request, job, published, roots, report):
    """Prepare full immutable staged closure or raise a concrete refusal."""
    from ..pipeline.job_queue import name, _own_asset
    from ..pipeline.reproduction_lineage import _pin, verify_phase1_origin
    if (request.record != "R" or published.commit != job.job["commit"]
            or published.plan_sha256 != job.plan_reference()["sha256"] or published.proof is not None):
        raise ValueError("Staged RAW reproduction requires the exact original producing revision")
    if len(published.configs) != len(job.runs):
        raise ValueError("Every original sealed N1 configuration is required")
    files, stages, runs, inputs_by_run, task_bytes, stage_jobs, inputs = _Files(), {}, {}, {}, {}, {}, {}
    for run, configured in zip(job.runs, published.configs):
        if configured.get("run") != run or job.tasks[run].get("pipeline") != "n1":
            raise ValueError("Every staged run must keep its original sealed N1 configuration")
        source = configured.get("phase1") or {}
        if "origin" not in source:
            raise ValueError(run + ": full RAW reproduction needs the persisted original P1 job binding; legacy ancestry is never guessed")
        original = verify_phase1_origin(source["origin"], configuration=source["configuration"],
            completion=source["completion"], expected_commit=source["code_commit"], ordered_case_ids=job.frames(run))
        if original["job"].get("series_id") != job.series_id:
            raise ValueError("The bound P1 job differs from the original selected series")
        if (original["task"].get("precision") != job.tasks[run].get("precision")
                or job.tasks[run].get("precision") != "float32"):
            raise ValueError("Staged RAW reproduction preserves the original float32 precision")
        p2root = Path(job.root)
        p2publication = {"job": _pin(p2root / "job.json", limit=MAX_METADATA_BYTES),
            "task": _pin(p2root / "results" / name(run) / "task.json", limit=MAX_METADATA_BYTES),
            "done": _pin(p2root / "done" / (name(run) + ".json"), limit=MAX_METADATA_BYTES),
            "products_index": _pin(_own_asset(p2root, job.done[run]["products_index"]), limit=MAX_METADATA_BYTES)}
        final = _p2_publication(p2publication, run)
        if final["job"] != job.job or final["task"] != job.tasks[run] or final["completed"] != job.completed[run]:
            raise ValueError("The final N1 job changed while staged export was prepared")
        combined = {}
        stage_readers = {}
        for phase, verified, publication in (("phase1", original, {key: source["origin"][key] for key in _PROOF_ROLES}),
                                              ("phase2", final, p2publication)):
            stage_id = ("p1-" if phase == "phase1" else "p2-") + name(run)
            name(stage_id)
            supplied = _verified_catalogue(verified["task"], roots, job.series_id)
            if phase == "phase2" and supplied.config != configured:
                raise ValueError("The packaged N1 configuration differs from the selected original DATA")
            if supplied.config.get("run") != run or [row["case_id"] for row in supplied.config["frames"]] != job.frames(run):
                raise ValueError("The original staged configuration changes the selected case order")
            stage_readers[phase] = supplied
            metadata_references = {
                (row.reference["root_alias"], row.reference["relative_path"], row.reference["sha256"], row.reference["bytes"])
                for pin, row in supplied.catalogue.transport.values()
                if (_key(pin["path"]), pin["sha256"]) in supplied.catalogue.documents}
            stage = _publication_files(files, stage_id, publication, verified)
            stage.update(pipeline=verified["task"]["pipeline"], run=run, commit=verified["job"]["commit"],
                precision=verified["task"]["precision"], ordered_case_ids=list(job.frames(run)),
                configuration=deepcopy(supplied.configuration),
                environment_lock=deepcopy(verified["job"]["environment_lock"]),
                output_bytes=sum(row["bytes"] for row in verified["products_index"]["products"]))
            if phase == "phase1":
                stage["origin"] = deepcopy(source["origin"])
            else:
                stage["phase1_origin"] = deepcopy(source["origin"])
            stages[stage_id], stage_jobs[stage_id] = stage, deepcopy(verified["job"])
            runs.setdefault(run, {})[phase] = stage_id
            by_reference = {}
            for row in supplied.rows:
                ref = row.reference
                camera = _camera_input(ref, row.uses, phase)
                item = StagedInput(ref["root_alias"], ref["relative_path"], ref["sha256"], ref["bytes"],
                                   ref["role"], tuple(row.uses), camera)
                key = (item.root_alias, item.relative_path)
                prior = combined.get(key)
                if prior is not None and (prior.sha256, prior.size_bytes) != (item.sha256, item.size_bytes):
                    raise ValueError("P1 and P2 require conflicting bytes at one original input path")
                if prior is not None:
                    item = replace(item, uses=tuple(dict.fromkeys((*prior.uses, *item.uses))), camera=prior.camera or item.camera)
                combined[key] = item
                target = _input_path(ref, item.camera)
                asset = dict(path=target, sha256=item.sha256, size_bytes=item.size_bytes)
                if not item.camera:
                    is_metadata = (ref["root_alias"], ref["relative_path"], ref["sha256"], ref["bytes"]) in metadata_references
                    if any(use in row.uses for use in ("run.configuration", "run.execution_series", "run.series_manifest")):
                        is_metadata = True
                    asset = files.add(target, row.path, item.sha256, item.size_bytes, sealed=is_metadata)
                by_reference[(ref["root_alias"], ref["relative_path"], ref["sha256"], ref["bytes"])] = asset
                if key in inputs and (inputs[key]["sha256"], inputs[key]["size_bytes"]) != (item.sha256, item.size_bytes):
                    raise ValueError("Different runs require conflicting bytes at one original input path")
                if key in inputs and inputs[key]["camera"] != item.camera:
                    raise ValueError("One shared input has contradictory camera and metadata roles across stages")
                inputs[key] = dict(root_alias=item.root_alias, relative_path=item.relative_path,
                    path=target, sha256=item.sha256, size_bytes=item.size_bytes, camera=item.camera)
            for pin, row in supplied.catalogue.transport.values():
                ref = row.reference
                asset = by_reference[(ref["root_alias"], ref["relative_path"], ref["sha256"], ref["bytes"])]
                files.alias(pin, asset, row.path, input_reference=ref, priority=2 if phase == "phase2" else 1)
        # Couple against every exact original P1 artifact, including outputs not
        # consumed by P2. Only named indexed products are followed; never a scan.
        p1index_root = Path(source["origin"]["products_index"]["path"]).parent
        for row in original["products_index"]["products"]:
            pin = dict(path=row["local_path"], sha256=row["sha256"], size_bytes=row["bytes"])
            key = (_key(pin["path"]), pin["sha256"], pin["size_bytes"])
            if key in files.transport:
                continue
            if "metadata_copy" in row:
                copy = row["metadata_copy"]
                path = p1index_root.joinpath(*copy["relative_path"].split("/"))
                shared_key = (_key(path), pin["sha256"], pin["size_bytes"])
                asset = files.transport[shared_key]["asset"]
            else:
                path = Path(pin["path"])
                asset = files.add("staged/expected/" + runs[run]["phase1"] + "/" + record_relative(row["product_id"]),
                    path, pin["sha256"], pin["size_bytes"])
            files.alias(pin, asset, path)
        inputs_by_run[run] = sorted(combined.values(), key=lambda item: (item.root_alias, item.relative_path))
        phase1 = stage_readers["phase1"].config
        for prior, current in zip(phase1["frames"], configured["frames"]):
            sha = prior["source"]["source_sha256"]
            acquisition = _frame_acquisition(original["task"], prior["case_id"])
            if acquisition["sha256"] != sha or current["source_binding"]["source_sha256"] != sha:
                raise ValueError("Full staged reproduction requires every original selected acquisition's RAW bytes")
        raw = _bytes(p2publication["task"]["path"], p2publication["task"])
        target = task_path(run)
        task_bytes[target] = raw
        files.add(target, p2publication["task"]["path"], p2publication["task"]["sha256"], len(raw), sealed=True, external=True)
    if published.plan_path is None:
        raise ValueError("The exact original P2 execution plan is required")
    plan_pin = _pin(published.plan_path, limit=MAX_METADATA_BYTES)
    if plan_pin["sha256"] != published.plan_sha256:
        raise ValueError("Original P2 execution plan changed")
    files.add("plan/adopted_plan.toml", published.plan_path, plan_pin["sha256"], plan_pin["size_bytes"], sealed=True, external=True)
    section = dict(format=FORMAT, require_bit_identical=True, stages=stages, runs=runs,
        camera_members={run: [item.root_alias + "/" + item.relative_path for item in rows if item.camera]
                        for run, rows in inputs_by_run.items()},
        roots={alias: "staged/roots/" + alias for alias in sorted({key[0] for key in inputs})},
        assets=sorted(files.assets.values(), key=lambda row: row["path"]),
        input_assets=sorted(inputs.values(), key=lambda row: (row["root_alias"], row["relative_path"])),
        transport=sorted(files.transport.values(), key=lambda row: (row["logical"]["path"], row["logical"]["sha256"])),
        sealed_metadata=sorted(files.sealed),
        provenance_policy=dict(retain_original_producer_metadata=getattr(request, "retain_producer_metadata", False) is True,
            scope="Only named immutable producer metadata and this staged provenance section; local paths and producer identifiers are retained verbatim"))
    return SimpleNamespace(inputs_by_run=inputs_by_run, members=list(files.members.values()),
                           tasks=task_bytes, section=section, stage_jobs=stage_jobs)


def _asset(value):
    import re
    if (not isinstance(value, dict) or not {"path", "sha256", "size_bytes"} <= value.keys()
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("Staged assets need exact path/SHA/size bindings")
    record_relative(value["path"])
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def attach_staged_software(staged, sections, members):
    """Record the explicit software files already prepared by the builder."""
    if set(sections) != set(staged.section["stages"]):
        raise ValueError("Every original stage requires its own exact software closure")
    assets = {row["path"]: row for row in staged.section["assets"]}
    additions = {}
    for member in members:
        path = record_relative(member.name)
        if not path.startswith("software/stages/") or path in assets or path in additions:
            raise ValueError("Staged software has conflicting or out-of-scope member names")
        if member.generate is not None:
            raise ValueError("Staged software must already have explicit immutable bytes")
        sha, size = hashlib.sha256(), 0
        for chunk in member.chunks():
            sha.update(chunk)
            size += len(chunk)
        value = dict(path=path, sha256=sha.hexdigest(), size_bytes=size)
        if ((member.sha256 is not None and member.sha256 != value["sha256"])
                or (member.size_bytes is not None and member.size_bytes != size)):
            raise ValueError("Staged software changed after its original verification")
        additions[path] = value
    assets.update(additions)
    expected = set()
    for stage_id, software in sections.items():
        stage = staged.section["stages"][stage_id]
        stage["software"] = deepcopy(software)
        expected.update(_software_assets(stage_id, stage, assets))
    if expected != set(additions):
        raise ValueError("Staged software omits or adds unreferenced files")
    staged.section["assets"] = sorted(assets.values(), key=lambda row: row["path"])
    # Only the original input inventory is a producer receipt. Generated public
    # software/version references and included code remain under strict scanning.
    staged.section["sealed_metadata"] = sorted(set(staged.section["sealed_metadata"]) | {
        stage["software"]["source_inventory"]["path"] for stage in staged.section["stages"].values()})


def _software_assets(stage_id, stage, assets):
    """Validate software references and return their exact archive path set."""
    import re
    from .export_software import _name, _url
    software = stage.get("software") or {}
    if (software.get("mode") not in {"references", "include"}
            or software.get("code", {}).get("commit") != stage["commit"]
            or re.fullmatch(r"\d+\.\d+\.\d+", str(software.get("python", ""))) is None):
        raise ValueError("Each stage needs its own original exact code/Python/software closure")
    prefix, paths = "software/stages/" + stage_id + "/", set()
    def add(bound, *, size_required=True):
        path = record_relative(bound["path"])
        archived = assets.get(path)
        if (not path.startswith(prefix) or archived is None or archived["sha256"] != bound["sha256"]
                or (size_required and archived != _asset(bound)) or path in paths):
            raise ValueError("Staged software file differs from its complete exact inventory")
        paths.add(path)
    for field in ("inventory", "source_inventory"):
        add(software[field])
    lock = software["environment_lock"]
    if lock["sha256"] != stage.get("environment_lock", {}).get("sha256"):
        raise ValueError("Stage software lock differs from its original job")
    add(lock, size_required=False)
    _url(software["code"]["url"])
    libraries = software.get("libraries")
    if not isinstance(libraries, list) or not libraries:
        raise ValueError("Each stage software closure needs every exact locked library")
    names = [_name(row["name"]) for row in libraries]
    if len(set(names)) != len(names) or any(not row.get("version") for row in libraries):
        raise ValueError("Stage software library names/versions are missing or duplicated")
    for row in libraries:
        if "url" in row:
            _url(row["url"])
    if software["mode"] == "include":
        for row in [software["code"], *libraries]:
            add(row)
    elif any("path" in row for row in [software["code"], *libraries]):
        raise ValueError("Reference-only software cannot hide unverified included assets")
    return paths


def _readback_software(datasetroot, stage, assets):
    """Recheck immutable inventory, original pins, wheels and code closure."""
    import re
    from .export_software import FORMAT as INPUT_FORMAT, _pins, _name, _bundle, _wheel
    software = stage["software"]
    document = _json_bytes(_bytes(inside(datasetroot, software["inventory"]["path"]), software["inventory"]))
    if document != {key: value for key, value in software.items() if key != "inventory"}:
        raise ValueError("Stage software versions document differs from its manifest")
    source = _json_bytes(_bytes(inside(datasetroot, software["source_inventory"]["path"]), software["source_inventory"]))
    if (source.get("format") != INPUT_FORMAT or source.get("python") != software["python"]
            or source.get("code", {}).get("commit") != stage["commit"]
            or source["code"].get("url") != software["code"]["url"]):
        raise ValueError("Original stage software inventory differs from its portable references")
    lock = assets[software["environment_lock"]["path"]]
    raw = _bytes(inside(datasetroot, lock["path"]), lock)
    pins = _pins(raw)
    python = re.search(r"Python\s+(\d+\.\d+)", raw.decode("utf-8-sig"))
    if (python is None or not software["python"].startswith(python[1] + ".")
            or source["environment_lock"]["sha256"] != lock["sha256"]
            or source["environment_lock"]["size_bytes"] != lock["size_bytes"]):
        raise ValueError("Original stage Python/lock bytes differ from their saved bindings")
    original = {_name(row["name"]): row for row in source["libraries"]}
    if len(original) != len(source["libraries"]) or set(original) != set(pins):
        raise ValueError("Original software inventory omits or adds locked libraries")
    if {_name(row["name"]): row["version"] for row in software["libraries"]} != pins:
        raise ValueError("Stage software differs from the exact original lock pins")
    for row in software["libraries"]:
        previous = original[_name(row["name"])]
        if previous["version"] != row["version"] or previous.get("url") != row.get("url"):
            raise ValueError("Original library reference changed in transport")
        if software["mode"] == "include":
            if (previous["sha256"], previous["size_bytes"]) != (row["sha256"], row["size_bytes"]):
                raise ValueError("Included library differs from its original inventory")
            _wheel(inside(datasetroot, row["path"]), _name(row["name"]), row["version"])
    if software["mode"] == "include":
        previous, row = source["code"]["bundle"], software["code"]
        if (previous["sha256"], previous["size_bytes"]) != (row["sha256"], row["size_bytes"]):
            raise ValueError("Included source differs from its original inventory")
        _bundle(inside(datasetroot, row["path"]), stage["commit"])


def validate_staged(manifest):
    """Validate exact stage/run/selection and portable closure without I/O."""
    from ..pipeline.job_queue import name
    from ..pipeline.reproduction_lineage import validate_phase1_origin
    section = manifest.get("staged_reproduction")
    if section is None:
        return None
    if (section.get("format") != FORMAT or manifest["identity"]["record"] != "R"
            or section.get("require_bit_identical") is not True):
        raise ValueError("Staged RAW reproduction belongs only to an R record")
    runs = {row["run"]: [frame["case_id"] for frame in row["frames"]] for row in manifest["runs"]}
    if set(section.get("runs", {})) != set(runs):
        raise ValueError("Staged replay runs differ from the saved frame selection")
    assets = {}
    for row in section.get("assets", ()):
        bound = _asset(row)
        if bound["path"] in assets:
            raise ValueError("Staged asset inventory has duplicate paths")
        assets[bound["path"]] = bound
    def included(bound):
        value = _asset(bound)
        if assets.get(value["path"]) != value:
            raise ValueError("Staged asset is omitted or differs from its complete inventory")
    known_stages = set()
    for run, phases in section["runs"].items():
        if set(phases) != {"phase1", "phase2"}:
            raise ValueError("Each staged run needs exactly P1 and P2")
        for phase, pipeline in (("phase1", "native-phase-statistics"), ("phase2", "n1")):
            expected = ("p1-" if phase == "phase1" else "p2-") + name(run)
            if phases[phase] != expected or expected in known_stages:
                raise ValueError("Staged identifiers differ from their exact original run")
            known_stages.add(expected)
            stage = section["stages"][expected]
            if (stage.get("pipeline") != pipeline or stage.get("run") != run
                    or stage.get("ordered_case_ids") != runs[run] or stage.get("precision") != "float32"
                    or (phase == "phase2" and stage.get("commit") != manifest["code"]["published_commit"])):
                raise ValueError("Staged task/selection/code/precision differs")
            if type(stage.get("output_bytes")) is not int or stage["output_bytes"] < 1:
                raise ValueError("Staged output estimate needs its exact original positive byte count")
            if stage.get("publication_root") != "staged/publications/" + expected:
                raise ValueError("Staged publication root differs from its named stage")
            if set(stage.get("publication", {})) != set(_PROOF_ROLES):
                raise ValueError("Staged publication needs every original ownership descriptor")
            proof_names = set()
            for proof in stage.get("proof_files", ()):
                relative = record_relative(proof["relative_path"])
                if relative in proof_names or proof["asset"]["path"] != stage["publication_root"] + "/" + relative:
                    raise ValueError("Staged publication proof is duplicated or relocated outside its exact tree")
                proof_names.add(relative)
                included(proof["asset"])
                if (proof["source"]["sha256"], proof["source"]["size_bytes"]) != (proof["asset"]["sha256"], proof["asset"]["size_bytes"]):
                    raise ValueError("Staged proof changed original metadata bytes")
            for role in _PROOF_ROLES:
                included(stage[role])
                original = stage["publication"][role]
                if (original["sha256"], original["size_bytes"]) != (stage[role]["sha256"], stage[role]["size_bytes"]):
                    raise ValueError("Staged publication role changes its original bytes")
                matches = [proof for proof in stage["proof_files"]
                           if proof["source"] == original and proof["asset"] == stage[role]]
                if len(matches) != 1:
                    raise ValueError("Staged publication role lacks its exact named proof file")
            origin = stage.get("origin") if phase == "phase1" else stage.get("phase1_origin")
            validate_phase1_origin(origin)
            if origin["run"] != run:
                raise ValueError("Staged P1 origin belongs to another run")
            if phase == "phase1" and {role: origin[role] for role in _PROOF_ROLES} != stage["publication"]:
                raise ValueError("Staged P1 publication differs from the exact persisted original job binding")
            _software_assets(expected, stage, assets)
        if section["stages"][phases["phase1"]]["origin"] != section["stages"][phases["phase2"]]["phase1_origin"]:
            raise ValueError("P1 and P2 retain different original producer bindings")
    if set(section.get("stages", {})) != known_stages:
        raise ValueError("Staged closure omits or adds a stage")
    inputs, camera = {}, {}
    for row in section.get("input_assets", ()):
        name(row["root_alias"])
        record_relative(row["relative_path"])
        key = (row["root_alias"], row["relative_path"])
        if key in inputs or type(row.get("camera")) is not bool:
            raise ValueError("Staged input catalogue paths are duplicated or have no explicit kind")
        inputs[key] = _asset(row)
        if row["path"] != _input_path(row, row["camera"]):
            raise ValueError("Staged input transport changes its original alias/relative path")
        if row["camera"]:
            camera[row["path"]] = _asset(row)
        else:
            included(row)
    if section.get("roots") != {alias: "staged/roots/" + alias for alias in sorted({key[0] for key in inputs})}:
        raise ValueError("Staged root aliases differ from the exact input closure")
    if set(section.get("camera_members", {})) != set(runs):
        raise ValueError("Staged RAW archive inventory differs from its exact selected runs")
    for members in section["camera_members"].values():
        if not isinstance(members, list) or len(set(members)) != len(members) or not set(members) <= camera.keys():
            raise ValueError("Staged RAW archive members differ from the original camera inputs")
    locators, identities = {}, {}
    for row in section.get("transport", ()):
        bound = _asset(row["asset"])
        if assets.get(bound["path"]) != bound and camera.get(bound["path"]) != bound:
            raise ValueError("Staged source transport points outside the exact archive closure")
        if (row["logical"]["sha256"], row["logical"]["size_bytes"]) != (bound["sha256"], bound["size_bytes"]):
            raise ValueError("Staged source transport changes SHA/size")
        logical = row["logical"]
        if not isinstance(logical.get("path"), str) or not logical["path"]:
            raise ValueError("Staged source transport needs its explicit original locator")
        key = (_key(logical["path"]), logical["sha256"], logical["size_bytes"])
        if key[0] in identities and identities[key[0]] != key[1:]:
            raise ValueError("Staged source transport conflicts with original DataReader locator identity")
        identities[key[0]] = key[1:]
        if key in locators:
            raise ValueError("Staged source transport repeats an original binding")
        locators[key] = bound
        if "input_reference" in row:
            ref = row["input_reference"]
            expected = inputs.get((ref["root_alias"], ref["relative_path"]))
            if (set(ref) != {"root_alias", "relative_path", "sha256", "bytes"} or expected != bound
                    or (ref["sha256"], ref["bytes"]) != (bound["sha256"], bound["size_bytes"])):
                raise ValueError("Staged transport differs from its explicit catalogue location")
    sealed = section.get("sealed_metadata")
    if not isinstance(sealed, list) or len(set(sealed)) != len(sealed) or not set(sealed) <= assets.keys():
        raise ValueError("Staged immutable metadata inventory differs")
    if section.get("provenance_policy", {}).get("retain_original_producer_metadata") is not True:
        raise ValueError("Original producer receipts require an explicit include choice")
    return section


def readback_staged(datasetroot, manifest):
    """Rehash packaged bytes and re-run original terminal guards on copied proofs."""
    from ..pipeline.reproduction_lineage import verify_phase1_origin
    section = validate_staged(manifest)
    if section is None:
        return None
    for asset in section["assets"]:
        path = inside(datasetroot, asset["path"])
        if path.stat().st_size != asset["size_bytes"] or file_sha256(path) != asset["sha256"]:
            raise ValueError("Packaged staged asset SHA/size differs: " + asset["path"])
    from ..pipeline.store_runtime import ResolvedInput
    import ntpath
    assets = {row["path"]: row for row in section["assets"]}
    inputs = {(row["root_alias"], row["relative_path"]): row for row in section["input_assets"]}
    transport = {(_key(row["logical"]["path"]), row["logical"]["sha256"], row["logical"]["size_bytes"]): row
                 for row in section["transport"]}
    verified, consumed, expected_transport, required, sealed, camera_roles = {}, set(), {}, set(), set(), {}
    for stage_id, stage in section["stages"].items():
        publication = {role: dict(stage["publication"][role], path=str(inside(datasetroot, stage[role]["path"])))
                       for role in _PROOF_ROLES}
        if stage["pipeline"] == "native-phase-statistics":
            origin = dict(stage["origin"], **publication)
            paired = section["stages"][section["runs"][stage["run"]]["phase2"]]
            p2task = _json_bytes(_bytes(inside(datasetroot, paired["task"]["path"]), paired["task"]))
            matches = [row for row in section["input_assets"] if row["root_alias"] == p2task["configuration"]["root_alias"]
                       and row["relative_path"] == p2task["configuration"]["relative_path"]]
            if len(matches) != 1:
                raise ValueError("Packaged P2 configuration is not in its exact input closure")
            config = _json_bytes(_bytes(inside(datasetroot, matches[0]["path"]), matches[0]))
            if config.get("phase1", {}).get("origin") != stage["origin"]:
                raise ValueError("Packaged N1 configuration changes the persisted P1 origin")
            result = verify_phase1_origin(origin, configuration=config["phase1"]["configuration"],
                completion=config["phase1"]["completion"], expected_commit=config["phase1"]["code_commit"],
                ordered_case_ids=stage["ordered_case_ids"])
        else:
            result = _p2_publication(publication, stage["run"])
        actual_task = _json_bytes(_bytes(inside(datasetroot, stage["task"]["path"]), stage["task"]))
        if (result["task"] != actual_task or result["task"]["ordered_case_ids"] != stage["ordered_case_ids"]
                or result["job"]["commit"] != stage["commit"] or result["task"]["precision"] != stage["precision"]
                or result["job"]["environment_lock"] != stage["environment_lock"]
                or result["job"]["series_id"] != manifest["identity"]["series_id"]
                or sum(row["bytes"] for row in result["products_index"]["products"]) != stage["output_bytes"]):
            raise ValueError("Packaged original stage differs from its verified publication")
        observed = {row["relative_path"]: row for row in result["proof_files"]}
        if set(observed) != {row["relative_path"] for row in stage["proof_files"]}:
            raise ValueError("Packaged publication proof omits or adds original metadata files")
        for proof in stage["proof_files"]:
            actual = observed[proof["relative_path"]]
            old_path = ntpath.join(ntpath.dirname(stage["publication"]["job"]["path"]), proof["relative_path"])
            if (_key(proof["source"]["path"]) != _key(old_path)
                    or (actual["sha256"], actual["size_bytes"]) != (proof["asset"]["sha256"], proof["asset"]["size_bytes"])):
                raise ValueError("Packaged proof changes an original publication locator or bytes")
            required.add(proof["asset"]["path"])
            sealed.add(proof["asset"]["path"])
        rows, by_use = [], {}
        for entry in result["task"]["input_catalog"]["inputs"]:
            ref = entry["reference"]
            matches = [row for row in section["input_assets"] if row["root_alias"] == ref["root_alias"]
                       and row["relative_path"] == ref["relative_path"]]
            if len(matches) != 1 or (matches[0]["sha256"], matches[0]["size_bytes"]) != (ref["sha256"], ref["bytes"]):
                raise ValueError("Packaged input inventory omits or alters an original stage consumer")
            consumed.add((ref["root_alias"], ref["relative_path"]))
            key = (ref["root_alias"], ref["relative_path"])
            phase = "phase1" if stage["pipeline"] == "native-phase-statistics" else "phase2"
            camera_roles[key] = camera_roles.get(key, False) or _camera_input(ref, entry["uses"], phase)
            row = ResolvedInput(deepcopy(ref), inside(datasetroot, matches[0]["path"]), tuple(entry["uses"]))
            rows.append(row)
            for use in row.uses:
                if use in by_use:
                    raise ValueError("Original staged consumer is duplicated")
                by_use[use] = row
            if not matches[0]["camera"]:
                required.add(matches[0]["path"])
        walked = _walk_catalogue(actual_task, tuple(rows), SimpleNamespace(for_use=by_use.__getitem__))
        if walked.configuration != stage["configuration"]:
            raise ValueError("Staged configuration locator differs from its original execution plan")
        for pin, row in walked.catalogue.transport.values():
            if ((_key(pin["path"]), pin["sha256"]) in walked.catalogue.documents
                    or any(use in row.uses for use in ("run.configuration", "run.execution_series", "run.series_manifest"))):
                item = inputs[(row.reference["root_alias"], row.reference["relative_path"])]
                if not item["camera"]:
                    sealed.add(item["path"])
        priority = 2 if stage["pipeline"] == "n1" else 1
        for pin, row in walked.catalogue.transport.values():
            if _key(pin["path"]) == _key(row.path):
                continue  # This is the newly materialized readback path, not producer history.
            key = (_key(pin["path"]), pin["sha256"], pin["size_bytes"])
            ref = {field: row.reference[field] for field in ("root_alias", "relative_path", "sha256", "bytes")}
            previous = expected_transport.get(key)
            if previous is None or priority > previous[0]:
                expected_transport[key] = (priority, ref)
            elif priority == previous[0] and ref != previous[1]:
                raise ValueError("Original stages have conflicting catalogue locations")
        if stage["pipeline"] == "native-phase-statistics":
            for product in result["products_index"]["products"]:
                key = (_key(product["local_path"]), product["sha256"], product["bytes"])
                if key not in transport:
                    raise ValueError("Packaged source transport omits an original P1 output")
                required.add(transport[key]["asset"]["path"])
        else:
            copy = assets.get(task_path(stage["run"]))
            if copy is None or (copy["sha256"], copy["size_bytes"]) != (stage["task"]["sha256"], stage["task"]["size_bytes"]):
                raise ValueError("Packaged prepared task differs from original P2 bytes")
            required.add(copy["path"])
            sealed.add(copy["path"])
            plan = by_use["run.execution_series"].reference
            copy = assets.get("plan/adopted_plan.toml")
            if copy is None or (copy["sha256"], copy["size_bytes"]) != (plan["sha256"], plan["bytes"]):
                raise ValueError("Packaged plan differs from original P2 bytes")
            required.add(copy["path"])
            sealed.add(copy["path"])
        required.update(_software_assets(stage_id, stage, assets))
        sealed.add(stage["software"]["source_inventory"]["path"])
        _readback_software(datasetroot, stage, assets)
        verified[stage_id] = result
    if consumed != {(row["root_alias"], row["relative_path"]) for row in section["input_assets"]}:
        raise ValueError("Packaged input inventory adds unconsumed resources")
    if any(inputs[key]["camera"] != camera for key, camera in camera_roles.items()):
        raise ValueError("Packaged camera classification differs from original named image consumers")
    if required != set(assets):
        raise ValueError("Staged asset inventory adds files outside the exact original closure")
    if sealed != set(section["sealed_metadata"]):
        raise ValueError("Staged producer-metadata retention scope differs from verified original receipts")
    for key, (_priority, ref) in expected_transport.items():
        if key not in transport or transport[key].get("input_reference") != ref:
            raise ValueError("Packaged source transport omits or redirects an original named consumer")
    for run in manifest["runs"]:
        stage = verified[section["runs"][run["run"]]["phase2"]]
        reference = stage["task"]["configuration"]
        configured = inputs[(reference["root_alias"], reference["relative_path"])]
        config = _json_bytes(_bytes(inside(datasetroot, configured["path"]), configured))
        stage_ids = section["runs"][run["run"]].values()
        run_inputs = {(row["reference"]["root_alias"], row["reference"]["relative_path"])
                      for stage_id in stage_ids for row in verified[stage_id]["task"]["input_catalog"]["inputs"]}
        cameras = sorted(inputs[key]["path"] for key in run_inputs if inputs[key]["camera"])
        if cameras != section["camera_members"][run["run"]]:
            raise ValueError("Staged RAW archive members differ from the exact per-run original catalogues")
        if len(run["frames"]) != len(config["frames"]):
            raise ValueError("Packaged RAW selection omits or adds an original frame")
        for frame, spec in zip(run["frames"], config["frames"]):
            raw = frame["raw"]
            original = _frame_acquisition(verified[section["runs"][run["run"]]["phase1"]]["task"], spec["case_id"])
            alias, separator, relative = raw["path"].partition("/")
            item = inputs.get((alias, relative))
            from ..image_formats import image_family
            if (not separator or item is None or item["camera"] is not True
                    or image_family(relative) != "raw"
                    or (alias, relative) != (original["root_alias"], original["relative_path"])
                    or (item["sha256"], item["size_bytes"]) != (raw["sha256"], raw["size_bytes"])
                    or raw["sha256"] != spec["source_binding"]["source_sha256"]
                    or frame["case_id"] != spec["case_id"] or frame["frame_index"] != spec["frame_index"]
                    or frame["elapsed_h"] != spec["elapsed_h"]):
                raise ValueError("Packaged RAW selection differs from the original sealed acquisition")
    return verified


def provenance_findings(request, staged, report):
    """Apply only the explicit immutable-producer-metadata inclusion choice."""
    acknowledged = getattr(request, "retain_producer_metadata", False) is True
    sealed = set(staged.section["sealed_metadata"])
    if not acknowledged:
        report.refuse("Original producer receipts contain local paths and producer identifiers; select Include original producer receipts for full staged RAW reproduction")
        return report.hits
    retained = []
    for hit in report.hits:
        member = hit.file.removeprefix(SERIES_ARCHIVE + ":") if hit.file.startswith(SERIES_ARCHIVE + ":") else None
        manifest_provenance = hit.file == MANIFEST and hit.where.startswith("key /staged_reproduction/")
        if not hit.allowed and (member in sealed or manifest_provenance):
            report.warnings.append("Retained by explicit original-producer-metadata choice: " + hit.describe())
            hit = replace(hit, allowed=True, source=hit.source + "; explicitly included immutable producer metadata")
        retained.append(hit)
    report.hits[:] = retained
    return retained
