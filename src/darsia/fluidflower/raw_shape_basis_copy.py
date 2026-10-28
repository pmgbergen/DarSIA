"""Explicit Root-bound, byte-exact received copies of closed RAW bases.

The first route deliberately retains the producer's absolute path spelling on
the receiver. The producer's ownership remains historical evidence; a separate
Root copy release binds a source-side validation and actual receiver ownership.
These helpers never create that Root release, rewrite producer metadata, fit a
registration, decode RAW or remove retained files.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket

from . import raw_shape_basis_reader as reader
from .raw_shape_basis_reader import _keys, _same, _identity, _key

INTENT = "darsia-root-raw-basis-copy-intent-v1"
OBSERVATION = "darsia-raw-basis-source-observation-v2"
RECEIPT = "darsia-raw-basis-received-copy-v2"
RELEASE = "darsia-root-raw-basis-copy-release-v2"
POOL = "darsia-raw-basis-retained-sources-v1"


class _ReceivedPins(reader._Pins):
    """Record the original reader's consumed identities separately from copy DATA."""

    def __init__(self, aliases=(), *, folder=None):
        """Keep normal physical hashing and begin with no scientific replay capture."""
        super().__init__(aliases, folder=folder)
        self.core_rows = None
        self.core_copyable = None

    def add(self, binding, *, metadata=False, copyable=False):
        """Use the unchanged hasher and record a core call when capture is active."""
        result = super().add(binding, metadata=metadata, copyable=copyable)
        if self.core_rows is not None:
            original = reader._identity(binding)
            self.core_rows[reader._key(original['path'])] = original
            if copyable:
                self.core_copyable[reader._key(original['path'])] = original
        return result


def host_identity():
    """Observe this machine and boot; never infer a remote machine identity."""
    import psutil
    return dict(hostname=socket.gethostname().casefold(), boot_epoch_s=int(psutil.boot_time()))


def _observer():
    """Record the actual process instance performing this local observation."""
    import psutil
    return dict(host=host_identity(), pid=os.getpid(),
                process_created_epoch_s=psutil.Process().create_time())


def _host(value):
    """Require an explicit finite machine/boot identity from Root's intent."""
    reader._keys(value, {"hostname", "boot_epoch_s"}, "RAW copy host")
    if (not isinstance(value["hostname"], str) or not value["hostname"]
            or value["hostname"] != value["hostname"].casefold()
            or type(value["boot_epoch_s"]) is not int or value["boot_epoch_s"] <= 0):
        raise ValueError("RAW copy host/boot identity is invalid")
    return value


def _process(value, expected):
    """Validate retained process facts without probing a historical remote PID."""
    reader._keys(value, {"host", "pid", "process_created_epoch_s"}, "RAW copy observer")
    if (not reader._same(_host(value["host"]), expected)
            or type(value["pid"]) is not int or value["pid"] <= 0
            or type(value["process_created_epoch_s"]) not in {int, float}
            or not expected['boot_epoch_s'] <= value["process_created_epoch_s"] < float("inf")):
        raise ValueError("RAW copy observer differs from its intended host/process")


def _metadata_destination(path, producer):
    """Keep new observation metadata local, exclusive and outside the basis."""
    from .rig_raw_write import _canonical
    from .job_filesystem import inspect_local_file
    target, basis = _canonical(path), _canonical(producer)
    if target == basis or target.is_relative_to(basis) or not target.parent.is_dir():
        raise ValueError("RAW copy metadata needs an existing separate local parent")
    if target.exists() or not inspect_local_file(target).allowed:
        raise ValueError("RAW copy metadata must be a fresh positively local file")
    return target


def _write(path, value):
    """Create one immutable observation receipt, without granting Root authority."""
    from .rig_raw_build import _descriptor
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return _descriptor(path)


def _intent(pins, binding, *, verify_sources=True):
    """Read Root intent, deferring source IO only until its retained pool is bound."""
    value = pins.add(binding, metadata=True, copyable=True)
    reader._keys(value, {"format", "copy_id", "issued_utc", "run_id", "certificate",
        "basis_authority", "parent_workspace", "source_host", "receiver_host",
        "source_observation", "receiver_receipt", "copy_source_closure"}, "Root RAW copy intent")
    if (value["format"] != INTENT or re.fullmatch(r"[0-9a-f]{32}", str(value["copy_id"])) is None
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(value["run_id"])) is None
            or "received_basis" in value["basis_authority"]):
        raise ValueError("Root copy intent needs one original producer operation")
    reader._time(value["issued_utc"], "Root copy intent")
    reader._identity(value["certificate"])
    reader._identity(value["parent_workspace"])
    _host(value["source_host"])
    _host(value["receiver_host"])
    for key in ("source_observation", "receiver_receipt"):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError("Root copy intent metadata paths must be explicit")
    closure = value["copy_source_closure"]
    reader._keys(closure, {"base_commit", "tree", "files"}, "Root copy source closure")
    if (any(re.fullmatch(r"[0-9a-f]{40}", str(closure[key])) is None for key in ("base_commit", "tree"))
            or not isinstance(closure["files"], list) or not closure["files"]
            or len({reader._key(reader._identity(row)["path"]) for row in closure["files"]}) != len(closure["files"])):
        raise ValueError("Root copy source closure is incomplete")
    if verify_sources:
        for row in closure["files"]:
            pins.add(row, copyable=True)
    return value


def _files(document, certificate):
    """Retain the complete original ordered file closure including certificate."""
    rows = [*document["outputs"], certificate]
    if len({reader._key(reader._identity(row)["path"]) for row in rows}) != len(rows):
        raise ValueError("RAW copy closure has missing or duplicate original paths")
    return deepcopy(rows)


def observe_source(intent_binding, *, aliases=(), folder=None):
    """Run the genuine original-owner reader and recheck on the source host.

    Root supplies the preexisting immutable intent. The returned observation
    is factual data for a later Root copy release, never a replacement for it.
    """
    pins = reader._Pins(aliases, folder=folder)
    intent = _intent(pins, intent_binding)
    if not reader._same(host_identity(), intent["source_host"]):
        raise ValueError("Source observation is running on another machine/boot")
    checked = reader.read_shape_basis(intent["certificate"], run_id=intent["run_id"],
        authority=intent["basis_authority"], parent_workspace=intent["parent_workspace"],
        aliases=aliases, folder=folder)
    ownership = checked.document["output_ownership"]
    path = _metadata_destination(intent["source_observation"], ownership["authority"]["destination"])
    checked.revalidate()
    pins.final()
    observed = dict(format=OBSERVATION, copy_id=intent["copy_id"], intent=deepcopy(intent_binding),
        observed_utc=datetime.now(timezone.utc).isoformat(), observer=_observer(),
        certificate=deepcopy(intent["certificate"]), basis_authority=deepcopy(intent["basis_authority"]),
        parent_workspace=deepcopy(intent["parent_workspace"]), ownership=deepcopy(ownership),
        files=_files(checked.document, intent["certificate"]),
        consumed_sources=deepcopy(list(checked.guards.rows.values())),
        copyable_sources=deepcopy(list(checked.guards.copyable.values())))
    if reader._time(observed["observed_utc"], "Source observation") < reader._time(intent["issued_utc"], "Root intent"):
        raise ValueError("Root intent was not issued before source observation")
    return _write(path, observed)


def _source_observation(pins, intent_binding, intent, binding, document):
    """Join original ownership and files to the exact source-side observation."""
    if Path(binding["path"]) != Path(intent["source_observation"]):
        raise ValueError("The source observation differs from Root's named path")
    value = pins.add(binding, metadata=True, copyable=True)
    reader._keys(value, {"format", "copy_id", "intent", "observed_utc", "observer", "certificate",
        "basis_authority", "parent_workspace", "ownership", "files", "consumed_sources",
        "copyable_sources"}, "RAW copy source observation")
    if (value["format"] != OBSERVATION or value["copy_id"] != intent["copy_id"]
            or not reader._same(value["intent"], intent_binding)
            or not reader._same(value["certificate"], intent["certificate"])
            or not reader._same(value["basis_authority"], intent["basis_authority"])
            or not reader._same(value["parent_workspace"], intent["parent_workspace"])
            or not reader._same(value["ownership"], document["output_ownership"])
            or not reader._same(value["files"], _files(document, intent["certificate"]))):
        raise ValueError("Source observation differs from its genuine original closure")
    _process(value["observer"], intent["source_host"])
    observed = reader._time(value["observed_utc"], "Source observation")
    if (observed < reader._time(intent["issued_utc"], "Root intent")
            or observed < reader._time(document["producer"]["finished_utc"], "Producer finish")
            or value['observer']['process_created_epoch_s'] > observed.timestamp()):
        raise ValueError("The copy observation precedes its intent or completed producer")
    if not isinstance(value["consumed_sources"], list) or not value["consumed_sources"]:
        raise ValueError("The source observation has no actual consumed source closure")
    for row in value["consumed_sources"]:
        reader._keys(row, {"original", "physical"}, "Source-observed consumed file")
        left, right = reader._identity(row["original"]), reader._identity(row["physical"])
        if any(left[key] != right[key] for key in ("sha256", "size_bytes")):
            raise ValueError("Source-observed physical bytes differ from their original identity")
    _pool_sources(intent, value)
    return value


def _pool_sources(intent, observed):
    """Whitelist only genuinely copyable original roles plus copy-helper sources."""
    consumed = {}
    for row in observed['consumed_sources']:
        original = reader._identity(row['original'])
        key = reader._key(original['path'])
        if key in consumed:
            raise ValueError('Source observation repeats a consumed original identity')
        consumed[key] = original
    if not isinstance(observed['copyable_sources'], list) or not observed['copyable_sources']:
        raise ValueError('Source observation has no explicit copyable source roles')
    allowed = {}
    for row in observed['copyable_sources']:
        original = reader._identity(row)
        key = reader._key(original['path'])
        if key in allowed or not reader._same(consumed.get(key), original):
            raise ValueError('Copyable source role is duplicate or outside the observed closure')
        allowed[key] = original
    for row in intent['copy_source_closure']['files']:
        original = reader._identity(row)
        key = reader._key(original['path'])
        if key in consumed and key not in allowed:
            raise ValueError('Copy-helper sources cannot promote an original physical-only role')
        if key in allowed and not reader._same(allowed[key], original):
            raise ValueError('Copy-helper source has a conflicting original identity')
        allowed[key] = original
    return list(allowed.values())


def _retained_sources(pins, binding, intent_binding, intent, observed_binding, observed):
    """Install a receipted pool for initial callers without changing original roles.

    The Root release and actual receiver receipt bind this complete manifest.
    Existing preparation copies may retain the same originals later; otherwise
    only this pool's confined relative copies are read on the receiving host.
    RAW acquisitions, original parent NPZ and producer outputs stay physical.
    """
    pool = pins.add(binding, metadata=True, copyable=True)
    reader._keys(pool, {'format', 'copy_id', 'intent', 'source_observation', 'folder', 'files'},
                 'Retained RAW-basis source pool')
    if not isinstance(pool['folder'], str):
        raise ValueError('Retained source pool needs its explicit absolute folder')
    root = Path(pool['folder'])
    if (pool['format'] != POOL or pool['copy_id'] != intent['copy_id']
            or not reader._same(pool['intent'], intent_binding)
            or not reader._same(pool['source_observation'], observed_binding)
            or not root.is_absolute() or '..' in root.parts
            or root != Path(binding['path']).parent
            or not isinstance(pool['files'], list)):
        raise ValueError('Retained source pool differs from its exact copy intent/observation')
    expected = _pool_sources(intent, observed)
    if len(pool['files']) != len(expected):
        raise ValueError('Retained source pool lacks its exact allowed copyable roles')
    for row, original in zip(pool['files'], expected):
        reader._keys(row, {'original', 'copy'}, 'Retained source pool file')
        copied = row['copy']
        if (not reader._same(reader._identity(row['original']), original)
                or not isinstance(copied, dict) or set(copied) != reader._IDENTITY_KEYS
                or not isinstance(copied['path'], str) or not Path(copied['path']).parts
                or Path(copied['path']).is_absolute() or '..' in Path(copied['path']).parts
                or any(copied[k] != original[k] for k in ('sha256', 'size_bytes'))):
            raise ValueError('Retained source pool changes an allowed original role or bytes')
        key = reader._key(original['path'])
        prior = pins.aliases.get(key)
        if prior is not None:
            if not reader._same(prior[0], original):
                raise ValueError('Preparation and receiver source pool disagree on original identity')
        else:
            checked = reader._Pins([row], folder=root)
            pins.aliases[key] = checked.aliases[key]
        pins.add(original, copyable=True)
    return pool


def receive_staged_basis(intent_binding, source_observation, staged_files, *, retained_sources):
    """Exclusively copy a complete staged byte closure on the intended receiver.

    Staging transport is external to this function. Its exact full list and
    source-observed retained metadata/runtime pool are verified here. No partial
    destination is reused or removed on failure.
    Only a later externally supplied Root release can admit these bytes.
    """
    from .rig_raw_build import _descriptor
    from .rig_raw_write import _canonical, _identity as filesystem_identity
    from .pipeline.products_publication import _no_links
    from .job_filesystem import inspect_local_file
    pins = reader._Pins()
    intent = _intent(pins, intent_binding, verify_sources=False)
    if not reader._same(host_identity(), intent["receiver_host"]):
        raise ValueError("Copy reception is running on another machine/boot")
    if not isinstance(staged_files, list) or not staged_files:
        raise ValueError("The receiver needs every exact staged original file")
    staged = {}
    for row in staged_files:
        reader._keys(row, {"original", "staged"}, "Staged RAW basis file")
        original, physical = reader._identity(row["original"]), reader._identity(row["staged"])
        key = reader._key(original["path"])
        if key in staged or any(original[name] != physical[name] for name in ("sha256", "size_bytes")):
            raise ValueError("Staged RAW basis files are duplicate or change original bytes")
        pins.add(physical)
        staged[key] = original, physical
    certificate = intent["certificate"]
    try:
        staged_certificate, physical_certificate = staged[reader._key(certificate["path"])]
    except KeyError as error:
        raise ValueError("Staged RAW basis lacks its original certificate") from error
    if not reader._same(staged_certificate, certificate):
        raise ValueError('Staged certificate differs from the exact Root intent')
    document = pins.add(physical_certificate, metadata=True)
    observation = _source_observation(pins, intent_binding, intent, source_observation, document)
    _retained_sources(pins, retained_sources, intent_binding, intent, source_observation, observation)
    _intent(pins, intent_binding)
    expected = _files(document, certificate)
    if {reader._key(row["path"]) for row in expected} != set(staged):
        raise ValueError("Staged RAW basis membership differs from the closed producer")
    if any(not reader._same(row, staged[reader._key(row['path'])][0]) for row in expected):
        raise ValueError('Staged output identity differs from the source-observed closure')
    authority = document["output_ownership"]["authority"]
    destination, root = _canonical(authority["destination"]), _canonical(authority["intended_root"])
    if (Path(certificate["path"]).parent != destination or destination == root
            or not destination.is_relative_to(root) or not root.is_dir() or destination.exists()
            or not inspect_local_file(destination / '.raw-copy-location').allowed):
        raise ValueError("The received basis needs its fresh same-path positive local destination")
    for protected in authority["forbidden_raw_roots"]:
        other = _canonical(protected)
        if destination == other or destination.is_relative_to(other) or other.is_relative_to(destination):
            raise ValueError("The received basis intersects original RAW acquisition data")
    if (any(row["size_bytes"] > authority["maximum_file_bytes"] for row in expected)
            or sum(row["size_bytes"] for row in expected) > authority["maximum_output_bytes"]):
        raise ValueError("The received basis exceeds its unchanged original byte ceilings")
    receipt_path = _metadata_destination(intent["receiver_receipt"], destination)
    root_identity = filesystem_identity(root)
    destination.mkdir()
    directory_identity = filesystem_identity(destination)
    copied_total = 0
    for original in expected:
        relative = Path(original["path"]).relative_to(destination)
        if not relative.parts or '..' in relative.parts:
            raise ValueError("A copied basis output leaves its original destination")
        target = destination / relative
        _no_links(target)
        if (filesystem_identity(root) != root_identity or filesystem_identity(destination) != directory_identity):
            raise ValueError("Receiver ownership changed during exact copying")
        target.parent.mkdir(parents=True, exist_ok=True)
        physical = staged[reader._key(original["path"])][1]
        digest = hashlib.sha256()
        copied_file = 0
        with Path(physical["path"]).open('rb') as incoming, target.open('xb') as outgoing:
            for block in iter(lambda: incoming.read(1024**2), b''):
                next_file, next_total = copied_file + len(block), copied_total + len(block)
                if (next_file > original['size_bytes']
                        or next_file > authority['maximum_file_bytes']
                        or next_total > authority['maximum_output_bytes']):
                    raise ValueError('Staged RAW basis grew beyond its declared copy byte ceiling; partial destination is retained')
                digest.update(block)
                outgoing.write(block)
                copied_file, copied_total = next_file, next_total
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if digest.hexdigest() != original["sha256"] or _descriptor(target) != original:
            raise ValueError("Received output bytes differ; partial destination is retained")
    pins.final()
    actual = set()
    for path in destination.rglob('*'):
        _no_links(path)
        if path.is_file():
            actual.add(path)
    if actual != {Path(row['path']) for row in expected}:
        raise ValueError('Received output membership changed before its receipt')
    for row in expected:
        if _descriptor(row['path']) != row:
            raise ValueError('Received output changed before its receipt')
    receipt = dict(format=RECEIPT, copy_id=intent["copy_id"], intent=deepcopy(intent_binding),
        source_observation=deepcopy(source_observation), retained_sources=deepcopy(retained_sources), received_utc=datetime.now(timezone.utc).isoformat(),
        observer=_observer(), certificate=deepcopy(certificate), files=expected,
        destination=str(destination), intended_root=str(root),
        directory_identity=list(directory_identity), intended_root_identity=list(root_identity),
        original_directory_identity=deepcopy(observation['ownership']['directory_identity']),
        original_intended_root_identity=deepcopy(observation['ownership']['intended_root_identity']))
    _receiver(pins, intent_binding, intent, source_observation, observation, receipt, document, retained_sources)
    return _write(receipt_path, receipt)


def _receiver(pins, intent_binding, intent, observed_binding, observed, receipt, document, retained_sources):
    """Verify actual receiver ownership independently of original sender IDs."""
    from .rig_raw_write import _canonical, _identity as filesystem_identity
    reader._keys(receipt, {"format", "copy_id", "intent", "source_observation", "received_utc", "observer",
        "certificate", "files", "retained_sources", "destination", "intended_root", "directory_identity", "intended_root_identity",
        "original_directory_identity", "original_intended_root_identity"}, "Received RAW basis receipt")
    owner = document['output_ownership']
    for key in ('directory_identity', 'intended_root_identity',
                'original_directory_identity', 'original_intended_root_identity'):
        if (not isinstance(receipt[key], list) or len(receipt[key]) != 2
                or any(type(value) is not int for value in receipt[key])):
            raise ValueError('Received ownership needs exact native integer identities')
    if (receipt['format'] != RECEIPT or receipt['copy_id'] != intent['copy_id']
            or not reader._same(receipt['intent'], intent_binding)
            or not reader._same(receipt['source_observation'], observed_binding)
            or not reader._same(receipt['retained_sources'], retained_sources)
            or not reader._same(receipt['certificate'], intent['certificate'])
            or not reader._same(receipt['files'], _files(document, intent['certificate']))
            or receipt['destination'] != owner['authority']['destination']
            or receipt['intended_root'] != owner['authority']['intended_root']
            or not reader._same(receipt['original_directory_identity'], owner['directory_identity'])
            or not reader._same(receipt['original_intended_root_identity'], owner['intended_root_identity'])):
        raise ValueError("Received basis receipt changes original ownership or file identities")
    _process(receipt['observer'], intent['receiver_host'])
    if (host_identity()['hostname'] != intent['receiver_host']['hostname']
            or reader._time(receipt['received_utc'], 'Basis reception') < reader._time(observed['observed_utc'], 'Source observation')
            or receipt['observer']['process_created_epoch_s'] > reader._time(receipt['received_utc'], 'Basis reception').timestamp()
            or list(filesystem_identity(_canonical(receipt['destination']))) != receipt['directory_identity']
            or list(filesystem_identity(_canonical(receipt['intended_root']))) != receipt['intended_root_identity']):
        raise ValueError("Received RAW basis host, chronology or current filesystem identity differs")


@dataclass
class ReceivedShapeBasis(reader.ReadShapeBasis):
    """Original scientific proof with a separately bound actual receiver owner."""
    receipt: dict | None = None

    def _current_owner(self):
        """Check actual receiver IDs and exact membership around long file hashing."""
        from .rig_raw_write import _canonical, _identity as filesystem_identity
        from .pipeline.products_publication import _no_links
        receipt = self.receipt
        destination = _canonical(receipt['destination'])
        if (host_identity()['hostname'] != receipt['observer']['host']['hostname']
                or list(filesystem_identity(destination)) != receipt['directory_identity']
                or list(filesystem_identity(_canonical(receipt['intended_root']))) != receipt['intended_root_identity']):
            raise ValueError("Received RAW-basis current ownership changed")
        expected = {Path(row['path']) for row in receipt['files']}
        actual = set()
        for path in destination.rglob('*'):
            _no_links(path)
            if path.is_file():
                actual.add(path)
        if actual != expected:
            raise ValueError("Received RAW-basis output membership changed")

    def revalidate(self):
        """Bracket all final source hashes with actual receiver ownership checks."""
        self._current_owner()
        self.guards.final()
        self._current_owner()


def _physical_lineage_inputs(pins, document, parent_workspace, run_id, request, proof):
    """Keep the strict lineage loader's exact current inputs physically original.

    Only an actually unmatched parent selects the existing lineage fallback.
    Its accepted/current TOMLs and series manifest are read again by their
    literal metadata loaders, so retained copies cannot stand in for them.
    Other explicitly copyable sources keep the existing receipted pool route.
    """
    revision = pins.add(parent_workspace, metadata=True, copyable=True)
    record = (revision['configuration']['frame_binding']['runs'] or {}).get(run_id) or {}
    configuration = reader._identity(
        {key: record.get('configuration', {}).get(key) for key in reader._IDENTITY_KEYS})
    if any(reader._same(configuration, row) for row in document['inputs']['configuration_files']):
        return
    from .raw_shape_configuration_lineage import _PROOF_KEYS
    reader._keys(proof, _PROOF_KEYS, 'Shape configuration lineage')
    accepted = proof['accepted_configurations']
    if not isinstance(accepted, list) or len(accepted) != 2:
        raise ValueError('Shape lineage requires the original ordered accepted run and merged files')
    for binding in [*accepted, *request['configuration_files'], request['series_context']['manifest']]:
        original = reader._identity(binding)
        key = reader._key(original['path'])
        alias = pins.aliases.get(key)
        if alias is not None and not reader._same(alias[0], original):
            raise ValueError('Received lineage input differs from its exact original binding')
        if key in pins.metadata_replay_keys:
            # An explicit historical replay routes the unchanged typed loader
            # through these verified bytes, rather than the live TOML path.
            pins.add(original, copyable=True)
            continue
        pins.aliases.pop(key, None)
        pins.add(original, copyable=True)


def read_received_basis(certificate, *, run_id, authority, parent_workspace,
                        reference_image=None, camera_profile=None, aliases=(), folder=None,
                        parent_configuration_lineage=None):
    """Replay a genuine Root-released copy through all original scientific joins."""
    original_authority = {key: deepcopy(value) for key, value in authority.items() if key != 'received_basis'}
    pins = _ReceivedPins(aliases, folder=folder)
    document = pins.add(certificate, metadata=True)
    released = pins.add(authority['received_basis'], metadata=True, copyable=True)
    reader._keys(released, {'format', 'intent', 'source_observation', 'receiver_receipt',
        'released_utc', 'certificate', 'original_authority', 'retained_sources'}, 'Root received-basis release')
    if (released['format'] != RELEASE or not reader._same(released['certificate'], certificate)
            or not reader._same(released['original_authority'], original_authority)):
        raise ValueError('Root copy release differs from this exact basis and original authority')
    intent_binding = released['intent']
    intent = _intent(pins, intent_binding, verify_sources=False)
    if (intent['run_id'] != run_id or not reader._same(intent['certificate'], certificate)
            or not reader._same(intent['basis_authority'], original_authority)
            or not reader._same(intent['parent_workspace'], parent_workspace)):
        raise ValueError('Root received-basis intent differs from its current exact inputs')
    observed = _source_observation(pins, intent_binding, intent, released['source_observation'], document)
    _retained_sources(pins, released['retained_sources'], intent_binding, intent, released['source_observation'], observed)
    _intent(pins, intent_binding)
    if Path(released['receiver_receipt']['path']) != Path(intent['receiver_receipt']):
        raise ValueError('Root copy release names another receiver receipt')
    receipt = pins.add(released['receiver_receipt'], metadata=True, copyable=True)
    _receiver(pins, intent_binding, intent, released['source_observation'], observed, receipt, document, released['retained_sources'])
    if reader._time(released['released_utc'], 'Root copy release') < reader._time(receipt['received_utc'], 'Basis reception'):
        raise ValueError('Root copy release predates actual reception')
    if (document.get('format') != reader.FORMAT or document.get('artifact_kind') != 'raw_shape_basis'
            or document.get('producer_role') != 'raw_shape_basis' or document.get('run_id') != run_id
            or document.get('run') != run_id or document.get('complete_rig_model') is not False
            or document.get('production_qualification_claimed') is not False
            or document.get('certificate_scope') != 'baseline and ordered shape chain to camera-U registration only'):
        raise ValueError('The received copy has no genuine explicit RAW shape-basis role/run')
    pins.core_rows = {reader._key(certificate['path']): reader._identity(certificate)}
    pins.core_copyable = {}
    started = reader._time(document['producer']['started_utc'], 'Shape-basis start')
    reader._authority(pins, original_authority, document['inputs']['operation_request'], document['renderer_implementation'], started)
    parent_configuration_lineage = reader._released_parent_lineage(
        pins, original_authority, parent_configuration_lineage)
    request = reader._request(pins, document, run_id)
    destination, _ = _received_outputs(pins, document, request, certificate, receipt)
    parameters = reader._camera(pins, document, request, run_id, reference_image, camera_profile)
    reader._runtime(pins, document)
    reader._saved_payload(pins, document, destination, parameters)
    if document['shape_basis_consistency'] != dict(required=False, basis_certificate=None, byte_exact_verified=False,
            mismatch_policy='STOP; no automatic placement or adoption changes'):
        raise ValueError('Received basis claims another full-build consistency role')
    if parent_configuration_lineage is None:
        labels, compatibility = reader._parent_grid(pins, document, parent_workspace, run_id)
    else:
        _physical_lineage_inputs(pins, document, parent_workspace, run_id, request,
                                 parent_configuration_lineage)
        labels, compatibility = reader._parent_grid(
            pins, document, parent_workspace, run_id, request=request,
            parent_configuration_lineage=parent_configuration_lineage)
    expected_core = {}
    for row in observed['consumed_sources']:
        original = reader._identity(row['original'])
        key = reader._key(original['path'])
        if key in expected_core:
            raise ValueError('Source observation repeats a consumed original identity')
        expected_core[key] = original
    if not reader._same(pins.core_rows, expected_core):
        raise ValueError('Received scientific replay differs from the source-observed full closure')
    expected_copyable = {reader._key(row['path']): reader._identity(row) for row in observed['copyable_sources']}
    if not reader._same(pins.core_copyable, expected_copyable):
        raise ValueError('Received source roles differ from the source-observed copyable closure')
    pins.core_rows = pins.core_copyable = None
    result = ReceivedShapeBasis(deepcopy(certificate), document, pins, labels, compatibility, deepcopy(receipt))
    result.revalidate()
    return result


def _received_outputs(pins, document, request, certificate, receipt):
    """Verify unchanged historical joins against the separately observed receiver."""
    from .rig_raw_write import _canonical, _identity as filesystem_identity
    from .job_filesystem import inspect_local_file
    owner = document["output_ownership"]
    _keys(owner, {"operation_id", "request", "authority", "directory_identity", "intended_root_identity",
                  "owner", "setup_started", "save_started", "save_completed"}, "Shape-basis ownership")
    authority = request["output"]
    _keys(authority, {"destination", "intended_root", "forbidden_raw_roots", "maximum_output_bytes", "maximum_file_bytes"}, "Shape-basis output authority")
    if any(type(authority[k]) is not int or authority[k] <= 0 for k in ("maximum_output_bytes", "maximum_file_bytes")) or authority["maximum_file_bytes"] > authority["maximum_output_bytes"]:
        raise ValueError("Shape-basis byte ceilings are invalid")
    destination, root = _canonical(authority["destination"]), _canonical(authority["intended_root"])
    for key in ("directory_identity", "intended_root_identity"):
        if (not isinstance(owner[key], list) or len(owner[key]) != 2
                or any(type(value) is not int for value in owner[key])):
            raise ValueError("Shape-basis filesystem ownership needs exact native integer identities")
    if (destination == root or not destination.is_relative_to(root) or Path(certificate["path"]).parent != destination
            or not _same(owner["authority"], authority) or not _same(owner["request"], document["inputs"]["operation_request"])
            or owner["operation_id"] != document["producer"]["operation_id"]
            or re.fullmatch(r"[0-9a-f]{32}", str(owner["operation_id"])) is None
            or list(filesystem_identity(destination)) != receipt["directory_identity"]
            or list(filesystem_identity(root)) != receipt["intended_root_identity"]):
        raise ValueError("Shape-basis request, operation or current filesystem ownership differs")
    if not inspect_local_file(destination / ".raw-shape-reader-location").allowed:
        raise ValueError("Shape-basis output lacks a positive local filesystem identity")
    protected = authority["forbidden_raw_roots"]
    if not isinstance(protected, list) or not protected:
        raise ValueError("Shape-basis output lacks its protected acquisition roots")
    for row in protected:
        other = _canonical(row)
        if destination == other or destination.is_relative_to(other) or other.is_relative_to(destination):
            raise ValueError("Shape-basis output intersects a protected RAW root")
    if not any(_canonical(document["inputs"]["raw"]["path"]).is_relative_to(_canonical(p)) for p in protected):
        raise ValueError("Shape-basis RAW acquisition is not protected by its request")
    files = document["outputs"]
    if not isinstance(files, list) or not files or len({_key(_identity(x)["path"]) for x in files}) != len(files):
        raise ValueError("Shape-basis owned output closure is missing or duplicate")
    paths = []
    for row in files:
        path = _canonical(row["path"])
        if not path.is_relative_to(destination) or path == destination or path == Path(certificate["path"]):
            raise ValueError("Shape-basis output leaves its owned operation")
        pins.add(row)
        paths.append(path)
    from .pipeline.products_publication import _no_links
    actual = []
    for path in destination.rglob("*"):
        _no_links(path)
        if path.is_file() and path != Path(certificate["path"]):
            actual.append(path)
    if set(actual) != set(paths):
        raise ValueError("Shape-basis owned output membership changed")
    if any(x["size_bytes"] > authority["maximum_file_bytes"] for x in [*files, certificate]) or sum(x["size_bytes"] for x in [*files, certificate]) > authority["maximum_output_bytes"]:
        raise ValueError("Shape-basis retained logical bytes exceed its request ceilings")
    for name in ("owner", "setup_started", "save_started", "save_completed"):
        if owner[name] not in files:
            raise ValueError("Shape-basis ownership marker is outside the original output closure")
    marker = pins.add(owner["owner"], metadata=True)
    if (marker.get("format") != "darsia-owned-raw-output-v1" or marker.get("operation_id") != owner["operation_id"]
            or not _same(marker.get("request"), owner["request"]) or marker.get("destination") != str(destination)
            or marker.get("intended_root") != str(root) or marker.get("directory_identity") != owner["directory_identity"]
            or not isinstance(marker.get("local_filesystem"), dict)):
        raise ValueError("Shape-basis owner marker belongs to another operation")
    for name in ("setup_started", "save_started"):
        if pins.add(owner[name], metadata=True) != dict(operation_id=owner["operation_id"]):
            raise ValueError("Shape-basis setup/save marker differs")
    finished = pins.add(owner["save_completed"], metadata=True)
    if (finished.get("format") != "darsia-owned-raw-save-completed-v1"
            or finished.get("operation_id") != owner["operation_id"] or finished.get("destination") != str(destination)
            or not _same(finished.get("request"), owner["request"])
            or finished.get("directory_identity") != owner["directory_identity"]
            or finished.get("no_clobber_save") is not True or finished.get("complete") is not True):
        raise ValueError("Shape-basis original save did not complete under the same ownership")
    metadata = [x for x in files if Path(x["path"]) == destination / "meta_data.json"]
    if len(metadata) != 1:
        raise ValueError("Shape-basis lacks its original saved operation metadata")
    saved = pins.add(metadata[0], metadata=True)
    if (saved.get("raw_build_id") != owner["operation_id"] or not _same(saved.get("raw_camera"), document["inputs"]["camera_binding"])
            or saved.get("raw_baseline_sha256") != document["inputs"]["raw"]["sha256"]
            or saved.get("baseline_path") != document["inputs"]["raw"]["path"]):
        raise ValueError("Shape-basis saved metadata has another RAW, camera or operation")
    return destination, authority
