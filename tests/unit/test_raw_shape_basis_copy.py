"""Real owned-file copy/replay with synthetic RAW/runtime/host fixture inputs.

The existing 2x3 producer fixture performs actual owned saves and reader checks.
Host labels are synthetic; these cases do not claim actual cross-host/native
binary qualification. No producer, payload, authority or ownership check is
stubbed by this file.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from darsia.fluidflower import raw_shape_basis_copy as copies
from darsia.fluidflower import raw_shape_basis_reader as reader
from .test_raw_shape_basis_reader import basis, _descriptor, _json, _changed_byte  # noqa: F401
from .test_raw_shape_configuration_lineage import synthetic_lineage  # noqa: F401

pytestmark = pytest.mark.parametrize('basis', ['owner-release'], indirect=True)


@pytest.fixture
def transport(basis, tmp_path, monkeypatch):
    """Observe the real original owner, then simulate a fresh same-path host."""
    return _transport_owned_basis(basis, tmp_path, monkeypatch)


@pytest.fixture
def lineage_transport(basis, synthetic_lineage, tmp_path, monkeypatch):
    """Own a second real 2x3 basis against the complete original lineage fixture.

    Only the inherited RAW/registration/runtime fixture suppliers are synthetic.
    The producer saves, five-key lineage validator, original typed metadata
    loaders, four original NPZ checks and source/copy readers are not replaced.
    """
    from darsia.fluidflower import rig_raw_build as build
    case = synthetic_lineage
    raw = Path(case['request']['baseline']['path'])
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b'unit-only RAW for the inherited synthetic decoder')
    output_root = tmp_path / 'lineage-products'
    output_root.mkdir()
    destination = output_root / 'new_basis'
    camera = deepcopy(basis.document['inputs']['camera_binding'])
    operation = dict(deepcopy(case['request']), format=build.BASIS_REQUEST_FORMAT,
        baseline=_descriptor(raw), profile=_descriptor(basis.profile),
        operation_overrides={'data.raw_camera': camera},
        output=dict(destination=str(destination), intended_root=str(output_root),
                    forbidden_raw_roots=[str(raw.parent)], maximum_output_bytes=10_000_000,
                    maximum_file_bytes=2_000_000), shape_basis_certificate=None)
    operation_binding = _json(tmp_path / 'lineage-operation.json', operation)
    proof_binding = _json(tmp_path / 'parent-lineage-proof.json', case['proof'])
    source_closure = deepcopy(basis.authority['source_closure'])
    released = dict(source_closure=source_closure, operation_request=operation_binding,
        released_utc=datetime.now(timezone.utc).isoformat(), parent_configuration_lineage=proof_binding)
    authority = dict(source_closure=source_closure, operation_request=operation_binding,
        root_release=_json(tmp_path / 'lineage-producer-release.json', released))
    context = build.begin_raw_rig_build(case['actual_path'], raw, camera, destination,
        configuration_sources=operation['configuration_files'], request_binding=operation_binding,
        artifact_kind='raw_shape_basis')
    rig = SimpleNamespace(**vars(basis.rig))
    rig._raw_build_id = context.operation_id
    rig.raw_baseline_sha256 = operation['baseline']['sha256']
    rig.baseline_path = raw
    context.begin_setup(rig, destination / 'figures')
    context.output.begin_save(rig, destination)
    context.output.image(rig.shape_corrected_baseline, destination / 'shape_corrected_baseline.npz')
    context.output.correction(rig.shape_corrections[0], destination / 'shape_correction_0_typecorrection.npz')
    with context.output.open(destination / 'meta_data.json') as stream:
        stream.write(json.dumps(dict(baseline_path=str(raw), raw_camera=camera,
            raw_baseline_sha256=rig.raw_baseline_sha256, raw_build_id=context.operation_id)).encode())
    context.output.finish_save()
    certificate = context.publish(rig)
    document = json.loads(Path(certificate['path']).read_bytes())
    parent = deepcopy(basis.parent)
    parent['configuration']['frame_binding']['runs']['sample']['configuration'] = deepcopy(case['parent'])
    parent_workspace = _json(tmp_path / 'lineage-parent-workspace.json', parent)
    new_basis = SimpleNamespace(certificate=certificate, document=document, authority=authority,
        parent_workspace=parent_workspace, parent=parent, context=context,
        source=raw, profile=basis.profile)
    value = _transport_owned_basis(new_basis, tmp_path, monkeypatch)
    value.lineage = case
    value.lineage_proof = proof_binding
    return value


def _transport_owned_basis(basis, tmp_path, monkeypatch):
    """Retain real owned bytes and source pool, with synthetic explicit host facts."""
    source_host = dict(hostname='synthetic-source', boot_epoch_s=1)
    receiver_host = dict(hostname='synthetic-receiver', boot_epoch_s=2)
    current = [source_host]
    monkeypatch.setattr(copies, 'host_identity', lambda: deepcopy(current[0]))
    code = [_descriptor(Path(module.__file__).resolve()) for module in (copies, reader)]
    intent = dict(format=copies.INTENT, copy_id='4' * 32,
        issued_utc=datetime.now(timezone.utc).isoformat(), run_id=basis.document['run_id'],
        certificate=deepcopy(basis.certificate), basis_authority=deepcopy(basis.authority),
        parent_workspace=deepcopy(basis.parent_workspace), source_host=source_host,
        receiver_host=receiver_host, source_observation=str(tmp_path / 'source-observation.json'),
        receiver_receipt=str(tmp_path / 'receiver-receipt.json'),
        copy_source_closure=dict(base_commit='5' * 40, tree='6' * 40, files=code))
    intent_binding = _json(tmp_path / 'root-copy-intent.json', intent)
    observed_binding = copies.observe_source(intent_binding)
    observed = json.loads(Path(observed_binding['path']).read_bytes())
    assert observed['ownership'] == basis.document['output_ownership']
    assert observed['observer']['host'] == source_host
    pool_root = tmp_path / 'retained-source-pool'
    pool_root.mkdir()
    originals = {reader._key(row['path']): deepcopy(row) for row in observed['copyable_sources']}
    for row in intent['copy_source_closure']['files']:
        originals[reader._key(row['path'])] = deepcopy(row)
    pool_files = []
    for index, original in enumerate(originals.values()):
        target = pool_root / f'{index:04d}.bin'
        shutil.copyfile(original['path'], target)
        copied = dict(_descriptor(target), path=target.name)
        pool_files.append(dict(original=original, copy=copied))
    pool_document = dict(format=copies.POOL, copy_id=intent['copy_id'], intent=intent_binding,
        source_observation=observed_binding, folder=str(pool_root), files=pool_files)
    pool_binding = _json(pool_root / 'retained-sources.json', pool_document)
    stage = tmp_path / 'transport-stage'
    stage.mkdir()
    staged_files = []
    for index, original in enumerate(observed['files']):
        target = stage / f'{index:04d}.bin'
        shutil.copyfile(original['path'], target)
        staged_files.append(dict(original=deepcopy(original), staged=_descriptor(target)))
    original_root = basis.context.destination.parent
    retained = tmp_path / 'retained-original-owner'
    original_root.rename(retained)
    original_root.mkdir()
    current[0] = receiver_host
    return SimpleNamespace(basis=basis, intent=intent, intent_binding=intent_binding,
        observed=observed, observed_binding=observed_binding, stage=stage,
        staged_files=staged_files, current=current, source_host=source_host,
        receiver_host=receiver_host, retained=retained, pool=pool_document,
        pool_binding=pool_binding, pool_root=pool_root)


def _receive(value):
    """Create only the receiver receipt; Root release remains separate fixture DATA."""
    return copies.receive_staged_basis(value.intent_binding, value.observed_binding, value.staged_files,
                                      retained_sources=value.pool_binding)


def _release(value, receipt):
    """Supply an explicit synthetic Root release after actual fixture reception."""
    document = dict(format=copies.RELEASE, intent=deepcopy(value.intent_binding),
        source_observation=deepcopy(value.observed_binding), receiver_receipt=deepcopy(receipt),
        released_utc=datetime.now(timezone.utc).isoformat(), certificate=deepcopy(value.basis.certificate),
        original_authority=deepcopy(value.basis.authority), retained_sources=deepcopy(value.pool_binding))
    binding = _json(Path(receipt['path']).parent / 'root-copy-release.json', document)
    return dict(deepcopy(value.basis.authority), received_basis=binding), document, binding


def _read(value, authority):
    """Use the public reader so every unchanged pixel/grid/source join runs."""
    return reader.read_shape_basis(value.basis.certificate, run_id='sample',
        authority=authority, parent_workspace=value.basis.parent_workspace)


def test_received_lineage_keeps_real_loader_inputs_and_bound_runtime_pool(lineage_transport, monkeypatch):
    """Real lineage replay reads original TOMLs while producer wrappers use copies."""
    f = lineage_transport
    blocked = {reader._key(row['path']) for row in f.basis.authority['source_closure']['files']}
    blocked.add(reader._key(f.lineage_proof['path']))
    actual_open = Path.open
    def receiver_only(path, *args, **kwargs):
        """Refuse unavailable producer/proof originals, not shared loader inputs."""
        if reader._key(path) in blocked:
            raise FileNotFoundError('Producer locator is unavailable on the receiver')
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', receiver_only)
    authority, _, _ = _release(f, _receive(f))
    checked = _read(f, authority)
    audit = checked.compatibility['configuration_lineage']
    assert audit['parent_configuration'] == f.lineage['parent']
    assert audit['consumed_configuration_files'] == f.lineage['request']['configuration_files']
    assert audit['injection_revision'][0]['changed_cells'] == 1
    assert audit['typed_consumption']['typed_protocol_differences'][0]['field'] == 'blacklist'
    physical = [*f.lineage['proof']['accepted_configurations'],
                *f.lineage['request']['configuration_files'], f.lineage['request']['series_context']['manifest']]
    for binding in physical:
        key = reader._key(binding['path'])
        assert key not in checked.guards.aliases
        assert checked.guards.rows[key]['physical'] == binding
    assert checked.guards.rows[reader._key(f.lineage_proof['path'])]['physical']['path'] != f.lineage_proof['path']
    checked.revalidate()


def test_received_lineage_refuses_explicit_override_of_released_proof(lineage_transport):
    """The public received dispatch preserves the strict release-proof equality."""
    f = lineage_transport
    authority, _, _ = _release(f, _receive(f))
    with pytest.raises(ValueError, match='differs from the bound release'):
        reader.read_shape_basis(f.basis.certificate, run_id='sample', authority=authority,
            parent_workspace=f.basis.parent_workspace, parent_configuration_lineage={'unbound': 'override'})


@pytest.mark.parametrize('part', ['accepted', 'current', 'series'])
def test_received_lineage_late_original_change_cannot_hide_behind_valid_pool(lineage_transport, part):
    """Final guards rehash actual typed-loader files despite retained identical copies."""
    f = lineage_transport
    authority, _, _ = _release(f, _receive(f))
    checked = _read(f, authority)
    binding = {'accepted': f.lineage['proof']['accepted_configurations'][0],
               'current': f.lineage['request']['configuration_files'][0],
               'series': f.lineage['request']['series_context']['manifest']}[part]
    key = reader._key(binding['path'])
    copied = next(row for row in f.pool['files'] if reader._key(row['original']['path']) == key)
    assert _descriptor(f.pool_root / copied['copy']['path'])['sha256'] == binding['sha256']
    _changed_byte(Path(binding['path']))
    with pytest.raises(ValueError, match='changed'):
        checked.revalidate()


def test_same_path_new_filesystem_owner_needs_positive_root_copy_release(transport):
    """Original physical ownership refuses a copy; the separate true receipt admits it."""
    f = transport
    receipt = _receive(f)
    received = json.loads(Path(receipt['path']).read_bytes())
    assert received['directory_identity'] != f.observed['ownership']['directory_identity']
    assert received['intended_root_identity'] != f.observed['ownership']['intended_root_identity']
    for original in f.observed['files']:
        assert _descriptor(original['path']) == original
    with pytest.raises(ValueError, match='filesystem ownership differs'):
        _read(f, f.basis.authority)
    authority, _, _ = _release(f, receipt)
    checked = _read(f, authority)
    assert isinstance(checked, copies.ReceivedShapeBasis)
    assert checked.document == f.basis.document
    assert checked.document['output_ownership'] == f.observed['ownership']
    checked.revalidate()


def test_an_observation_or_free_alias_is_not_a_root_copy_release(transport):
    """A factual sidecar cannot serve as the externally bound Root copy release."""
    f = transport
    _receive(f)
    with pytest.raises(ValueError, match='Root received-basis release'):
        _read(f, dict(f.basis.authority, received_basis=f.observed_binding))


@pytest.mark.parametrize('part', ['certificate', 'output'])
def test_staged_tamper_refuses_before_creating_receiver_destination(transport, part):
    """Transport corruption never creates a plausible partial receiver from stale pins."""
    f = transport
    row = f.staged_files[-1 if part == 'certificate' else 0]
    _changed_byte(Path(row['staged']['path']))
    with pytest.raises(ValueError, match='changed'):
        _receive(f)
    assert not f.basis.context.destination.exists()


def test_rehashed_staging_cannot_substitute_another_certificate(transport):
    """Even a current stage hash cannot replace Root's exact original descriptor."""
    f = transport
    row = f.staged_files[-1]
    _changed_byte(Path(row['staged']['path']))
    row['staged'] = _descriptor(row['staged']['path'])
    row['original'] = dict(row['original'], sha256=row['staged']['sha256'])
    with pytest.raises(ValueError, match='certificate differs'):
        _receive(f)
    assert not f.basis.context.destination.exists()


@pytest.mark.parametrize('part', ['original_authority', 'intent', 'receiver_receipt'])
def test_rehashed_wrong_root_copy_record_still_refuses(transport, part):
    """All three independent closure joins are checked after their JSON hash passes."""
    f = transport
    receipt = _receive(f)
    authority, released, binding = _release(f, receipt)
    if part == 'original_authority':
        released['original_authority']['source_closure']['tree'] = '7' * 40
    elif part == 'intent':
        released['intent'] = dict(released['intent'], sha256='0' * 64)
    else:
        received = json.loads(Path(receipt['path']).read_bytes())
        received['copy_id'] = '8' * 32
        released['receiver_receipt'] = _json(Path(receipt['path']), received)
    authority['received_basis'] = _json(Path(binding['path']), released)
    with pytest.raises(ValueError):
        _read(f, authority)


@pytest.mark.parametrize('part', ['copied_output', 'root_copy_release', 'pool_member', 'pool_manifest'])
def test_late_consumed_byte_change_refuses_publication(transport, part):
    """Revalidation streams copied assets and the original bound Root record again."""
    f = transport
    authority, _, binding = _release(f, _receive(f))
    checked = _read(f, authority)
    path = {'root_copy_release': Path(binding['path']),
            'copied_output': Path(f.observed['files'][0]['path']),
            'pool_manifest': Path(f.pool_binding['path']),
            'pool_member': f.pool_root / f.pool['files'][0]['copy']['path']}[part]
    _changed_byte(path)
    with pytest.raises(ValueError, match='changed'):
        checked.revalidate()


def test_late_byte_identical_directory_substitution_is_not_the_receipted_receiver(transport):
    """A second same-path directory has different receiver IDs despite equal file bytes."""
    f = transport
    authority, _, _ = _release(f, _receive(f))
    checked = _read(f, authority)
    path = f.basis.context.destination
    retained_receiver = path.parent / 'retained-receiver'
    path.rename(retained_receiver)
    shutil.copytree(retained_receiver, path)
    with pytest.raises(ValueError, match='ownership changed'):
        checked.revalidate()


def test_receiver_never_reuses_an_existing_destination(transport):
    """Existing data remain byte-identical when the exclusive receiver refuses."""
    f = transport
    f.basis.context.destination.mkdir()
    sentinel = f.basis.context.destination / 'sentinel.txt'
    sentinel.write_bytes(b'preserve existing receiver data')
    with pytest.raises(ValueError, match='fresh same-path'):
        _receive(f)
    assert sentinel.read_bytes() == b'preserve existing receiver data'


def test_receiver_rejects_wrong_actual_machine_or_boot(transport):
    """The receive action observes its own host rather than trusting a supplied label."""
    f = transport
    f.current[0] = f.source_host
    with pytest.raises(ValueError, match='another machine/boot'):
        _receive(f)
    assert not f.basis.context.destination.exists()


@pytest.mark.parametrize('entrypoint', ['reference_model', 'preflight_geometry'])
def test_initial_consumers_automatically_use_bound_crosshost_source_pool(transport, monkeypatch, entrypoint):
    """Actual default callers work without access to producer code/runtime paths.

    Original descriptor strings remain in the model and source observation.
    An IO trap rejects every original copyable source locator, simulating a
    receiver without that producer environment; actual retained bytes are read.
    RAW and original parent NPZ remain at their actual physical locations.
    """
    import numpy as np
    from darsia.fluidflower import round5_geometry, map_placement
    f = transport
    blocked = {reader._key(row['original']['path']) for row in f.pool['files']}
    actual_open = Path.open
    def receiver_only(path, *args, **kwargs):
        if reader._key(path) in blocked:
            raise FileNotFoundError('Producer locator is unavailable on the receiver: ' + str(path))
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', receiver_only)
    receipt = _receive(f)
    authority, _, _ = _release(f, receipt)
    frame = dict(H_camera_to_mm=[[1., 0., 0.], [0., -1., 2.], [0., 0., 1.]],
                 sand_area_factor14=1.0, explicit_certificate={'synthetic': True})
    root = Path(f.intent_binding['path']).parent
    saved = _json(root / 'saved-model.json', dict(run='sample', conditional_frame=frame))
    edges = _json(root / 'edges.json', {'synthetic': True})
    configuration = deepcopy(f.basis.parent['configuration']['frame_binding']['runs']['sample']['configuration'])
    profile = dict(f.basis.document['inputs']['profile'], profile_id='camera', camera_wh=[3, 2])
    model = round5_geometry.reference_model('sample', frame,
        reference_image=deepcopy(f.basis.document['inputs']['raw']), camera_profile=profile,
        source_model=saved, edge_review=edges, baseline_configuration=configuration,
        baseline_native_renderer=f.basis.certificate, raw_shape_basis_authority=authority,
        raw_shape_basis_workspace=f.basis.parent_workspace)
    assert model['baseline_native_grid']['raw_shape_basis_authority'] == authority
    assert model['baseline_configuration'] == configuration
    if entrypoint == 'preflight_geometry':
        bound = _json(root / 'reference-model.json', model)
        index = _json(root / 'reference-index.json', {'synthetic': True})
        geometry = map_placement.RunGeometry(run_id='sample',
            metric_to_grid=np.asarray(model['H_metric_mm_to_camera_u']),
            reference_image=model['reference_image'], shape_hw=(2, 3),
            source=bound['path'], model=model, entry={'reference_model': bound})
        progress = []
        checked = reader.preflight_geometry(index, {'sample': geometry}, ['sample'],
                                            run_progress=progress.append)['sample']
        assert progress == [dict(phase='raw_basis', completed=n, total=1, run_id='sample')
                            for n in (0, 1)]
    else:
        checked = reader.read_model_basis(model)
    for original in f.observed['copyable_sources']:
        row = checked.guards.rows[reader._key(original['path'])]
        assert row['original'] == original
        assert Path(row['physical']['path']).is_relative_to(f.pool_root)
    assert checked.guards.rows[reader._key(f.basis.source)]['physical']['path'] == str(f.basis.source)
    for field in f.basis.fields.values():
        assert checked.guards.rows[reader._key(field['path'])]['physical']['path'] == field['path']
    checked.revalidate()


@pytest.mark.parametrize('part', ['raw', 'parent_npz', 'producer_output'])
def test_retained_pool_cannot_alias_original_physical_only_roles(transport, part):
    """A bound pool may retain observed metadata roles, never acquisition/NPZ/output roles."""
    f = transport
    if part == 'raw':
        original = deepcopy(f.basis.document['inputs']['raw'])
    elif part == 'parent_npz':
        original = {k: f.basis.fields['labels'][k] for k in ('path', 'sha256', 'size_bytes')}
    else:
        original = deepcopy(f.observed['files'][0])
    target = f.pool_root / 'not-a-copyable-role.bin'
    shutil.copyfile(original['path'] if part != 'producer_output' else f.staged_files[0]['staged']['path'], target)
    pool = deepcopy(f.pool)
    pool['files'][0] = dict(original=original, copy=dict(_descriptor(target), path=target.name))
    f.pool_binding = _json(Path(f.pool_binding['path']), pool)
    with pytest.raises(ValueError, match='original role|copyable roles'):
        _receive(f)
    assert not f.basis.context.destination.exists()


def test_late_staged_growth_is_refused_before_any_oversized_write(transport, monkeypatch):
    """Late EOF growth cannot consume file/destination budget before refusal."""
    f = transport
    original = f.observed['files'][0]
    staged = Path(f.staged_files[0]['staged']['path'])
    target = Path(original['path'])
    actual_open = Path.open
    triggered = []
    def grow_at_exclusive_open(path, mode='r', *args, **kwargs):
        result = actual_open(path, mode, *args, **kwargs)
        if Path(path) == target and mode == 'xb':
            with actual_open(staged, 'ab') as writer:
                writer.write(b'x' * (f.basis.document['output_ownership']['authority']['maximum_file_bytes'] + 1))
            triggered.append(True)
        return result
    monkeypatch.setattr(Path, 'open', grow_at_exclusive_open)
    with pytest.raises(ValueError, match='grew beyond its declared copy byte ceiling'):
        _receive(f)
    assert triggered == [True]
    assert target.exists() and target.stat().st_size <= original['size_bytes']
    actual_size = sum(path.stat().st_size for path in f.basis.context.destination.rglob('*') if path.is_file())
    assert actual_size <= f.basis.document['output_ownership']['authority']['maximum_output_bytes']
    assert not Path(f.intent['receiver_receipt']).exists()
    assert staged.stat().st_size > original['size_bytes']


@pytest.mark.parametrize('scope', ['destination', 'intended_root'])
def test_identical_directory_substitution_during_final_hash_is_detected(transport, monkeypatch, scope):
    """Changed native IDs after the first hash cannot hide behind identical bytes."""
    f = transport
    authority, _, _ = _release(f, _receive(f))
    checked = _read(f, authority)
    actual_add = checked.guards.add
    triggered = []
    path = Path(checked.receipt[scope])
    retained = path.parent / ('retained-during-hash-' + scope)
    def swap_after_first_hash(binding, **kwargs):
        result = actual_add(binding, **kwargs)
        if not triggered:
            path.rename(retained)
            shutil.copytree(retained, path)
            triggered.append(True)
        return result
    monkeypatch.setattr(checked.guards, 'add', swap_after_first_hash)
    with pytest.raises(ValueError, match='ownership changed'):
        checked.revalidate()
    assert triggered == [True]
    assert all(_descriptor(row['path']) == row for row in f.observed['files'])


def test_extra_output_added_during_final_hash_is_detected(transport, monkeypatch):
    """Membership is checked again after long hashes as well as before them."""
    f = transport
    authority, _, _ = _release(f, _receive(f))
    checked = _read(f, authority)
    actual_add = checked.guards.add
    triggered = []
    def add_member_after_first_hash(binding, **kwargs):
        result = actual_add(binding, **kwargs)
        if not triggered:
            (f.basis.context.destination / 'late-unreceipted.txt').write_bytes(b'late member')
            triggered.append(True)
        return result
    monkeypatch.setattr(checked.guards, 'add', add_member_after_first_hash)
    with pytest.raises(ValueError, match='membership changed'):
        checked.revalidate()
    assert triggered == [True]


def test_source_pool_or_root_pool_substitution_refuses(transport):
    """The final Root release and actual receipt must name the same exact pool."""
    f = transport
    authority, release, binding = _release(f, _receive(f))
    pool = deepcopy(f.pool)
    pool['copy_id'] = 'e' * 32
    release['retained_sources'] = _json(f.pool_root / 'substituted-pool.json', pool)
    authority['received_basis'] = _json(Path(binding['path']), release)
    with pytest.raises(ValueError, match='source pool differs'):
        _read(f, authority)


# These fixtures retain the existing registered parent's actual 2x3 NPZ grids
# and real approved manual-version/Prepare route. Only the original RAW decoder,
# registration and runtime identities are synthetic, as in their shared fixture.
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: E402,F401
from .test_geometry_registered_export_apply import registered_setup  # noqa: E402,F401


@pytest.mark.parametrize('outcome', ['complete', 'late-current-pool-drift'])
def test_genuine_prepare_uses_received_sources_and_finally_rechecks_current_pool(
        registered_setup, tmp_path, monkeypatch, outcome, basis):
    """Real Prepare writes history copies but may never use them to hide late pool drift."""
    from . import test_painted_adoption_table as shared
    from darsia.fluidflower import painted_adoption_table as table
    from darsia.fluidflower import painted_adoption_geometry as geometry
    actual_models = shared._models
    transports, blocked, changed = {}, set(), []
    actual_open = Path.open
    def receiver_only(path, *args, **kwargs):
        if reader._key(path) in blocked:
            raise FileNotFoundError('Foreign producer runtime/source is unavailable: ' + str(path))
        return actual_open(path, *args, **kwargs)
    def received_model(run, model, folder):
        grid = model['baseline_native_grid']
        certificate = grid['renderer']
        basis_document = json.loads(Path(certificate['path']).read_bytes())
        local = tmp_path / ('received-' + run)
        local.mkdir()
        source = SimpleNamespace(certificate=certificate, document=basis_document,
            authority=grid['raw_shape_basis_authority'], parent_workspace=grid['parent_workspace_revision'],
            context=SimpleNamespace(destination=Path(certificate['path']).parent))
        value = _transport_owned_basis(source, local, monkeypatch)
        authority, _, _ = _release(value, _receive(value))
        grid['raw_shape_basis_authority'] = authority
        transports[run] = value
    def models_then_foreign_source_barrier(*args, **kwargs):
        assert kwargs.get('mutate') is None
        index = actual_models(*args, **dict(kwargs, mutate=received_model))
        for value in transports.values():
            document = value.basis.document
            blocked.update(reader._key(row['path']) for row in document['renderer_implementation'])
            runtime = document['producer']['runtime']
            blocked.add(reader._key(runtime['executable']['path']))
            blocked.update(reader._key(row['module']['path']) for row in runtime['libraries'].values())
        monkeypatch.setattr(Path, 'open', receiver_only)
        return index
    def late_drift(message):
        if outcome != 'late-current-pool-drift' or 'TWO' not in message.upper() or changed:
            return
        value = transports['one']
        executable = value.basis.document['producer']['runtime']['executable']
        row = next(row for row in value.pool['files'] if row['original'] == executable)
        path = value.pool_root / row['copy']['path']
        _changed_byte(path)
        changed.append(path)
    monkeypatch.setattr(shared, '_models', models_then_foreign_source_barrier)
    if outcome == 'late-current-pool-drift':
        with pytest.raises(ValueError, match='changed'):
            shared.prepared_painting(registered_setup, tmp_path, during_prepare=late_drift)
        assert changed
        folders = list((tmp_path / 'preparation').iterdir())
        assert len(folders) == 1 and not (folders[0] / 'index.json').exists()
        assert (folders[0] / 'configuration_sources' / 'one.toml').is_file()
        assert list((folders[0] / 'raw_basis_sources').glob('*.bin'))
    else:
        prepared, _, _, _, _ = shared.prepared_painting(registered_setup, tmp_path)
        folder = Path(prepared['folder'])
        document = json.loads((folder / 'index.json').read_bytes())
        assert set(transports) == {'one', 'two'}
        assert blocked
        table.verify_preparation_publication(document, folder)
        source_rows, checks = geometry.received_source_rows(document['painting_geometry'])
        for value in transports.values():
            for row in value.pool['files']:
                original = row['original']
                key = (original['path'], original['sha256'], original['size_bytes'])
                assert any(Path(source_rows[key]['path']).is_relative_to(item.pool_root)
                           for item in transports.values()) or source_rows[key] == original
        for checked in checks:
            checked.revalidate()
        for row in document['painting_geometry']['raw_shape_basis']['source_copies']:
            copied = row['copy']
            physical = _descriptor(folder / copied['path'])
            assert physical['sha256'] == row['original']['sha256']
            assert physical['size_bytes'] == row['original']['size_bytes']
