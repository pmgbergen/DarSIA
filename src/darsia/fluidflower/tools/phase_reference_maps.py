"""SHA-admitted saved reference maps for Studio; never historical inference."""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import canonical_metadata_paths, local_cache_root
from darsia.fluidflower.native.config import require_parameters
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import shutil
import threading
import uuid

_HASH_CACHE = OrderedDict()
_HASH_LOCK = threading.RLock()
_RESOLVE_LOCK = threading.RLock()


def _parameters():
    parameters = require_parameters("saved_review_metadata")["reference_views"]
    current = parameters["current_sources"]["reference_maps"]
    if not isinstance(current, dict) or Path(current["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("Saved reference reader is not explicitly bound")
    if _sha(Path(__file__)) != current["sha256"] or Path(__file__).stat().st_size != current["size_bytes"]:
        raise ValueError("Saved reference reader implementation changed")
    return parameters


def _setting(name):
    value = _parameters()[name]
    return set(value) if name == "layers" else value


def _windows_shared_cache():
    return local_cache_root() / _parameters()["cache_suffix"]





def _producer_identity(role):
    """Use original self-hash provenance DATA without executing old producers."""
    return _parameters()["producer_identities"][role]["sha256"]


def _compatible_methods():
    return set(_parameters()["compatible_method_sha256"])





def __getattr__(name):
    names = {"FORMAT": "format", "POLICY": "policy", "LAYERS": "layers"}
    if name not in names:
        raise AttributeError(name)
    return _setting(names[name])


def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding='utf-8-sig'))

def _sha(path):
    path = resolve_review_path(path)
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _HASH_LOCK:
        cached = _HASH_CACHE.get(path)
        if cached and cached[0] == signature:
            _HASH_CACHE.move_to_end(path)
            return cached[1]
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'Reference asset changed while hashing: {path}')
    result = h.hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE[path] = (signature, result)
        while len(_HASH_CACHE) > 4096:
            _HASH_CACHE.popitem(last=False)
    return result

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def _write(path, value):
    from darsia.fluidflower.phase_saved_review_metadata import _write as _canonical_saved_write
    return _canonical_saved_write(path, value)

def cache_root():
    override = os.environ.get('FFAC_PHASE_REFERENCE_MAP_CACHE')
    if override:
        return Path(override)
    if os.name == 'nt':
        return _windows_shared_cache()
    from darsia.fluidflower.execution import default_execution_cache_root
    return default_execution_cache_root() / 'phase-reference-maps'

def _receipt_cache_roots():
    current = cache_root()
    roots = [current]
    if os.name == 'nt' and current == _windows_shared_cache():
        local = Path.home() / 'AppData' / 'Local'
        roots.append(local / 'DarSIA/FluidFlower/execution/phase-reference-maps')
        roots.extend((local / 'Packages').glob('*/LocalCache/Local/DarSIA/FluidFlower/execution/phase-reference-maps'))
    return roots

def _import_cached_layers(layers, folder):
    """Move verified display assets out of a private cache without changing pixels."""
    result = {}
    for layer, item in layers.items():
        if layer not in _setting('layers'):
            raise ValueError('Unknown cached reference layer')
        source = Path(item['path'])
        if _sha(source) != item['sha256']:
            raise ValueError('Legacy reference map differs from its receipt')
        target = folder / 'imported_layers' / (layer + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
        try:
            shutil.copyfile(source, temporary)
            if _sha(temporary) != item['sha256']:
                raise ValueError('Reference map changed while importing')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        result[layer] = {**item, **_asset(target)}
    return result

def _frame(root, run, index):
    return Path(root) / 'native_frames' / run / f'frame_{index:03d}.jpg'

def _asset(path):
    path = resolve_review_path(path)
    return {'path': str(path), 'sha256': _sha(path)}

def _strictly_earlier(frames, elapsed):
    previous = float('-inf')
    result = []
    for index, row in enumerate(frames):
        value = float(row['elapsed_h'])
        if not previous < value:
            raise ValueError('Reviewed reference history must have strictly increasing explicit times')
        previous = value
        if value < elapsed:
            result.append({**row, 'frame_index': index})
    return result

def _original_layers(job, index):
    run, fingerprint = (job['run'], job['params_sha256'])
    folder = Path(job['phase_map_output_dir']) / 'native_phase_maps' / run / f'A_{fingerprint[:16]}'
    name = f'frame_{index:03d}.png'
    paths = {'all': folder / name, 'gas': folder / 'gas' / name, 'aqueous': folder / 'aqueous' / name}
    if job.get('raw_phase_map_output_dir'):
        paths['raw'] = Path(job['raw_phase_map_output_dir']) / 'native_phase_maps' / run / f"A_{job['optical_model_sha256'][:16]}" / name
    if job.get('presence_diagnostics_dir'):
        for layer, folder_name in (('context_added', 'added_aqueous'), ('context_removed', 'suppressed_presence'), ('context_uncertain', 'unresolved')):
            paths[layer] = Path(job['presence_diagnostics_dir']) / run / folder_name / name
    return {key: _asset(value) for key, value in paths.items() if value.is_file()}

def _display_layers(layers):
    from darsia.fluidflower.phase_rendering import two_phase_overlay_path
    result = {}
    for layer, source in layers.items():
        if layer in {'all', 'gas', 'aqueous', 'raw'}:
            phase = 'all' if layer == 'raw' else layer
            display = two_phase_overlay_path(Path(source['path']), phase, cache_root=cache_root() / 'display')
            result[layer] = {**_asset(display), 'source': source}
        else:
            result[layer] = source
    return result

def build_plan(reference, review_root, run, candidate):
    """Bind one immutable request to the actual candidate's validation package."""
    from darsia.fluidflower.tools.phase_injection_references import resolve_reference_asset
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_training.radiometry import analysis_review_root
    candidate = dict(candidate) if isinstance(candidate, Mapping) else candidate.provenance()
    run = run.casefold()
    if candidate.get('run', run).casefold() != run:
        raise ValueError('Reference candidate belongs to a different experiment')
    validation = Path(candidate.get('phase_map_output_dir') or '').parent
    binding_path, manifest_path = (validation / 'validation_binding.json', validation / 'postflight_manifest.json')
    binding, campaign = (_read(binding_path), _read(manifest_path))
    if binding.get('selected_candidates', {}).get(run) != candidate.get('candidate_id') or binding.get('pipeline_sha256') != candidate.get('params_sha256'):
        raise ValueError('Reference candidate does not match its frozen validation binding')
    matches = [j for j in campaign['jobs'] if j.get('run') == run and j.get('candidate_id') == candidate['candidate_id']]
    if len(matches) != 1:
        raise ValueError('Reference candidate has no unique original validation job')
    job = matches[0]
    model_root = resolve_review_path(binding['model_root'])
    if _model_identity(model_root) != (binding['model_id'], binding['model_sha256']):
        raise ValueError('Frozen reference model identity changed')
    if candidate.get('optical_model_sha256') and candidate['optical_model_sha256'] != binding['model_sha256']:
        raise ValueError('Reference candidate optical model differs')
    profile = load_physics_profile(model_root)
    if physics_runtime_identity(profile) != binding.get('physics_runtime'):
        raise ValueError('Frozen reference physical runtime changed')
    if binding.get('radiometry_profile') is not None:
        raise ValueError('Exact reference maps for a separate model radiometry profile are not yet available')
    native_root = resolve_review_path(analysis_review_root(resolve_review_path(job.get('analysis_source_review_root') or job['review_output_dir']), run))
    native_path = native_root / 'native_frames' / run / 'manifest.json'
    native, native_sha = (_read(native_path), _sha(native_path))
    supplied_native = Path(review_root) / 'native_frames' / run / 'manifest.json'
    if _sha(supplied_native) != native_sha:
        raise ValueError("Displayed reference native geometry differs from candidate's actual input")
    frames = native['frame_sources']
    baseline = _asset(native_root / 'native_references' / run / 'pre_injection.jpg')
    exact_index = None
    if reference is None:
        ref_metadata = native.get('reference') or {}
        index = ref_metadata.get('timeline_frame_index')
        elapsed = ref_metadata.get('elapsed_h')
        if index is not None:
            index = int(index)
            if not float(frames[index]['elapsed_h']) < 0:
                raise ValueError('Baseline timeline index is not a pre-injection image')
            elapsed = frames[index]['elapsed_h']
            if _sha(_frame(native_root, run, index)) == baseline['sha256']:
                exact_index = index
        ref = {'id': 'baseline', 'label': 'Before injection', 'source_name': ref_metadata.get('source_name', 'Pre-injection baseline'), 'elapsed_h': elapsed, 'phase_time_h': float(elapsed) if elapsed is not None else -1e-06, 'time_basis': 'native reference metadata' if elapsed is not None else 'pre-injection control; acquisition time not recorded', 'photo': baseline, 'native_manifest_sha256': native_sha}
        replay = []
    else:
        reference = dict(reference)
        if reference.get('native_manifest_sha256') != native_sha:
            raise ValueError('Selected reference native identity is stale')
        photo = _asset(resolve_reference_asset(reference, Path(review_root), run))
        elapsed = float(reference['elapsed_h'])
        index = reference.get('frame_index')
        if index is not None:
            index = int(index)
            original = frames[index]
            if abs(float(original['elapsed_h']) - elapsed) > 1e-09 or original.get('source_name', '').casefold() != reference['source_name'].casefold():
                raise ValueError('Reference frame index does not identify the selected acquisition')
            if _sha(_frame(native_root, run, index)) == photo['sha256']:
                exact_index = index
        ref = {**reference, 'photo': photo, 'phase_time_h': elapsed}
        replay = _strictly_earlier(frames, elapsed)
    history = [{'frame_index': row['frame_index'], 'source_name': row.get('source_name'), 'elapsed_h': float(row['elapsed_h']), 'photo': _asset(_frame(native_root, run, row['frame_index']))} for row in replay]
    plan = {'format': _setting('format'), 'policy': _setting('policy'), 'method_sha256': _producer_identity('reference_maps'), 'run': run, 'dataset': job['dataset'], 'candidate_id': candidate['candidate_id'], 'pipeline_sha256': binding['pipeline_sha256'], 'model_root': str(model_root), 'model_id': binding['model_id'], 'model_sha256': binding['model_sha256'], 'physics_runtime': binding.get('physics_runtime'), 'binding': _asset(binding_path), 'campaign': _asset(manifest_path), 'native_manifest': _asset(native_path), 'native_root': str(native_root), 'baseline': baseline, 'canvas_wh': native['canvas_wh'], 'content_box_xywh': native['content_box_xywh'], 'reference': ref, 'replay': history, 'job': job, 'inference_provenance': binding.get('inference_provenance'), 'exact_reviewed_frame_index': exact_index}
    if exact_index is not None:
        plan['reviewed_layers'] = _original_layers(job, exact_index)
    if profile is not None and 'unified_phase_decision' in profile:
        plan['decision_cache_adapter_sha256'] = _producer_identity('decision_reference_adapter')
        plan['decision_evidence_reader_sha256'] = _producer_identity('decision_evidence_reader')
    plan['key'] = _digest(plan)
    return plan

def _check_assets(plan):
    assets = [plan['binding'], plan['campaign'], plan['native_manifest'], plan['baseline'], plan['reference']['photo']]
    assets += [row['photo'] for row in plan['replay']]
    for item in assets:
        if _sha(item['path']) != item['sha256']:
            raise ValueError(f"Reference input changed: {item['path']}")

def _computation_inputs(plan):
    result = {key: value for key, value in plan.items() if key not in {'key', 'method_sha256'}}
    result['reference'] = {key: plan['reference'].get(key) for key in ('source_name', 'elapsed_h', 'phase_time_h', 'native_manifest_sha256', 'photo')}
    return result

def _reuse_compatible_receipt(plan):
    expected = _computation_inputs(plan)
    paths = (path for root in _receipt_cache_roots() for path in root.glob('*/receipt.json'))
    for receipt_path in paths:
        try:
            receipt = _read(receipt_path)
            if receipt.get('method_sha256') not in _compatible_methods() | {plan.get('method_sha256')}:
                continue
            original = {key: receipt[key] for key in plan if key != 'key'}
            if _digest(original) != receipt.get('key') or canonical_metadata_paths(_computation_inputs(original)) != canonical_metadata_paths(expected):
                continue
            for item in receipt['layers'].values():
                if _sha(item['path']) != item['sha256']:
                    raise ValueError('Compatible reference map differs from its receipt')
            migrated = {**receipt, **plan, 'reused_receipt': _asset(receipt_path)}
            if receipt_path.parent.parent.resolve() != cache_root().resolve():
                migrated['layers'] = _import_cached_layers(receipt['layers'], cache_root() / plan['key'])
            _write(cache_root() / plan['key'] / 'receipt.json', migrated)
            return migrated
        except (OSError, KeyError, ValueError):
            continue
    return None

def _cached(plan, layer):
    receipt_path = cache_root() / plan['key'] / 'receipt.json'
    if receipt_path.is_file():
        receipt = _read(receipt_path)
    else:
        receipt = _reuse_compatible_receipt(plan) if 'reference' in plan else None
        if receipt is None:
            return None
    if receipt.get('key') != plan['key']:
        raise ValueError('Reference map receipt identity differs')
    item = receipt.get('layers', {}).get(layer)
    if item is None:
        raise ValueError(f'The exact reference has no {layer} phase layer')
    if _sha(item['path']) != item['sha256']:
        raise ValueError('Reference map asset differs from its receipt')
    return Path(item['path'])


from darsia.fluidflower.phase_saved_presentation import _worker_lock

def resolve_saved_request(plan):
    """Admit saved reviewed/single-pass maps after all original checks."""
    if plan.get('format') != _setting('format') or plan.get('policy') != _setting('policy') or plan.get('method_sha256') != _producer_identity('reference_maps'):
        raise ValueError('Reference worker method does not match its request')
    if _digest({k: v for k, v in plan.items() if k != 'key'}) != plan['key']:
        raise ValueError('Reference request identity changed')
    folder = cache_root() / plan['key']
    single_pass = Path(plan['binding']['path']).parent / 'single_pass_references' / plan['run'] / plan['reference']['id'] / 'single_pass.json'
    if not (folder / 'receipt.json').is_file() and not plan.get('reviewed_layers', {}).get('all') and not single_pass.is_file():
        return None
    with _worker_lock(folder / 'worker.lock'):
        _check_assets(plan)
        if (folder / 'receipt.json').is_file():
            return _read(folder / 'receipt.json')
        compatible = _reuse_compatible_receipt(plan)
        if compatible is not None:
            return compatible
        if plan.get('reviewed_layers', {}).get('all'):
            layers = plan['reviewed_layers']
            for asset in layers.values():
                if _sha(asset['path']) != asset['sha256']:
                    raise ValueError('Existing exact reviewed map changed')
            receipt = {**plan, 'mode': 'exact-reviewed-image-reuse', 'layers': _display_layers(layers), 'replay_validation': 'original validation maps; no fresh inference'}
            _write(folder / 'receipt.json', receipt)
            return receipt
        single_pass = Path(plan['binding']['path']).parent / 'single_pass_references' / plan['run'] / plan['reference']['id'] / 'single_pass.json'
        if single_pass.is_file():
            record = _read(single_pass)
            method = (plan.get('inference_provenance') or {}).get('single_pass_reference_sha256')
            snapshot = Path(plan['binding']['path']).parent / 'code_snapshot' / 'phase_reference_single_pass.py'
            if method is None or record.get('method_sha256') != method or (not snapshot.is_file()) or (_sha(snapshot) != method):
                raise ValueError('Single-pass reference method differs from frozen validation')
            excluded = set(plan['job'].get('independent_reference_frames') or [])
            prior = [row['frame_index'] for row in plan['replay'] if row['frame_index'] not in excluded]
            expected_previous = prior[-1] if prior else None
            if record.get('id') != plan['reference']['id'] or record.get('elapsed_h') != plan['reference']['phase_time_h'] or record.get('photo', {}).get('sha256') != plan['reference']['photo']['sha256'] or (record.get('previous_reviewed_frame_index') != expected_previous) or (record.get('canonical_history_unchanged') is not True):
                raise ValueError('Single-pass reference does not match the exact causal request')
            generated = {**plan['job'], 'phase_map_output_dir': record['final_root'], 'raw_phase_map_output_dir': record['raw_root'], 'presence_diagnostics_dir': record['diagnostics_root']}
            layers = _original_layers(generated, 0)
            _check_assets(plan)
            receipt = {**plan, 'mode': 'single-pass-exact-reference', 'layers': _display_layers(layers), 'replay_validation': 'causal state from the same reviewed validation pass', 'single_pass_provenance': _asset(single_pass), 'previous_reviewed_frame_index': expected_previous, 'canonical_history_unchanged': True, 'canonical_history_sha256': record['canonical_history_sha256'], 'physical_report': record.get('physical_report')}
            _write(folder / 'receipt.json', receipt)
            return receipt
        return None


def resolve_reference_overlay(reference, review_root, run, candidate, side, layer, *, cached_only=False, progress=None):
    """Show exact saved output, or explicitly report unavailable; never replay."""
    if side not in {"A", "B"} or layer not in _setting("layers"):
        raise ValueError("Unknown candidate side or exact reference phase layer")
    if cached_only and reference is not None and reference.get("needs_materialization"):
        return None
    with _RESOLVE_LOCK:
        plan = build_plan(reference, Path(review_root), run, candidate)
        ready = _cached(plan, layer)
        if ready:
            return ready
        if cached_only and not plan.get('reviewed_layers', {}).get('all'):
            return None
        if resolve_saved_request(plan) is not None:
            return _cached(plan, layer)
        if cached_only:
            return None
        message = _parameters()["unavailable_reference"]
        if progress is not None:
            progress(message)
        raise FileNotFoundError(message)


# Legacy callers may request saved admission, but never a historical calculation.
compute_request = resolve_saved_request
