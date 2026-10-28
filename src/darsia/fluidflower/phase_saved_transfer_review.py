"""Saved transfer metadata, source admission and review-session readers; no calculation."""
from __future__ import annotations
import copy
import json
import math
import re
import tempfile
import time
from pathlib import Path
from .phase_saved_review_metadata import configured_parameters, read_json, load_native_source, bound_path
from .phase_runtime import load_physics_profile as _load_physics_profile_uncached, physics_runtime_identity
from .project import sha256_file as _sha256_file_uncached
from .phase_training.radiometry import analysis_review_root
from .phase_training.scope import series_id
from .phase_validation import _candidate_paths
from .review_storage import resolve_review_path
_DISCOVERY_CACHE = None


def _parameters():
    return configured_parameters("saved_transfer_review")


def _typed(item):
    from .phase_saved_review_metadata import typed_saved_data_value
    return typed_saved_data_value(item)


def _literal(key):
    return _typed(_parameters()["literals"][key])


def _global(name):
    return _typed(_parameters()["globals"][name])


def _cached(bucket, key, produce):
    """Memoise one pure read for the duration of a single discovery pass.

    Discovery asks the same questions once per candidate experiment: the same manifests, the
    same training scope, the same physics profile, the same digests. It never writes to those
    files while it runs, so reading each of them once gives the same answer for a fraction of
    the time, which matters when the model lives on a share and every miss is a network round
    trip. Outside a discovery pass nothing is cached, so the batch still sees each file as it
    is at that moment. Deep copies are handed out so a caller that modifies what it received
    cannot reach into the cache.
    """
    cache = _DISCOVERY_CACHE
    if cache is _literal('_cached:73:16'):
        return produce()
    store = cache.setdefault(bucket, {})
    if key not in store:
        store[key] = produce()
    return copy.deepcopy(store[key])

def _read_uncached(path):
    return read_json(resolve_review_path(path))

def read(path):
    return _cached(_literal('read:86:19'), str(path), lambda: _read_uncached(path))

def sha256_file(path):
    return _cached(_literal('sha256_file:94:19'), str(path), lambda: _sha256_file_uncached(resolve_review_path(path)))

def load_physics_profile(root):
    return _cached(_literal('load_physics_profile:98:19'), str(root), lambda: _load_physics_profile_uncached(resolve_review_path(root)))

def event(stage, **values):
    print(_global('EVENT_PREFIX') + json.dumps(dict(stage=stage, **values)), flush=_literal('event:109:72'))

def progress_event(stage, **values):
    """Report progress to the window that spawned this process; importers get silence.

    The batch imports this module and runs discovery once per experiment. It has its own
    progress reporting and must not have its log filled with thousands of these lines.
    """
    if __name__ == _literal('progress_event:118:19'):
        event(stage, **values)

def target_sources(manifest_path, selected):
    """Inspect prepared target metadata; do not load frames or predictions."""
    return _cached(_literal('target_sources:154:19'), (str(manifest_path), str(selected)), lambda: _target_sources_uncached(manifest_path, selected))

def _prepared_native_photographs(review_root, run, native):
    """Check the fixed photographic files while retaining the shadow identity."""
    if _literal('_prepared_native_photographs:160:7') in native:
        from .phase_saved_transfer_review import _rig_prepared_photographs as prepared_photographs
        return prepared_photographs(review_root, run, native)
    from .phase_source_alias import KEY, resolve_original_native_source
    aliased = KEY in native
    photo_root, _original_path, original = resolve_original_native_source(review_root, run, native)

    def photograph(record, directory, stem):
        expected = photo_root / directory / run / f'{stem}.jpg'
        declaration = (record.get(_literal('_prepared_native_photographs:170:34')) or {}).get(_literal('_prepared_native_photographs:170:65'))
        if aliased or declaration is not _literal('_prepared_native_photographs:171:41'):
            if not isinstance(declaration, dict) or not isinstance(declaration.get(_literal('_prepared_native_photographs:173:54')), str) or (not isinstance(declaration.get(_literal('_prepared_native_photographs:174:54')), str)) or (not re.fullmatch(_literal('_prepared_native_photographs:175:40'), declaration[_literal('_prepared_native_photographs:175:72')])):
                raise ValueError(_literal('_prepared_native_photographs:176:33'))
            path = resolve_review_path(declaration[_literal('_prepared_native_photographs:177:51')])
            if path.resolve() != expected.resolve() or not path.is_file():
                raise ValueError(_literal('_prepared_native_photographs:179:33'))
            return path
        path = expected.with_suffix(_literal('_prepared_native_photographs:181:36')) if expected.with_suffix(_literal('_prepared_native_photographs:181:68')).is_file() else expected
        if not path.is_file():
            raise ValueError(_literal('_prepared_native_photographs:183:29'))
        return path
    baseline = photograph(original.get(_literal('_prepared_native_photographs:189:39')) or {}, _literal('_prepared_native_photographs:189:59'), _literal('_prepared_native_photographs:189:80'))
    for index, record in enumerate(original.get(_literal('_prepared_native_photographs:190:48')) or []):
        photograph(record, _literal('_prepared_native_photographs:191:27'), f'frame_{index:03d}')
    return baseline

def _target_sources_uncached(manifest_path, selected):
    manifest_path = resolve_review_path(manifest_path)
    manifest = read(manifest_path)
    rows = []
    jobs = manifest.get(_literal('_target_sources_uncached:199:24'), [])
    for job_index, job in enumerate(jobs, _literal('_target_sources_uncached:200:42')):
        if not isinstance(job, dict):
            continue
        dataset = job.get(_literal('_target_sources_uncached:203:26')) or manifest.get(_literal('_target_sources_uncached:203:53'))
        if dataset and series_id(dataset) != selected:
            continue
        run = str(job.get(_literal('_target_sources_uncached:206:26')) or _literal('_target_sources_uncached:206:36')).strip().casefold()
        if not run:
            continue
        progress_event(_literal('_target_sources_uncached:209:23'), run=run, job_index=job_index, job_total=len(jobs))
        candidate_root, review_root = _candidate_paths(manifest_path.parent, job)
        if job.get(_literal('_target_sources_uncached:211:19')):
            review_root = Path(job[_literal('_target_sources_uncached:212:35')])
            if not review_root.is_absolute():
                review_root = manifest_path.parent / review_root
        try:
            review_root = resolve_review_path(analysis_review_root(resolve_review_path(review_root), run))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        native_path = review_root / _literal('_target_sources_uncached:219:36') / run / _literal('_target_sources_uncached:219:60')
        if not native_path.is_file():
            continue
        native = read(native_path)
        legacy_provenance = _literal('_target_sources_uncached:223:28')
        if not dataset:
            sources = native.get(_literal('_target_sources_uncached:228:33')) or []
            source_pattern = f'(?:^|/)albus/raw data/{re.escape(run)}(?:/|$)'
            if not (selected == _literal('_target_sources_uncached:230:32') and manifest.get(_literal('_target_sources_uncached:230:56')) == _literal('_target_sources_uncached:230:69') and str(manifest.get(_literal('_target_sources_uncached:231:41'), _literal('_target_sources_uncached:231:64'))).startswith(_literal('_target_sources_uncached:231:80')) and (native.get(_literal('_target_sources_uncached:232:35')) == _literal('_target_sources_uncached:232:46')) and (str(native.get(_literal('_target_sources_uncached:233:39'), _literal('_target_sources_uncached:233:46'))).casefold() == run) and re.fullmatch(_literal('_target_sources_uncached:234:37'), run) and sources and all((re.search(source_pattern, str(row.get(_literal('_target_sources_uncached:235:66'), _literal('_target_sources_uncached:235:81'))).replace(_literal('_target_sources_uncached:235:94'), _literal('_target_sources_uncached:235:100')).casefold()) for row in sources))):
                continue
            dataset = selected
            legacy_provenance = _literal('_target_sources_uncached:239:32')
        count = int(native.get(_literal('_target_sources_uncached:240:31')) or _literal('_target_sources_uncached:240:49'))
        if count <= _literal('_target_sources_uncached:241:20') or len(native.get(_literal('_target_sources_uncached:241:40'), [])) != count:
            continue
        frames = native[_literal('_target_sources_uncached:243:24')]
        try:
            times = [float(frame[_literal('_target_sources_uncached:245:33')]) for frame in frames]
            if any((not math.isfinite(value) for value in times)) or any((later <= earlier for earlier, later in zip(times, times[_literal('_target_sources_uncached:247:83'):]))) or any((frame.get(_literal('_target_sources_uncached:248:37')) != index for index, frame in enumerate(frames))):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        controls = job.get(_literal('_target_sources_uncached:252:27')) or []
        if controls and (controls != [_literal('_target_sources_uncached:253:38')] or times[_literal('_target_sources_uncached:253:50')] >= _literal('_target_sources_uncached:253:56') or native.get(_literal('_target_sources_uncached:254:30'), {}).get(_literal('_target_sources_uncached:254:59')) != _literal('_target_sources_uncached:254:70')):
            continue
        try:
            baseline = _prepared_native_photographs(review_root, run, native)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        target_job = dict(job, run=run, dataset=selected, candidate_folder=str(candidate_root), review_output_dir=str(review_root), analysis_source_review_root=str(review_root))
        rows.append(dict(run=run, dataset=selected, source_manifest=str(manifest_path), review_root=str(review_root), native_manifest=str(native_path), native_manifest_sha256=sha256_file(native_path), baseline=str(baseline), frame_count=count, job=target_job, prepared=_literal('_target_sources_uncached:268:56'), has_registered_domain=bool((native.get(_literal('_target_sources_uncached:269:51')) or {}).get(_literal('_target_sources_uncached:269:72'))), dataset_provenance=legacy_provenance or _literal('_target_sources_uncached:270:52')))
    return rows

def registered_target_physics(model_root, target):
    """Require the chosen model's own frozen context for this exact target grid."""
    root = Path(model_root).resolve()
    profile = load_physics_profile(root)
    key = f"{target[_literal('registered_target_physics:278:20')]}/{target[_literal('registered_target_physics:278:40')]}"
    entry = _literal('registered_target_physics:279:12') if profile is _literal('registered_target_physics:279:31') else profile.get(_literal('registered_target_physics:279:53'), {}).get(key)
    if entry is _literal('registered_target_physics:280:16') or entry.get(_literal('registered_target_physics:280:34')) != target[_literal('registered_target_physics:280:70')]:
        raise ValueError(_literal('registered_target_physics:281:25'))
    profiles = [root / name for name in _global('PHYSICS_FILES') if (root / name).is_file()]
    if len(profiles) != _literal('registered_target_physics:283:24'):
        raise ValueError(_literal('registered_target_physics:284:25'))
    context = (root / entry[_literal('registered_target_physics:285:28')]).resolve()
    if Path(entry[_literal('registered_target_physics:286:18')]).is_absolute() or not context.is_relative_to(root):
        raise ValueError(_literal('registered_target_physics:287:25'))
    if sha256_file(context) != entry[_literal('registered_target_physics:288:37')]:
        raise ValueError(_literal('registered_target_physics:289:25'))
    return dict(profile=profile, profile_path=profiles[_literal('registered_target_physics:290:55')], context_path=context, source_runtime=physics_runtime_identity(profile), key=key)

def _session_frame_progress(session, run, frame_count):
    """Count valid explicit frame judgments in one saved session."""
    try:
        decisions = read(session[_literal('_session_frame_progress:480:33')])[_literal('_session_frame_progress:480:51')]
        decision = decisions.get(run)
        if not isinstance(decision, dict):
            return (_literal('_session_frame_progress:483:19'), _literal('_session_frame_progress:483:22'), _literal('_session_frame_progress:483:29'))
        from darsia.fluidflower.tools.calibration_pairwise_review import normalize_frame_choices
        choices = normalize_frame_choices(decision.get(_literal('_session_frame_progress:485:55')), frame_count=frame_count, comparison_mode=_literal('_session_frame_progress:486:53'))
    except (OSError, KeyError, TypeError, ValueError):
        return (_literal('_session_frame_progress:488:15'), _literal('_session_frame_progress:488:18'), _literal('_session_frame_progress:488:25'))
    activity = bool(choices or decision.get(_literal('_session_frame_progress:489:44')) or decision.get(_literal('_session_frame_progress:489:75')))
    resolved = sum((row[_literal('_session_frame_progress:490:23')] != _literal('_session_frame_progress:490:36') for row in choices))
    return (resolved, resolved == frame_count, activity)

def _matches_review_session(session, binding, run, source_model_id):
    if not binding or not binding.get(_literal('_matches_review_session:581:38')):
        return _literal('_matches_review_session:582:15')
    if session.get(_literal('_matches_review_session:583:19')) == _global('VISIBLE_EVIDENCE_ROLE'):
        if session.get(_literal('_matches_review_session:584:23')) != source_model_id:
            return _literal('_matches_review_session:585:19')
    elif session.get(_literal('_matches_review_session:586:21')) != source_model_id:
        return _literal('_matches_review_session:587:15')
    try:
        return Path(session.get(_literal('_matches_review_session:589:33')) or _literal('_matches_review_session:589:53')).resolve() == Path(binding[_literal('_matches_review_session:589:83')]).resolve() and session.get(_literal('_matches_review_session:590:28')) == binding[_literal('_matches_review_session:590:58')] and ((session.get(_literal('_matches_review_session:591:29')) or {}).get(run) == binding[_literal('_matches_review_session:591:79')])
    except (OSError, TypeError, ValueError):
        return _literal('_matches_review_session:593:15')

def saved_validation_sources(model_root, binding, review_sessions=()):
    """List already-verified saved maps without reconstructing training inputs.

    The Studio catalogue supplies its validated model/pipeline binding. Small
    native manifests are still hash checked here; individual photographs and
    numeric maps retain their full checks when the selected frame is opened.
    Older exports without source hashes use the existing discovery route.
    """
    if binding.get(_literal('saved_validation_sources:662:20')) and resolve_review_path(binding[_literal('saved_validation_sources:663:44')]).resolve() != resolve_review_path(model_root).resolve():
        raise ValueError(_literal('saved_validation_sources:665:25'))
    combined = resolve_review_path(binding[_literal('saved_validation_sources:666:43')])
    manifest_path = combined / _literal('saved_validation_sources:667:31')
    manifest = read(manifest_path)
    expected = (binding.get(_literal('saved_validation_sources:669:28')), binding.get(_literal('saved_validation_sources:669:53')), binding.get(_literal('saved_validation_sources:670:28')), binding.get(_literal('saved_validation_sources:670:60')))
    actual = tuple((manifest.get(key) for key in (_literal('saved_validation_sources:672:8'), _literal('saved_validation_sources:672:20'), _literal('saved_validation_sources:672:36'), _literal('saved_validation_sources:672:55'))))
    if actual != expected or not binding.get(_literal('saved_validation_sources:673:45')):
        raise ValueError(_literal('saved_validation_sources:674:25'))
    selected = {str(run).casefold(): str(candidate) for run, candidate in (binding.get(_literal('saved_validation_sources:676:51')) or {}).items()}
    if not selected or len(selected) != int(binding.get(_literal('saved_validation_sources:677:56')) or _literal('saved_validation_sources:677:72')):
        raise ValueError(_literal('saved_validation_sources:678:25'))
    jobs = [job for job in manifest.get(_literal('saved_validation_sources:679:40'), []) if str(job.get(_literal('saved_validation_sources:680:24')) or _literal('saved_validation_sources:680:43')) == selected.get(str(job.get(_literal('saved_validation_sources:680:75')) or _literal('saved_validation_sources:680:85')).casefold())]
    if len(jobs) != len(selected) or {str(job[_literal('saved_validation_sources:681:46')]).casefold() for job in jobs} != set(selected):
        raise ValueError(_literal('saved_validation_sources:682:25'))
    if any((not job.get(_literal('saved_validation_sources:683:23')) for job in jobs)):
        return _literal('saved_validation_sources:684:15')
    rows = []
    for job in jobs:
        run = str(job[_literal('saved_validation_sources:687:22')]).casefold()
        for key, wanted in ((_literal('saved_validation_sources:688:29'), expected[_literal('saved_validation_sources:688:62')]), (_literal('saved_validation_sources:689:29'), expected[_literal('saved_validation_sources:689:55')]), (_literal('saved_validation_sources:689:61'), expected[_literal('saved_validation_sources:689:89')])):
            if job.get(key) != wanted:
                raise ValueError(_literal('saved_validation_sources:691:33') + run)
        directory = Path(job.get(_literal('saved_validation_sources:692:33')) or job[_literal('saved_validation_sources:692:71')])
        if not directory.is_absolute():
            directory = combined / directory
        directory = resolve_review_path(directory)
        native_path = directory / _literal('saved_validation_sources:696:34') / run / _literal('saved_validation_sources:696:58')
        if sha256_file(native_path) != job[_literal('saved_validation_sources:697:43')]:
            raise ValueError(_literal('saved_validation_sources:698:29') + run)
        native = read(native_path)
        frames = native.get(_literal('saved_validation_sources:700:28')) or []
        count = int(native.get(_literal('saved_validation_sources:701:31')) or _literal('saved_validation_sources:701:49'))
        if count <= _literal('saved_validation_sources:702:20') or count != len(frames):
            raise ValueError(_literal('saved_validation_sources:703:29') + run)
        times = [float(frame[_literal('saved_validation_sources:704:29')]) for frame in frames]
        if any((not math.isfinite(value) for value in times)) or any((right <= left for left, right in zip(times, times[_literal('saved_validation_sources:706:73'):]))) or any((frame.get(_literal('saved_validation_sources:707:33')) != index for index, frame in enumerate(frames))):
            raise ValueError(_literal('saved_validation_sources:708:29') + run)
        dataset = series_id(job.get(_literal('saved_validation_sources:709:36')) or manifest.get(_literal('saved_validation_sources:709:63')))
        parent = dict(combined_root=str(combined), target_model_id=binding[_literal('saved_validation_sources:710:75')], candidate=selected[run])
        row = dict(run=run, dataset=dataset, source_manifest=str(manifest_path), review_root=str(directory), native_manifest=str(native_path), native_manifest_sha256=job[_literal('saved_validation_sources:714:39')], frame_count=count, total_frames=count, job=dict(job), prepared=_literal('saved_validation_sources:715:56'), preparation_status=_literal('saved_validation_sources:715:81'), parent_review_binding=parent, completed_evaluations={}, available_physics_modes=[], physics_mode=_literal('saved_validation_sources:717:53'), training_seen=_literal('saved_validation_sources:717:71'), optical_training_status=_literal('saved_validation_sources:718:36'), calibration_evidence_status=_literal('saved_validation_sources:718:78'), review_status=_literal('saved_validation_sources:719:26'), reviewed_frames=_literal('saved_validation_sources:719:58'), label=f'{run.upper()} · saved phase maps · {count} frames')
        activity = complete = _literal('saved_validation_sources:721:30')
        for session in review_sessions:
            if _matches_review_session(session, parent, run, binding[_literal('saved_validation_sources:723:69')]):
                reviewed, resolved, active = _session_frame_progress(session, run, count)
                row[_literal('saved_validation_sources:725:20')] = max(row[_literal('saved_validation_sources:725:49')], reviewed)
                complete |= resolved
                activity |= active
        row[_literal('saved_validation_sources:728:12')] = _literal('saved_validation_sources:728:31') if complete else _literal('saved_validation_sources:728:59') if activity else _literal('saved_validation_sources:728:90')
        rows.append(row)
    return sorted(rows, key=lambda row: (int(re.search(_literal('saved_validation_sources:730:55'), row[_literal('saved_validation_sources:730:67')]).group()), row[_literal('saved_validation_sources:730:89')]))

def verify_hashes(records):
    for name, expected in records.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f'Evaluation input changed: {name}')

def validate_review_binding(plan, binding):
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness
    if binding.get(_literal('validate_review_binding:958:19'), _literal('validate_review_binding:958:34')) != _literal('validate_review_binding:958:47'):
        raise ValueError(_literal('validate_review_binding:959:25'))
    root = Path(plan[_literal('validate_review_binding:960:21')])
    require_reference_readiness(root / _literal('validate_review_binding:961:39'), allow_pending=_literal('validate_review_binding:961:67'))
    verify_hashes(plan[_literal('validate_review_binding:962:23')])
    if physics_runtime_identity(load_physics_profile(root)) != plan[_literal('validate_review_binding:963:68')]:
        raise ValueError(_literal('validate_review_binding:964:25'))
    if set(binding[_literal('validate_review_binding:965:19')]) != {plan[_literal('validate_review_binding:965:52')]} or binding.get(_literal('validate_review_binding:965:75')) != plan[_literal('validate_review_binding:965:102')]:
        raise ValueError(_literal('validate_review_binding:966:25'))
    if (binding[_literal('validate_review_binding:967:16')], binding[_literal('validate_review_binding:967:37')]) != (plan[_literal('validate_review_binding:967:63')], plan[_literal('validate_review_binding:967:92')]):
        raise ValueError(_literal('validate_review_binding:968:25'))

def _rig_prepared_photographs(review_root, run, manifest=None):
    """Availability hashes only small certificates; selected reads hash JPEGs."""
    state = load_native_source(review_root, run, manifest)
    for row in [state[_literal('rig_prepared_photographs:125:22')][_literal('rig_prepared_photographs:125:32')], *state[_literal('rig_prepared_photographs:125:52')][_literal('rig_prepared_photographs:125:62')]]:
        for mode in (_literal('rig_prepared_photographs:126:21'), _literal('rig_prepared_photographs:126:31'), _literal('rig_prepared_photographs:126:39'), _literal('rig_prepared_photographs:126:47'), _literal('rig_prepared_photographs:126:56'), _literal('rig_prepared_photographs:126:69')):
            bound_path(row[_literal('rig_prepared_photographs:127:27')][mode], base=state[_literal('rig_prepared_photographs:127:59')].parent, verify=_literal('rig_prepared_photographs:127:87'))
    return bound_path(state[_literal('rig_prepared_photographs:128:28')][_literal('rig_prepared_photographs:128:38')][_literal('rig_prepared_photographs:128:51')][_literal('rig_prepared_photographs:128:66')], base=state[_literal('rig_prepared_photographs:129:33')].parent, verify=_literal('rig_prepared_photographs:129:61'))

def write_metadata(path, value, *, replace_attempts=10, retry_delay=0.05):
    """Atomically publish JSON despite short Windows reader sharing conflicts."""
    if replace_attempts < _literal('write_metadata:47:26') or retry_delay < _literal('write_metadata:47:45'):
        raise ValueError(_literal('write_metadata:48:25'))
    path = Path(path)
    temporary = _literal('write_metadata:50:16')
    try:
        with tempfile.NamedTemporaryFile(mode=_literal('write_metadata:52:46'), encoding=_literal('write_metadata:52:60'), dir=path.parent, prefix=f'.{path.name}.', suffix=_literal('write_metadata:53:48'), delete=_literal('write_metadata:53:63')) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=_literal('write_metadata:55:50'), indent=_literal('write_metadata:55:64'), allow_nan=_literal('write_metadata:55:77'))
        for attempt in range(replace_attempts):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == replace_attempts - _literal('write_metadata:61:49'):
                    raise
                time.sleep(min(retry_delay * _literal('write_metadata:63:45') ** attempt, _literal('write_metadata:63:59')))
    finally:
        if temporary is not _literal('write_metadata:65:28'):
            try:
                temporary.unlink(missing_ok=_literal('write_metadata:67:44'))
            except OSError:
                pass
