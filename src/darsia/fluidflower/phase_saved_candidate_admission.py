"""Exact saved candidate and deferred-presentation admission; never materialize."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .phase_saved_review_metadata import configured_parameters
from .phase_validation import _model_identity
from .project import sha256_file
from .review_storage import resolve_review_path


def _parameters():
    return configured_parameters("saved_candidate_admission")


def _typed(item):
    from .phase_saved_review_metadata import typed_saved_data_value
    return typed_saved_data_value(item)


def _literal(key):
    return _typed(_parameters()["literals"][key])


def _global(name):
    return _typed(_parameters()["globals"][name])


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding=_literal('_read:56:47')))
    if not isinstance(value, dict):
        raise ValueError(f'Expected an object: {path}')
    return value

def _require(condition: bool, message: str) -> _literal('_require:62:47'):
    if not condition:
        raise ValueError(message)

def saved_review_candidate(selection_dir: Path, job: dict):
    """Adapt registered saved maps with their native timeline, without mass files."""
    from darsia.fluidflower.tools.calibration_pairwise_review import Candidate
    from darsia.fluidflower.phase_saved_registration import registration_path, registration_warning
    selection_dir = resolve_review_path(selection_dir).resolve()

    def path(value):
        value = Path(value)
        return resolve_review_path(value if value.is_absolute() else selection_dir / value).resolve()

    def same_directory(left, right):
        try:
            return Path(left).samefile(right)
        except OSError:
            return _literal('saved_review_candidate:248:19')
    root = path(job[_literal('saved_review_candidate:249:20')])
    warning = registration_warning(root)
    seam_registration = _literal('saved_review_candidate:251:24')
    combined_registration = _literal('saved_review_candidate:252:28')
    if warning.startswith(_literal('saved_review_candidate:253:26')):
        value = _read(registration_path(root.name))
        if value.get(_literal('saved_review_candidate:255:21')) == _literal('saved_review_candidate:255:45'):
            from darsia.fluidflower.phase_saved_revision_review import t4_saved_review
            checked = t4_saved_review(root)
            _require(checked[_literal('saved_review_candidate:258:29')] == job.get(_literal('saved_review_candidate:258:59')), _literal('saved_review_candidate:259:21'))
            combined_registration = checked[_literal('saved_review_candidate:260:44')]
        elif value.get(_literal('saved_review_candidate:261:23')) == _literal('saved_review_candidate:261:47'):
            from darsia.fluidflower.phase_saved_revision_review import l1_neighbourhood_saved_review
            checked = l1_neighbourhood_saved_review(root)
            _require(checked[_literal('saved_review_candidate:264:29')] == job.get(_literal('saved_review_candidate:264:59')), _literal('saved_review_candidate:265:21'))
            combined_registration = checked[_literal('saved_review_candidate:266:44')]
        elif value.get(_literal('saved_review_candidate:267:23')) == _literal('saved_review_candidate:267:47'):
            from darsia.fluidflower.phase_saved_revision_review import t3_saved_review
            checked = t3_saved_review(root)
            _require(checked[_literal('saved_review_candidate:270:29')] == job.get(_literal('saved_review_candidate:270:59')), _literal('saved_review_candidate:271:21'))
            combined_registration = checked[_literal('saved_review_candidate:272:44')]
    if warning.startswith(_literal('saved_review_candidate:273:26')) and (not warning.startswith(_literal('saved_review_candidate:273:81'))):
        value = _read(registration_path(root.name))
        if value.get(_literal('saved_review_candidate:275:21')) == _literal('saved_review_candidate:275:45'):
            seam_registration = value
        elif value.get(_literal('saved_review_candidate:277:23')) == _literal('saved_review_candidate:277:47'):
            from darsia.fluidflower.phase_saved_revision_review import combined_saved_review
            combined = combined_saved_review(root)
            _require(combined[_literal('saved_review_candidate:280:30')] == job.get(_literal('saved_review_candidate:280:60')), _literal('saved_review_candidate:281:21'))
            combined_registration = combined[_literal('saved_review_candidate:282:45')]
    _require(warning.startswith(_literal('saved_review_candidate:283:32')) or seam_registration is not _literal('saved_review_candidate:283:100') or combined_registration is not _literal('saved_review_candidate:283:137'), _literal('saved_review_candidate:284:13'))
    identity = _model_identity(root)
    if seam_registration is not _literal('saved_review_candidate:286:32'):
        seal = _read(root / _literal('saved_review_candidate:287:28'))
        plan = seal.get(_literal('saved_review_candidate:288:24')) or {}
        _require(seal.get(_literal('saved_review_candidate:292:26')) is _literal('saved_review_candidate:292:41') and seal.get(_literal('saved_review_candidate:292:59')) == list(identity) and (seal.get(_literal('saved_review_candidate:293:30')) == job.get(_literal('saved_review_candidate:293:60'))) and same_directory(path(plan.get(_literal('saved_review_candidate:294:50')) or _literal('saved_review_candidate:294:61')), root / _literal('saved_review_candidate:294:73')) and (plan.get(_literal('saved_review_candidate:295:30')) == seam_registration.get(_literal('saved_review_candidate:295:65'))) and (seal.get(_literal('saved_review_candidate:296:30')) is _literal('saved_review_candidate:296:50')) and (seal.get(_literal('saved_review_candidate:296:69')) is _literal('saved_review_candidate:296:95')), _literal('saved_review_candidate:297:17'))
    binding = _read(root / _literal('saved_review_candidate:298:27'))
    manifest = _read(root / _literal('saved_review_candidate:299:28'))
    run = str(job[_literal('saved_review_candidate:300:18')]).casefold()
    flags = job.get(_literal('saved_review_candidate:301:20')) or {}
    _require(flags.get(_literal('saved_review_candidate:302:23')) is _literal('saved_review_candidate:302:52') and flags.get(_literal('saved_review_candidate:303:23')) is _literal('saved_review_candidate:303:51') and (flags.get(_literal('saved_review_candidate:304:23')) is _literal('saved_review_candidate:304:51')) and (job.get(_literal('saved_review_candidate:305:21')) == _literal('saved_review_candidate:305:39')) and (job.get(_literal('saved_review_candidate:305:60')) is _literal('saved_review_candidate:305:83')), _literal('saved_review_candidate:306:13'))
    _require((binding.get(_literal('saved_review_candidate:307:26')), binding.get(_literal('saved_review_candidate:307:51'))) == identity and (manifest.get(_literal('saved_review_candidate:308:27')), manifest.get(_literal('saved_review_candidate:308:53'))) == identity and (binding.get(_literal('saved_review_candidate:309:25')) == manifest.get(_literal('saved_review_candidate:309:60')) == job.get(_literal('saved_review_candidate:309:90'))) and (job.get(_literal('saved_review_candidate:310:21')) == identity[_literal('saved_review_candidate:310:57')]) and ((binding.get(_literal('saved_review_candidate:311:26')) or {}).get(run) == job.get(_literal('saved_review_candidate:311:76'))), _literal('saved_review_candidate:312:13'))
    matches = [value for value in manifest[_literal('saved_review_candidate:313:43')] if str(value[_literal('saved_review_candidate:313:64')]).casefold() == run]
    _require(len(matches) == _literal('saved_review_candidate:314:29') and matches[_literal('saved_review_candidate:314:43')] == job, _literal('saved_review_candidate:315:13'))
    selected_binding = _read(selection_dir / _literal('saved_review_candidate:316:45'))
    selected_manifest = _read(selection_dir / _literal('saved_review_candidate:317:46'))
    if same_directory(selection_dir, path(binding[_literal('saved_review_candidate:318:50')])) or same_directory(selection_dir, root / _literal('saved_review_candidate:318:109')):
        _require(selected_binding == binding and selected_manifest == manifest and (_read(path(binding[_literal('saved_review_candidate:320:36')]) / _literal('saved_review_candidate:320:56')) == binding) and (_read(path(binding[_literal('saved_review_candidate:321:36')]) / _literal('saved_review_candidate:321:56')) == manifest), _literal('saved_review_candidate:322:17'))
    else:
        package = root / _literal('saved_review_candidate:324:25') / run
        source_sha = sha256_file(root / _literal('saved_review_candidate:325:40') / run / _literal('saved_review_candidate:325:64'))
        _require(same_directory(selection_dir, package) and same_directory(path(selected_binding[_literal('saved_review_candidate:327:54')]), package) and (selected_manifest.get(_literal('saved_review_candidate:328:39')) == [job]) and (selected_binding.get(_literal('saved_review_candidate:329:38')) == {run: job[_literal('saved_review_candidate:329:73')]}) and all((value.get(_literal('saved_review_candidate:330:31')) == identity[_literal('saved_review_candidate:330:55')] and value.get(_literal('saved_review_candidate:330:72')) == identity[_literal('saved_review_candidate:330:100')] and (value.get(_literal('saved_review_candidate:331:31')) == binding[_literal('saved_review_candidate:331:61')]) and (value.get(_literal('saved_review_candidate:332:31')) == binding.get(_literal('saved_review_candidate:332:65'))) and (value.get(_literal('saved_review_candidate:333:31')) == _literal('saved_review_candidate:333:49')) and (value.get(_literal('saved_review_candidate:333:72')) == source_sha) for value in (selected_binding, selected_manifest))), _literal('saved_review_candidate:335:17'))
    review, maps = (path(job[_literal('saved_review_candidate:336:28')]), path(job[_literal('saved_review_candidate:336:60')]))
    _require(review.is_dir() and maps.is_dir() and maps.is_relative_to(root / _literal('saved_review_candidate:337:78')), _literal('saved_review_candidate:338:13'))
    native = review / _literal('saved_review_candidate:339:22') / run / _literal('saved_review_candidate:339:46')
    registration = combined_registration if combined_registration is not _literal('saved_review_candidate:340:74') else seam_registration if seam_registration is not _literal('saved_review_candidate:341:66') else _read(registration_path(identity[_literal('saved_review_candidate:341:109')]))
    _require(sha256_file(native) == (registration.get(_literal('saved_review_candidate:342:54')) or {}).get(run), _literal('saved_review_candidate:343:13'))
    source = _read(native)
    count = source.get(_literal('saved_review_candidate:345:23'))
    _require(type(count) is int and count > _literal('saved_review_candidate:346:44') and (len(source.get(_literal('saved_review_candidate:346:65')) or []) == count) and (source.get(_literal('saved_review_candidate:347:24')) == job.get(_literal('saved_review_candidate:347:46')) == _literal('saved_review_candidate:347:60')) and (source.get(_literal('saved_review_candidate:347:82')) == run), _literal('saved_review_candidate:348:13'))
    return Candidate(run=run, role=job[_literal('saved_review_candidate:349:39')], family=job.get(_literal('saved_review_candidate:349:63'), _literal('saved_review_candidate:349:73')), variant=job[_literal('saved_review_candidate:350:20')], candidate_id=job[_literal('saved_review_candidate:350:49')], grid_order=job.get(_literal('saved_review_candidate:350:85'), _literal('saved_review_candidate:350:99')), params_sha256=job[_literal('saved_review_candidate:351:26')], normalized_mae=_literal('saved_review_candidate:351:59'), mass_plot=root / _literal('saved_review_candidate:352:25'), contour_movie=root / _literal('saved_review_candidate:352:74'), per_frame_csv=native, summary_path=root / _literal('saved_review_candidate:353:50'), plot_seconds=_literal('saved_review_candidate:353:79'), image_seconds=_literal('saved_review_candidate:353:97'), fps=_literal('saved_review_candidate:353:105'), frame_count=count, quality_flags=flags, dataset=job[_literal('saved_review_candidate:354:60')], review_output_dir=review, phase_map_output_dir=maps, phase_model_only=_literal('saved_review_candidate:355:52'), optical_model_sha256=job[_literal('saved_review_candidate:355:83')], raw_phase_map_output_dir=_literal('saved_review_candidate:356:33'), presence_diagnostics_dir=_literal('saved_review_candidate:356:64'), phase_support_format=job.get(_literal('saved_review_candidate:357:37'), _literal('saved_review_candidate:357:61')))

def validate_deferred_presentation(summary, *, run, candidate_id, params_sha256, review_output_dir, phase_map_output_dir, per_frame_csv):
    """Accept absent display media only for an explicit, identity-bound native package."""
    descriptor = summary.get(_literal('validate_deferred_presentation:226:29')) or {}
    if summary.get(_literal('validate_deferred_presentation:227:19')) != _literal('validate_deferred_presentation:227:45') or descriptor.get(_literal('validate_deferred_presentation:227:74')) != configured_parameters('presentation')['deferred_format']:
        raise ValueError(_literal('validate_deferred_presentation:228:25'))
    for key, expected in ((_literal('validate_deferred_presentation:229:27'), run), (_literal('validate_deferred_presentation:229:41'), candidate_id), (_literal('validate_deferred_presentation:229:73'), params_sha256)):
        if descriptor.get(key) != expected:
            raise ValueError(f'Deferred presentation identifies another {key}')
    native_path = resolve_review_path(review_output_dir) / _literal('validate_deferred_presentation:232:59') / run / _literal('validate_deferred_presentation:232:83')
    if resolve_review_path(descriptor[_literal('validate_deferred_presentation:233:39')]) != native_path.resolve() or sha256_file(native_path) != descriptor[_literal('validate_deferred_presentation:234:54')] or resolve_review_path(descriptor[_literal('validate_deferred_presentation:235:46')]) != resolve_review_path(phase_map_output_dir):
        raise ValueError(_literal('validate_deferred_presentation:236:25'))
    native = _read(native_path)
    count = int(descriptor[_literal('validate_deferred_presentation:238:27')])
    with Path(per_frame_csv).open(encoding=_literal('validate_deferred_presentation:239:43'), newline=_literal('validate_deferred_presentation:239:64')) as stream:
        times = [float(row[_literal('validate_deferred_presentation:240:27')]) for row in csv.DictReader(stream)]
    expected = [float(row[_literal('validate_deferred_presentation:241:26')]) for row in native[_literal('validate_deferred_presentation:241:58')]]
    if count < _literal('validate_deferred_presentation:242:15') or len(expected) != count or int(native[_literal('validate_deferred_presentation:242:57')]) != count or (times != expected):
        raise ValueError(_literal('validate_deferred_presentation:243:25'))
    if not np.isfinite(times).all():
        raise ValueError(_literal('validate_deferred_presentation:245:25'))
    if int((summary.get(_literal('validate_deferred_presentation:246:24')) or {}).get(_literal('validate_deferred_presentation:246:63'), _literal('validate_deferred_presentation:246:90'))) != count:
        raise ValueError(_literal('validate_deferred_presentation:247:25'))
    return (descriptor, native, times)
