"""One recorded display acquisition for a run; current scientific geometry is unchanged.

The series configuration names the decision and the photograph
(``[decisions.clearwater_source]`` in series.toml; see series_decisions). The
code checks only that the recorded decision exists and is unchanged. The
alternate instrument mapping is source-bound, not a replacement geometry or a
calibration for other photographs. The default RAW reader is unchanged.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

FORMAT = 'darsia-ac45-clearwater-source-policy-v1'  # The saved policy format (unchanged).
AUTHORIZATION_FORMAT = 'darsia-clearwater-source-authorization-v1'
SETTING = 'clearwater_source'
OUTPUT_GRID = 'existing calibrated camera-U pixel centres; x right, y down'
_SETTING_KEYS = {'record_id', 'record_sha256', 'actor', 'confirmed_utc', 'run_id', 'reason', 'selected_rule',
                 'raw', 'copied_raw', 'camera_jpeg', 'current_model', 'current_profile', 'instrument'}
_INSTRUMENT_KEYS = ('profile_id', 'intrinsics_source_profile_id', 'distortion', 'ca_red', 'ca_blue', 'camera_wh')
_AUTHORIZATION_KEYS = {'format', 'actor', 'confirmed_utc', 'run_id', 'reason', 'selected_raw', 'camera_jpeg',
                       'current_model_sha256', 'current_profile_sha256'}
_POLICY_KEYS = {'format', 'run_id', 'authority', 'parent_catalog', 'parent_review', 'raw', 'copied_raw',
                'camera_jpeg', 'current_model', 'current_profile', 'instrument', 'display_only',
                'historical_geometry_used'}


def _api():
    from . import painted_review_comparison as comparison
    return comparison


def _text(value):
    """True for nonempty text."""
    return isinstance(value, str) and bool(value.strip())


def _grid(value):
    """True for a [width, height] pair of positive integers."""
    return isinstance(value, list) and len(value) == 2 and all(type(item) is int and item > 0 for item in value)


def _numbers(value):
    """True for a nonempty list of integers, such as a lens table."""
    return isinstance(value, list) and bool(value) and all(type(item) is int for item in value)


def _asset(value):
    """An exact path/SHA/size descriptor in its canonical key order."""
    result = _api()._descriptor(value)
    if set(value) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('A recorded photograph needs exactly its path, SHA-256 and size')
    return dict(path=result['path'], sha256=result['sha256'], size_bytes=result['size_bytes'])


def choice_settings(decisions):
    """Validate a series' recorded water-photograph choice from its decision settings."""
    from .series_decisions import DecisionSettingUnavailable, checked_pin, decision_setting, utc_text
    value = decision_setting(decisions, SETTING)
    if not isinstance(value, dict) or set(value) != _SETTING_KEYS:
        raise DecisionSettingUnavailable('decisions.clearwater_source needs exactly: ' + ', '.join(sorted(_SETTING_KEYS)))
    if (any(not _text(value[key]) for key in ('record_id', 'actor', 'run_id', 'reason', 'selected_rule'))
            or not isinstance(value['record_sha256'], str) or len(value['record_sha256']) != 64
            or any(character not in '0123456789abcdef' for character in value['record_sha256'])):
        raise DecisionSettingUnavailable('decisions.clearwater_source needs its record, decider, run, reason and rule')
    utc_text(value['confirmed_utc'], 'recorded water-photograph decision time')
    raw, copied, jpeg = (_asset(value[key]) for key in ('raw', 'copied_raw', 'camera_jpeg'))
    if (copied['sha256'], copied['size_bytes']) != (raw['sha256'], raw['size_bytes']):
        raise DecisionSettingUnavailable('The copied photograph must have the recorded photograph bytes')
    profile = value['current_profile']
    if (not isinstance(profile, dict) or set(profile) != {'sha256', 'size_bytes', 'profile_id', 'camera_wh'}
            or not _text(profile['profile_id']) or not _grid(profile['camera_wh'])):
        raise DecisionSettingUnavailable('decisions.clearwater_source.current_profile needs its pin, profile and grid')
    instrument = value['instrument']
    if (not isinstance(instrument, dict) or set(instrument) != set(_INSTRUMENT_KEYS)
            or not _text(instrument['profile_id']) or not _text(instrument['intrinsics_source_profile_id'])
            or not all(_numbers(instrument[key]) for key in ('distortion', 'ca_red', 'ca_blue'))
            or instrument['camera_wh'] != profile['camera_wh']):
        raise DecisionSettingUnavailable('decisions.clearwater_source.instrument needs its mapping on the profile grid')
    pin = checked_pin({key: profile[key] for key in ('sha256', 'size_bytes')}, 'recorded current profile')
    return dict(record_id=value['record_id'], record_sha256=value['record_sha256'], actor=value['actor'],
        confirmed_utc=value['confirmed_utc'], run_id=value['run_id'], reason=value['reason'],
        selected_rule=value['selected_rule'], raw=raw, copied_raw=copied, camera_jpeg=jpeg,
        current_model=checked_pin(value['current_model'], 'recorded current model'),
        current_profile=dict(pin, profile_id=profile['profile_id'], camera_wh=list(profile['camera_wh'])),
        instrument={key: deepcopy(instrument[key]) for key in _INSTRUMENT_KEYS})


def series_choice(series_root):
    """The recorded water-photograph choice named in a series' configuration."""
    from .series_decisions import read_series_decisions
    return choice_settings(read_series_decisions(series_root))


def human_authorization(choice):
    """Plain human-form authorization of the recorded choice; record identifiers stay private."""
    return dict(format=AUTHORIZATION_FORMAT, actor=choice['actor'],
        confirmed_utc=choice['confirmed_utc'], run_id=choice['run_id'], reason=choice['reason'],
        selected_raw=deepcopy(choice['raw']), camera_jpeg=deepcopy(choice['camera_jpeg']),
        current_model_sha256=choice['current_model']['sha256'],
        current_profile_sha256=choice['current_profile']['sha256'])


def _checked_human_authorization(value, policy, choice=None):
    """The saved authorization names who decided, when and why, and matches its policy.

    With the series' recorded ``choice`` it must be exactly that decision.
    """
    from .series_decisions import utc_text
    c = _api()
    try:
        utc_text(value.get('confirmed_utc') if isinstance(value, dict) else None)
        valid = (isinstance(value, dict) and set(value) == _AUTHORIZATION_KEYS
                 and value['format'] == AUTHORIZATION_FORMAT
                 and _text(value['actor']) and _text(value['reason'])
                 and value['run_id'] == policy['run_id']
                 and c._same(value['selected_raw'], policy['raw'])
                 and c._same(value['camera_jpeg'], policy['camera_jpeg'])
                 and value['current_model_sha256'] == policy['current_model']['sha256']
                 and value['current_profile_sha256'] == policy['current_profile']['sha256']
                 and (choice is None or c._same(value, human_authorization(choice))))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('The recorded water-photograph authorization changed')
    return deepcopy(value)


def authority_record(value, choice):
    """Accept the exact recorded answer, including from the append-only answer file."""
    from .series_decisions import find_decision_record
    return find_decision_record(value, choice['record_id'], digest=choice['record_sha256'],
                                where='decision for the water photograph')


def _same_asset(first, second):
    c = _api()
    return (c._same({key: first.get(key) for key in ('sha256', 'size_bytes')},
                    {key: second.get(key) for key in ('sha256', 'size_bytes')})
            and Path(first['path']).resolve() == Path(second['path']).resolve())


def _companion_jpeg(raw, jpeg):
    """True when the JPG is the camera's companion of the RAW: same folder and name."""
    return Path(jpeg['path']).with_suffix('').resolve() == Path(raw['path']).with_suffix('').resolve()


def validate_policy_document(value, *, run_id=None, model=None, profile=None, raw=None, choice=None):
    """Literal source-contract checks; file admission remains in checked_policy.

    Without ``choice`` the saved policy must be complete and consistent; with
    the series' recorded ``choice`` it must also name exactly that choice.
    """
    c = _api(); value = c._json(value)
    if (not isinstance(value, dict) or set(value) != _POLICY_KEYS or value['format'] != FORMAT
            or not _text(value['run_id']) or value['display_only'] is not True
            or value['historical_geometry_used'] is not False
            or run_id is not None and value['run_id'] != run_id
            or choice is not None and value['run_id'] != choice['run_id']):
        raise ValueError('This recorded water-photograph policy applies to another run')
    assets = {key: c._descriptor(value[key]) for key in ('raw', 'copied_raw', 'camera_jpeg')}
    if (not c._same({key: assets['copied_raw'][key] for key in ('sha256', 'size_bytes')},
                    {key: assets['raw'][key] for key in ('sha256', 'size_bytes')})
            or not _companion_jpeg(assets['raw'], assets['camera_jpeg'])
            or choice is not None and any(not _same_asset(assets[key], choice[key]) for key in assets)):
        raise ValueError('The recorded policy names another acquisition or companion JPG')
    current = c._descriptor(value['current_model'])
    camera = c._descriptor(value['current_profile'])
    if (not _text(value['current_profile'].get('profile_id'))
            or not _grid(value['current_profile'].get('camera_wh'))
            or choice is not None and (
                current['sha256'] != choice['current_model']['sha256']
                or current['size_bytes'] != choice['current_model']['size_bytes']
                or camera['sha256'] != choice['current_profile']['sha256']
                or camera['size_bytes'] != choice['current_profile']['size_bytes']
                or value['current_profile'].get('profile_id') != choice['current_profile']['profile_id']
                or not c._same(value['current_profile'].get('camera_wh'), choice['current_profile']['camera_wh']))):
        raise ValueError('The water-photograph choice needs its exact current own geometry and configuration')
    if (model is not None and not c._same(c._descriptor(model), current)
            or profile is not None and not c._same(profile, value['current_profile'])
            or raw is not None and not _same_asset(c._descriptor(raw), assets['raw'])):
        raise ValueError('The water-photograph choice belongs to another source, model or profile')
    instrument = value['instrument']
    if (not isinstance(instrument, dict) or set(instrument) != {*_INSTRUMENT_KEYS, 'output_grid'}
            or instrument['output_grid'] != OUTPUT_GRID
            or not _text(instrument['profile_id']) or not _text(instrument['intrinsics_source_profile_id'])
            or not all(_numbers(instrument[key]) for key in ('distortion', 'ca_red', 'ca_blue'))
            or not c._same(instrument['camera_wh'], value['current_profile']['camera_wh'])
            or choice is not None and not c._same(instrument, dict(choice['instrument'], output_grid=OUTPUT_GRID))):
        raise ValueError('The one-image instrument mapping differs from the recorded choice')
    for key in ('authority', 'parent_catalog', 'parent_review'):
        c._descriptor(value[key])
    return value


def checked_policy(binding, *, row=None, choice=None):
    """Authenticate the decision and all active source bytes, never old sand DATA."""
    c = _api(); value = validate_policy_document(c._checked(binding, metadata=True),
        run_id=None if row is None else row['run_id'],
        model=None if row is None else row['target_model'], choice=choice)
    _checked_human_authorization(c._checked(value['authority'], metadata=True), value, choice)
    model = c._checked(value['current_model'], metadata=True)
    if (model.get('run_id') != value['run_id'] or not c._same(model.get('camera_profile'), value['current_profile'])):
        raise ValueError('The current model/profile binding of the recorded photograph changed')
    c._checked(value['current_profile'], metadata=True)
    c._checked(value['parent_catalog'], metadata=True)
    for key in ('raw', 'copied_raw', 'camera_jpeg'):
        c._checked(value[key])
    return value


def instrument_parameters(parameters, policy):
    """Private copy: the recorded D/CA and today's existing intrinsics source coefficients."""
    value = validate_policy_document(policy)
    instrument = value['instrument']
    result = deepcopy(parameters)
    common = result['common']
    if not _api()._same(common['sensor']['camera_wh'], instrument['camera_wh']):
        raise ValueError('The one-image mapping changes the declared camera-U grid')
    if instrument['profile_id'] in common['profiles']:
        raise ValueError('The current configuration already defines this source-specific instrument')
    common['lens']['ca_red'] = deepcopy(instrument['ca_red'])
    common['lens']['ca_blue'] = deepcopy(instrument['ca_blue'])
    common['profiles'][instrument['profile_id']] = dict(distortion=deepcopy(instrument['distortion']),
        coefficients=deepcopy(parameters['common']['profiles'][instrument['intrinsics_source_profile_id']]['coefficients']))
    return result, instrument['profile_id']


def source_instrument_parameters(parameters, *, binding, path, expected_sha256,
                                 profile_path, profile_sha256, profile_id):
    """The optional display loader cannot expand this authority to another file."""
    c = _api(); value = checked_policy(binding)
    supplied = dict(path=str(profile_path), sha256=profile_sha256, size_bytes=value['current_profile']['size_bytes'],
                    profile_id=profile_id, camera_wh=list(value['current_profile']['camera_wh']))
    validate_policy_document(value, profile=supplied,
        raw=dict(path=str(path), sha256=expected_sha256, size_bytes=value['raw']['size_bytes']))
    original = c._checked(value['current_profile'], metadata=True)
    original = original.get('registration', original)
    if not c._same(original, parameters):
        raise ValueError('The source-specific display received changed current instrument parameters')
    return instrument_parameters(parameters, value)


def selection_from_policy(binding, context, row):
    """The display selection of a recorded policy; the series configuration names the choice."""
    c = _api(); document = validate_policy_document(c._checked(binding, metadata=True))
    choice = series_choice(Path(c._descriptor(context['series_manifest'])['path']).parent)
    # Every unmodified run goes through the complete original selection reader.
    if row['run_id'] != document['run_id']:
        validate_policy_document(document, choice=choice)
        _checked_human_authorization(c._checked(document['authority'], metadata=True), document, choice)
        parent = dict(context, image_roles=dict(context['image_roles'], clearwater=document['parent_catalog']))
        return c._clear_role(parent, row)
    value = checked_policy(binding, row=row, choice=choice)
    return dict(format='darsia-recorded-clearwater-display-selection-v1', run_id=value['run_id'],
        selected_raw=deepcopy(value['raw']), camera_jpeg=deepcopy(value['camera_jpeg']),
        selected_rule=choice['selected_rule'],
        selection_only=True, historical_render_not_reused=True, source_exception=deepcopy(binding),
        instrument=deepcopy(value['instrument']), authority=deepcopy(value['authority']))


def prepare_clearwater_review(source, destination, *, authority, choice):
    """Save only a fresh display-context review. Never replace the open review."""
    from .painted_analysis_exclusions import _review_metadata, MAX_REVIEW_BYTES
    from .pipeline.job_queue import _immutable_bytes, _publish_bytes
    c = _api(); destination = Path(destination)
    folder = destination.with_name(destination.stem + '_water_source')
    if (destination.exists() or destination.is_symlink() or folder.exists() or folder.is_symlink()
            or destination.suffix.casefold() != '.json'):
        raise FileExistsError('Choose a fresh saved review and water-source namespace')
    destination = destination.resolve(); folder = folder.resolve()
    review = c.read_review(source)
    rows = [row for row in review['runs'] if row['run_id'] == choice['run_id']]
    if len(rows) != 1:
        raise ValueError('The saved review has no unique accepted own model for ' + choice['run_id'])
    row = rows[0]; geometry, _, _ = c._model(row)
    context = c.validate_display_context(review['display_context'], review)
    parent_catalog = context['image_roles'].get('clearwater')
    if parent_catalog is None or c._checked(parent_catalog, metadata=True).get('format') != 'darsia-clearwater-display-selections-v1':
        raise ValueError('Choose the original saved review before applying the recorded water-photograph choice')
    authority_record(c._checked(authority, metadata=True), choice)  # Private decision verification only.
    authority_data = _immutable_bytes(human_authorization(choice))
    authority_path = folder / 'authority.json'
    authority_copy = dict(path=str(authority_path), sha256=hashlib.sha256(authority_data).hexdigest(), size_bytes=len(authority_data))
    instrument = choice['instrument']
    value = dict(format=FORMAT, run_id=choice['run_id'], authority=authority_copy,
        parent_catalog=deepcopy(parent_catalog), parent_review=deepcopy(review['review_binding']),
        raw=deepcopy(choice['raw']), copied_raw=deepcopy(choice['copied_raw']), camera_jpeg=deepcopy(choice['camera_jpeg']),
        current_model=deepcopy(row['target_model']), current_profile=deepcopy(geometry.model['camera_profile']),
        instrument=dict(profile_id=instrument['profile_id'],
            intrinsics_source_profile_id=instrument['intrinsics_source_profile_id'],
            distortion=deepcopy(instrument['distortion']), ca_red=deepcopy(instrument['ca_red']),
            ca_blue=deepcopy(instrument['ca_blue']), camera_wh=deepcopy(instrument['camera_wh']),
            output_grid=OUTPUT_GRID),
        display_only=True, historical_geometry_used=False)
    validate_policy_document(value, model=row['target_model'], choice=choice)
    for bound in (value['raw'], value['copied_raw'], value['camera_jpeg'], value['current_model'], value['current_profile'], parent_catalog):
        c._checked(bound)
    policy_data = _immutable_bytes(value); policy_path = folder / 'source_policy.json'
    policy_binding = dict(path=str(policy_path), sha256=hashlib.sha256(policy_data).hexdigest(), size_bytes=len(policy_data))
    result = {key: deepcopy(item) for key, item in review.items() if key != 'review_binding'}
    result['display_context']['image_roles']['clearwater'] = policy_binding
    data = _immutable_bytes(result)
    if len(data) > MAX_REVIEW_BYTES:
        raise ValueError('The fresh saved review exceeds its original read bound')
    # All original scientific metadata, row hashes and source selections remain literal.
    scientific = lambda doc: {key: item for key, item in doc.items() if key not in {'review_binding', 'display_context'}}
    if not c._same(scientific(result), scientific(review)):
        raise ValueError('The display-source review changes original scientific metadata')
    c._checked(review['review_binding']); c._checked(authority)
    for bound in (value['raw'], value['copied_raw'], value['camera_jpeg'], value['current_model'], value['current_profile'], parent_catalog):
        c._checked(bound)
    folder.mkdir(parents=False, exist_ok=False)
    _publish_bytes(authority_path, authority_data, create_parents=False)
    _publish_bytes(policy_path, policy_data, create_parents=False)
    _publish_bytes(destination, data, create_parents=False)
    fresh = c.read_review(destination)
    _review_metadata(fresh)
    return dict(gap_review=deepcopy(fresh['review_binding']), source_policy=policy_binding,
                run_id=choice['run_id'], original_source_files_changed=False, clearwater_selection_changed=True,
                current_review_changed=False, adopted=False, worker_started=False)


class ClearwaterReviewBuilder:
    """Prepare a water-photograph review for the one run the series' recorded choice names."""

    def __init__(self, choice):
        self.choice = deepcopy(choice)
        self.run_id = self.choice['run_id']

    def __call__(self, source, destination, *, authority):
        """Prepare one review with the recorded choice (see prepare_clearwater_review)."""
        return prepare_clearwater_review(source, destination, authority=authority, choice=self.choice)


def clearwater_review_builder(series_root):
    """A builder for the series' recorded water-photograph choice, or None when it names none."""
    from .series_decisions import read_series_decisions
    if series_root is None:
        return None
    try:
        decisions = read_series_decisions(series_root)
    except FileNotFoundError:
        return None
    if SETTING not in decisions:
        return None
    return ClearwaterReviewBuilder(choice_settings(decisions))
