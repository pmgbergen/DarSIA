"""Durable exact human chart decisions beside one immutable transfer review.

Saved proposal bytes retain their v1/v2/v3 meaning. Loading never creates an
approval, recomputes a mask or changes an existing review/version/geometry.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile

from .painting_chart_transfer import _json, _validate_approval
from .storage_paths import resolve_storage_path, same_storage_descriptor

FORMAT = 'darsia-chart-review-decisions-v1'
MAX_BYTES = 16 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _same_sources(first, second):
    """Retain literal source DATA, resolving only explicit file descriptors."""
    if isinstance(first, dict) and isinstance(second, dict):
        if {'path', 'sha256', 'size_bytes'} <= first.keys() and {'path', 'sha256', 'size_bytes'} <= second.keys():
            return same_storage_descriptor(first, second)
        return first.keys() == second.keys() and all(_same_sources(first[key], second[key]) for key in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(_same_sources(a, b) for a, b in zip(first, second))
    return _json(first) == _json(second)


def _sources(value, seen=None):
    """Verify explicit source descriptors without decoding RAWs or arrays."""
    from .painting_chart_transfer_reader import _checked
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if {'path', 'sha256', 'size_bytes'} <= value.keys():
            identity = (value['path'], value['sha256'], value['size_bytes'])
            if identity not in seen:
                _checked(value)
                seen.add(identity)
        for item in value.values():
            _sources(item, seen)
    elif isinstance(value, list):
        for item in value:
            _sources(item, seen)
    return seen


class ChartApprovalStore:
    def __init__(self, session, path=None, *, decision_proposal_format=None):
        review = session.review
        bound = review.get('review_binding')
        if not isinstance(bound, dict) or set(bound) != {'path', 'sha256', 'size_bytes'}:
            raise ValueError('Persistent chart decisions require an exact saved review binding')
        self.session = session
        decision_format = decision_proposal_format or getattr(session, 'decision_proposal_format', None)
        filename = 'chart_review_decisions.json'
        if decision_format is not None:
            from .painting_chart_transfer import DETECTED_FORMAT
            if decision_format != DETECTED_FORMAT:
                raise ValueError('Unsupported current chart decision policy')
            filename = 'chart_review_decisions_v3.json'
        self.path = resolve_storage_path(path) if path is not None else resolve_storage_path(bound['path']).parent / filename
        self.binding = dict(review=deepcopy(bound), review_content_sha256=_digest(review),
            geometry_index=deepcopy(review['geometry_index']),
            painting_facies=deepcopy(review['source']['painting_facies']),
            detection_manifest=deepcopy(session.manifest_binding),
            runs=[row['run_id'] for row in review['runs']])
        if decision_format is not None:
            self.binding['decision_proposal_format'] = decision_format
        if len(set(self.binding['runs'])) != len(self.binding['runs']):
            raise ValueError('Persistent chart decisions require distinct selected runs')
        self._seen = None

    @classmethod
    def for_session(cls, session, *, decision_proposal_format=None):
        # Synthetic/in-memory previews have no on-disk review to persist to.
        # The ordinary ChartTransferSession always supplies this binding.
        if 'review_binding' not in session.review:
            return None
        return cls(session, decision_proposal_format=decision_proposal_format)

    def _read(self):
        from .painted_adoption_table import _json as strict_json
        if not self.path.exists():
            return None, None
        with self.path.open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('Saved chart decisions exceed their bounded metadata format')
        return strict_json(raw), hashlib.sha256(raw).hexdigest()

    def _validate(self, state):
        stored = state.get('binding') if type(state) is dict else None
        expected = deepcopy(self.binding)
        same_review = isinstance(stored, dict) and same_storage_descriptor(stored.get('review'), self.binding['review'])
        if same_review:
            # The retained digest includes its transient review locator. Reuse
            # that exact locator only after its full descriptor matches; every
            # serialized review field still participates in the original digest.
            expected['review_content_sha256'] = _digest(dict(self.session.review, review_binding=stored['review']))
        if (type(state) is not dict or set(state) != {'format', 'binding', 'approvals', 'payloads', 'adjustments', 'confirmations'}
                or state['format'] != FORMAT or not same_review or not _same_sources(stored, expected)):
            raise ValueError('Saved chart decisions bind another review, map version or geometry')
        _sources(self.binding)
        expected = set(self.binding['runs'])
        for field in ('approvals', 'payloads', 'adjustments', 'confirmations'):
            if type(state[field]) is not dict or not set(state[field]) <= expected:
                raise ValueError('Saved chart decisions contain a foreign selected run')
        if set(state['approvals']) != set(state['payloads']):
            raise ValueError('Saved chart decisions lost an original approved proposal')
        for run, correction in state['adjustments'].items():
            import math
            if (type(correction) is not list or len(correction) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in correction)):
                raise ValueError('Saved chart adjustment must retain finite millimetre values')
        for checks in state['confirmations'].values():
            if type(checks) is not list or len(checks) != 2 or any(type(item) is not bool for item in checks):
                raise ValueError('Saved chart checks must retain two explicit choices')
        rows = {row['run_id']: row for row in self.session.review['runs']}
        reference = self.session.review['source']['reference_run']
        seen = set()
        for run, approval in state['approvals'].items():
            from .painted_adoption_table import _json as strict_json
            payload = state['payloads'][run]
            if type(payload) is not str:
                raise ValueError('Saved approval must retain the original proposal bytes')
            raw = payload.encode('utf8')
            metadata = strict_json(raw)
            if _json(metadata) != raw:
                raise ValueError('Saved proposal bytes are not the original canonical payload')
            _validate_approval(metadata, raw, approval)
            if (self.binding.get('decision_proposal_format') is not None
                    and metadata['format'] != self.binding['decision_proposal_format']):
                raise ValueError('Historical chart decisions cannot approve a new detected whole-chart review')
            if (metadata['run_id'] != run or state['confirmations'].get(run) != [True, True]
                    or _json(metadata['correction_mm']) != _json(state['adjustments'].get(run, [0., 0.]))):
                raise ValueError('Saved approval differs from its run adjustment or confirmations')
            source = metadata['source_binding']
            expected_source = dict(review=self.binding['review'], painting_facies=self.binding['painting_facies'],
                run_id=run, reference_model=rows[reference]['target_model'],
                target_model=rows[run]['target_model'], base_product=rows[run]['product'],
                detection_manifest=self.binding['detection_manifest'])
            if any(not _same_sources(source.get(key), value) for key, value in expected_source.items()):
                raise ValueError('Saved approval lost its exact review/model/map/detection source')
            _sources(source, seen)
        return state

    def load(self):
        state, digest = self._read()
        if state is None:
            _sources(self.binding)
            self._seen = None
            return dict(approvals={}, payloads={}, adjustments={}, confirmations={})
        self._validate(state)
        self._seen = digest
        return {key: deepcopy(state[key]) for key in ('approvals', 'payloads', 'adjustments', 'confirmations')}

    def confirmed_proposal(self, proposal, approval, *, saved_state=None):
        """Replay exact confirmed bytes only when fresh source/result DATA match."""
        from .painted_adoption_table import _json as strict_json
        from .painting_chart_transfer import ChartTransferProposal
        state = self.load() if saved_state is None else saved_state
        run = approval.get('run_id')
        if (run not in state['approvals'] or _json(state['approvals'][run]) != _json(approval)
                or type(state['payloads'].get(run)) is not str):
            raise ValueError('The original confirmed proposal is unavailable for this exact approval')
        payload = state['payloads'][run].encode('utf8')
        retained = strict_json(payload)
        _validate_approval(retained, payload, approval)
        fresh = proposal.validate()
        if (_json({key: value for key, value in retained.items() if key != 'source_binding'})
                != _json({key: value for key, value in fresh.items() if key != 'source_binding'})
                or not _same_sources(retained.get('source_binding'), fresh.get('source_binding'))):
            raise ValueError('The confirmed proposal source or result changed beyond an explicit storage relocation')
        _sources(retained['source_binding'])
        confirmed = ChartTransferProposal(payload, proposal.result, proposal.display_outlines)
        _validate_approval(confirmed.validate(), confirmed.payload, approval)
        return confirmed

    def save(self, approvals, payloads, adjustments, confirmations):
        state = dict(format=FORMAT, binding=deepcopy(self.binding), approvals=deepcopy(approvals),
            payloads={run: payloads[run] for run in approvals}, adjustments=deepcopy(adjustments),
            confirmations=deepcopy(confirmations))
        self._validate(state)
        raw = _json(state)
        if len(raw) > MAX_BYTES:
            raise ValueError('Saved chart decisions exceed their bounded metadata format')
        _previous, digest = self._read()
        if digest != self._seen:
            raise ValueError('Chart decisions changed in another view; reopen this review before saving')
        handle, name = tempfile.mkstemp(prefix='.chart-decisions-', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(handle, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if Path(name).exists():
                Path(name).unlink()
        self._seen = hashlib.sha256(raw).hexdigest()
