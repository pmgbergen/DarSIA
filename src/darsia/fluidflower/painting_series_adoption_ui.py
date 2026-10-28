"""One explicit painting action: prepare, check, review, confirm, then adopt."""
from copy import deepcopy
from importlib.util import find_spec
import json
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot

from .painting_series_use_ui import matches_version


_VERSION_KEYS = ('folder', 'sha256', 'name', 'kind', 'run_id', 'files', 'proposal_method')
_BACKEND_MODULES = ('darsia.fluidflower.painted_adoption_table',
    'darsia.fluidflower.painted_adoption_geometry', 'darsia.fluidflower.raw_shape_basis_reader')


def automatic_backend_available():
    """Report installed capability only; every operation still runs real admission."""
    return all(find_spec(name) is not None for name in _BACKEND_MODULES)


def _version_identity(record):
    """Keep the literal persisted identity separate from mutable display labels."""
    return {key: deepcopy(record.get(key)) for key in _VERSION_KEYS}


def _chart_source_binding(source):
    """Use the literal chart source; current findings may retain its locator."""
    if not isinstance(source, dict):
        return None
    bound = source.get('chart_source_review', source.get('review_binding', source))
    return (deepcopy(bound) if isinstance(bound, dict)
            and set(bound) == {'path', 'sha256', 'size_bytes'} else None)


class SeriesAdoptionController(QObject):
    """Chain existing GUI workers without granting authority from status metadata."""

    findings_changed = Signal()
    chart_review_unavailable = Signal(object, str)

    def __init__(self, view, *, gap_dialog_factory=None, catalogue_refresher=None):
        """Retain no scientific inputs and start no I/O before the user's action."""
        super().__init__(view)
        self.view = view
        self._gap_dialog_factory = gap_dialog_factory
        self.available = automatic_backend_available()
        self.context = self.preparation = self.dialog = None
        self.error = None
        self.gap_dialog = self._gap_review_source = None
        self.recorded_dialog = None
        self.chart_dialog = None
        self._chart_review_binding = None
        self._chart_generation = 0
        self._chart_failed = False
        self._chart_opening_message = ''
        self._automatic_chart_request = None
        self._automatic_chart_retry_queued = False
        self.chart_review_unavailable.connect(self._chart_review_unavailable)
        self._chart_approvals = {}
        self._pending_local_reuse = None
        self._area_approval = None
        self._placement_review_source = None
        self._findings_required = False
        self._findings_context = self._findings_review_binding = None
        self.catalogue_refresher = catalogue_refresher
        self._catalogue_generation = 0
        self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
        self._catalogue_refresh_queued = False
        self._findings_update_reason = ''
        self._findings_draft_contexts = {}
        self._registered_gap_reviews = []
        self._source_choices = []
        self._warning_choices = {}
        self._target_bindings = {}
        self._preparation_action_generation = 0
        self._completed_preparation_generation = None
        self._water_surface_confirmations = {}
        self.display_context = None
        self._water_surface_tolerances = dict(height_tolerance_mm=3.,angle_tolerance_degrees=.5)
        self._position_tolerances = dict(method='camera registration (map placement)',shift_tolerance_mm=2.,angle_tolerance_degrees=.5)
        self._suspended = False
        self._confirmation_pending = False
        self._abandoned_preparation = False
        self._confirm_timer = QTimer(self)
        self._confirm_timer.setSingleShot(True)
        self._confirm_timer.timeout.connect(self._confirm)
        preview_retired = getattr(getattr(view, 'run_preview', None), 'reader_retired', None)
        if preview_retired is not None:
            preview_retired.connect(self._queue_automatic_chart_review)
        for owner, name in ((view, 'busy_changed'), (getattr(view, 'series_use_status', None), 'changed')):
            signal = getattr(owner, name, None)
            if signal is not None:
                signal.connect(self._findings_changed)
                signal.connect(self._queue_automatic_chart_review)

    def _findings_context_snapshot(self):
        """Capture already loaded GUI identities without reading approval files."""
        view = self.view
        record = view._selected_version()
        if view.project is None or record is None:
            raise ValueError('Select the saved map and its series first.')
        approval, frame = view._approval_for_version(record), view.current_frame()
        if not approval or frame is None:
            raise ValueError('The selected map needs its exact approval and current frame.')
        return dict(project=view.project, project_root=str(view.project.root),
            version=_version_identity(record), approval=deepcopy(approval),
            frame=deepcopy(frame.to_dict()), selection=deepcopy(view.run_preview.geometry_selection()))

    def _same_findings_context(self, snapshot):
        """Require the original project object as well as every literal selection."""
        saved = self._findings_context
        return bool(saved is not None and saved['project'] is snapshot['project'] and saved == snapshot)

    def _retained_findings_state(self, snapshot):
        """Read only the exact completed passive status request, never adoption proof."""
        status = getattr(self.view, 'series_use_status', None)
        desired = getattr(status, 'desired', None)
        if desired is None:
            return ''
        from . import layer_painting as painting
        root = painting.painting_folder(snapshot['project']).parent
        current = Path(snapshot['project'].root) / 'overrides/geometry_facies/current.json'
        expected = json.dumps([str(root), str(current), snapshot['version']['folder'],
            snapshot['version']['sha256'], snapshot['approval']], sort_keys=True)
        if (desired != expected or getattr(status, 'cached', None) != expected
                or _version_identity(getattr(status, 'record', None) or {}) != snapshot['version']
                or getattr(status, 'approval', None) != snapshot['approval']):
            return 'loading'
        result = getattr(status, 'result', None)
        if (not isinstance(result, dict)
                or set(result) != {'transfer', 'preparation', 'adopted', 'gap_reviews', 'errors'}
                or not isinstance(result['errors'], list) or result['errors']
                or not isinstance(result['gap_reviews'], list)
                or any(not isinstance(item, dict) for item in result['gap_reviews'])
                or any(result[key] is not None and not isinstance(result[key], dict)
                       for key in ('transfer', 'preparation', 'adopted'))):
            return 'unavailable'
        return 'known' if result['transfer'] is not None or result['gap_reviews'] else ''

    def findings_block_reason(self):
        """Keep current Transfer findings mandatory even after closing their window."""
        from .manual_map_approval import has_manual_map_approval
        saved = (getattr(getattr(self.view, 'series_use_status', None), 'result', None) or {}).get('preparation')
        if not self._findings_draft_contexts and has_manual_map_approval(saved or {}):
            # Readiness only: start and the original Check authenticate bytes.
            return ''
        status = getattr(self.view, 'series_use_status', None)
        engaged = (self._findings_required or bool(self._findings_draft_contexts)
                   or getattr(status, 'desired', None) is not None)
        try:
            snapshot = self._findings_context_snapshot()
            retained = self._retained_findings_state(snapshot)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return ('The selected project, map, approval or geometry changed. Open Review gaps and findings.'
                    if engaged else '')
        if retained == 'loading':
            return 'Wait for this selected map\'s saved findings status to finish loading.'
        if retained == 'unavailable':
            return ('Saved Transfer and findings history is unavailable. Resolve the reported status read error '
                    'before Check or Apply and Build, then open Review gaps and findings.')
        if self._findings_draft_contexts:
            return ('Unconfirmed area choices remain. Open Review gaps and findings and confirm the '
                    'current area choices before Check, Adopt, or Apply and Build.')
        if not self._findings_required and retained != 'known':
            return ''
        if not self._same_findings_context(snapshot):
            return 'Open Review gaps and findings for the current project, map, approval and geometry first.'
        if self._findings_update_reason:
            return self._findings_update_reason
        findings = self.gap_dialog
        if findings is None or findings.review is None:
            return 'Open Review gaps and findings and wait for the current bound list to finish loading.'
        if findings.review.get('review_binding') != self._findings_review_binding:
            return 'The bound findings changed. Open Review gaps and findings again.'
        return findings.unresolved_findings_reason()

    def set_findings_draft_pending(self, dialog, payload, *, confirmed=False):
        """Retain exact draft edits; only the actual confirmed-C callback may clear them.

        Ordinary draft signals omit confirmed. A genuine new C result may pass
        confirmed=True while its original review row is current.
        """
        keys = {'review_binding', 'run_id', 'component_id', 'review_row_sha256', 'pending'}
        if (dialog is not self.gap_dialog or not isinstance(payload, dict) or set(payload) != keys
                or type(payload['pending']) is not bool):
            return False
        try:
            if (dialog.review is None or not self._same_findings_context(self._findings_context_snapshot())
                    or not self._same_findings_inputs(payload['review_binding'], dialog.review.get('review_binding'))):
                return False
            row = next(component for run in dialog.review['runs'] if run['run_id'] == payload['run_id']
                for component in run['components'] if component['component_id'] == payload['component_id'])
            if (not isinstance(payload['review_row_sha256'], str) or not payload['review_row_sha256']
                    or row['review_row_sha256'] != payload['review_row_sha256']):
                return False
            context = {key: deepcopy(payload[key]) for key in keys if key != 'pending'}
            identity = json.dumps(context, sort_keys=True, allow_nan=False, ensure_ascii=False)
            if payload['pending']:
                pending = getattr(dialog, 'pending_draft_choices', None)
                if callable(pending) and not any(self._same_findings_inputs(item, payload) for item in pending()):
                    return False
                self._findings_draft_contexts[identity] = context
                self._catalogue_generation += 1
                self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
                self._catalogue_refresh_queued = False
            elif confirmed is True and identity in self._findings_draft_contexts:
                del self._findings_draft_contexts[identity]
            else:
                return False
            self.findings_changed.emit()
            return True
        except (AttributeError, KeyError, StopIteration, TypeError, ValueError):
            return False

    @Slot()
    def _findings_changed(self):
        """Relay current list actions to the view and new Setup preparation."""
        self.findings_changed.emit()
        self._try_findings_refresh()

    def _queue_findings_refresh(self):
        """Wait for the current preview to retire before dispatching changed inputs."""
        self._catalogue_generation += 1
        self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
        self._catalogue_refresh_queued = True
        self._findings_update_reason = 'Updating findings for the changed human choices. Wait for the original replay before Check or Apply and Build.'
        self.findings_changed.emit()
        self._try_findings_refresh()

    def _choices_ready_for_refresh(self):
        """Batch human gap/source choices independently of unresolved diagnostics."""
        findings = self.gap_dialog
        if findings is None or findings.review is None or self._findings_draft_contexts:
            return False
        gaps = [row for row in findings._rows if row['kind'] == 'gap']
        needs_source = any(row['kind'] == 'source_state' for row in findings._rows)
        return (all(findings._resolved(row) for row in gaps)
                and (not needs_source or bool(findings._acquisition_actions)))

    def _findings_choices_changed(self):
        """Invalidate every edit, then refresh once the current human choices are complete."""
        self._catalogue_generation += 1
        self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
        self._catalogue_refresh_queued = False
        if self.dialog is not None:
            self.dialog.candidate = self.dialog.checked_candidate = self.dialog.review_resolutions = None
        self._findings_update_reason = ('Finish the current gap choices and choose the source water state. '
            'The original findings will refresh together before Check or Apply and Build.')
        if self._choices_ready_for_refresh():
            self._queue_findings_refresh()
        else:
            if self.gap_dialog is not None:
                self.gap_dialog.status.setText(self._findings_update_reason)
            self.view.use_map_progress.setText(self._findings_update_reason)
            self.findings_changed.emit()

    def _try_findings_refresh(self):
        """Use the existing retirement notification instead of polling busy readers."""
        findings = self.gap_dialog
        if (self._catalogue_refresh_queued and not self._suspended and findings is not None
                and not self._findings_draft_contexts
                and not getattr(findings, '_closing', False) and not findings.busy()):
            self._catalogue_refresh_queued = False
            self.refresh_findings()

    def _allow_findings_check(self):
        """Explain a refused direct Check without dispatching any preparation."""
        reason = self.findings_block_reason()
        if reason:
            self.view.use_map_progress.setText(reason)
            if self.gap_dialog is not None:
                self.gap_dialog.status.setText(reason)
            return False
        return True

    def start(self, record, selection, _displayed_preparation=None, *, analysis_exclusions=None, source_choices=None, warning_choices=None, target_corrections=None, local_reuse=None):
        """Prepare the exact selected map, including explicitly reviewed local reuse."""
        # A newly approved chart changes the effective pixels. Prepare those
        # products before asking for decisions on their fresh findings; the
        # previous mask's findings cannot govern this read-only preparation.
        # Unconfirmed human edits still block, and Check/Save/Build retain
        # their original findings guards.
        if (local_reuse is None or self._findings_draft_contexts) and not self._allow_findings_check():
            return False
        if not self.available:
            self.view.use_map_progress.setText(
                'The rigs need their RAW basis and the automatic adoption backend before the map can be adopted.')
            return False
        if (self.context is not None or self.view.busy()
                or getattr(self.view, "_adoption_thread", None) is not None):
            self.view.use_map_progress.setText('Wait for the current adoption operation to finish.')
            return False
        self._suspended = False
        self._abandoned_preparation = False
        if self.dialog is not None:
            old_dialog = self.dialog
            old_dialog.close()
            if self.view._adoption_dialog is old_dialog:
                self.view._adoption_dialog = None
            self.dialog = None
        self.preparation = self.error = self._gap_review_source = None
        self._completed_preparation_generation = None
        self._area_approval = deepcopy(analysis_exclusions)
        self._source_choices = deepcopy(source_choices or [])
        self._warning_choices = deepcopy(warning_choices or {})
        if target_corrections is None:
            self._target_bindings = {}
        try:
            self.view._read_approvals(str(record['run_id']))
            approval = self.view._approval_for_version(record)
            if not approval:
                raise ValueError('Approve this exact saved version in Versions before adoption.')
            frame = self.view.current_frame()
            if frame is None:
                raise ValueError('The selected saved map needs its current rig frame.')
            self.context = dict(project=self.view.project,
                project_root=str(self.view.project.root),
                version=_version_identity(record), approval=deepcopy(approval),
                frame=deepcopy(frame.to_dict()), selection=deepcopy(selection),
                preparation_generation=self._preparation_action_generation,
                preparation_actions=dict(analysis_exclusions=deepcopy(analysis_exclusions),
                                         target_corrections=deepcopy(target_corrections or [])))
            from .manual_map_approval import has_manual_map_approval, verify_manual_map_approval
            if (_displayed_preparation is not None and local_reuse is None
                    and analysis_exclusions is None and not target_corrections
                    and has_manual_map_approval(_displayed_preparation)):
                from .painted_adoption_table import _checked, _bound
                folder = Path(_displayed_preparation['folder'])
                saved = _checked(_bound(folder / 'index.json'), metadata=True)
                saved['folder'] = str(folder)
                if not verify_manual_map_approval(saved):
                    raise ValueError('The saved preparation has no complete manual map approval')
                self.context['manual_map_approved'] = True
                self.prepared(saved)
                self.preparation_retired()
                return self.preparation is not None
            options = dict(continue_photo_edges=True)
            if local_reuse is not None:
                options["local_reuse"] = deepcopy(local_reuse)
                self.context["chart_approvals"] = deepcopy(local_reuse["approvals"])
            if analysis_exclusions is not None:
                options["analysis_exclusions"] = deepcopy(analysis_exclusions)
            if target_corrections is not None:
                options["target_corrections"] = deepcopy(target_corrections)
            if not self.view.prepare_adoption_of(record, selection.index_path, **options):
                self.context = None
                self.view.use_map_progress.setText(self.view.versions_note.text())
                return False
            return True
        except Exception as error:
            self.context = None
            self.view.use_map_progress.setText(f'The adoption could not start: {error}')
            return False

    def prepared(self, result):
        """Keep the actual preparation until its original worker has retired."""
        if self._abandoned_preparation:
            self._abandoned_preparation = False
            return True
        if self.context is None:
            return False
        try:
            record = self.context['version']
            if (not matches_version(result, record)
                    or result['map_version'].get('approval') != self.context['approval']
                    or result.get('reference_run') != record['run_id']
                    or 'painting_facies' not in result):
                raise ValueError('The preparation does not bind this exact approved painting version.')
            bound = result['geometry_index']
            selected = self.context['selection']
            from .storage_paths import same_storage_path
            if not same_storage_path(bound['path'], selected.index_path):
                raise ValueError('The preparation used another measured geometry index.')
            if getattr(selected, 'sha256', '') and (
                    bound.get('sha256') != selected.sha256
                    or bound.get('size_bytes') != selected.size_bytes):
                raise ValueError('The selected series geometry changed during preparation.')
            expected = self.context.get('preparation_actions', dict(analysis_exclusions=None,target_corrections=[]))
            actual = result.get('gui_preparation_actions')
            if actual is None and expected == dict(analysis_exclusions=None,target_corrections=[]):
                actual = expected  # Preserve old no-action injected worker results.
            if actual != expected:
                raise ValueError('The successful preparation did not include the requested exact area/correction actions.')
            self.preparation = deepcopy(result)
            self._completed_preparation_generation = self.context.get('preparation_generation',self._preparation_action_generation)
            if result.get("gap_review") is not None:
                self.view._last_gap_review_source = deepcopy(result["gap_review"])
            return True
        except (KeyError, TypeError, ValueError) as error:
            self.error = str(error)
            self.preparation = None
            return True

    def failed(self, message):
        """Retain a plain error and let the valid owned worker retire naturally."""
        if self.context is not None:
            self.error, self.preparation = str(message), None

    def _current(self):
        """Check current GUI context; scientific admission remains with the backend."""
        context = self.context
        if context is None or self._suspended:
            return False
        if "chart_approvals" in context and context["chart_approvals"] != self._chart_approvals:
            return False
        view = self.view
        try:
            view._read_approvals(str(context['version']['run_id']))
            selected = view._selected_version()
            frame = view.current_frame()
            return bool(view.project is context['project']
                and str(view.project.root) == context['project_root']
                and selected is not None and _version_identity(selected) == context['version']
                and view._approval_for_version(selected) == context['approval']
                and frame is not None and frame.to_dict() == context['frame']
                and view.run_preview.geometry_selection() == context['selection'])
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return False

    def preparation_retired(self):
        """Open only the automatic existing dialog after native worker retirement."""
        if self.context is None:
            return
        if self._gap_review_source is not None and not self._suspended and self._current():
            source, self._gap_review_source = self._gap_review_source, None
            self.open_gap_review(source)
            return
        if self._suspended or self.error or self.preparation is None or not self._current():
            message = self.error or 'The selected version, approval or geometry changed; start adoption again.'
            self.view.use_map_progress.setText('The map was not adopted: ' + message)
            self.context = None
            return
        if self.context.get('manual_map_approved'):
            self._open_adoption_dialog(self.preparation)
            return
        # The saved preparation is read, and every run's labels verified, in
        # the background (seconds to minutes on the shared drive); the dialog
        # then opens with exactly what was read.
        from .ui_metadata_preview import MetadataPreview
        from . import map_adoption as ma
        reader = self.__dict__.get('_preparation_reader')
        if reader is None:
            reader = self._preparation_reader = MetadataPreview(self.view)
            reader.completed.connect(self._preparation_read)
        folder, context = Path(self.preparation['folder']), self.context
        self.view.use_map_progress.setText('Reading the prepared adoption and its saved labels…')
        reader.start(lambda _cancelled: (context, ma.read_preparation(folder)))

    def _preparation_read(self, value, error):
        """Open the adoption dialog on the preparation read in the background."""
        if self.context is None or (value is not None and value[0] is not self.context):
            return  # Adoption was cancelled or started again meanwhile.
        if error or value is None:
            self.view.use_map_progress.setText('The map was not adopted: ' + (error or 'no answer was returned'))
            self.context = None
            return
        if self._suspended or self.error or self.preparation is None or not self._current():
            message = self.error or 'The selected version, approval or geometry changed; start adoption again.'
            self.view.use_map_progress.setText('The map was not adopted: ' + message)
            self.context = None
            return
        self._open_adoption_dialog(value[1])

    def _open_adoption_dialog(self, preparation_metadata):
        """Open the automatic adoption dialog on the saved preparation and start its Check."""
        from .map_adoption_view import AdoptionDialog
        from .series_decisions import series_actor
        from .series_geometry import default_actor
        try:
            actor = (series_actor(self.view._series_root()) if 'chart_approvals' in self.context
                     else self.view._reviewer() or default_actor())
            dialog = AdoptionDialog(self.view.project, self.preparation['folder'],
                actor=actor, parent=self.view,
                automatic_only=True, findings_managed=not self.context.get('manual_map_approved', False),
                preparation_metadata=preparation_metadata,
                adoption_scope_note=(self.preparation.get('gui_adoption_cohort') or {}).get('summary', ''))
            dialog.check_button.hide()
            dialog.save_button.hide()
            dialog.operation_retired.connect(self._dialog_retired)
            dialog.confirmation_decided.connect(self._confirmation_decided)
            dialog.finished.connect(self._dialog_closed)
            self.dialog = self.view._adoption_dialog = dialog
            if self.display_context is not None:
                dialog.set_display_preferences(self._display_preferences())
            if self._source_choices:
                dialog._facies_source_choice = deepcopy(self._source_choices[-1])
            self._show_adoption_panel()
            if dialog.revision is not None:
                self.view.use_map_progress.setText(dialog.status.text())
                self.context = None
                self.findings_changed.emit()
                return
            if not dialog.check():
                raise ValueError(dialog.status.text())
            self.view.use_map_progress.setText('Checking the automatic facies table and the saved sources…')
        except Exception as error:
            self.view.use_map_progress.setText(f'The map was not adopted: {error}')
            self.context = None

    def _show_adoption_panel(self):
        """Present the retained adoption owner inside a capable workspace."""
        present = getattr(self.view, 'show_adoption_panel', None)
        if callable(present):
            present(self.dialog)
        else:
            self.dialog.show()

    @Slot(object)
    def gap_review_ready(self, source):
        """Retain typed review metadata; opening waits for worker retirement."""
        if self.context is not None and not self._suspended:
            self._findings_required = True
            self._gap_review_source = deepcopy(source)
            self.view._last_gap_review_source = deepcopy(source)
            self.findings_changed.emit()

    @Slot(object)
    def transfer_review_ready(self, source):
        """Retain original apply-all findings until the transfer thread retires."""
        incoming = _chart_source_binding(source)
        dialog = self.chart_dialog
        current = (_chart_source_binding(getattr(getattr(dialog, 'session', None), 'review', None))
                   or self._chart_review_binding
                   or _chart_source_binding(getattr(self.view, '_last_gap_review_source', None)))
        if incoming is None or incoming != current:
            # A new transfer can keep the same painting and geometry while its
            # raster policy/products and exact signed proposal source differ.
            # Retire the old reader before another panel can reuse its state.
            self.chart_dialog = None
            self._chart_context = None
            self._chart_generation += 1
            self._chart_failed = False
            self._chart_approvals = {}
            self._pending_local_reuse = None
            self._preparation_action_generation += 1
            if dialog is not None:
                dialog.close()
        self._chart_review_binding = incoming
        self._findings_required = True
        self._findings_context = self._findings_review_binding = None
        self._area_approval = None
        self._source_choices, self._warning_choices = [], {}
        self._target_bindings = {}
        self._catalogue_generation += 1
        self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
        self._catalogue_refresh_queued = False
        self._findings_update_reason = ''
        self._placement_review_source = deepcopy(source)
        self.view._last_gap_review_source = deepcopy(source)
        self.findings_changed.emit()

    def placement_retired(self):
        """Review actual retained findings; never launch another transfer."""
        if self._placement_review_source is not None and not self._suspended:
            source, self._placement_review_source = self._placement_review_source, None
            if getattr(self.view, '_workflow_scope', None) in {'all', 'apply'}:
                self.open_chart_review(automatic=True)
            else:
                self.open_gap_review(source)

    def _review_context(self, review):
        """A catalogue is reviewable only for the still-current selected sources."""
        if (not self._current() or not matches_version(review, self.context['version'])
                or review['map_version'].get('approval') != self.context['approval']):
            raise ValueError('The gap review belongs to another selected painting version or context.')
        selected, bound = self.context['selection'], review['geometry_index']
        from .storage_paths import same_storage_path
        if (not same_storage_path(bound['path'], selected.index_path)
                or getattr(selected, 'sha256', '') and (
                    bound.get('sha256') != selected.sha256
                    or bound.get('size_bytes') != selected.size_bytes)):
            raise ValueError('The gap review belongs to another measured geometry index.')
        self.display_context = deepcopy(review.get('display_context'))
        self.view._last_gap_review_source = deepcopy(review)
        snapshot = self._findings_context_snapshot()
        self._findings_required = True
        self._findings_context = snapshot
        self._findings_review_binding = deepcopy(review.get('review_binding'))
        if self.gap_dialog is not None:
            self.gap_dialog.set_resolved_target_rows([])
        self.findings_changed.emit()

    def register_gap_review(self, binding):
        """Retain one transient technical descriptor; grant no review authority."""
        if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256', 'size_bytes'}
                or not isinstance(binding['path'], str) or not Path(binding['path']).is_absolute()
                or not isinstance(binding['sha256'], str) or len(binding['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in binding['sha256'])
                or type(binding['size_bytes']) is not int or not 0 < binding['size_bytes'] <= 128 * 1024 * 1024):
            raise ValueError('A literal bounded review path, SHA-256 and size are required.')
        project = self.view.project
        if project is None:
            raise ValueError('Load the series before registering its review descriptor.')
        item = dict(project_root=str(project.root), binding=deepcopy(binding))
        if item not in self._registered_gap_reviews:
            if len(self._registered_gap_reviews) >= 16:
                raise ValueError('Too many transient review descriptors.')
            self._registered_gap_reviews.append(item)

    def current_gap_review(self):
        """Return only the live list for the unchanged, already loaded context.

        Inspecting that existing list is safe during its finite reader. It must
        not become another preparation, collector, or source admission.
        """
        findings = self.gap_dialog
        if self._suspended or findings is None or findings._closing or findings.review is None:
            return None
        try:
            if (self._same_findings_context(self._findings_context_snapshot())
                    and findings.review.get('review_binding') == self._findings_review_binding):
                return findings
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            pass
        return None

    def open_latest_gap_review(self):
        """Resolve known records for the real selection, then use original gates."""
        findings = self.current_gap_review()
        if findings is not None:
            # A modeless review can be hidden behind Main or minimized. Keep
            # its answers and original worker; no busy gate is bypassed for a
            # new operation, and the exact context was checked above.
            if findings.isMinimized():
                findings.showNormal()
            else:
                findings.show()
            findings.raise_()
            findings.activateWindow()
            return findings
        from .painted_review_comparison import read_review
        record = self.view._selected_version()
        if record is None:
            self.view.use_map_progress.setText('Select the approved saved map version in Versions first.')
            return None
        project_root = str(self.view.project.root)
        sources = [deepcopy(item['binding']) for item in self._registered_gap_reviews
                   if item['project_root'] == project_root]
        sources.extend(deepcopy(self.view.series_use_status.result.get('gap_reviews') or []))
        cached = getattr(self.view, '_last_gap_review_source', None)
        if matches_version(cached, record):
            sources.append(deepcopy(cached))
        if not sources:
            self.view.use_map_progress.setText('No review is recorded for this selected version. Use Transfer map to all runs to create one, or Open other review…')
            return None
        expected = _version_identity(record)

        def reader(_source):
            from .storage_paths import resolve_storage_path, same_storage_descriptor
            def bound(value):
                return value.get('review_binding', value)
            ordered = sorted(sources, key=lambda value: (resolve_storage_path(bound(value)['path']).stat().st_mtime_ns,
                                                         bound(value)['path']), reverse=True)
            for value in ordered:
                descriptor = bound(value)
                review = read_review(descriptor['path'])
                if not same_storage_descriptor(review['review_binding'], descriptor):
                    raise ValueError('The known review changed its exact path, SHA-256 or size.')
                if matches_version(review, expected):
                    return review
            raise ValueError('No known review matches this selected map version. Use Transfer map to all runs, or Open other review…')

        return self.open_gap_review(sources, review_reader=reader)

    @Slot(object, str)
    def _chart_review_unavailable(self, request, reason):
        generation, snapshot = request
        if self._chart_generation == generation and getattr(self, '_chart_context', None) == snapshot:
            self._chart_failed = True
            if getattr(self.view, '_workflow_scope', None) == 'apply':
                self.view.show_transfer_review_prompt(reason)
            else:
                self.view.show_mark_layers()
            self.view.use_map_progress.setText(reason)

    def _automatic_chart_identity(self):
        view = self.view
        return (id(view.project), str(getattr(view.project, 'root', '')),
                getattr(view.session, 'run_id', None),
                json.dumps(_version_identity(view._selected_version() or {}), sort_keys=True))

    def _chart_readers_busy(self):
        return (self.view.busy() or self.view.run_preview.busy()
                or getattr(self.view, '_adoption_thread', None) is not None
                or self._versions_reading())

    def _versions_reading(self):
        return bool(getattr(self.view, 'versions_reading', lambda: False)())

    def background_reading_answered(self):
        """Try a waiting automatic chart review again after the view's versions or point check arrived.

        Arriving versions may select the remembered version, so the waiting
        request is renewed for the same project and run before it is tried.
        """
        request = self._automatic_chart_request
        if request is None:
            return
        current = self._automatic_chart_identity()
        if current[:3] == request[:3]:
            self._automatic_chart_request = current
            self._queue_automatic_chart_review()

    def _queue_automatic_chart_review(self, *_args):
        if self._automatic_chart_request is not None and not self._automatic_chart_retry_queued:
            self._automatic_chart_retry_queued = True
            QTimer.singleShot(0, self._retry_automatic_chart_review)

    def _retry_automatic_chart_review(self):
        self._automatic_chart_retry_queued = False
        if self._automatic_chart_request is None:
            return
        if (getattr(self.view, '_workflow_scope', None) != 'apply'
                or self._automatic_chart_identity() != self._automatic_chart_request):
            self._automatic_chart_request = None
            return
        if not self._chart_readers_busy():
            self.open_chart_review(automatic=True)

    def _chart_opening_note(self, message):
        self._chart_opening_message = message
        self.view.use_map_progress.setText(message)
        if getattr(self.view, '_workflow_scope', None) == 'apply':
            self.view.show_transfer_review_prompt(message)

    def open_chart_review(self, *, automatic=False):
        """Review retained map inputs without starting another RAW preparation."""
        from .painting_chart_review_ui import ChartReviewDialog
        from .painting_chart_transfer_reader import selected_chart_session, saved_chart_review_paths
        if getattr(self.view, '_workflow_scope', None) == 'apply':
            self.view.show_transfer_review_prompt()
        if automatic:
            self._automatic_chart_request = self._automatic_chart_identity()
        busy = self._chart_readers_busy() if automatic else (
            self.view.busy() or getattr(self.view, '_adoption_thread', None) is not None)
        if busy:
            self._chart_opening_note('Reading the saved versions…' if automatic and self._versions_reading()
                                     else 'Wait for the current map operation to finish.')
            return None
        if automatic:
            point_blocker = getattr(self.view, '_reference_point_review_blocker', lambda: None)()
            if point_blocker:
                self._chart_opening_note(point_blocker)
                return None
        if automatic:
            self._automatic_chart_request = None
        if self.view._selected_version() is None:
            table = getattr(self.view, 'versions_table', None)
            if table is not None:
                approved = {row['folder'] for row in self.view.versions
                            if self.view._approval_for_version(row)}
                for row in range(table.rowCount()):
                    if table.item(row, 0).data(Qt.ItemDataRole.UserRole) in approved:
                        table.selectRow(row)
                        break
        try:
            snapshot = self._findings_context_snapshot()
        except ValueError as error:
            self._chart_opening_note(str(error))
            return None
        if self.chart_dialog is not None and not self.chart_dialog._closing:
            if getattr(self, '_chart_context', None) == snapshot and not self._chart_failed:
                self.view.show_transfer_review(self.chart_dialog)
                return self.chart_dialog
            self.chart_dialog.close()
        sources = [deepcopy(item['binding']) for item in self._registered_gap_reviews
                   if item['project_root'] == snapshot['project_root']]
        sources.extend(deepcopy((self.view.series_use_status.result or {}).get('gap_reviews') or []))
        sources.extend(source for source in (getattr(self.gap_dialog, 'review', None),
                       getattr(self.view, '_last_gap_review_source', None)) if isinstance(source, dict))
        if automatic and not sources:
            available = bool(saved_chart_review_paths(snapshot['project_root']))
            if not available:
                self._chart_opening_note('No saved transfer preparation is available for this map. Use Transfer map to all runs first.')
                return None
        self._chart_generation += 1
        generation = self._chart_generation
        def validate(review):
            from .storage_paths import same_storage_path
            selected, bound = snapshot['selection'], review['geometry_index']
            if (self._chart_generation != generation or self._findings_context_snapshot() != snapshot
                    or not matches_version(review, snapshot['version'])
                    or review['source']['painting_facies']['approval'] != snapshot['approval']
                    or not same_storage_path(bound['path'], selected.index_path)
                    or getattr(selected, 'sha256', '') and (
                        bound.get('sha256') != selected.sha256
                        or bound.get('size_bytes') != selected.size_bytes)):
                raise ValueError('The saved ColorChecker review does not match the selected approved map.')
        def read_session(known):
            try:
                return selected_chart_session(snapshot['project_root'], snapshot['version'], snapshot['selection'], known)
            except Exception as error:
                if automatic:
                    self.chart_review_unavailable.emit((generation, snapshot), str(error))
                raise
        from .series_decisions import series_actor
        try:
            actor = series_actor(self.view._series_root())
        except ValueError as error:
            self._chart_opening_note(str(error))
            return None
        dialog = ChartReviewDialog(sources, actor=actor, parent=self.view.run_preview.stack,
            session_factory=read_session,
            review_validator=validate, registered_runs=self.view._series_runs())
        self.chart_dialog = dialog
        self._chart_review_binding = _chart_source_binding(getattr(self.view, '_last_gap_review_source', None))
        self._chart_context = snapshot
        self._chart_failed = False
        self._chart_approvals = {}
        def changed(approvals):
            if self.chart_dialog is not dialog or dialog._closing:
                return
            bound = _chart_source_binding(getattr(getattr(dialog, 'session', None), 'review', None))
            if bound is not None:
                self._chart_review_binding = bound
            self._chart_approvals = deepcopy(approvals)
            if self.dialog is not None:
                self.dialog.candidate = self.dialog.checked_candidate = None
            self._preparation_action_generation += 1
        dialog.approvals_changed.connect(changed)
        dialog.approvals_changed.connect(self.view._refresh_transfer_runs)
        opening_message = self._chart_opening_message
        def loaded():
            if (self.chart_dialog is dialog and not dialog._closing
                    and self._chart_generation == generation and self._chart_context == snapshot
                    and self._chart_opening_message == opening_message):
                if opening_message and self.view.use_map_progress.text() == opening_message:
                    self.view.use_map_progress.clear()
                self._chart_opening_message = ''
        review_loaded = getattr(dialog, 'review_loaded', None)
        if review_loaded is not None:
            review_loaded.connect(loaded)
        for signal_name in ('review_loaded', 'run_changed'):
            signal = getattr(dialog, signal_name, None)
            if signal is not None:
                signal.connect(self.view._refresh_transfer_runs)
        def reuse(request):
            if self.chart_dialog is not dialog or dialog._closing:
                return
            try:
                validate(request['review'])
                if self._target_bindings:
                    raise ValueError('Open the saved review containing the completed sand corrections first.')
                self._pending_local_reuse = (self._findings_context_snapshot(), deepcopy(request))
                started = self.start(self.view._selected_version(), self.view.run_preview.geometry_selection(),
                    analysis_exclusions=self._area_approval, source_choices=self._source_choices,
                    warning_choices=self._warning_choices, local_reuse=request)
                if not started and self.findings_block_reason():
                    self.open_latest_gap_review()
            except (ValueError, KeyError) as error:
                dialog.status.setText(str(error))
        dialog.reuse_requested.connect(reuse)
        self.view.show_transfer_review(dialog)
        return dialog

    def open_recorded_review(self, source):
        """Independent inspection window; never enters the adoption context."""
        if self.view.busy():
            self.view.use_map_progress.setText("Wait for the current reader to finish.")
            return None
        from .painting_recorded_review_ui import RecordedGuidedReviewDialog
        if self.recorded_dialog is not None:
            self.recorded_dialog.close()
        dialog = RecordedGuidedReviewDialog(source, parent=self.view)
        self.recorded_dialog = self.view._recorded_review_dialog = dialog
        dialog.show()
        return dialog

    def open_gap_review(self, source, *, review_reader=None):
        """Open saved verified review metadata without starting preparation."""
        from .painting_gap_review_ui import GapReviewDialog
        from .series_geometry import default_actor
        if (review_reader is None and isinstance(source, dict)
                and set(source) == {'path', 'sha256', 'size_bytes'}):
            from .painted_review_comparison import read_review
            expected = deepcopy(source)
            def read_bound_review(_source):
                review = read_review(expected['path'])
                if not self._same_findings_inputs(review['review_binding'], expected):
                    raise ValueError('The review differs from its supplied path, SHA-256 or size.')
                return review
            review_reader = read_bound_review
        try:
            if self.view.busy() or getattr(self.view, '_adoption_thread', None) is not None:
                raise ValueError('Wait for the current operation to retire before reviewing gaps.')
            if self.context is not None and not self._current():
                self.context = None
            if self.context is None:
                record = self.view._selected_version()
                if record is None:
                    raise ValueError('Select the approved saved map version in Versions first.')
                self.view._read_approvals(str(record['run_id']))
                approval = self.view._approval_for_version(record)
                frame = self.view.current_frame()
                if not approval or frame is None:
                    raise ValueError('The selected map needs its exact approval and current frame.')
                self.context = dict(project=self.view.project, project_root=str(self.view.project.root),
                    version=_version_identity(record), approval=deepcopy(approval),
                    frame=deepcopy(frame.to_dict()), selection=deepcopy(self.view.run_preview.geometry_selection()))
            if not self._same_findings_context(self._findings_context_snapshot()):
                self._area_approval = None
                self._source_choices, self._warning_choices = [], {}
                self._target_bindings = {}
            self._findings_required = True
            if self.gap_dialog is not None:
                old_dialog, self.gap_dialog = self.gap_dialog, None
                old_dialog.close()
            self._suspended = False
            factory = self._gap_dialog_factory or GapReviewDialog
            from .series_decisions import gap_review_decision_ref
            series_root = self.view._series_root() if hasattr(self.view, '_series_root') else None
            dialog = factory(source, actor=self.view._reviewer() or default_actor(),
                decision_ref=gap_review_decision_ref(series_root),
                parent=self.view, review_validator=self._review_context,
                preparation_dirty=self._preparation_requires_refresh,
                additional_result_consumer=lambda key, result:
                    self._consume_findings_result(key, result, owner=dialog),
                **({'review_reader': review_reader} if review_reader is not None else {}))
            if self._gap_dialog_factory is not None:
                initial_requested = False
                def initial_findings():
                    nonlocal initial_requested
                    if (initial_requested or dialog is not self.gap_dialog
                            or dialog._closing or dialog.busy() or dialog.review is None):
                        return
                    initial_requested = True
                    self._queue_findings_refresh()
                dialog.readiness_changed.connect(initial_findings)
                dialog.region_confirmation_requested.connect(self._confirm_gap_regions)
                dialog.map_fix_requested.connect(self._remember_region_preview)
                dialog.region_preview_reset_requested.connect(self._reset_region_preview)
                dialog.draft_choices_changed.connect(lambda payload, owned=dialog:
                    self.set_findings_draft_pending(owned, payload))
            dialog.paint_requested.connect(self._paint_gap)
            dialog.recheck_requested.connect(self._recheck_gaps)
            dialog.approval_changed.connect(self._areas_changed)
            dialog.display_recheck_requested.connect(self._recheck_display)
            dialog.state_requested.connect(self._state_choice)
            dialog.fix_requested.connect(self._fix_issue)
            dialog.adopt_requested.connect(self._findings_adopt)
            dialog.water_surface_confirmed.connect(self._water_surface_confirmed)
            dialog.water_surface_edited.connect(self._water_surface_edited)
            dialog.water_surfaces_cleared.connect(self._water_surfaces_cleared)
            dialog.water_surface_tolerances_changed.connect(self._water_surface_tolerances_changed)
            dialog.position_tolerances_changed.connect(self._position_tolerances_changed)
            dialog.readiness_changed.connect(self._findings_changed)
            dialog.refresh_findings_requested.connect(self._refresh_findings)
            dialog.water_surface_confirmations=deepcopy(self._water_surface_confirmations)
            dialog.position_tolerances=dict(shift_mm=self._position_tolerances['shift_tolerance_mm'],
                rotation_degrees=self._position_tolerances['angle_tolerance_degrees'])
            dialog.water_surface.height_tolerance.setValue(self._water_surface_tolerances['height_tolerance_mm'])
            dialog.water_surface.angle_tolerance.setValue(self._water_surface_tolerances['angle_tolerance_degrees'])
            dialog.approval = deepcopy(self._area_approval)
            dialog._acquisition_actions = deepcopy(self._source_choices)
            dialog._warning_actions = deepcopy(self._warning_choices)
            dialog.pending_corrections = deepcopy(list(self._target_bindings.values()))
            dialog.finished.connect(lambda _result, owned=dialog: self._gap_closed(owned))
            self.gap_dialog = self.view._gap_review_dialog = dialog
            self.findings_changed.emit()
            dialog.show()
            self.view.use_map_progress.setText('Review the gap outlines and make an explicit choice. Nothing was adopted.')
            return dialog
        except Exception as error:
            self.view.use_map_progress.setText('The gap review could not open: ' + str(error))
            self.context = None
            return None

    def _water_surface_confirmed(self, record):
        """Retain only a literal action from the human confirmation button."""
        if self._current():
            self._water_surface_confirmations[record['action']['run_id']]=deepcopy(record)
            self._invalidate_display_candidate()

    def _water_surface_tolerances_changed(self, values):
        if self._current():
            self._water_surface_tolerances=deepcopy(values)
            self._invalidate_display_candidate()

    def _water_surface_edited(self, run):
        if self._current():
            self._water_surface_confirmations.pop(run,None)
            self._invalidate_display_candidate()

    def _water_surfaces_cleared(self):
        if self._current():
            self._water_surface_confirmations = {}
            self._invalidate_display_candidate()

    def _position_tolerances_changed(self, values):
        if self._current():
            self._position_tolerances=dict(method='camera registration (map placement)',
                shift_tolerance_mm=values['shift_mm'],angle_tolerance_degrees=values['rotation_degrees'])
            self._invalidate_display_candidate()

    def _display_preferences(self):
        """Pure UI snapshot; the owned Check reconstructs genuine source bindings."""
        return dict(position_policy=deepcopy(self._position_tolerances),water_policy=deepcopy(self._water_surface_tolerances),
            water_actions={run:deepcopy(record['action']) for run,record in self._water_surface_confirmations.items()})

    def _preparation_requires_refresh(self):
        return self.preparation is None or self._completed_preparation_generation != self._preparation_action_generation

    def _areas_changed(self, approval):
        if self._current() and self._area_approval != approval:
            self._area_approval = deepcopy(approval)
            self._preparation_action_generation += 1
            request = (self.gap_dialog.confirmed_region_request()
                       if self.gap_dialog is not None and hasattr(self.gap_dialog, 'confirmed_region_request') else None)
            if request is not None:
                self._remember_region_preview(request)
                payload = {key: deepcopy(request[key]) for key in (
                    "review_binding", "run_id", "component_id", "review_row_sha256")}
                self.set_findings_draft_pending(self.gap_dialog, dict(payload, pending=False), confirmed=True)
            self._findings_choices_changed()

    def _invalidate_display_candidate(self):
        self._warning_choices = {}
        if self.dialog is not None:
            self.dialog.set_display_preferences(self._display_preferences())
        self.findings_changed.emit()

    def _recheck_display(self):
        """Reuse a completed preparation; tolerance edits do not create new gaps."""
        if not self._allow_findings_check():
            return
        if self._preparation_requires_refresh() and self.gap_dialog is not None:
            self._recheck_gaps(self.gap_dialog.approval)
            return
        if (not self._current() or self.gap_dialog is None or self.gap_dialog.busy()
                or self.dialog is None or self.preparation is None or self.dialog.busy()):
            if self.gap_dialog is not None:
                self.gap_dialog.status.setText('A completed preparation is required. Resolve the areas and run the original preparation first.')
            return
        if not self.dialog.set_display_preferences(self._display_preferences() or None):
            return
        # This slot is the explicit Check action, so the latest human source
        # answer must enter that original Check even when preparation is reused.
        started = (self.dialog.choose_facies_source_state(deepcopy(self._source_choices[-1]))
                   if self._source_choices else self.dialog.check())
        if started:
            self.gap_dialog.operation_waiting = True
            self.gap_dialog._buttons()

    def _consume_findings_result(self, key, result, *, owner):
        """Deliver a collector result after the original native thread has retired."""
        if not isinstance(key, tuple) or len(key) != 2 or key[0] != 'findings-catalogue':
            return False
        if owner is self.gap_dialog and not owner._closing:
            installed = self.install_findings_result(result)
            if not installed and not owner.busy():
                # The original reader really retired, even when an intervening
                # edit invalidated its request. Persistent draft/context guards
                # remain responsible for Check and adoption.
                owner.operation_waiting = owner._catalogue_refresh_pending = False
        return True

    def _remember_region_preview(self, request):
        """Remember only local human markings; destination choices remain explicit."""
        dialog = self.gap_dialog
        if not self._current() or dialog is None or dialog.review is None:
            return
        session = getattr(dialog, 'region_preview_session', None)
        if session is not None:
            try:
                session.remember(request, review=dialog.review)
            except (KeyError, TypeError, ValueError, OSError) as error:
                dialog.status.setText('These markings cannot be suggested elsewhere: ' + str(error))

    def _reset_region_preview(self, request):
        """An explicit reset uses the same finite reader and exact current context."""
        dialog = self.gap_dialog
        if (not self._current() or dialog is None or dialog.busy() or dialog._closing
                or dialog.operation_waiting):
            return
        try:
            key = dialog.region_context_key()
            binding = request['review_binding']
            expected = (binding['path'], binding['sha256'], binding['size_bytes'],
                request['run_id'], request['component_id'], request['review_row_sha256'])
            if key is None or key != expected or dialog.review['review_binding'] != binding:
                raise ValueError('The selected deviation changed before the reset')
            session = getattr(dialog, 'region_preview_session', None)
            if session is None:
                raise ValueError('The verified suggestion reader is unavailable')
            session.suppress(request)
            review = deepcopy(dialog.review)
            dialog._start(('region-preview-reset', key),
                lambda: dialog._read_preview(review, request['run_id'], request['component_id']))
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
            dialog.status.setText('The suggestion could not be reset: ' + str(error))

    def _confirm_gap_regions(self, request):
        """Only an explicit human Confirm starts the original finite owned reader."""
        dialog = self.gap_dialog
        if (not self._current() or dialog is None or dialog.busy() or dialog._closing
                or dialog.operation_waiting):
            if dialog is not None:
                dialog.status.setText('Wait for the current operation or reopen the current review before confirming.')
            return
        try:
            from . import layer_painting as painting
            from .painted_gap_regions import validate_gap_region_snapshot, save_gap_region_approval
            review, submitted = deepcopy(dialog.review), deepcopy(request)
            key = dialog.region_context_key()
            bound = submitted['review_binding']
            expected = (bound['path'], bound['sha256'], bound['size_bytes'],
                submitted['run_id'], submitted['component_id'], submitted['review_row_sha256'])
            if key is None or key != expected or review['review_binding'] != bound:
                raise ValueError('The selected deviation changed before your confirmation')
            action = {name: submitted[name] for name in ('actor', 'approved_utc', 'reason', 'decision_ref')}
            if action['actor'] != dialog.actor or any(type(value) is not str or not value.strip() for value in action.values()):
                raise ValueError('The confirmation needs the actual reviewer, time and separate reason')
            previous = deepcopy(submitted['previous'])
            if dialog.approval != previous:
                raise ValueError('The earlier confirmed areas changed before this request')
            from datetime import datetime, timezone
            import uuid
            destination = (painting.painting_folder(self.view.project).parent / 'gap_region_approvals'
                / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:12]))
            def publish():
                snapshot = validate_gap_region_snapshot(submitted['snapshot'], review=review)
                return save_gap_region_approval(snapshot, destination, **action,
                    other_text=submitted.get('other_text', ''), previous=previous)
            dialog._start(('region-approval', key), publish)
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
            dialog.status.setText('The classifications could not be submitted: ' + str(error))

    @Slot()
    def _refresh_findings(self):
        """Route the existing list action through its generation-bound dispatcher."""
        self.refresh_findings()

    def _findings_input_bindings(self, review):
        """Freeze only literal collector inputs, without granting result authority."""
        findings = self.gap_dialog
        targets = [deepcopy(binding) for run in review['runs']
                   for binding in run.get('target_correction_bindings') or []]
        for binding in self._target_bindings.values():
            if not any(self._same_findings_inputs(binding, prior) for prior in targets):
                targets.append(deepcopy(binding))
        return dict(review=deepcopy(review['review_binding']),
            target_bindings=targets,
            analysis_exclusions=deepcopy(findings.approval),
            source_choices=deepcopy(findings._acquisition_actions),
            display_preferences=self._display_preferences())

    def _same_findings_inputs(self, first, second):
        """Keep literal input types/values, with aliases only for typed file descriptors."""
        from .storage_paths import same_storage_descriptor
        if isinstance(first, dict) and isinstance(second, dict):
            if set(first) == set(second) == {'path', 'sha256', 'size_bytes'}:
                return same_storage_descriptor(first, second)
            return set(first) == set(second) and all(
                self._same_findings_inputs(first[key], second[key]) for key in first)
        if isinstance(first, list) and isinstance(second, list):
            return len(first) == len(second) and all(
                self._same_findings_inputs(left, right) for left, right in zip(first, second))
        return json.dumps(first, sort_keys=True, allow_nan=False, ensure_ascii=False) == json.dumps(
            second, sort_keys=True, allow_nan=False, ensure_ascii=False)

    def _catalogue_failed(self, message):
        """Keep original replay mandatory while making a failed refresh retryable."""
        self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
        self._findings_update_reason = 'Updated findings are required before Check or Apply and Build. ' + str(message)
        if self.gap_dialog is not None:
            self.gap_dialog.catalogue_refresh_failed(str(message))
        self.view.use_map_progress.setText(self._findings_update_reason)
        self.findings_changed.emit()

    @Slot()
    def refresh_findings(self):
        """Dispatch detached generation/review/input_bindings DATA to an owned collector.

        The callback starts finite worker work and returns; it must not collect
        synchronously. Its result is installed separately by install_findings_result.
        """
        findings = self.gap_dialog
        if self._findings_draft_contexts:
            reason = self.findings_block_reason()
            if findings is not None:
                findings.status.setText(reason)
            self.view.use_map_progress.setText(reason)
            self.findings_changed.emit()
            return False
        self._catalogue_refresh_queued = False
        try:
            snapshot = self._findings_context_snapshot()
            if (findings is None or findings.review is None
                    or not self._same_findings_context(snapshot)
                    or findings.review.get('review_binding') != self._findings_review_binding):
                raise ValueError('Open the bound findings list for the current selection first.')
            self._catalogue_generation += 1
            request = dict(generation=self._catalogue_generation, review=deepcopy(findings.review),
                input_bindings=self._findings_input_bindings(findings.review))
            self._catalogue_request, self._catalogue_context, self._catalogue_dialog = request, snapshot, findings
            self._findings_update_reason = 'Updating findings from the original replay. Wait for the fresh residual list before Check or Apply and Build.'
            findings.operation_waiting = True
            findings._buttons()
            findings.status.setText(self._findings_update_reason)
            self.view.use_map_progress.setText(self._findings_update_reason)
            self.findings_changed.emit()
            if self.catalogue_refresher is None:
                raise ValueError('The original-check findings collector is not connected. No Check or preparation was started.')
            try:
                self.catalogue_refresher(deepcopy(request))
            except Exception as error:
                self._catalogue_failed(f'{type(error).__name__}: {error}')
                return False
            return True
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            self._catalogue_failed(str(error))
            return False

    @Slot(object)
    def install_findings_result(self, value):
        """Install authenticated residual review/report DATA for one exact request.

        A successful result has generation, review, and report; the report's
        input_bindings must equal the captured request. An error result has
        generation, input_bindings, and error. The collector authenticates the
        fresh review off the GUI thread before delivering either result.
        """
        request = self._catalogue_request
        if (request is None or not isinstance(value, dict)
                or type(value.get('generation')) is not int
                or value['generation'] != request['generation']):
            return False
        try:
            snapshot = self._findings_context_snapshot()
            if (self._catalogue_context is None or snapshot['project'] is not self._catalogue_context['project']
                    or snapshot != self._catalogue_context
                    or self.gap_dialog is not self._catalogue_dialog
                    or getattr(self.gap_dialog, '_closing', False)
                    or not self._same_findings_inputs(
                        self._findings_input_bindings(self.gap_dialog.review), request['input_bindings'])):
                raise ValueError('The current findings context or collector inputs changed. Refresh the current list again.')
            report = value.get('report')
            inputs = value.get('input_bindings') if value.get('error') else (report or {}).get('input_bindings')
            if not self._same_findings_inputs(inputs, request['input_bindings']):
                raise ValueError('The findings result does not bind the exact requested inputs.')
            if value.get('error'):
                raise ValueError(str(value['error']))
            review = value.get('review')
            if (not isinstance(review, dict) or not isinstance(report, dict)
                    or not isinstance(report.get('issues'), list)
                    or any(not isinstance(issue, dict) for issue in report['issues'])
                    or not isinstance(review.get('review_binding'), dict)
                    or not matches_version(review, snapshot['version'])
                    or review['map_version'].get('approval') != snapshot['approval']
                    or review.get('geometry_index') != request['review'].get('geometry_index')):
                raise ValueError('The collector did not return the current bound residual review and report.')
            install = getattr(self.gap_dialog, 'install_catalogue', None)
            if not callable(install):
                raise ValueError('The findings view cannot install a fresh residual review yet.')
            if install(deepcopy(review), deepcopy(report)) is False:
                raise ValueError('Wait for the findings view to become available, then refresh again.')
            self._findings_context, self._findings_review_binding = snapshot, deepcopy(review['review_binding'])
            self._catalogue_request = self._catalogue_context = self._catalogue_dialog = None
            self._findings_update_reason = ''
            self.findings_changed.emit()
            return True
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            self._catalogue_failed(str(error))
            return False

    def _paint_gap(self, request):
        """Open an isolated target-native draft, without a reference switch."""
        from .painting_target_correction_ui import TargetCorrectionDialog
        from . import layer_painting as painting
        from .series_geometry import default_actor
        try:
            if not self._current():
                raise ValueError('The selected version, approval or geometry changed.')
            photo_options = {}
            comparison_reader = getattr(self.gap_dialog, '_comparison_reader', None)
            if comparison_reader is not None:
                photo_options['comparison_reader'] = comparison_reader
            target_reader = getattr(self.gap_dialog, 'target_paint_reader', None)
            if target_reader is not None:
                photo_options['reader'] = target_reader
            packet = getattr(self.gap_dialog, '_base_packet', None)
            if (packet is not None and packet.get('image_role') == 'current'
                    and packet.get('run_id') == request['run_id']
                    and packet.get('component_id') == request['component_id']):
                photo_options['current_display_packet'] = deepcopy({
                    key: value for key, value in packet.items() if key != 'region_draft'})
            dialog = TargetCorrectionDialog(request['review'], request['run_id'], request['component_id'],
                actor=self.view._reviewer() or default_actor(), parent=self.view,
                destination=painting.painting_folder(self.view.project).parent / 'target_corrections',
                point_native_xy=request['point_native_xy'], **photo_options)
            dialog.saved.connect(self._target_saved)
            dialog.finished.connect(self._target_closed)
            self.view._target_correction_dialog = dialog
            self.gap_dialog.hide()
            dialog.show()
        except (AttributeError, KeyError, TypeError, ValueError, OSError, ImportError) as error:
            self.gap_dialog.status.setText(str(error))

    def _target_saved(self, result):
        """A new human-approved version invalidates affected old resolutions."""
        if not self._current() or not self._ingest_target_binding(result):
            return
        if self.dialog is not None:
            self.dialog._update_buttons()
        findings = self.gap_dialog
        if findings is not None:
            findings._refresh_rows()
        self._queue_findings_refresh()

    def _verified_target_row(self, result):
        """Match an actual completed target Save to the exact current review row."""
        target = getattr(self.view, '_target_correction_dialog', None)
        findings = self.gap_dialog
        if target is None or findings is None or findings.review is None:
            return None
        try:
            if (target.run_id != result['run_id'] or target.component_id != result['component_id']
                    or target.binding is None or target.binding != result['binding']
                    or not self._same_findings_context(self._findings_context_snapshot())):
                return None
            bound = findings.review.get('review_binding')
            if not bound or target.review.get('review_binding') != bound:
                return None
            def row(review):
                return next(component for run in review['runs'] if run['run_id'] == result['run_id']
                    for component in run['components'] if component['component_id'] == result['component_id'])
            current, saved = row(findings.review), row(target.review)
            if (not current.get('review_row_sha256')
                    or current['review_row_sha256'] != saved.get('review_row_sha256')):
                return None
            return dict(run_id=result['run_id'], component_id=result['component_id'],
                review_row_sha256=current['review_row_sha256'])
        except (AttributeError, KeyError, StopIteration, TypeError, ValueError, OSError):
            return None

    def _ingest_target_binding(self, result):
        """Retain a completed explicit Save once, including during shutdown."""
        if self._verified_target_row(result) is None:
            return False
        key = (result['run_id'], result['component_id'])
        if self._target_bindings.get(key) == result['binding']:
            return False
        self._target_bindings[key] = deepcopy(result['binding'])
        self._preparation_action_generation += 1
        findings = self.gap_dialog
        approval = deepcopy(findings.approval if findings is not None else self._area_approval)
        if approval is not None:
            approval['runs'] = [row for row in approval['runs']
                if (row['run_id'], row['component_id']) != key]
            approval = approval if approval['runs'] else None
        self._area_approval = deepcopy(approval)
        if findings is not None:
            findings.approval = approval
            affected = {row['issue_index'] for row in findings._issues
                        if (row['issue'].get('details') or {}).get('run_id') == result['run_id']}
            findings._warning_actions = {identity: action for identity, action in findings._warning_actions.items()
                                        if action['issue_index'] not in affected}
            self._warning_choices = deepcopy(findings._warning_actions)
            findings.pending_corrections = deepcopy(list(self._target_bindings.values()))
            findings._preparation_changed = True
            findings._candidate = findings._report = None
            findings.set_resolved_target_rows([])
        self._findings_update_reason = 'Updating findings after the saved target correction. The original replay must produce the residual list before Check or Apply and Build.'
        if self.dialog is not None:
            self.dialog.candidate = self.dialog.checked_candidate = self.dialog.review_resolutions = None
        return True

    def _target_closed(self, _result):
        if self._current() and self.gap_dialog is not None:
            self.gap_dialog.show()

    def _recheck_gaps(self, approval):
        """A human click starts a fresh original preparation with its exact choices."""
        if not self._allow_findings_check():
            return
        if not self._current() or self.gap_dialog is None or self.gap_dialog.busy():
            self.view.use_map_progress.setText('The gap-review context changed. Nothing was adopted.')
            return
        record, selection = deepcopy(self.view._selected_version()), self.context['selection']
        choices = deepcopy(self.gap_dialog._acquisition_actions)
        warnings = deepcopy(self.gap_dialog._warning_actions)
        targets = self._findings_input_bindings(self.gap_dialog.review)['target_bindings']
        local_reuse = None
        if self._pending_local_reuse is not None:
            snapshot, request = self._pending_local_reuse
            if snapshot == self._findings_context_snapshot():
                local_reuse = deepcopy(request)
            else:
                self._pending_local_reuse = None
        self.gap_dialog.close()
        self.context = None
        self.start(record, selection, analysis_exclusions=deepcopy(approval), source_choices=choices, warning_choices=warnings,
                   target_corrections=targets or None, local_reuse=local_reuse)

    def _state_choice(self, choice):
        """Use only the enum action just entered by the human in this window."""
        if not self._current():
            self.view.use_map_progress.setText('The source context changed. Nothing was adopted.')
            return
        self._source_choices = deepcopy(self.gap_dialog._acquisition_actions)
        self._findings_choices_changed()

    def _findings_adopt(self, records):
        """Only the human Adopt action enters the original final confirmation."""
        if not self._allow_findings_check():
            return
        if (not self._current() or self.dialog is None or self.dialog.busy()
                or self.gap_dialog is None or not self.gap_dialog.can_adopt()):
            return
        self.dialog.set_review_resolutions(deepcopy(records))
        self.gap_dialog.operation_waiting = True
        self.gap_dialog._buttons()
        self._show_adoption_panel()
        self._confirmation_pending = True
        self._confirm_timer.start(0)

    def _fix_issue(self, issue):
        """Open the existing relevant controls; never repair scientific data here."""
        from .painting_gap_review_ui import focus_issue_controls
        if not self._current():
            return
        if issue.get('code') in {'run_geometry_acquisition_unknown', 'indicator_facies_carryover_risk'}:
            self.gap_dialog.source_state.setFocus()
            self.gap_dialog.status.setText('Choose the water state yourself in this row, enter a reason and apply your choice.')
            return
        focus_issue_controls(self.view, issue)
        self.gap_dialog.close()
        if self.dialog is not None:
            self.dialog.close()

    def _gap_closed(self, expected):
        """Closing gap review does not continue into Check or adoption."""
        if self.gap_dialog is expected:
            self.context = None
            self.findings_changed.emit()

    def _dialog_retired(self, operation):
        """Review only a fully retired check; report a save from its actual receipt."""
        if self.context is None or self._suspended:
            return
        dialog = self.dialog
        if operation == 'check':
            if not self._current():
                dialog.status.setText('The selected version, approval or geometry changed. Nothing was adopted.')
                self.view.use_map_progress.setText(dialog.status.text())
                self.context = None
                return
            if dialog.validation_report is None:
                self.view.use_map_progress.setText(dialog.status.text())
                self.context = None
                if self.gap_dialog is not None:
                    self.gap_dialog.operation_waiting = False
                    self.gap_dialog._buttons()
                return
            if self.context.get('manual_map_approved'):
                if dialog.candidate is None:
                    self.view.use_map_progress.setText(dialog.status.text())
                    return
                from .manual_map_approval import approved_candidate
                if not approved_candidate(dialog.candidate):
                    self.view.use_map_progress.setText('The saved manual map approvals changed. Nothing was adopted.')
                    return
                self.view.use_map_progress.setText('The approved maps passed Check. Confirm saving this geometry revision.')
                self._confirmation_pending = True
                self._confirm_timer.start(0)
                return
            source = self.preparation.get('gap_review')
            findings = self.gap_dialog
            if findings is None or findings._closing:
                findings = self.open_gap_review(source)
            if findings is None:
                return
            findings.set_check_result(dialog.checked_candidate, dialog.validation_report)
            self.findings_changed.emit()
            findings.show()
            dialog.hide()
            self.view.use_map_progress.setText('Resolve the findings in the list. Adopt becomes available only when all findings are resolved.')
        elif operation == 'save':
            if dialog.revision is None:
                self.view.use_map_progress.setText(dialog.status.text())
            else:
                self.view.use_map_progress.setText(
                    f"Adopted this map as geometry revision {dialog.revision['revision_id']}. "
                    'Apply and Build have not run.')
                self.view.series_use_status.invalidate()
            self.context = None
            self.view._refresh_versions()
            self.findings_changed.emit()

    def _confirm(self):
        """Ask once, only for the still-current checked automatic candidate."""
        if not self._confirmation_pending:
            return
        self._confirmation_pending = False
        if not self._current() or self.dialog is None or not self.dialog.isVisible():
            self.view.use_map_progress.setText('The adoption context changed. Nothing was adopted.')
            self.context = None
            return
        if not self._allow_findings_check():
            return
        self.dialog._save_clicked()

    def _confirmation_decided(self, accepted):
        """A declined final question leaves all Store revisions unchanged."""
        if accepted and (not self._current() or self.findings_block_reason()):
            # The final question has a nested event loop. A queued context
            # change during that loop must not save the previous candidate.
            self.dialog.candidate = None
            self.dialog.status.setText('The selected sources or findings changed. Nothing was adopted.')
            self.view.use_map_progress.setText(self.dialog.status.text())
            self.context = None
        elif not accepted:
            if self.gap_dialog is not None:
                self.gap_dialog.close()
            self.view.use_map_progress.setText('Adoption cancelled. The checked preparation is retained; nothing was adopted.')
            self.context = None

    def _dialog_closed(self, _result):
        """Ignore any pending confirmation after the user closes its result window."""
        self._confirmation_pending = False
        self._confirm_timer.stop()
        if self.dialog is not None and not self.dialog.busy():
            self.context = None

    def wait_for_workers(self):
        """Suppress continuation and naturally retire any owned dialog worker."""
        self._abandoned_preparation = self.context is not None and self.dialog is None
        self._suspended = True
        self._confirmation_pending = False
        self._confirm_timer.stop()
        if self.dialog is not None:
            self.dialog.wait_for_workers()
        if self.gap_dialog is not None:
            self.gap_dialog.wait_for_workers()
        if self.recorded_dialog is not None:
            self.recorded_dialog.wait_for_workers()
        if self.chart_dialog is not None:
            self.chart_dialog.wait_for_workers()
        target = getattr(self.view, "_target_correction_dialog", None)
        if target is not None:
            target.wait_for_workers()
            if target.binding is not None:
                # The human Save already completed; suspension must not lose
                # its binding merely because no queued GUI slot remains.
                self._ingest_target_binding(dict(run_id=target.run_id,
                    component_id=target.component_id, binding=target.binding))
        self.context = None
