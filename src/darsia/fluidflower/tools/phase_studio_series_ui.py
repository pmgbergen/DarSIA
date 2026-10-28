"""Embedded series-calibration workspace; heavy work runs outside the GUI."""

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root
from darsia.fluidflower.series_paths import studio_series_keys
import json
import hashlib
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import Qt, QProcess, QTimer, QUrl, QThread, Signal
from PySide6.QtGui import QPixmap, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QTabWidget,
    QScrollArea, QSlider, QTextEdit, QAbstractItemView, QDialog)

from darsia.fluidflower import phase_saved_review_metadata as backend
from darsia.fluidflower.tools.phase_studio_mass_comparison import (
    result_variant_label, matching_frame_index, ac_compatibility_note)
from darsia.fluidflower.tools.phase_studio_saved_results import (
    discover_saved_series_entries, model_browser_metadata, verify_saved_series_entry)
from darsia.fluidflower.tools.phase_studio_saved_archive import (
    family_key, load_archive, set_archived)
from darsia.fluidflower.tools.phase_studio_saved_method import (
    explain_ffum_scenario, format_saved_method_summary)
from darsia.fluidflower.tools.phase_studio_photometry_policy import load_policy, variant_status


class ModelMassLookup(QThread):
    ready = Signal(object, str)

    def __init__(self, root, parent):
        super().__init__(parent)
        self.root = root

    def run(self):
        from darsia.fluidflower.concentration_inputs import discover_model_inputs
        from darsia.fluidflower.concentration_configuration import describe_run
        try:
            if self.isInterruptionRequested():
                return
            catalog = discover_model_inputs(self.root, include_saved_results=False)
            for job in catalog['jobs']:
                if self.isInterruptionRequested():
                    return
                job['concentration'] = describe_run(job)
            catalog['saved_results_deferred'] = True
            if not self.isInterruptionRequested():
                self.ready.emit(catalog, '')
        except Exception as error:
            if not self.isInterruptionRequested():
                self.ready.emit(None, str(error))


class SavedSeriesResultLookup(QThread):
    """Find immutable concentration variants without depending on the active phase model."""
    ready = Signal(str, object, str)

    def __init__(self, dataset, parent, cache_root):
        super().__init__(parent)
        self.dataset = str(dataset).casefold()
        self.cache_root = Path(cache_root)

    def run(self):
        try:
            root = series_path(self.dataset, 'results') / 'concentration_mass' / 'catalog'
            entries = discover_saved_series_entries(
                self.dataset, root, self.cache_root,
                cancelled=self.isInterruptionRequested,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(self.dataset, entries, '')
        except Exception as error:
            if not self.isInterruptionRequested():
                self.ready.emit(self.dataset, [], str(error))


def _add_model_browser_metadata(entry, result, result_path):
    """Use the same display metadata for imported and catalogued results."""
    entry.update(model_browser_metadata(result, result_path))


def _photometry_filter_values(entry):
    """Read saved correction settings without inferring missing scientific metadata."""
    settings = entry.get('method') or {}
    variant = settings.get('photometry_variant_id')
    parts = variant.rsplit('_cc_', 1) if isinstance(variant, str) else []
    variant_light = parts[0] if len(parts) == 2 and parts[0] in ('none', 'historical', 'facies') else None
    variant_color = ({'on': True, 'off': False}.get(parts[1])
                     if variant_light is not None else None)
    if variant_color is None:
        variant_light = None

    light = settings.get('illumination')
    if light is None:
        light = variant_light
    elif light not in ('none', 'historical', 'facies') or (variant_light is not None and light != variant_light):
        light = None

    color = settings.get('colorchecker_enabled')
    if color is None:
        color = variant_color
    elif type(color) is not bool or (variant_color is not None and color != variant_color):
        color = None
    return light or 'unknown', ('on' if color is True else 'off' if color is False else 'unknown')


_FILTERED_DISPLAY_NOTICE = ('The displayed concentration model is hidden by the current light/color or scope '
                            'filters. Its image remains open; adjust filters or select another model.')


class _FullscreenPixmap(QLabel):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self._source_pixmap = pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self._rescale()

    def set_source_pixmap(self, pixmap):
        self._source_pixmap = pixmap
        self.setText('')
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if not self._source_pixmap.isNull():
            self.setPixmap(self._source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


class SeriesMassWorkspace(QWidget):
    def show_saved_overflow_comparison(self):
        """Open the original saved comparison in this workspace's presentation."""
        self.views.setCurrentWidget(self.overflow_plot)

    def set_overflow_comparison_mode(self, mode: str) -> None:
        """Show or hide the original saved overflow figure; no calculation or setting write."""
        if mode not in ("off", "compare"):
            raise ValueError("The overflow comparison display must be Off or Compare only")
        self.views.setTabVisible(self.views.indexOf(self.overflow_plot), mode == "compare")

    def __init__(self, parent=None, *, cache_root=local_cache_root() / 'phase_mass_workspace', selected_model=None):
        super().__init__(parent)
        self.cache_root = Path(cache_root)
        self.selected_model = selected_model
        self.entries = []; self.overflow = {}; self.process = None; self.job = None
        self.checked = set()
        self.result = None; self.result_path = None; self.operation = None
        self._active_preview_path = None
        self._preferred_preview_field = 'preview'
        self.calibration_file = None
        self.phase_catalog = None; self.lookup = None; self._view_cache = {}
        self.series_lookup = None; self._series_catalog_loaded = set()
        self._archive_by_series = {}
        self._photometry_policies = {}
        self._active_model_menu = 'current'
        layout = QVBoxLayout(self)
        bar = QHBoxLayout(); layout.addLayout(bar)
        self.series = QComboBox()
        for name in studio_series_keys(): self.series.addItem(name.upper(), name)
        self.scope = QComboBox(); self.scope.addItem('Shared series model (default)', 'series')
        self.scope.addItem('Individual experiment (comparison)', 'individual')
        self.import_button = QPushButton('Import prepared results / report')
        self.import_button.clicked.connect(self.choose_results)
        for w in (self.series, self.import_button): bar.addWidget(w)
        self.import_model = QPushButton('Load selected phase model and concentration results')
        self.import_model.setEnabled(selected_model is not None)
        self.import_model.clicked.connect(self.load_model_results);bar.addWidget(self.import_model)

        self.setup_toggle = QPushButton('Prepare calculations and manage saved results')
        self.setup_toggle.setCheckable(True); self.setup_toggle.setChecked(False)
        layout.addWidget(self.setup_toggle)
        self.setup_body = QWidget(); setup_layout = QVBoxLayout(self.setup_body)
        setup_layout.setContentsMargins(4, 2, 4, 4); layout.addWidget(self.setup_body)
        self.setup_body.setVisible(False)
        self.setup_toggle.toggled.connect(self.setup_body.setVisible)
        notice = QLabel('Calculations use the reviewed photographs and keep their selected phase maps fixed. '
            'A fit to injected mass is a calibration, not independent mass validation.')
        notice.setWordWrap(True); setup_layout.addWidget(notice)
        prepare_bar = QHBoxLayout(); setup_layout.addLayout(prepare_bar)
        self.phase_experiment = QComboBox(); self.phase_experiment.setMinimumContentsLength(12)
        for widget in (QLabel('Registered phase experiments'), self.phase_experiment):
            prepare_bar.addWidget(widget)
        self.phase_binding = QLabel('Load the selected phase model to use its exact review images.')
        self.phase_binding.setWordWrap(True); setup_layout.addWidget(self.phase_binding)
        self.settings_toggle=QPushButton('Legacy comparison method and imported results')
        self.settings_toggle.setCheckable(True);setup_layout.addWidget(self.settings_toggle)
        self.settings_body=QWidget();setup_layout.addWidget(self.settings_body)
        self.settings_body.setVisible(False)
        self.settings_toggle.toggled.connect(self.settings_body.setVisible)
        layout=QVBoxLayout(self.settings_body);layout.setContentsMargins(0,0,0,0)
        calibration_bar=QHBoxLayout();layout.addLayout(calibration_bar)
        calibration_bar.addWidget(self.scope)
        self.load_calibration_button=QPushButton('Load frozen series calibration')
        self.load_calibration_button.clicked.connect(self.load_calibration)
        self.new_calibration_button=QPushButton('Fit a new calibration instead')
        self.new_calibration_button.clicked.connect(self.clear_calibration)
        self.calibration_label=QLabel('New calibration: parameters fitted jointly to checked experiments.')
        self.calibration_label.setWordWrap(True)
        for w in (self.load_calibration_button,self.new_calibration_button,self.calibration_label):calibration_bar.addWidget(w)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Use','Experiment','Phase model','Calculated','Overflow targets','Concentration model'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(230); layout.addWidget(self.table)
        self.table.itemChanged.connect(self.check_changed)
        self.series.currentIndexChanged.connect(self._series_changed)
        refs = QHBoxLayout(); layout.addLayout(refs)
        self.import_overflow = QPushButton('Attach overflow targets to selected run')
        self.import_overflow.clicked.connect(self.choose_overflow)
        self.clear_overflow = QPushButton('Remove attached targets'); self.clear_overflow.clicked.connect(self.remove_overflow)
        self.overflow_mode = QComboBox()
        self.overflow_mode.addItem('Overflow: compare with calculated gas', 'compare')
        self.overflow_mode.addItem('Overflow: constrain gas within measured interval', 'constrain')
        for w in (self.import_overflow,self.clear_overflow,self.overflow_mode): refs.addWidget(w)
        info = QLabel('Overflow targets represent liquid displacement → gas volume/mass, not CO₂ loss. '
            'Only matching measured times are used; no extrapolation. Inspect feed-rate and zero-gas-anchor '
            'assumptions before enabling constraints. Conflicts with the phase map are reported, not hidden.')
        info.setWordWrap(True); layout.addWidget(info)
        actions = QHBoxLayout(); layout.addLayout(actions)
        self.open_selected = QPushButton('View selected prepared result'); self.open_selected.clicked.connect(self.view_selected)
        self.stop_button = QPushButton('Stop'); self.stop_button.clicked.connect(self.stop); self.stop_button.setEnabled(False)
        for w in (self.open_selected,self.stop_button): actions.addWidget(w)
        layout=self.layout()
        self.status = QLabel('')
        self.status.setWordWrap(True); self.status.setMaximumHeight(34); layout.addWidget(self.status)
        browser = QHBoxLayout(); layout.addLayout(browser)
        self.view_experiment = QComboBox(); self.view_model = QComboBox()
        self.history_model = QComboBox()
        self.view_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.view_model.setMinimumContentsLength(20)
        self.history_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.history_model.setMinimumContentsLength(12)
        self.model_scope_filter = QComboBox()
        self.model_scope_filter.addItem('All calibration scopes', 'all')
        self.model_scope_filter.addItem('Shared / global', 'shared')
        self.model_scope_filter.addItem('Individual', 'individual')
        self.overflow_filter = QComboBox()
        self.overflow_filter.addItem('All overflow uses', 'all')
        self.overflow_filter.addItem('Used in fit', 'fit')
        self.overflow_filter.addItem('Compared only', 'compare')
        self.overflow_filter.addItem('No overflow', 'off')
        self.overflow_filter.addItem('Unknown', 'unknown')
        self.overflow_filter.setToolTip(
            'Used in fit includes constrained and hard-constrained models. '
            'Compared only means the overflow measurement did not influence the fitted model.')
        correction_filters = QHBoxLayout(); layout.addLayout(correction_filters)
        self.illumination_filter = QComboBox()
        self.illumination_filter.addItem('All light corrections', 'all')
        self.illumination_filter.addItem('No spatial correction', 'none')
        self.illumination_filter.addItem('Historical correction', 'historical')
        self.illumination_filter.addItem('Facies correction', 'facies')
        self.illumination_filter.addItem('Unknown / not recorded', 'unknown')
        self.colorchecker_filter = QComboBox()
        self.colorchecker_filter.addItem('All ColorChecker settings', 'all')
        self.colorchecker_filter.addItem('ColorChecker on', 'on')
        self.colorchecker_filter.addItem('ColorChecker off', 'off')
        self.colorchecker_filter.addItem('Unknown / not recorded', 'unknown')
        correction_filters.addWidget(QLabel('Light correction'))
        correction_filters.addWidget(self.illumination_filter)
        correction_filters.addWidget(QLabel('Color correction'))
        correction_filters.addWidget(self.colorchecker_filter)
        correction_filters.addWidget(QLabel('Overflow'))
        correction_filters.addWidget(self.overflow_filter)
        correction_filters.addStretch(1)
        self.model_sort = QComboBox()
        self.model_sort.addItem('Newest first', 'newest')
        self.model_sort.addItem('Lowest mass-fit NMAE · fit data', 'fit_error')
        self.model_sort.addItem('Highest peak free-gas mass (g)', 'peak_gas')
        self.model_sort.setToolTip(
            'Mass-fit NMAE is in-sample calibration error, not independent validation. '
            'Peak free-gas mass is the highest modeled free-CO₂ mass in one saved image '
            'of the selected experiment; it does not compare gas saturation or a whole-series total.')
        browser.addWidget(QLabel('Experiment')); browser.addWidget(self.view_experiment)
        browser.addWidget(QLabel('Scope')); browser.addWidget(self.model_scope_filter)
        browser.addWidget(QLabel('Sort')); browser.addWidget(self.model_sort)
        browser.addWidget(QLabel('Current models')); browser.addWidget(self.view_model, 1)
        browser.addWidget(QLabel('History')); browser.addWidget(self.history_model, 1)
        self.archive_button = QPushButton('Move to history')
        self.archive_button.clicked.connect(lambda: self.move_selected_model(True))
        self.restore_button = QPushButton('Restore')
        self.restore_button.clicked.connect(lambda: self.move_selected_model(False))
        browser.addWidget(self.archive_button); browser.addWidget(self.restore_button)
        self.model_assumption = QLabel()
        self.model_assumption.setWordWrap(True)
        self.model_assumption.setVisible(False)
        layout.addWidget(self.model_assumption)
        self.view_experiment.currentIndexChanged.connect(self.experiment_changed)
        self.view_model.currentIndexChanged.connect(self.variant_changed)
        self.history_model.currentIndexChanged.connect(self.history_variant_changed)
        self.model_scope_filter.currentIndexChanged.connect(lambda *_: self.refresh_browser())
        self.overflow_filter.currentIndexChanged.connect(lambda *_: self.refresh_browser())
        self.illumination_filter.currentIndexChanged.connect(lambda *_: self.refresh_browser())
        self.colorchecker_filter.currentIndexChanged.connect(lambda *_: self.refresh_browser())
        self.model_sort.currentIndexChanged.connect(lambda *_: self.refresh_browser())
        preview_bar = QHBoxLayout(); layout.addLayout(preview_bar)
        self.preview_field = QComboBox()
        self.preview_field.addItem('Concentration', 'preview'); self.preview_field.setEnabled(False)
        self.preview_field.setToolTip('View saved field previews for this photograph. Switching fields does not run a calculation.')
        preview_bar.addWidget(QLabel('Map')); preview_bar.addWidget(self.preview_field)
        self.color_scale = QComboBox()
        self.color_scale.addItem('Fixed linear scale', 'fixed_linear')
        self.color_scale.addItem('Nonlinear scale (γ = 0.5)', 'power')
        self.color_scale.addItem('Auto linear per image', 'frame_linear')
        self.color_scale.setToolTip('Display-only concentration mapping; mass calculations are unchanged. Auto linear rescales each image independently, so colors are not directly comparable between frames.')
        preview_bar.addWidget(QLabel('Concentration scale')); preview_bar.addWidget(self.color_scale)
        self.fullscreen_button = QPushButton('Full screen')
        self.fullscreen_button.setEnabled(False)
        self.fullscreen_button.clicked.connect(self.open_fullscreen_viewer)
        preview_bar.addWidget(self.fullscreen_button)
        self.compare_photometry = QPushButton('Before / after corrections')
        self.compare_photometry.setEnabled(False)
        self.compare_photometry.setToolTip(
            'Compare the same registered photograph before and after a saved illumination / '
            'ColorChecker correction, or compare two saved corrections. The phase map stays fixed.')
        self.compare_photometry.clicked.connect(self.open_photometry_comparison)
        preview_bar.addWidget(self.compare_photometry)
        preview_bar.addStretch(1)
        self.preview_field.currentIndexChanged.connect(self.preview_changed)
        self.color_scale.currentIndexChanged.connect(self.preview_changed)
        # Why an alternate scale could not be drawn, per (frame, scale).
        self._scale_errors = {}
        self.image_position = QLabel('No image selected');
        self.image_position.setMinimumWidth(210)
        self.frame_summary = QLabel('Select a saved concentration result to browse its images.')
        self.frame_summary.setWordWrap(True)
        navigation = QHBoxLayout(); layout.addLayout(navigation)
        self.previous_image = QPushButton('◀ Previous')
        self.next_image = QPushButton('Next ▶')
        self.previous_image.setEnabled(False); self.next_image.setEnabled(False)
        self.previous_image.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value()-1)))
        self.next_image.clicked.connect(lambda: self.slider.setValue(min(self.slider.maximum(), self.slider.value()+1)))
        navigation.addWidget(self.previous_image)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0,0)
        self.slider.setEnabled(False); self.slider.valueChanged.connect(self.show_frame)
        navigation.addWidget(self.slider, 1)
        navigation.addWidget(self.next_image)
        navigation.addWidget(self.image_position)
        layout.addWidget(self.frame_summary)
        self.details_toggle=QPushButton('Show method and quality notes')
        self.details_toggle.setCheckable(True); layout.addWidget(self.details_toggle)
        self.details_body=QWidget(); detail_layout=QVBoxLayout(self.details_body)
        detail_layout.setContentsMargins(2,0,2,2)
        self.compatibility = QLabel(); self.compatibility.setWordWrap(True); detail_layout.addWidget(self.compatibility)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(105)
        detail_layout.addWidget(self.details); layout.addWidget(self.details_body)
        self.details_body.setVisible(False)
        self.details_toggle.toggled.connect(self.details_body.setVisible)
        self._fullscreen_refresh = None
        self.views = QTabWidget(); layout.addWidget(self.views, 1)
        # Keep the figures readable in the embedded workspace. Its host pages
        # scroll vertically when the available screen height is smaller. The
        # figures rescale to their tab and need no scroll area of their own.
        self.views.setMinimumHeight(700)
        self.photo = _FullscreenPixmap(QPixmap())
        self.photo.setMinimumSize(500, 300)
        self.curve = _FullscreenPixmap(QPixmap())
        self.overflow_plot = _FullscreenPixmap(QPixmap())
        self.dilution_plot = _FullscreenPixmap(QPixmap())
        self._figure_keys = {}
        for title, widget in [('Photograph + concentration',self.photo),('Mass curves',self.curve),
                              ('Area and dilution',self.dilution_plot),('Overflow / free gas',self.overflow_plot)]:
            tab_index=self.views.addTab(widget,title)
            if widget is self.photo:
                self.photo_index=tab_index
            else:
                self._figure_keys[tab_index]={self.curve:'mass_curve',self.dilution_plot:'dilution_curve',
                                              self.overflow_plot:'overflow_curve'}[widget]
        self.comparison = QTableWidget(0, 6)
        self.comparison.setHorizontalHeaderLabels(['Concentration model','Injected (g)','Dissolved (g)','Free gas (g)','Total (g)','Total − injected (g)'])
        self.comparison.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comparison.horizontalHeader().setStretchLastSection(True)
        self.views.addTab(self.comparison, 'Compare models at this image')
        self.mass_comparison_plot = _FullscreenPixmap(QPixmap())
        self.mass_comparison_plot.setText('Open this tab to compare injected, dissolved and free-gas mass for the saved calibrations.')
        self.mass_comparison_index = self.views.addTab(self.mass_comparison_plot, 'Compare mass curves')
        self.local_diagnostics_plot=_FullscreenPixmap(QPixmap())
        self.local_diagnostics_index=self.views.addTab(self.local_diagnostics_plot,'Local diagnostics')
        self._figure_keys[self.local_diagnostics_index]='local_diagnostics_curve'
        self.views.setTabVisible(self.local_diagnostics_index,False)
        self.views.currentChanged.connect(self.show_mass_comparison)
        self.export_toggle=QPushButton('Export…'); self.export_toggle.setCheckable(True)
        layout.addWidget(self.export_toggle)
        self.export_body=QWidget(); exports=QHBoxLayout(self.export_body);layout.addWidget(self.export_body)
        self.export_toggle.toggled.connect(self.export_body.setVisible); self.export_body.setVisible(False)
        self.export_buttons=[]
        for text,kind in [('Export MP4','movie'),('Export current figure','figure'),('Export CSV','csv'),('Export figures + numerical fields','bundle')]:
            button=QPushButton(text);button.setEnabled(False);button.clicked.connect(lambda _=False,k=kind:self.export(k))
            exports.addWidget(button);self.export_buttons.append(button)
        self.timer=QTimer(self);self.timer.setInterval(1000);self.timer.timeout.connect(self.poll)
        QApplication.instance().aboutToQuit.connect(self.stop)

    def ensure_selected_model(self):
        """Refresh on entering this tab after changing the Studio phase model."""
        if self.process is not None and self.process.state()!=QProcess.NotRunning:return
        model = self.selected_model() if self.selected_model else None
        if model and (not self.phase_catalog or str(model['root']) != self.phase_catalog.get('model_root')):
            self.load_model_results()

    def _series_changed(self, *_):
        dataset = str(self.series.currentData() or '').casefold()
        if dataset and dataset not in self._archive_by_series:
            try:
                root = series_path(dataset, 'results') / 'concentration_mass'
                self._archive_by_series[dataset] = load_archive(root, dataset)
            except (OSError, ValueError) as error:
                self._archive_by_series[dataset] = {}
                self.status.setText('Saved history could not be loaded: ' + str(error))
        self.refresh()
        self._sync_series_controls()
        if not dataset or dataset in self._series_catalog_loaded:
            return
        if self.series_lookup is not None and self.series_lookup.isRunning():
            if self.series_lookup.dataset != dataset:
                self.status.setText(f'Loading saved {dataset.upper()} results after the current lookup…')
            return
        self.status.setText(f'Finding saved {dataset.upper()} concentration results…')
        self.series_lookup = SavedSeriesResultLookup(dataset, self, self.cache_root / 'saved_series_indexes')
        self.series_lookup.ready.connect(self._series_results_ready)
        self.series_lookup.start()

    def _series_results_ready(self, dataset, entries, error):
        if not error:
            positions = {old['path']: index for index, old in enumerate(self.entries)}
            for entry in entries:
                if entry['path'] in positions:
                    self.entries[positions[entry['path']]] = entry
                else:
                    self.entries.append(entry)
            self._series_catalog_loaded.add(dataset)
            if self.series.currentData() == dataset:
                # A late background lookup must not replace an image that the
                # researcher has already opened or overwrite its frame status.
                self.refresh(keep_view=self.result is not None)
                if self.result is None:
                    self.status.setText(f'{len(entries)} saved {dataset.upper()} concentration results found.')
        elif self.series.currentData() == dataset:
            self.status.setText('Saved results could not be read: ' + error)
        self.series_lookup = None
        if self.series.currentData() != dataset:
            self._series_changed()

    def _archive_key(self, entry):
        root = series_path(entry['dataset'], 'results') / 'concentration_mass'
        return family_key(entry['path'], root)

    def _photometry_policy(self, dataset):
        dataset = str(dataset).casefold()
        if dataset not in self._photometry_policies:
            self._photometry_policies[dataset] = load_policy(dataset)
        return self._photometry_policies[dataset]

    def _is_excluded_method(self, entry):
        dataset = str(entry.get('dataset') or '').casefold()
        return variant_status(dataset, entry.get('method') or {},
                              self._photometry_policy(dataset)) == 'excluded'

    def _is_archived(self, entry):
        if self._is_excluded_method(entry):
            return True
        try:
            return self._archive_key(entry) in self._archive_by_series.get(entry['dataset'], {})
        except ValueError:
            # Explicitly imported external comparison files can still be
            # viewed, but cannot be placed in a series-owned NAS archive.
            return False

    def _sorted_model_entries(self, entries, *, history=False):
        scope = self.model_scope_filter.currentData()
        if scope != 'all':
            scopes = ('shared', 'series') if scope == 'shared' else (scope,)
            entries = [entry for entry in entries if entry.get('calibration_scope') in scopes]
        overflow = self.overflow_filter.currentData()
        if overflow == 'fit':
            entries = [entry for entry in entries
                       if entry.get('overflow_usage') in ('constrain', 'hard')]
        elif overflow != 'all':
            entries = [entry for entry in entries if entry.get('overflow_usage') == overflow]
        light = self.illumination_filter.currentData()
        color = self.colorchecker_filter.currentData()
        if light != 'all' or color != 'all':
            entries = [entry for entry in entries
                       if ((light == 'all' or _photometry_filter_values(entry)[0] == light)
                           and (color == 'all' or _photometry_filter_values(entry)[1] == color))]
        entries = [entry for entry in entries if self._is_archived(entry) == history]
        if self.model_sort.currentData() == 'fit_error':
            return sorted(entries, key=lambda entry: (
                entry.get('mass_fit_nmae') is None,
                entry.get('mass_fit_nmae') if entry.get('mass_fit_nmae') is not None else float('inf'),
                entry.get('created_utc', ''), entry['path']))
        if self.model_sort.currentData() == 'peak_gas':
            return sorted(entries, key=lambda entry: (
                entry.get('peak_free_gas_mass_g') is not None,
                entry.get('peak_free_gas_mass_g') if entry.get('peak_free_gas_mass_g') is not None else float('-inf'),
                entry.get('created_utc', ''), entry['path']), reverse=True)
        return sorted(entries, key=lambda entry: (entry.get('created_utc', ''), entry['path']), reverse=True)

    def _sync_series_controls(self):
        dataset = str(self.series.currentData() or '').casefold()
        bound = str((self.phase_catalog or {}).get('dataset') or '').casefold()
        matches = bool(bound and dataset == bound)
        self.phase_experiment.setEnabled(matches)
        if bound and not matches:
            self.phase_binding.setText(
                f"Browsing saved {dataset.upper()} concentration results. The loaded phase model is "
                f"for {bound.upper()}; select a {dataset.upper()} phase model before calculating new results.")
        elif matches:
            self.phase_binding.setText('Phase model: '+str(self.phase_catalog.get('model_id') or '')+
                ' · same images as phase validation. Loading this list does not calculate concentration or alter the phase maps.')

    def open_photometry_comparison(self):
        if self.result is None:
            return
        from darsia.fluidflower.tools.phase_studio_photometry_view import (
            PhotometryComparisonDialog, variant_choices,
        )
        choices = variant_choices(self.visible, self.result, self.result_path)
        if not choices:
            self.status.setText('No saved correction variant is available for this phase model and run.')
            return
        existing = getattr(self, '_photometry_comparison_dialog', None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        dialog = PhotometryComparisonDialog(
            self.result, self.result_path, self.slider.value(), self.visible, self)
        self._photometry_comparison_dialog = dialog
        dialog.showMaximized()

    def open_fullscreen_viewer(self):
        if self.result is None:
            self.status.setText('Select a saved concentration result before opening the full-screen viewer.')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Concentration maps and figures')
        outer = QVBoxLayout(dialog)
        controls = QHBoxLayout()
        summary = QLabel(self.image_position.text() + '\n' + self.frame_summary.text())
        summary.setWordWrap(True)
        controls.addWidget(summary, 1)
        close = QPushButton('Exit full screen')
        close.clicked.connect(dialog.close)
        controls.addWidget(close)
        outer.addLayout(controls)
        # Keep frame navigation above the figure. A large saved image can make
        # the scroll area request more height than the full-screen dialog has;
        # controls below it were then pushed outside the visible screen.
        navigation = QHBoxLayout()
        previous = QPushButton('◀ Previous image', dialog)
        next_image = QPushButton('Next image ▶', dialog)
        position = QLabel(self.image_position.text(), dialog)
        slider = QSlider(Qt.Orientation.Horizontal, dialog)
        slider.setRange(self.slider.minimum(), self.slider.maximum())
        slider.setValue(self.slider.value())
        previous.clicked.connect(lambda: slider.setValue(max(slider.minimum(), slider.value()-1)))
        next_image.clicked.connect(lambda: slider.setValue(min(slider.maximum(), slider.value()+1)))
        slider.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(slider.setValue)
        navigation.addWidget(previous); navigation.addWidget(slider, 1)
        navigation.addWidget(next_image); navigation.addWidget(position)
        outer.addLayout(navigation)
        tabs = QTabWidget(dialog)
        outer.addWidget(tabs, 1)
        labels = []
        titles = [self.views.tabText(i) for i in range(self.views.count())]

        def refresh_images(*_):
            summary.setText(self.image_position.text() + '\n' + self.frame_summary.text())
            position.setText(self.image_position.text())
            previous.setEnabled(slider.value() > slider.minimum())
            next_image.setEnabled(slider.value() < slider.maximum())
            index = tabs.currentIndex()
            if index < 0 or index >= len(labels):
                return
            label = labels[index]
            path = None
            if index == self.photo_index:
                path = self._current_figure_path()
            elif self.result is not None and index in self._figure_keys:
                path = self._saved_figure_path(self.result.get(self._figure_keys[index]))
            elif index == self.mass_comparison_index:
                path = self.mass_comparison_plot.toolTip() or None
            path_key = str(path) if path else None
            if path_key is not None and path_key == getattr(label, '_loaded_path', object()):
                return
            pixmap = QPixmap(path_key) if path_key else QPixmap()
            if pixmap.isNull():
                tab = self.views.widget(index)
                source = tab.widget() if isinstance(tab, QScrollArea) else tab
                pixmap = source.grab() if source is not None else QPixmap()
            label._loaded_path = path_key
            label._source_pixmap = pixmap
            label._rescale()

        for title in titles:
            area = QScrollArea(dialog)
            area.setWidgetResizable(True)
            label = _FullscreenPixmap(QPixmap(), area)
            area.setWidget(label)
            labels.append(label)
            tabs.addTab(area, title)
        tabs.setCurrentIndex(self.views.currentIndex())
        def sync_fullscreen_tab(index):
            if self.views.currentIndex() != index:
                self.views.setCurrentIndex(index)
            refresh_images()

        tabs.currentChanged.connect(sync_fullscreen_tab)
        refresh_images()
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), dialog)
        shortcut.activated.connect(dialog.close)
        previous_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), dialog)
        previous_shortcut.activated.connect(previous.click)
        next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), dialog)
        next_shortcut.activated.connect(next_image.click)
        dialog.finished.connect(lambda _=0: self._close_fullscreen_viewer(
            dialog, tabs, slider, refresh_images, sync_fullscreen_tab))
        self.views.currentChanged.connect(refresh_images)
        self._fullscreen_refresh = refresh_images
        self._fullscreen_dialog = dialog
        dialog.showFullScreen()

    def _close_fullscreen_viewer(self, dialog, tabs, slider, refresh_images, sync_fullscreen_tab):
        try:
            self.views.currentChanged.disconnect(refresh_images)
        except (RuntimeError, TypeError):
            pass
        try:
            tabs.currentChanged.disconnect(sync_fullscreen_tab)
        except (RuntimeError, TypeError):
            pass
        try:
            self.slider.valueChanged.disconnect(slider.setValue)
        except (RuntimeError, TypeError):
            pass
        if getattr(self, '_fullscreen_dialog', None) is dialog:
            self._fullscreen_dialog = None
            self._fullscreen_refresh = None

    def import_paths(self, path):
        paths = path if isinstance(path, (list, tuple)) else backend.result_paths(path)
        for p in paths:
            value = backend.read_result(p)
            entry = backend.inspect_result(p, value)
            _add_model_browser_metadata(entry, value, p)
            if not any(e['path']==entry['path'] for e in self.entries):
                for old in self.entries:
                    if (old['dataset'],old['run'])==(entry['dataset'],entry['run']):self.checked.discard(old['path'])
                self.entries.append(entry)
                if not self._is_excluded_method(entry):
                    self.checked.add(entry['path'])
        self.refresh()

    def choose_results(self):
        path,_=QFileDialog.getOpenFileName(self,'Prepared result or delivery report',str(local_cache_root() / 'phase_diagnostics'),'JSON (*.json)')
        if not path:return
        try:self.import_paths(path);self.status.setText('Prepared results imported. Check phase-model versions before fitting.')
        except Exception as error:self.status.setText(str(error))

    def load_model_results(self):
        if self.lookup is not None and self.lookup.isRunning():return
        try:
            model=self.selected_model() if self.selected_model else None
            if not model:raise ValueError('Select a phase model in Review and label evidence first')
            self.import_model.setEnabled(False)
            self.lookup = ModelMassLookup(Path(model['root']), self)
            self.lookup.ready.connect(self.model_results_ready)
            self.lookup.finished.connect(lambda: self.import_model.setEnabled(True))
            self.status.setText('Loading registered phase images and saved concentration models…')
            self.lookup.start()
        except Exception as error:self.status.setText(str(error))

    def model_results_ready(self, catalog, error):
        if error:
            self.phase_catalog = None; self.phase_experiment.clear()
            self.phase_binding.setText('Phase-model inputs unavailable: '+error)
            self.status.setText(error); return
        previous_path = str(self.result_path.resolve()) if self.result_path else None
        self.phase_catalog = catalog
        ready = sum(bool(job.get('concentration',{}).get('ready')) for job in catalog['jobs'])
        self.series.setCurrentIndex(self.series.findData(catalog['dataset']))
        self.phase_experiment.clear()
        for job in catalog['jobs']:self.phase_experiment.addItem(job['run'].upper(), job)
        self.phase_binding.setText('Phase model: '+catalog['model_id']+' · same images as phase validation. '
            'Loading this list does not calculate concentration or alter the phase maps.')
        if catalog['results']:
            self.import_paths(catalog['results'])
        result_paths = {str(Path(path).resolve()) for path in catalog['results']}
        if previous_path not in result_paths and result_paths:
            latest = max((entry for entry in self.entries if entry['path'] in result_paths),
                         key=lambda entry:(entry.get('created_utc',''),entry['path']))
            self.load_result(latest['path'])
        if catalog.get('saved_results_deferred'):
            self._series_changed()
            if catalog['dataset'] in self._series_catalog_loaded:
                count = sum(entry['dataset'] == catalog['dataset'] for entry in self.entries)
                self.status.setText(f"{len(catalog['jobs'])} experiments available · {count} saved concentration results. "
                    'Select registered native inputs above to calculate; saved results can be compared below.')
        else:
            self.status.setText(f"{len(catalog['jobs'])} experiments available · {len(catalog['results'])} saved concentration results. "
                'Select registered native inputs above to calculate; saved results can be compared below.')
        self._sync_series_controls()


    def attach_modern_overflow(self, run):
        try:
            if not self.phase_catalog or self.phase_catalog['dataset'] != self.series.currentData():
                raise ValueError('Load the matching phase model before attaching overflow.')
            path, _ = QFileDialog.getOpenFileName(
                self, f'Measured, time-linked overflow targets for {run.upper()}',
                str(series_path(self.phase_catalog['dataset'], 'results')), 'JSON (*.json)')
            if path:
                self.overflow[(self.phase_catalog['dataset'], run)] = path
                self.status.setText(f'{run.upper()}: measured overflow target attached. '
                                    'The calculation will verify times, uncertainty and source.')
        except (OSError, ValueError) as error:
            self.status.setText(str(error))





    def refresh(self, *, keep_view=False):
        self.visible=[e for e in self.entries if e['dataset']==self.series.currentData()]
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.visible))
        for row,e in enumerate(self.visible):
            excluded = self._is_excluded_method(e)
            check=QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled if excluded else Qt.ItemIsUserCheckable|Qt.ItemIsEnabled)
            if excluded:
                self.checked.discard(e['path'])
                check.setToolTip('Excluded FF_UM light correction; available for historical review only.')
            check.setCheckState(Qt.Checked if e['path'] in self.checked else Qt.Unchecked)
            self.table.setItem(row,0,check)
            target=self.overflow.get((e['dataset'],e['run']))
            for col,text in enumerate([e['run'].upper(),e['model'],e['created_utc'],Path(target).name if target else 'None — shared gas model'],1):
                item=QTableWidgetItem(text);item.setToolTip(str(target) if col==4 and target else text);self.table.setItem(row,col,item)
            item=QTableWidgetItem(e.get('label','Saved model'));item.setToolTip(item.text());self.table.setItem(row,5,item)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(2,220);self.table.setColumnWidth(5,340)
        self.table.blockSignals(False)
        if not keep_view:
            self.refresh_browser()

    def refresh_browser(self):
        selected = self.view_experiment.currentData()
        runs = sorted({e['run'] for e in self.visible})
        self.view_experiment.blockSignals(True);self.view_experiment.clear()
        for run in runs:self.view_experiment.addItem(run.upper(),run)
        if selected in runs:self.view_experiment.setCurrentIndex(self.view_experiment.findData(selected))
        self.view_experiment.blockSignals(False)
        self.experiment_changed()

    def experiment_changed(self):
        old_path = str(self.result_path) if self.result_path else None
        selected_run = self.view_experiment.currentData()
        entries = [e for e in self.visible if e['run']==selected_run]
        loaded_run = ((self.result or {}).get('selected_inputs', {}).get('candidate') or {}).get('run')
        preserve_loaded = bool(old_path and loaded_run == selected_run)
        self._fill_model_menus(entries, old_path if preserve_loaded else None,
                               preserve_unmatched=preserve_loaded)
        if old_path and self.view_model.findData(old_path) >= 0:
            self.variant_changed()
        elif old_path and self.history_model.findData(old_path) >= 0:
            self.history_variant_changed()
        elif preserve_loaded:
            self.status.setText(_FILTERED_DISPLAY_NOTICE)
        elif self.view_model.count():
            self.variant_changed()
        elif self.history_model.count():
            self.history_variant_changed()
        else:
            self.clear_view()

    def _fill_model_menus(self, entries, selected_path=None, *, preserve_unmatched=False):
        for menu, history in ((self.view_model, False), (self.history_model, True)):
            menu.blockSignals(True)
            menu.clear()
            for entry in self._sorted_model_entries(entries, history=history):
                label = entry.get('label', 'Saved model')
                tooltip = label
                if str(entry.get('dataset') or '').casefold() == 'ffum':
                    scenario = explain_ffum_scenario({'method': {'settings': entry.get('method') or {}}})
                    if scenario:
                        scope = 'Shared' if entry.get('calibration_scope') in ('shared', 'series') else 'Individual'
                        peak = entry.get('peak_free_gas_mass_g')
                        peak_text = f' · peak free gas {peak:.3f} g' if isinstance(peak, (int, float)) else ''
                        date = str(entry.get('created_utc') or 'undated').replace('T', ' ')[:16]
                        tag = hashlib.sha256(entry['path'].encode('utf-8')).hexdigest()[:8]
                        label = f"{scope} · {scenario['short_label']}{peak_text} · {date} / {tag}"
                        tooltip = f"{scenario['detail']}\n\n{tooltip}"
                if self._is_excluded_method(entry):
                    label += ' · Excluded for FF_UM: light correction'
                menu.addItem(label, entry['path'])
                menu.setItemData(menu.count() - 1, tooltip, Qt.ToolTipRole)
            index = menu.findData(selected_path)
            if index >= 0:
                menu.setCurrentIndex(index)
            elif preserve_unmatched:
                menu.setCurrentIndex(-1)
            menu.blockSignals(False)
        if selected_path and self.view_model.findData(selected_path) >= 0:
            self._active_model_menu = 'current'
        elif selected_path and self.history_model.findData(selected_path) >= 0:
            self._active_model_menu = 'history'
        elif preserve_unmatched and selected_path:
            self._active_model_menu = None
        elif self.view_model.count():
            self._active_model_menu = 'current'
        elif self.history_model.count():
            self._active_model_menu = 'history'
        else:
            self._active_model_menu = None
        self._update_archive_buttons()

    def _update_archive_buttons(self):
        self.archive_button.setEnabled(self._active_model_menu == 'current' and bool(self.view_model.currentData()))
        selected = self.history_model.currentData()
        entry = next((item for item in self.visible if item['path'] == selected), None)
        self.restore_button.setEnabled(self._active_model_menu == 'history' and bool(selected)
                                       and entry is not None and not self._is_excluded_method(entry))

    def clear_view(self):
        self.result=None;self.result_path=None;self._active_preview_path=None
        for widget in (self.photo,self.curve,self.dilution_plot,self.overflow_plot,
                       self.mass_comparison_plot,self.local_diagnostics_plot):
            widget.set_source_pixmap(QPixmap())
            widget.clear()
        self.details.clear();self.views.setTabVisible(self.local_diagnostics_index,False)
        self.preview_field.blockSignals(True);self.preview_field.clear()
        self.preview_field.addItem('Concentration','preview');self.preview_field.setEnabled(False)
        self.preview_field.blockSignals(False)
        self.slider.setRange(0,0);self.comparison.setRowCount(0)
        self.slider.setEnabled(False)
        self.previous_image.setEnabled(False); self.next_image.setEnabled(False)
        self.fullscreen_button.setEnabled(False)
        self.compare_photometry.setEnabled(False)
        self.image_position.setText('No image selected')
        self.frame_summary.setText('No saved concentration result for this series.')
        self.model_assumption.clear()
        self.model_assumption.setVisible(False)
        self.compatibility.clear()
        for button in self.export_buttons:button.setEnabled(False)

    def variant_changed(self):
        path=self.view_model.currentData()
        if path:
            self._active_model_menu = 'current'
            self._update_archive_buttons()
            try:self.load_result(path)
            except Exception as error:
                self.clear_view();self.status.setText(str(error))

    def history_variant_changed(self):
        path=self.history_model.currentData()
        if path:
            self._active_model_menu = 'history'
            self._update_archive_buttons()
            try:self.load_result(path)
            except Exception as error:
                self.clear_view();self.status.setText(str(error))

    def move_selected_model(self, archived):
        path = self.view_model.currentData() if archived else self.history_model.currentData()
        if not path:
            self.status.setText('Select a saved calibration first.')
            return
        dataset = str(self.series.currentData() or '').casefold()
        try:
            entry = next((item for item in self.visible if item['path'] == path), None)
            if entry is None:
                raise ValueError('Selected calibration is no longer in this series.')
            if not archived and self._is_excluded_method(entry):
                raise ValueError('This FF_UM light-correction method is excluded and cannot be restored.')
            if entry.get('receipt_path'):
                verify_saved_series_entry(entry)
            root = series_path(dataset, 'results') / 'concentration_mass'
            _, families = set_archived(root, dataset, path, archived)
            self._archive_by_series[dataset] = families
            self.experiment_changed()
            self.status.setText('Calibration moved to History for all its experiments.' if archived
                                else 'Calibration restored to Current models for all its experiments.')
        except (OSError, ValueError, KeyError) as error:
            self.status.setText('Could not change model visibility: ' + str(error))

    def check_changed(self,item):
        if item.column()!=0:return
        path=self.visible[item.row()]['path']
        if item.checkState()==Qt.Checked:self.checked.add(path)
        else:self.checked.discard(path)

    def selected_entry(self):
        row=self.table.currentRow()
        if row<0 or row>=len(self.visible):raise ValueError('Select an experiment row')
        return self.visible[row]

    def choose_overflow(self):
        try:
            entry=self.selected_entry()
            if entry['dataset']!='ffac':raise ValueError('This target importer currently supports reviewed FF_AC overflow only')
            path,_=QFileDialog.getOpenFileName(self,'Reviewed derived overflow gas targets','','JSON (*.json)')
            if not path:return
            source=backend.read_result(entry['path'])
            refs,binding=backend.references_for(entry['run'],[r['elapsed_h'] for r in source['frames']],path)
            self.overflow[(entry['dataset'],entry['run'])]=path
            self.refresh()
            self.status.setText(f"Overflow: {sum(r is not None for r in refs)}/{len(refs)} selected frames covered. No extrapolation.")
            self.details.setPlainText(json.dumps(binding,indent=2))
        except Exception as error:self.status.setText(str(error))

    def remove_overflow(self):
        try:
            e=self.selected_entry();self.overflow.pop((e['dataset'],e['run']),None);self.refresh()
        except ValueError as error:self.status.setText(str(error))


    def load_calibration(self):
        path,_=QFileDialog.getOpenFileName(self,'Frozen shared-series calibration','','JSON (*.json)')
        if not path:return
        try:
            fit=backend.read(path)
            if fit.get('contract') == 'ffac-common-mass-round3-frozen-v1':
                if fit.get('frozen') is not True:
                    raise ValueError('Select a frozen calibration document')
                from PySide6.QtWidgets import QDialog, QVBoxLayout
                from darsia.fluidflower.mass_calibration_display import MassCalibrationDisplay, read_mass_calibration_display
                dialog = QDialog(self)
                dialog.setWindowTitle('Saved calibration — read only')
                view = MassCalibrationDisplay(dialog)
                view.set_display(read_mass_calibration_display('phase_model', path))
                layout = QVBoxLayout(dialog)
                notice = QLabel('Viewing this file does not select a calibration for the series or change the legacy calculation settings.')
                notice.setWordWrap(True)
                layout.addWidget(notice)
                layout.addWidget(view)
                dialog.resize(760, 420)
                dialog.exec()
                return
            if fit.get('format')!=backend.series_mass.VERSION or fit.get('scope')!='series':
                raise ValueError('Select a shared calibration.json generated by this workspace')
            if fit.get('dataset')!=self.series.currentData():raise ValueError('Calibration belongs to another series')
            self.calibration_file=Path(path)
            self.scope.setCurrentIndex(self.scope.findData('series'))
            self.overflow_mode.setCurrentIndex(self.overflow_mode.findData(fit['overflow_mode']))
            self.calibration_label.setText('Frozen parameters — no refit: '+str(path))
        except Exception as error:self.status.setText(str(error))

    def clear_calibration(self):
        self.calibration_file=None
        self.calibration_label.setText('New calibration: parameters fitted jointly to checked experiments.')

    def launch(self,arguments,operation):
        if self.process is not None and self.process.state()!=QProcess.NotRunning:return
        root=process_directory()
        from darsia.fluidflower.phase_workflow import phase_python_executable
        python=Path(phase_python_executable())
        self.operation=operation;self.process=QProcess(self);self.process.setWorkingDirectory(str(root))
        if self.job is None:self.job=self.cache_root/uuid.uuid4().hex;self.job.mkdir(parents=True)
        self.process.setStandardOutputFile(str(self.job/'stdout.log'))
        self.process.setStandardErrorFile(str(self.job/'stderr.log'))
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.stop_button.setEnabled(True)
        if operation != 'export':
            raise ValueError("This workspace only exports completed saved results")
        module = 'darsia.fluidflower.tools.phase_studio_series_export'
        self.process.start(str(python),['-B','-m',module,*arguments])
        self.status.setText('Working in background; phase validation remains available.');self.timer.start()

    def process_error(self,error):
        if error==QProcess.FailedToStart:
            self.timer.stop();self.stop_button.setEnabled(False)
            self.status.setText(self.process.errorString())

    def poll(self):
        if self.operation not in ('calculate','native','joint','photometry','local_full_depth') or self.job is None:return
        try:state=backend.read(self.job/'status.json')
        except (OSError,ValueError):return
        self.status.setText(state['stage'])
        if state.get('error'):self.details.setPlainText(state['error'])

    def finished(self,code,_status):
        self.poll();self.timer.stop();self.stop_button.setEnabled(False)
        if code:
            self.status.setText('Operation stopped or failed. Existing results are unchanged.')
            try:
                state = backend.read(self.job/'status.json')
                if state.get('error'):
                    self.status.setText(state['error'].strip().splitlines()[-1]+' Existing results are unchanged.')
            except (OSError,ValueError):pass
            try:self.details.setPlainText((self.job/'stderr.log').read_text()[-6000:])
            except OSError:pass
        elif self.operation in ('calculate','native'):
            delivery=self.job/'results/delivery.json'
            self.import_paths(delivery)
            if self.phase_catalog:
                from darsia.fluidflower.tools.phase_studio_mass_inputs import register_variant
                for path in backend.result_paths(delivery):register_variant(path,self.phase_catalog['results_root'])
            result=backend.result_paths(delivery)[0];self.load_result(result)
            self.status.setText('Calculation complete. Calibration status, parameter fitting and measurement use are recorded in the results.')
        elif self.operation in ('joint','photometry'):
            try:
                delivery = backend.read(self.job / 'results' / 'delivery.json')
                rows = (delivery.get('experiments', []) if self.operation == 'joint'
                        else delivery.get('deliveries', []))
                paths = [Path(row['result']) for row in rows if row.get('result')]
                if not delivery.get('complete') or not paths:
                    raise ValueError('The new calibration has no complete Studio results')
                self.import_paths(paths)
                self.load_result(paths[0])
                if self.phase_catalog:
                    self._series_catalog_loaded.discard(self.phase_catalog['dataset'])
                    self._series_changed()
                self.status.setText(f'{len(paths)} saved calibration results ready in Studio. '
                                    'Inspect held-out checks and method details before choosing a model.')
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.status.setText('Calibration finished but Studio could not load its delivery: ' + str(error))
        elif self.operation == 'local_full_depth':
            try:
                delivery = backend.read(self.job / 'delivery.json')
                paths = [Path(path) for path in delivery.get('results', [])]
                if not delivery.get('complete') or not paths:
                    raise ValueError('The depth-saturation comparison has no complete Studio results')
                self.import_paths(paths)
                self.load_result(paths[0])
                self._series_catalog_loaded.discard('ffum')
                self._series_changed()
                self.status.setText('Experimental visible-gas depth comparison ready. '
                                    'Inspect the global and local variants and their held-out diagnostics.')
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.status.setText('Depth-saturation comparison finished but Studio could not load it: ' + str(error))
        else:self.status.setText('Export complete.')

    def stop(self):
        if self.process is not None and self.process.state()!=QProcess.NotRunning:self.process.kill()
        running = [thread for thread in (self.lookup, self.series_lookup)
                   if thread is not None and thread.isRunning()]
        for thread in running:
            thread.requestInterruption()
        for thread in running:
            thread.wait()

    def view_selected(self):
        try:self.load_result(self.selected_entry()['path'])
        except Exception as error:self.status.setText(str(error))

    def load_result(self,path):
        registered = next((entry for entry in self.entries
                           if entry['path'] == str(Path(path).resolve()) and entry.get('receipt_path')), None)
        if registered is not None:
            verify_saved_series_entry(registered)
        value=backend.read_result(path)
        if not value.get('complete') or not value.get('frames'):raise ValueError('Result is incomplete')
        candidate=value['selected_inputs']['candidate']
        old=self.result; old_index=self.slider.value()
        same_run=bool(old and all(old['selected_inputs']['candidate'].get(key)==candidate.get(key) for key in ('dataset','run')))
        index=matching_frame_index(old,old_index,value) if same_run else 0
        self.result,self.result_path=value,Path(path)
        self._scale_errors = {}
        self._view_cache[str(Path(path).resolve())]=value
        self.view_experiment.blockSignals(True)
        self.view_experiment.setCurrentIndex(self.view_experiment.findData(candidate['run']))
        self.view_experiment.blockSignals(False)
        # Keep the browser bound to explicitly opened table rows as well.
        entries=[e for e in self.visible if e['run']==candidate['run']]
        displayed_path = str(Path(path).resolve())
        self._fill_model_menus(entries, displayed_path, preserve_unmatched=True)
        hidden_by_filters = (self.view_model.findData(displayed_path) < 0
                             and self.history_model.findData(displayed_path) < 0)
        self.slider.blockSignals(True);self.slider.setRange(0,len(value['frames'])-1)
        self.slider.setValue(index if index is not None else 0);self.slider.blockSignals(False)
        self.slider.setEnabled(len(value['frames']) > 1)
        self.previous_image.setEnabled(len(value['frames']) > 1)
        self.next_image.setEnabled(len(value['frames']) > 1)
        self.fullscreen_button.setEnabled(True)
        from darsia.fluidflower.tools.phase_studio_photometry_view import variant_choices
        self.compare_photometry.setEnabled(
            bool(variant_choices(self.visible, value, path)))
        scenario = explain_ffum_scenario(value) if candidate['dataset'].casefold() == 'ffum' else None
        self.model_assumption.setText(
            f"What this model assumes: {scenario['short_label']}. {scenario['detail']}" if scenario else '')
        self.model_assumption.setVisible(bool(scenario))
        fit_warnings = value.get('calibration',{}).get('warnings',[])
        spatial_note = ('Exploratory calibration: good total-mass agreement does not validate local concentration. '
            'Inspect concentration jumps at sand-layer boundaries.'
            if value.get('method',{}).get('settings',{}).get('response') in ('native_preflash_monotone', 'joint_reference_dilution') else '')
        self.compatibility.setText('\n'.join(filter(None,[
            ('Excluded for FF_UM: this light-correction method is retained for historical review only.'
             if variant_status(candidate['dataset'], value.get('method', {}).get('settings') or {},
                               self._photometry_policy(candidate['dataset'])) == 'excluded' else ''),
            format_saved_method_summary(value),
            ac_compatibility_note(candidate['dataset'],candidate['run']),
            spatial_note,
            *('Calibration: '+str(warning) for warning in fit_warnings)])))
        self.show_frame(self.slider.value())
        self.status.setText('Matching image is absent from this saved calibration; showing its first available image.'
                            if same_run and index is None else _FILTERED_DISPLAY_NOTICE if hidden_by_filters else '')
        self.views.setTabVisible(self.local_diagnostics_index,bool(value.get('local_diagnostics_curve')))
        self._load_saved_figures(value)
        for button in self.export_buttons:button.setEnabled(True)

        self.show_mass_comparison()

    def _load_saved_figures(self, value):
        """Load the retained tab figures only for a presentation that shows them."""
        for widget,key in [(self.curve,'mass_curve'),(self.dilution_plot,'dilution_curve'),
                           (self.overflow_plot,'overflow_curve'),(self.local_diagnostics_plot,'local_diagnostics_curve')]:
            if value.get(key):
                widget.set_source_pixmap(QPixmap(self._saved_figure_path(value[key])))
            else:
                widget.set_source_pixmap(QPixmap())
                widget.setText({'mass_curve':'No mass curve in this saved result.',
                    'dilution_curve':'No area and mean-concentration curve in this saved result.',
                    'overflow_curve':'No overflow comparison in this saved result. Attach targets and calculate to add it.',
                    'local_diagnostics_curve':'No local diagnostics curve in this saved result.'}[key])
    def show_mass_comparison(self, *_):
        """Plot saved scalar results only when this comparison tab is requested."""
        if self.views.currentIndex() != self.mass_comparison_index:return
        self.mass_comparison_plot.set_source_pixmap(QPixmap())
        self.mass_comparison_plot.setToolTip('')
        if self.result is None:return
        try:
            from darsia.fluidflower.tools.phase_studio_mass_phase_plot import render_mass_phase_comparison
            candidate = self.result['selected_inputs']['candidate']
            values, identities = {}, []
            entries = sorted(self.visible, key=lambda e:(e.get('created_utc',''),e['path']), reverse=True)
            # The viewed calibration takes precedence over a newer saved fit.
            entries.sort(key=lambda e:str(Path(e['path']).resolve()) != str(self.result_path.resolve()))
            for entry in entries:
                if entry['run'] != candidate['run']:continue
                path = str(Path(entry['path']).resolve())
                if path not in self._view_cache:
                    if entry.get('receipt_path'):
                        verify_saved_series_entry(entry)
                    self._view_cache[path] = backend.read_result(path)
                value = self._view_cache[path]
                if value.get('selected_inputs',{}).get('candidate') != candidate:continue
                if value.get('method',{}).get('settings') != self.result.get('method',{}).get('settings'):continue
                if self.result.get('cross_validation') and value.get('cross_validation') != self.result['cross_validation']:continue
                scope = value.get('calibration',{}).get('scope')
                scope = 'shared' if scope in ('shared','series') else scope
                if scope not in ('individual','shared') or scope in values:continue
                values[scope] = value
                identities.append((scope,path,hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()))
            if not values:
                self.mass_comparison_plot.setText('No individual or shared calibration is saved for this phase candidate.');return
            root = Path(self.phase_catalog['results_root']) if self.phase_catalog else self.result_path.parent
            key = hashlib.sha256(json.dumps(sorted(identities)).encode()).hexdigest()[:20]
            target = root/'comparisons'/('mass-phases-'+key)/'mass_phase_comparison.png'
            if not target.is_file():
                target.parent.mkdir(parents=True,exist_ok=True)
                render_mass_phase_comparison(values,target)
            self.mass_comparison_plot.set_source_pixmap(QPixmap(str(target)))
            self.mass_comparison_plot.setToolTip(str(target))
        except (OSError,ValueError,KeyError) as error:
            self.mass_comparison_plot.setText('Mass comparison unavailable: '+str(error))

    def _saved_figure_path(self,path,base=None):
        """A saved file's path, relative to the result file ``base`` (by default the shown one)."""
        if not path:return None
        path=Path(path)
        return str(path if path.is_absolute() else Path(base or self.result_path).parent/path)

    def _scale_cache_path(self, row, mode, base=None):
        """The saved fields of ``row`` and the kept image of their ``mode`` scale; (None, None) without fields.

        ``base`` is the result file the paths are relative to (by default the shown one).
        """
        fields_path = self._saved_figure_path(row.get('fields'), base)
        if not fields_path or not Path(fields_path).is_file():
            return None, None
        cache_key = hashlib.sha256(f"{fields_path}|{mode}|scale-v1".encode()).hexdigest()[:24]
        return fields_path, Path(local_cache_root()) / 'phase_mass_display' / f'{cache_key}.png'

    def _start_scale_render(self, index, row, mode):
        """Draw ``row``'s alternate scale off the GUI thread; the frame is shown again when it is ready."""
        from darsia.fluidflower.ui_metadata_preview import MetadataPreview
        reader = self.__dict__.get('_scale_reader')
        if reader is None:
            reader = self._scale_reader = MetadataPreview(self)
            reader.completed.connect(self._scale_rendered)
        elif reader.busy and self.__dict__.get('_scale_rendering') == (index, mode):
            return
        self._scale_rendering = (index, mode)
        base = self.result_path

        def render(_cancelled):
            from darsia.fluidflower.production_results_view import _RENDER_LOCK
            with _RENDER_LOCK:  # One matplotlib figure is drawn at a time.
                return index, mode, self._scaled_concentration_preview(row, mode, base)

        reader.start(render)

    def _scale_rendered(self, value, error):
        """Show the frame again with its drawn scale, if it is still the one shown."""
        rendering = self.__dict__.pop('_scale_rendering', None)
        if error:
            self._scale_errors[rendering] = error
        if rendering == (self.slider.value(), self.color_scale.currentData()):
            self.show_frame(self.slider.value())

    def _scaled_concentration_preview(self, row, mode, base=None):
        """Render a display-only scale from saved numerical fields, without recalculating mass.

        No widget is touched, so this can run off the GUI thread; ``base`` is
        the result file its saved paths are relative to.
        """
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import PowerNorm

        fields_path, cache_path = self._scale_cache_path(row, mode, base)
        if fields_path is None:
            return None
        if cache_path.is_file():
            return str(cache_path)
        with np.load(fields_path, allow_pickle=False) as archive:
            names = set(archive.files)
            concentration = np.asarray(archive['concentration_g_l'], dtype=np.float32)
            if 'photograph_rgb' in names:
                rgb = np.asarray(archive['photograph_rgb'], dtype=np.float32)
            elif 'rgb' in names:
                rgb = np.asarray(archive['rgb'], dtype=np.float32)
            else:
                photo_path = self._saved_figure_path(row.get('photograph_arrays'), base)
                if not photo_path or not Path(photo_path).is_file():
                    return None
                with np.load(photo_path, allow_pickle=False) as photograph:
                    rgb = np.asarray(photograph['rgb'], dtype=np.float32)
            support = None
            for name in ('aqueous_volume_m3', 'av', 'aqueous_fraction'):
                if name in names:
                    support = np.asarray(archive[name]) > 0
                    break
            if support is None:
                support = concentration > 0
            if 'unquantified_phase_mask' in names:
                support &= ~np.asarray(archive['unquantified_phase_mask'], dtype=bool)

        finite = np.isfinite(concentration)
        support &= finite
        limit = row.get('color_limit_g_l')
        try:
            limit = float(limit) if limit is not None else float('nan')
        except (TypeError, ValueError):
            limit = float('nan')
        if not np.isfinite(limit) or limit <= 0:
            values = concentration[support & (concentration > 0)]
            limit = float(np.percentile(values, 99.5)) if values.size else 1.0
        if mode == 'frame_linear':
            values = concentration[support & (concentration > 0)]
            vmax = float(np.percentile(values, 99.5)) if values.size else limit
            vmax = max(vmax, 1e-6)
            norm = None
            scale_title = f'Auto linear per image · 0–{vmax:.3g} g/L'
        else:
            vmax = limit
            norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax) if mode == 'power' else None
            scale_title = f'Nonlinear γ=0.5 · 0–{vmax:.3g} g/L' if norm else f'Fixed linear · 0–{vmax:.3g} g/L'
        step = max(1, int(np.ceil(rgb.shape[1] / 1200)))
        displayed = np.ma.masked_where(~support[::step, ::step], concentration[::step, ::step])
        fig = Figure(figsize=(12, 5), layout='constrained')
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        axes[0].imshow(rgb[::step, ::step]); axes[0].set_title('Corrected photograph')
        axes[1].imshow(rgb[::step, ::step], alpha=.28)
        mapped = axes[1].imshow(displayed, cmap='viridis', vmin=None if norm else 0,
                                vmax=None if norm else vmax, norm=norm)
        axes[1].set_title('Estimated dissolved CO₂ · ' + scale_title)
        fig.colorbar(mapped, ax=axes[1], label='Concentration (g/L)')
        for axis in axes:
            axis.set_axis_off()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cache_path, dpi=110)
        return str(cache_path)

    def preview_changed(self, *_):
        if self.preview_field.currentData():
            self._preferred_preview_field=self.preview_field.currentData()
        self.show_frame(self.slider.value())

    def _show_field_preview(self,row):
        choices=[('Concentration','preview'),('Gas saturation','gas_saturation_preview'),
                 ('Local dilution','local_dilution_preview')]
        self.preview_field.blockSignals(True);self.preview_field.clear()
        for title,key in choices:
            if key=='preview' or row.get(key):self.preview_field.addItem(title,key)
        selected=self.preview_field.findData(self._preferred_preview_field)
        self.preview_field.setCurrentIndex(selected if selected>=0 else 0)
        self.preview_field.setEnabled(self.preview_field.count()>1);self.preview_field.blockSignals(False)
        key=self.preview_field.currentData();path=self._saved_figure_path(row.get(key))
        scale_mode = self.color_scale.currentData()
        scale_note = ''
        if key == 'preview' and scale_mode != 'fixed_linear':
            # A kept scale is shown at once; another is drawn in the background
            # while the fixed scale is shown.
            index = self.slider.value()
            fields_path, cache_path = self._scale_cache_path(row, scale_mode)
            error = self._scale_errors.get((index, scale_mode))
            if fields_path is None:
                scale_note = ' · alternate scale unavailable for this saved result'
            elif cache_path.is_file():
                path = str(cache_path)
            elif error:
                scale_note = f' · alternate scale unavailable: {error}'
            else:
                self._start_scale_render(index, row, scale_mode)
                scale_note = ' · drawing the alternate scale…'
        self.views.setTabText(self.photo_index,'Photograph + '+self.preview_field.currentText().lower())
        # Reuse the existing display scale when switching fields or matching
        # photographs between saved models; never load dense arrays.
        pixmap=QPixmap(str(path)) if path else QPixmap()
        self._active_preview_path = path
        self.photo.setToolTip((str(path or '') + scale_note) if path else scale_note)
        if pixmap.isNull():
            self.photo.set_source_pixmap(QPixmap())
            self.photo.setText('Saved field preview is unavailable.')
        else:
            self.photo.set_source_pixmap(pixmap)
        return ('Requested field is not saved for this image; showing concentration.' if selected<0 else '')

    def show_frame(self,index):
        if self.result is None or not 0<=index<len(self.result['frames']):return
        row=self.result['frames'][index]
        field_note=self._show_field_preview(row)
        fit=self.result['calibration']
        injected=row.get('injected_mass_g')
        reference=f"Injected {injected:.4f} g · Difference {row['total_mass_g']-injected:+.4f} g" if injected is not None else 'Injected mass unavailable'
        qualifier = 'PARTIAL: uncalibrated phase areas remain · ' if not row.get('quantification_complete',True) else ''
        self.image_position.setText(f"Image {index+1}/{len(self.result['frames'])} · {row['elapsed_h']:.3f} h")
        self.image_position.setToolTip(str(row.get('source_name','')))
        self.frame_summary.setText(f"{qualifier}{reference} · Detected {row['total_mass_g']:.4f} g "
            f"(dissolved {row['aqueous_mass_g']:.4f} g + free gas {row['gas_mass_g']:.4f} g)"
            + (f" · {field_note}" if field_note else ''))
        bounds = (f"Mass interval: {row['total_mass_lower_g']:.4f}–{row['total_mass_upper_g']:.4f} g\n"
                  if row.get('unknown_mass_upper_g',0)>0 else '')
        mean = row.get('mean_aqueous_concentration_g_l')
        dilution = (f"Aqueous support area: {row['aqueous_support_area_m2']:.4f} m² · Mean concentration: {mean:.4f} g/L\n"
                    if mean is not None and row.get('aqueous_support_area_m2') is not None else '')
        self.details.setPlainText((f"{field_note}\n" if field_note else '') +
            f"{index+1}/{len(self.result['frames'])} · {row['elapsed_h']:.3f} h · {row.get('calibration_role','')}\n"
            f"{bounds}{dilution}"
            f"{result_variant_label(self.result,self.result_path)}\n"
            f"Parameters: {json.dumps(fit['parameters'])}\nGas reference: {json.dumps(row.get('gas_reference') or row.get('gas_constraint',{}))}\n"
            f"{fit.get('validation_scope','')}\n{fit.get('fitting_scope_description','')}\n"
            f"{fit.get('parameter_identifiability_warning') or ''}\n" +
            '\n'.join([*fit.get('warnings',[]),*self.result.get('limitations',[])]))
        self.update_comparison(index)
        if self._fullscreen_refresh is not None:
            self._fullscreen_refresh()

    def update_comparison(self,index):
        candidate=self.result['selected_inputs']['candidate']
        entries=[e for e in self.visible if e['run']==candidate['run']]
        self.comparison.setRowCount(len(entries))
        for table_row,entry in enumerate(entries):
            path=str(Path(entry['path']).resolve())
            try:
                if path not in self._view_cache:
                    if entry.get('receipt_path'):
                        verify_saved_series_entry(entry)
                    self._view_cache[path]=backend.read_result(path)
                value=self._view_cache[path]
                match=matching_frame_index(self.result,index,value)
                if match is None:
                    cells=[entry.get('label','Saved model')]+['Image unavailable']*5
                else:
                    row=value['frames'][match];injected=row.get('injected_mass_g')
                    cells=[entry.get('label','Saved model'),f'{injected:.4f}' if injected is not None else 'Unavailable',
                        f"{row['aqueous_mass_g']:.4f}",f"{row['gas_mass_g']:.4f}",f"{row['total_mass_g']:.4f}",
                        f"{row['total_mass_g']-injected:+.4f}" if injected is not None else 'Unavailable']
            except (OSError,ValueError,KeyError):cells=[entry.get('label','Saved model')]+['Result unavailable']*5
            for col,text in enumerate(cells):
                item=QTableWidgetItem(text);item.setToolTip(text);self.comparison.setItem(table_row,col,item)
        self.comparison.resizeColumnsToContents();self.comparison.setColumnWidth(0,470)

    def _current_figure_path(self):
        if self.result is None:return None
        index=self.views.currentIndex()
        if index==self.photo_index:
            return self._active_preview_path or self._saved_figure_path(
                self.result['frames'][self.slider.value()].get(self.preview_field.currentData() or 'preview'))
        if index==self.mass_comparison_index:
            return self.mass_comparison_plot.toolTip() or None
        key=self._figure_keys.get(index)
        return self._saved_figure_path(self.result.get(key)) if key else None

    def export(self,kind):
        if self.process is not None and self.process.state()!=QProcess.NotRunning:
            self.status.setText('Finish or stop the current operation before exporting.');return
        if self.result is None:return
        if kind=='bundle':
            parent=QFileDialog.getExistingDirectory(self,'Choose parent folder for a new export bundle')
            if not parent:return
            destination=Path(parent)/('mass-export-'+uuid.uuid4().hex[:8])
        else:
            suffix={'movie':'mp4','csv':'csv','figure':'png'}[kind]
            name,_=QFileDialog.getSaveFileName(self,'Export result','result.'+suffix,'Result (*.'+suffix+')')
            if not name:return
            destination=Path(name)
        if destination.exists():self.status.setText('Choose a new filename; existing exports are preserved.');return
        if kind=='figure':
            import shutil
            source=self._current_figure_path()
            if not source:self.status.setText('Choose an available saved field, mass or diagnostics figure to export.');return
            shutil.copy2(source,destination);self.status.setText('Figure exported.');return
        self.job=self.cache_root/uuid.uuid4().hex;self.job.mkdir(parents=True)
        self.launch(['--result',str(self.result_path),'--destination',str(destination),'--export',kind],'export')

    def closeEvent(self,event):
        if self.lookup is not None and self.lookup.isRunning():
            self.status.setText('Waiting for the input lookup to finish before closing.');event.ignore();return
        self.stop();event.accept()
