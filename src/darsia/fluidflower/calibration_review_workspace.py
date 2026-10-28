"""Calibration-review presentation on the original saved-result workspace."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .calibration_review_panels import CalibrationReviewPanels
from .tools.phase_studio_series_ui import SeriesMassWorkspace


def calibration_label(entry):
    name = entry.get("review_calibration_name") or "Unnamed saved calibration"
    chemistry = entry.get("review_chemistry") or "Chemistry not recorded"
    return f"{name} · {chemistry}"


class _LocalReviewRead(QThread):
    ready = Signal(object, object, str)

    def __init__(self, request, parent):
        super().__init__(parent)
        self.request = request

    def run(self):
        context, bound, dataset, name = self.request
        try:
            from .calibration_review_gas import local_reviews
            rows, error = local_reviews(bound, dataset=dataset), ""
        except Exception as problem:
            rows, error = [], str(problem)
        if not self.isInterruptionRequested():
            self.ready.emit(self.request, rows, error)


def approval_blocker(result, result_path, run, series_root, calibration, *, series_id, native_review=None):
    """Never approve the series reference while viewing a different calibration."""
    from .calibration_approvals import calibration_identity

    if native_review is not None:
        return "This native result has not been verified against the series approval record; approval is unavailable."
    if not result or not result_path:
        return "Open a saved result bound to the series' approved calibration before approving this run."
    candidate = result.get("selected_inputs", {}).get("candidate", {})
    if str(candidate.get("dataset", "")).casefold().replace("_", "") != str(series_id).casefold().replace("_", ""):
        return "The displayed result belongs to another experimental series."
    if str(candidate.get("run", "")).casefold() != str(run).casefold():
        return "The displayed result belongs to another run."
    reference = result.get("series_calibration")
    if not isinstance(reference, str) or not reference:
        return "The displayed result has no explicit saved calibration reference; run approval is unavailable."
    # These are the existing binding fields read by mass_calibration_display.
    # A path hashed today cannot establish which bytes produced an old result.
    digests = [mapping.get(key) for mapping, key in (
        (result.get("bindings") or {}, "calibration_sha256"),
        (result.get("scalar_result") or {}, "calibration_sha256"),
        (result.get("calibration") or {}, "sha256"), (result, "calibration_sha256"))
        if isinstance(mapping, dict) and mapping.get(key)]
    if not digests:
        return "The saved result does not record its calibration checksum; run approval is unavailable."
    if any(digest != calibration["sha256"] for digest in digests):
        return "The displayed result's recorded calibration checksum differs from the series' approved calibration."
    path = Path(reference)
    if not path.is_absolute():
        path = Path(result_path).resolve().parent / path
    try:
        displayed = calibration_identity(series_root, path)
    except (OSError, ValueError) as error:
        return "The displayed calibration could not be verified: " + str(error)
    if displayed["sha256"] != calibration["sha256"]:
        return "The displayed calibration differs from the series' approved calibration. Select its saved result to approve this run."
    return ""


class CalibrationReviewWorkspace(SeriesMassWorkspace):
    """Change presentation only; retain original discovery, result and frame readers."""

    review_selection_changed = Signal()

    def __init__(self, *args, review_series_root=None, **kwargs):
        self.review_mode = False
        self._review_closed = False
        self._review_calibration = None
        self._review_dataset = None
        self._native_entries = {}
        self._native_readers = {}
        self._native_registrations = {}
        self._native_source_registrations = {}
        self._native_extra_collections = {}
        self._native_review = None
        self._local_review = None
        self._local_readers, self._local_entries = {}, {}
        self._local_loaded, self._local_pending = set(), []
        self._local_lookup = None
        self._native_series_root = Path(review_series_root) if review_series_root is not None else None
        self._native_catalog_loaded = set()
        self._native_catalog_paths = {}
        self.visible = []
        super().__init__(*args, **kwargs)
        self.review_header = QWidget()
        header = QVBoxLayout(self.review_header)
        header.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        header.addLayout(row)
        row.addWidget(QLabel("Mass calibration"))
        self.mass_calibration = QComboBox()
        self.mass_calibration.setMinimumContentsLength(24)
        self.mass_calibration.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mass_calibration.currentIndexChanged.connect(self._calibration_changed)
        row.addWidget(self.mass_calibration, 1)
        self.refresh_saved = QPushButton("Refresh saved results")
        self.refresh_saved.clicked.connect(self._refresh_saved_results)
        row.addWidget(self.refresh_saved)
        self.open_collection = QPushButton("Open collected products…")
        self.open_collection.setToolTip("Open an existing collection receipt for this result; no files are copied.")
        self.open_collection.clicked.connect(self._choose_native_collection)
        self.open_collection.setEnabled(False)
        row.addWidget(self.open_collection)
        self.open_local_calibration = QPushButton("Open saved calibration result…")
        self.open_local_calibration.setToolTip("Open an existing frozen local calibration and its saved gas-contribution records.")
        self.open_local_calibration.clicked.connect(self._choose_local_calibration)
        row.addWidget(self.open_local_calibration)
        self.review_note = QLabel("Choose the saved calibration to review, then a run and image. "
            "Photograph corrections affect only the photograph panel. "
            "Each map names its saved quantity and units; mass maps use an explicit fixed scale.")
        self.review_note.setWordWrap(True)
        header.addWidget(self.review_note)
        self.native_notice = QLabel()
        self.native_notice.setWordWrap(True)
        self.native_notice.setTextFormat(Qt.TextFormat.PlainText)
        header.addWidget(self.native_notice)
        from .calibration_review_native_catalog_ui import NativeReviewCatalogue
        self.native_catalog = NativeReviewCatalogue(self)
        self.native_catalog.ready.connect(self._native_catalog_ready)
        from .calibration_review_native_collection_ui import NativeCollectionLookup
        self.native_collections = NativeCollectionLookup(self)
        self.native_collections.ready.connect(self._native_collection_ready)
        self.layout().insertWidget(0, self.review_header)
        self.review_header.hide()
        # Group the retained selection rows so Results/export can restore them
        # intact, with the same widgets and callbacks.
        self._legacy_review_rows = []
        for anchor in (self.series, self.view_experiment, self.illumination_filter, self.preview_field):
            for index in range(self.layout().count()):
                item = self.layout().itemAt(index)
                child = item.layout()
                if child is not None and child.indexOf(anchor) >= 0:
                    self.layout().takeAt(index)
                    wrapper = QWidget()
                    wrapper.setLayout(child)
                    self.layout().insertWidget(index, wrapper)
                    self._legacy_review_rows.append(wrapper)
                    break
        self.review_panels = CalibrationReviewPanels(self)
        self.review_panels.layout_choice.setCurrentIndex(
            self.review_panels.layout_choice.findData("side_by_side"))
        self.layout().insertWidget(self.layout().indexOf(self.views), self.review_panels, 1)
        self.review_panels.hide()
        from .calibration_review_playback import CalibrationReviewPlayback
        self.playback = CalibrationReviewPlayback(self.slider, self.review_panels, self)
        self.layout().insertWidget(self.layout().indexOf(self.frame_summary), self.playback)
        self.playback.hide()

    def _choose_local_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open saved calibration result", "", "JSON (*.json)")
        if path:
            self.attach_local_calibration(path)

    def attach_local_calibration(self, frozen, *, name="Saved local calibration"):
        """Use the exact completed refit snapshot or explicit user selection."""
        if self._review_closed:
            return
        dataset, context = str(self.series.currentData() or "").casefold(), self._native_context()
        identity = frozen.get("sha256", frozen.get("path")) if isinstance(frozen, dict) else str(frozen)
        key = (context, identity)
        if key in self._local_loaded or any(request[0] == key for request in self._local_pending):
            return
        if self._local_lookup is not None and self._local_lookup.request[0] == key:
            return
        self._local_pending.append((key, deepcopy(frozen), dataset, str(name)))
        self._start_local_lookup()

    def _start_local_lookup(self):
        if self._local_lookup is not None or not self._local_pending or self._review_closed:
            return
        worker = _LocalReviewRead(self._local_pending.pop(0), self)
        self._local_lookup = worker
        worker.ready.connect(self._local_ready)
        worker.finished.connect(lambda: self._local_finished(worker))
        worker.start()

    def _local_ready(self, request, readers, error):
        key, _bound, dataset, name = request
        if self._review_closed or key[0] != self._native_context() or dataset != str(self.series.currentData() or "").casefold():
            return
        if error:
            self.native_notice.setText("Local gas contribution review unavailable: " + error)
            return
        self._local_loaded.add(key)
        for reader in readers:
            path, run = str(Path(reader.frozen["path"]).resolve()), reader.run
            entry_key = (key[0], path, run)
            self._local_readers[(path, run)] = reader
            self._local_entries[entry_key] = dict(path=path, dataset=dataset, run=run,
                model="Local calibration", created_utc=reader.document.get("frozen_utc", ""), method={},
                frames=len(reader["frames"]), review_context=key[0],
                review_calibration_id="sha256:" + reader.frozen["sha256"],
                review_calibration_sha256=reader.frozen["sha256"], review_calibration_name=name,
                review_chemistry=reader.document.get("chemistry_model", "Chemistry not recorded"),
                calibration_scope=reader.document.get("calibration_scope", "shared"), label=name)
        if readers:
            self.mass_calibration.blockSignals(True)
            self.mass_calibration.setCurrentIndex(-1)
            self.mass_calibration.blockSignals(False)
            self._review_calibration = "sha256:" + readers[0].frozen["sha256"]
        self.refresh(keep_view=True)

    def _local_finished(self, worker):
        if self._local_lookup is worker:
            self._local_lookup = None
        worker.deleteLater()
        self._start_local_lookup()

    def set_review_mode(self, active):
        self.playback.pause()
        self.playback.setVisible(active)
        self.review_mode = bool(active)
        self.review_header.setVisible(active)
        for widget in self._legacy_review_rows:
            widget.setVisible(not active)
        self.views.setVisible(not active)
        if active:
            self.refresh()
        else:
            self._native_review = None
            self._local_review = None
            super().refresh()
        dataset = str(self.series.currentData() or "").casefold()
        if dataset and dataset not in self._series_catalog_loaded:
            self._series_changed()
        self.review_panels.set_active(active)
        self._sync_review_panels()
        if active:
            self._start_native_lookup()

    def register_native_review(self, reader, *, dataset, calibration_name, registration=None,
                               source_registrations=(), refresh=True):
        """Attach an explicitly admitted index; no discovery or series inference.

        The caller supplies its existing job/collection bindings to
        NativeSavedReview before calling this on the UI thread. The dataset is
        a display association, not evidence permitting a series approval.
        """
        from .calibration_review_native import NativeSavedReview
        if not isinstance(reader, NativeSavedReview) or not str(calibration_name).strip():
            raise ValueError("An admitted native reader and explicit calibration name are required")
        path = str(Path(reader.index["path"]).resolve())
        entry = dict(path=path, dataset=str(dataset).casefold(), run=reader.expected_run,
            model="Saved native result", created_utc="", method={}, frames=len(reader.case_ids),
            review_context=self._native_context(str(dataset).casefold()),
            review_calibration_id="sha256:" + reader.expected_calibration_sha256,
            review_calibration_name=str(calibration_name).strip(), review_chemistry="Chemistry not recorded",
            calibration_scope="unknown", label=str(calibration_name).strip())
        self._native_entries[path], self._native_readers[path] = entry, reader
        self._native_registrations[path] = deepcopy(registration)
        self._native_source_registrations[path] = deepcopy(tuple(source_registrations))
        if refresh:
            self.refresh(keep_view=True)

    def native_registration(self):
        if self._native_review is None or self.result_path is None:
            return None
        return deepcopy(self._native_registrations.get(str(self.result_path.resolve())))

    def native_source_registrations(self):
        if self._native_review is None or self.result_path is None:
            return ()
        return deepcopy(self._native_source_registrations.get(str(self.result_path.resolve()), ()))

    def _choose_native_collection(self):
        reader = self._native_review
        if self._review_closed or reader is None or not self.review_mode or self.native_collections.busy():
            return
        context = (self._native_context(), reader)
        path, _filter = QFileDialog.getOpenFileName(self, "Open existing collected products", "",
            "Collection or publication receipt (*.json)")
        if (self._review_closed or not path or not self.review_mode
                or context != (self._native_context(), self._native_review)):
            return
        self.open_collection.setEnabled(False)
        self.native_notice.setText("Checking the selected collection against this saved result…")
        self.native_collections.request(context, reader, path)

    def _native_collection_ready(self, context, reader, error):
        if self._review_closed:
            return
        if context != (self._native_context(), self._native_review) or not self.review_mode:
            self.open_collection.setEnabled(self._native_review is not None and not self.native_collections.busy())
            return
        self.open_collection.setEnabled(True)
        if error:
            self.native_notice.setText("Collected products unavailable: " + error)
            return
        previous = context[1]
        path = str(Path(previous.index["path"]).resolve())
        entry = self._native_entries[path]
        self._native_extra_collections.setdefault(context[0], {})[reader.index["sha256"]] = reader.collections
        # Supersede any catalogue read that started before this explicit
        # binding; its old attachment must not replace the newly opened copy.
        self._start_native_lookup(force=True)
        self.register_native_review(reader, dataset=entry["dataset"],
            calibration_name=entry["review_calibration_name"],
            registration=self._native_registrations.get(path),
            source_registrations=self._native_source_registrations.get(path, ()))
        self.native_notice.setText("Existing collected products are open for this saved result.")

    def set_native_series_root(self, root):
        root = Path(root).resolve()
        if self._native_series_root is not None and root == self._native_series_root.resolve():
            return
        self._native_series_root = root
        self.native_notice.clear()
        if self._native_review is not None:
            self.clear_view()
        if self.review_mode:
            self.refresh(keep_view=True)
            self._start_native_lookup(force=True)

    def _native_context(self, dataset=None):
        dataset = str(self.series.currentData()) if dataset is None else str(dataset)
        return (dataset, str(self._native_series_root.resolve())) if self._native_series_root else None

    def _start_native_lookup(self, *, force=False):
        if self._review_closed:
            return
        context = self._native_context()
        if (context is not None and (force or context not in self._native_catalog_loaded)
                and (force or not self.native_catalog.is_loading(context))):
            self.native_catalog.refresh(*context,
                extra_collections=self._native_extra_collections.get(context, {}))

    def _series_changed(self, *_):
        if self._review_closed:
            return
        super()._series_changed()
        if self.review_mode and hasattr(self, "native_catalog"):
            self._start_native_lookup()

    @Slot(str, object, str)
    def _series_results_ready(self, dataset, entries, error):
        if self._review_closed:
            return
        super()._series_results_ready(dataset, entries, error)

    def load_model_results(self):
        # Retain the original reader and launch behavior, with owned completion
        # slots so a queued finished signal cannot change a closed review.
        if self._review_closed or self.lookup is not None and self.lookup.isRunning():
            return
        from .tools.phase_studio_series_ui import ModelMassLookup
        try:
            model = self.selected_model() if self.selected_model else None
            if not model:
                raise ValueError('Select a phase model in Review and label evidence first')
            self.import_model.setEnabled(False)
            self.lookup = ModelMassLookup(Path(model['root']), self)
            self.lookup.ready.connect(self.model_results_ready, Qt.ConnectionType.QueuedConnection)
            self.lookup.finished.connect(self._model_lookup_finished, Qt.ConnectionType.QueuedConnection)
            self.status.setText('Loading registered phase images and saved concentration models…')
            self.lookup.start()
        except Exception as error:
            self.status.setText(str(error))

    @Slot(object, str)
    def model_results_ready(self, catalog, error):
        if self._review_closed:
            return
        super().model_results_ready(catalog, error)

    @Slot()
    def _model_lookup_finished(self):
        if not self._review_closed:
            self.import_model.setEnabled(True)

    def _native_catalog_ready(self, context, rows, errors):
        if self._review_closed or context != self._native_context():
            return
        for path in self._native_catalog_paths.get(context, ()):
            if self._native_entries.get(path, {}).get("review_context") == context:
                self._native_entries.pop(path, None)
                self._native_readers.pop(path, None)
                self._native_registrations.pop(path, None)
                self._native_source_registrations.pop(path, None)
        paths = []
        for row in rows:
            self.register_native_review(**row, refresh=False)
            paths.append(str(Path(row["reader"].index["path"]).resolve()))
        self._native_catalog_paths[context] = paths
        if not errors:
            self._native_catalog_loaded.add(context)
        self.native_notice.setText("Native saved results unavailable: " + "; ".join(errors) if errors else "")
        self.refresh(keep_view=True)

    def set_overflow_comparison_mode(self, mode):
        """Follow the Mass calibration page's existing display-only choice."""
        if mode not in ("off", "compare"):
            raise ValueError("The overflow comparison display must be Off or Compare only")
        self.views.setTabVisible(self.views.indexOf(self.overflow_plot), mode == "compare")
        self.review_panels.set_overflow_enabled(mode == "compare")

    def show_saved_overflow_comparison(self):
        """Open the saved comparison in the currently visible presentation."""
        if self.review_mode:
            panel = self.review_panels.panels[0]
            panel.content.setCurrentIndex(panel.content.findData("overflow_curve"))
            self.review_panels.refresh()
        else:
            self.views.setCurrentWidget(self.overflow_plot)

    def _entry_calibration(self, entry):
        # Older/imported metadata has a distinct result identity until its
        # original reader supplies an explicit common calibration reference.
        base = entry.get("review_calibration_id") or str(Path(entry["path"]).resolve())
        if entry.get("review_calibration_binding_conflict"):
            return "Conflicting saved calibration checksums: " + str(Path(entry["path"]).resolve())
        related = [other for other in self.visible if other.get("review_calibration_id") == base]
        digests = {other.get("review_calibration_sha256") for other in related}
        if len(digests) > 1:
            return base + " · " + (entry.get("review_calibration_sha256") or "Unverified: " + entry["path"])
        return base

    def _refresh_calibrations(self):
        previous = self.mass_calibration.currentData() or self._review_calibration
        selected_path = str(self.result_path) if self.result_path else None
        dataset = (self.series.currentData(), self._native_context())
        if self._review_dataset is not None and dataset != self._review_dataset:
            previous, selected_path = None, None
        self._review_dataset = dataset
        groups = {}
        for entry in getattr(self, "visible", []):
            key = self._entry_calibration(entry)
            groups.setdefault(key, entry)
            if previous is None and str(entry["path"]) == selected_path:
                previous = key
        if previous not in groups:
            previous = next((self._entry_calibration(entry) for entry in getattr(self, "visible", [])
                             if str(entry["path"]) == selected_path), previous)
        self.mass_calibration.blockSignals(True)
        self.mass_calibration.clear()
        for key, entry in groups.items():
            label = calibration_label(entry)
            if sum(calibration_label(other) == label for other in groups.values()) > 1:
                label += " · " + key
            self.mass_calibration.addItem(label, key)
            self.mass_calibration.setItemData(self.mass_calibration.count() - 1, key, Qt.ToolTipRole)
        if previous is not None and previous not in groups:
            self.mass_calibration.addItem("Previously selected calibration · unavailable", previous)
            self.mass_calibration.setItemData(self.mass_calibration.count() - 1, previous, Qt.ToolTipRole)
        index = self.mass_calibration.findData(previous)
        self.mass_calibration.setCurrentIndex(index if index >= 0 else (0 if groups else -1))
        self.mass_calibration.blockSignals(False)
        self.mass_calibration.setEnabled(bool(groups))
        self._review_calibration = self.mass_calibration.currentData()

    def _calibration_changed(self, *_):
        self._review_calibration = self.mass_calibration.currentData()
        self.refresh_browser()

    def _refresh_saved_results(self):
        self._series_catalog_loaded.discard(str(self.series.currentData() or "").casefold())
        self._series_changed()
        self._start_native_lookup(force=True)

    def refresh(self, *, keep_view=False):
        if self.review_mode:
            super().refresh(keep_view=True)
            self.visible.extend(entry for entry in self._native_entries.values()
                                if entry["dataset"] == self.series.currentData()
                                and entry["review_context"] == self._native_context())
            self.visible.extend(entry for entry in self._local_entries.values()
                                if entry["dataset"] == str(self.series.currentData() or "").casefold()
                                and entry["review_context"] == self._native_context())
            self.refresh_browser()
        else:
            super().refresh(keep_view=keep_view)

    def refresh_browser(self):
        if not self.review_mode:
            return super().refresh_browser()
        self._refresh_calibrations()
        entries = [entry for entry in getattr(self, "visible", [])
                   if self._entry_calibration(entry) == self._review_calibration]
        selected = self.view_experiment.currentData()
        runs = sorted({entry["run"] for entry in entries})
        self.view_experiment.blockSignals(True)
        self.view_experiment.clear()
        for run in runs:
            self.view_experiment.addItem(run.upper(), run)
        if selected in runs:
            self.view_experiment.setCurrentIndex(self.view_experiment.findData(selected))
        self.view_experiment.blockSignals(False)
        self.experiment_changed()

    def _sorted_model_entries(self, entries, *, history=False):
        if not self.review_mode:
            return super()._sorted_model_entries(entries, history=history)
        # The review choice selects one saved calibration; legacy light,
        # colour, scope and overflow filters must never hide its mass result.
        if history:
            return []
        return sorted((entry for entry in entries if self._entry_calibration(entry) == self._review_calibration),
                      key=lambda entry: (entry.get("created_utc", ""), entry["path"]), reverse=True)

    def experiment_changed(self):
        if not self.review_mode:
            return super().experiment_changed()
        entries = self._sorted_model_entries([entry for entry in getattr(self, "visible", [])
                                               if entry["run"] == self.view_experiment.currentData()])
        previous = str(self.result_path) if self.result_path else None
        self._fill_model_menus(entries, previous)
        path = next((entry["path"] for entry in entries if entry["path"] == previous),
                    entries[0]["path"] if entries else None)
        if path:
            try:
                self.load_result(path)
            except Exception as error:
                self.clear_view()
                self.status.setText(str(error))
        else:
            self.clear_view()
        self.review_selection_changed.emit()

    def load_result(self, path):
        if hasattr(self, "playback"):
            self.playback.pause()
        native_path = str(Path(path).resolve())
        local = self._local_readers.get((native_path, self.view_experiment.currentData())) if self.review_mode else None
        if local is not None:
            self._load_local_review(local, path)
            return
        native = self._native_readers.get(native_path) if self.review_mode else None
        if native is not None:
            if self._native_entries[native_path]["review_context"] != self._native_context():
                raise ValueError("This native result belongs to another series root.")
            self._load_native_review(native, path)
            return
        self._native_review = None
        self._local_review = None
        super().load_result(path)
        if hasattr(self, "review_panels"):
            self._sync_review_panels()

    def show_frame(self, index):
        if self._local_review is not None:
            row = self._local_review["frames"][index]
            time = f" · {row['elapsed_h']:.3f} h" if isinstance(row.get("elapsed_h"), (int, float)) else " · Time not recorded"
            self.image_position.setText(f"Image {index+1}/{len(self._local_review['frames'])}" + time)
            self.image_position.setToolTip(row["case_id"])
            self.frame_summary.setText("Saved local calibration · fixed gas-contribution scales across images.")
            self.previous_image.setEnabled(index > 0)
            self.next_image.setEnabled(index+1 < len(self._local_review["frames"]))
            self._sync_review_panels()
            return
        if self._native_review is not None:
            metadata = self._native_review.frame_metadata(index)
            self.image_position.setText(f"Image {index + 1}/{len(self._native_review.case_ids)} · {metadata['elapsed_h']:.3f} h")
            self.image_position.setToolTip(self._native_review.case_ids[index])
            self.frame_summary.setText("Native saved mass products. Map scales are fixed display choices; unknown areas remain marked.")
            self.previous_image.setEnabled(index > 0)
            self.next_image.setEnabled(index + 1 < len(self._native_review.case_ids))
            self._sync_review_panels()
            return
        super().show_frame(index)
        if hasattr(self, "review_panels"):
            self._sync_review_panels()

    def _show_field_preview(self, row):
        if self.review_mode:
            return ""
        return super()._show_field_preview(row)

    def _load_saved_figures(self, value):
        if not self.review_mode:
            super()._load_saved_figures(value)

    def update_comparison(self, index):
        if not self.review_mode:
            super().update_comparison(index)

    def show_mass_comparison(self, *_):
        if not self.review_mode:
            super().show_mass_comparison()

    def clear_view(self):
        if hasattr(self, "playback"):
            self.playback.pause()
        self._native_review = None
        self._local_review = None
        if hasattr(self, "open_collection"):
            self.open_collection.setEnabled(False)
        super().clear_view()
        if hasattr(self, "review_panels"):
            self._sync_review_panels()

    def _sync_review_panels(self):
        if self._local_review is not None:
            self.review_panels.set_local_result(self._local_review, self.slider.value())
            return
        if self._native_review is not None:
            self.review_panels.set_native_result(self._native_review, self.slider.value())
            return
        self.review_panels.set_result(self.result, self.result_path, self.slider.value(),
                                      getattr(self, "visible", []))

    def _load_local_review(self, reader, path):
        previous = self._local_review
        previous_case = previous["frames"][self.slider.value()]["case_id"] if previous is not None else None
        index = next((i for i, row in enumerate(reader["frames"]) if row["case_id"] == previous_case), 0)
        self.clear_view()
        self._local_review, self.result, self.result_path = reader, reader, Path(path)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(reader["frames"])-1)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.slider.setEnabled(len(reader["frames"]) > 1)
        self.details.setPlainText("Frozen local calibration: " + reader.frozen["path"] +
            "\nCalibration SHA-256: " + reader.frozen["sha256"] +
            "\nSaved contribution preparation: " + reader.document["overflow_preparation"]["path"])
        self.status.setText("Saved gas interpretation and overflow contributions; no fitting or RAW decoding.")
        self.show_frame(index)
        self.review_selection_changed.emit()

    def _load_native_review(self, reader, path):
        previous = self._native_review
        previous_frame = previous.frame_metadata(self.slider.value()) if previous is not None else None
        index = 0
        if previous_frame is not None and previous.expected_run == reader.expected_run:
            index = next((position for position in range(len(reader.case_ids))
                if all(reader.frame_metadata(position).get(key) == previous_frame.get(key)
                       for key in ("source_binding", "elapsed_h", "frame_index"))), 0)
        self.clear_view()
        self._native_review, self.result_path = reader, Path(path)
        self.open_collection.setEnabled(not self.native_collections.busy())
        self.view_experiment.blockSignals(True)
        self.view_experiment.setCurrentIndex(self.view_experiment.findData(reader.expected_run))
        self.view_experiment.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(reader.case_ids) - 1)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.slider.setEnabled(len(reader.case_ids) > 1)
        self.details.setPlainText("Saved native index: " + reader.index["path"] +
            "\nCalibration SHA-256: " + reader.expected_calibration_sha256)
        self.status.setText("Native saved products are attached through their original index and collection bindings.")
        self.show_frame(index)

    def closeEvent(self, event):
        self.stop_review_reads()
        super().closeEvent(event)

    def stop_review_reads(self):
        self.playback.pause()
        self._review_closed = True
        self._local_pending.clear()
        # Queued ready/finished signals can outlive wait(). Invalidate their
        # callbacks first and detach all retained pending lookup ownership.
        readers = [reader for reader in (self.lookup, self.series_lookup, self._local_lookup) if reader is not None]
        self.lookup, self.series_lookup = None, None
        self._local_lookup = None
        self.review_panels.stop()
        self.native_catalog.stop()
        self.native_collections.stop()
        # These are the retained metadata readers, not calculation workers.
        for reader in readers:
            reader.requestInterruption()
        for reader in readers:
            reader.wait()
