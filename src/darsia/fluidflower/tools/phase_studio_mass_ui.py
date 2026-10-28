"""Nonblocking concentration/mass window for the current Studio candidate."""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSlider, QTextEdit, QVBoxLayout, QWidget, QTabWidget)


class PreparedResultLookup(QThread):
    ready = Signal(object, object, str)

    def __init__(self, request, method, parent):
        super().__init__(parent)
        self.request, self.method = request, method

    def run(self):
        from darsia.fluidflower.tools.phase_studio_mass_cache import find_result
        try:
            path = find_result(self.request, self.method)
            self.ready.emit(path, self.method, "")
        except (OSError, ValueError, KeyError) as error:
            self.ready.emit(None, self.method, str(error))


class StudioMassWindow(QMainWindow):
    def __init__(self, request, cache_root, parent=None):
        super().__init__(parent)
        from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
        install_interaction_safety()
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.request = json.loads(json.dumps(request, default=str))
        from darsia.fluidflower.tools.phase_studio_mass_cache import DEFAULT_MASS_POLICY
        self.request.setdefault("mass_policy", dict(DEFAULT_MASS_POLICY))
        self.cache_root = Path(cache_root)
        self.result = None; self.result_path = None
        self.lookup = None
        candidate = request["candidate"]
        self.setWindowTitle(f"Concentration and mass · {candidate['dataset'].upper()} · {candidate['run'].upper()}")
        self.resize(1360, 860)
        body = QWidget(); self.setCentralWidget(body); layout = QVBoxLayout(body)
        self.binding = QLabel(f"Selected phase map: {candidate['candidate_id']}\n"
            "Fixed Studio photographs and phase maps · registered baseline + clear-water reference")
        self.binding.setWordWrap(True); layout.addWidget(self.binding)
        form = QFormLayout(); layout.addLayout(form)
        self.optical = QComboBox()
        self.optical.addItem("Reference-normalized soft log (damped dark channels)", "reference_soft_log")
        self.optical.addItem("Reference-normalized log", "reference_log")
        self.restoration = QComboBox(); self.restoration.addItem("DarSIA scalar TVD", "darsia_tvd")
        self.restoration.addItem("No additional scalar restoration", "none")
        self.chemistry = QComboBox(); self.chemistry.addItem("Empirical power response · mass calibrated", "empirical_power")
        self.chemistry.addItem("DarSIA SimpleFlash · mass calibrated optical range", "darsia_simple_flash")
        for title, widget in (("Optical response", self.optical), ("Restoration", self.restoration), ("Concentration response", self.chemistry)):
            form.addRow(title, widget)
        notice = QLabel("Individual comparison: use the Series calibration tab for a shared model. "
            "No buffer/pH calibration is assumed. Injected mass fits one curve for the run; "
            "held-out times check its mass prediction. This does not independently validate local concentration. "
            "The dilution option is an empirical volume factor, not a transport solver. The mutually exclusive "
            "free-gas class is treated as gas-dominant (Sg 0.50–1.00); the unmeasured liquid concentration in "
            "those cells is retained as a reported mass interval.")
        notice.setWordWrap(True); layout.addWidget(notice)
        actions = QHBoxLayout(); layout.addLayout(actions)
        self.prepared = QPushButton("Load prepared results")
        self.prepared.clicked.connect(self.load_prepared); actions.addWidget(self.prepared)
        self.open_files = QPushButton("Open result files"); self.open_files.setEnabled(False)
        self.open_files.clicked.connect(self.open_result); actions.addWidget(self.open_files)
        self.status = QLabel("Ready to load an existing saved result.")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(115)
        layout.addWidget(self.details)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        self.image = QLabel("Concentration images will appear here."); self.image.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.image)
        self.tabs = QTabWidget(); self.tabs.addTab(scroll, "Concentration fields")
        self.mass_curve = QLabel("Mass curves will appear here."); self.mass_curve.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.mass_curve, "Mass versus injected reference"); layout.addWidget(self.tabs, 1)
        self.series_workspace = None
        self.series_workspace_tab = QWidget()
        self.series_workspace_layout = QVBoxLayout(self.series_workspace_tab)
        self.tabs.addTab(self.series_workspace_tab, "Series calibration, overflow and export")
        self.tabs.currentChanged.connect(self.open_series_workspace)
        playback = QHBoxLayout(); layout.addLayout(playback)
        self.play = QPushButton("Play"); self.play.setCheckable(True); self.play.setEnabled(False)
        self.play.setToolTip("One reviewed image every 1.5 seconds; acquisition times are shown in each image.")
        self.play.toggled.connect(self.toggle_playback); playback.addWidget(self.play)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.show_frame); playback.addWidget(self.slider, 1)
        self.playback_timer = QTimer(self); self.playback_timer.setInterval(1500)
        self.playback_timer.timeout.connect(self.advance_frame)
        for widget in (self.optical, self.restoration, self.chemistry):
            widget.currentIndexChanged.connect(self.method_changed)
        QTimer.singleShot(0, self.load_prepared)

    def method(self):
        return dict(optical=self.optical.currentData(), restoration=self.restoration.currentData(),
                    chemistry=self.chemistry.currentData())

    def open_series_workspace(self,index):
        if self.tabs.widget(index) is not self.series_workspace_tab:return
        if self.series_workspace is None:
            from darsia.fluidflower.tools.phase_studio_series_ui import SeriesMassWorkspace
            self.series_workspace=SeriesMassWorkspace(self.series_workspace_tab,cache_root=self.cache_root/'phase_mass_workspace')
            self.series_workspace_scroll = QScrollArea(self.series_workspace_tab)
            self.series_workspace_scroll.setWidgetResizable(True)
            self.series_workspace_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.series_workspace_scroll.setWidget(self.series_workspace)
            self.series_workspace_layout.addWidget(self.series_workspace_scroll)
            dataset=self.request['candidate']['dataset']
            self.series_workspace.series.setCurrentIndex(self.series_workspace.series.findData(dataset))
        if self.result_path is not None:self.series_workspace.import_paths(self.result_path)

    def toggle_playback(self, enabled):
        self.play.setText("Pause" if enabled else "Play")
        if enabled:
            if self.slider.value() == self.slider.maximum(): self.slider.setValue(0)
            self.playback_timer.start()
        else:
            self.playback_timer.stop()

    def advance_frame(self):
        if self.slider.value() >= self.slider.maximum():
            self.play.setChecked(False)
        else:
            self.slider.setValue(self.slider.value()+1)

    def method_changed(self):
        self.play.setChecked(False); self.play.setEnabled(False)
        self.result = None; self.result_path = None
        self.image.clear(); self.mass_curve.clear(); self.details.clear()
        self.open_files.setEnabled(False)
        self.load_prepared()

    def load_prepared(self):
        if self.lookup is not None and self.lookup.isRunning(): return
        self.prepared.setEnabled(False)
        self.status.setText("Checking prepared results for this exact phase model and method…")
        self.lookup = PreparedResultLookup(self.request, self.method(), self)
        self.lookup.ready.connect(self.prepared_ready)
        self.lookup.finished.connect(lambda: self.prepared.setEnabled(True))
        self.lookup.start()

    def prepared_ready(self, path, method, error):
        if method != self.method():
            QTimer.singleShot(100, self.load_prepared)
            return
        if error:
            self.status.setText("Prepared result could not be verified; inspect the saved files.")
            self.details.setPlainText(error)
        elif path is not None:
            self.load_result(path)
            self.status.setText("Prepared results loaded · no mass recalculation · Play or select an image below.")
        else:
            self.status.setText("No saved result for this exact selection and method. Historical calculation is unavailable.")






    def load_result(self, path):
        result = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if result.get("complete") is not True: raise ValueError("Incomplete concentration result")
        if any(result["selected_inputs"]["candidate"].get(key) != self.request["candidate"].get(key)
               for key in ("params_sha256", "candidate_id", "run", "dataset")):
            raise ValueError("Concentration result belongs to another phase model")
        if bool(result['selected_inputs'].get('local_baseline_search')) != bool(self.request.get('local_baseline_search')):
            raise ValueError("Concentration result uses a different local baseline search setting")
        if any(result['method']['settings'].get(k) != v for k,v in self.method().items()):
            raise ValueError("Concentration result uses another mass method")
        from darsia.fluidflower.tools.phase_studio_mass_cache import _selected_index
        if _selected_index(self.request, result) is None:
            raise ValueError("The selected photograph is not present in this mass result")
        self.result, self.result_path = result, Path(path)
        self.slider.setRange(0, len(result["frames"])-1)
        selected = int(self.request.get("frame_index", 0))
        if self.request.get("reference_mode"):
            reference = self.request.get("reference")
            matches = [i for i, row in enumerate(result["frames"]) if (
                row.get("source_name") == reference.get("source_name") and
                abs(float(row["elapsed_h"])-float(reference["elapsed_h"])) < 1e-7)
                ] if reference is not None else [i for i, row in enumerate(result["frames"]) if row["elapsed_h"] <= 0]
        else:
            matches = [i for i, row in enumerate(result["frames"]) if row.get("frame_index") == selected]
        self.slider.setValue(matches[0] if matches else min(selected, len(result["frames"])-1))
        if result.get("mass_curve"):
            self.mass_curve.setPixmap(QPixmap(result["mass_curve"]).scaled(1250, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.open_files.setEnabled(True); self.play.setEnabled(True); self.show_frame(self.slider.value())

    def show_frame(self, index):
        if self.result is None: return
        row = self.result["frames"][index]
        self.image.setPixmap(QPixmap(row["preview"]).scaled(1250, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        p = self.result["calibration"]["parameters"]
        self.details.setPlainText(f"Frame {index+1}/{len(self.result['frames'])} · {row['elapsed_h']:.4f} h · {row['calibration_role']}\n"
            f"Injected {row['injected_mass_g']:.4f} g | Dissolved {row['aqueous_mass_g']:.4f} g | "
            f"Gas {row['gas_mass_g']:.4f} g | Total {row['total_mass_g']:.4f} g\n"
            f"Partition uncertainty: gas {row['gas_mass_min_g']:.4f}–{row['gas_mass_max_g']:.4f} g | "
            f"total {row['total_mass_min_g']:.4f}–{row['total_mass_max_g']:.4f} g\n"
            f"Dilution factor {row['dilution_factor']:.4f} | Parameters: {json.dumps(p)}\n"
            f"Method: {json.dumps(self.result['method']['settings'])}\n"
            f"Gas policy: {json.dumps(self.result['mass_policy'])}")
        bounds = self.result['calibration'].get('parameters_at_bounds', [])
        if bounds:
            self.details.append('Fit reached parameter bounds: ' + ', '.join(bounds) + '. Local concentration remains underdetermined by total mass.')

    def open_result(self):
        if self.result_path: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_path.parent)))

    def closeEvent(self, event):
        # Closing the viewer does not interrupt numerical work or Studio saves.
        self.play.setChecked(False)
        if getattr(self, '_standalone', False):
            if self.lookup is not None: self.lookup.wait()
            event.accept(); return
        self.hide(); event.ignore()


def open_for_review(window):
    if window.current_pair_object is None: return
    from dataclasses import asdict
    candidate = window.current_pair_object.candidate("A")
    request = dict(review_root=str(window.current_review_root), candidate=asdict(candidate),
                   frame_index=int(window.frame_index), reference_mode=bool(window.reference_mode),
                   local_baseline_search=bool(window.local_reference_search.isChecked()),
                   reference=(window.injection_references[window.active_injection_reference]
                              if window.active_injection_reference is not None else None))
    dialog = StudioMassWindow(request, window.cache_root, window)
    from darsia.fluidflower.mass_saved_review import attach_saved_common_review
    attach_saved_common_review(dialog, candidate, int(window.frame_index))
    if not hasattr(window, "_mass_windows"): window._mass_windows = []
    window._mass_windows.append(dialog)
    dialog.show(); dialog.raise_()


if __name__ == '__main__':
    import argparse
    from PySide6.QtWidgets import QApplication
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    request = json.loads(args.request.read_text(encoding='utf-8-sig'))
    dialog = StudioMassWindow(request, args.request.parent)
    dialog._standalone = True
    dialog.show()
    sys.exit(app.exec())
