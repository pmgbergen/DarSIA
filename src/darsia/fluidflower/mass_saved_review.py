"""Explicit read-only Studio viewing of a saved common concentration layer.

The ordinary native workspace remains the default. The additional button reads
only bound receipts and pre-rendered files, in workers, never RAW arrays or a fit.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from .mass_saved_metadata import FrozenCalibrationEvidence
from .native.io import sha256_file as sha_file
from .phase_saved_review_metadata import configured_parameters

def _setting(name):
    return configured_parameters("mass_saved_review")[name]


def __getattr__(name):
    fields = {"CONTRACT": "contract", "ROUND2_CONTRACT": "round2_contract"}
    if name not in fields:
        raise AttributeError(name)
    return _setting(fields[name])


def _verify_registration_freeze(registration):
    frozen=registration["calibration_freeze"]
    if sha_file(frozen["path"])!=frozen["sha256"]:
        raise ValueError("Saved common concentration freeze changed")
    if registration.get("contract")==_setting('round2_contract'):
        from .mass_saved_metadata import FrozenEvidence
        evidence=FrozenEvidence.verify(frozen["path"])
        check_asset=registration["final_check"]
        if sha_file(check_asset["path"])!=check_asset["sha256"]:
            raise ValueError("Saved round-2 final check changed")
        check=json.loads(Path(check_asset["path"]).read_text(encoding="utf-8-sig"))
        if check.get("full_production_allowed") is not True or check.get("freeze",{}).get("sha256")!=evidence.sha256:
            raise ValueError("The round-2 final check did not permit full production")
        return evidence
    return FrozenCalibrationEvidence.verify(frozen["path"])


def _value(candidate, name, default=None):
    return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)


def series_root_for_candidate(candidate):
    summary = Path(_value(candidate, "summary_path")).resolve()
    for folder in summary.parents:
        if folder.name == "results":
            return folder.parent
    return None


def matching_registrations(candidate):
    root = series_root_for_candidate(candidate)
    if root is None or _value(candidate, "dataset") != _setting('dataset'):
        return []
    model_id = Path(_value(candidate, "summary_path")).parent.name
    model_sha = _value(candidate, "optical_model_sha256")
    matches = []
    for path in sorted((root / "results" / "concentration_layer_registrations").glob("*.json"), reverse=True):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if (value.get("contract") in {_setting('contract'),_setting('round2_contract')} and value.get("candidate_id") == model_id
                and value.get("candidate_sha256") == model_sha and value.get("complete") is True
                and value.get("default") is False):
            matches.append((path, value))
    return matches


@lru_cache(maxsize=8)
def _native_timeline(path, signature):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {int(row["frame_index"]): (row.get("source_sha256"), float(row["elapsed_h"])) for row in value["frame_sources"]}


def admit_registration(candidate, path, registration):
    model_id = Path(_value(candidate, "summary_path")).parent.name
    if (registration.get("contract") not in {_setting('contract'),_setting('round2_contract')} or registration.get("candidate_id") != model_id
            or registration.get("candidate_sha256") != _value(candidate, "optical_model_sha256")
            or registration.get("complete") is not True or registration.get("default") is not False):
        raise ValueError("Saved common concentration does not match the selected phase candidate")
    evidence = _verify_registration_freeze(registration)
    timeline_path = Path(_value(candidate, "per_frame_csv"))
    stamp = timeline_path.stat()
    timeline = _native_timeline(str(timeline_path), (stamp.st_size, stamp.st_mtime_ns))
    run = _value(candidate, "run")
    rows = sorted((row for row in registration["frames"] if row["run"] == run), key=lambda x: x["frame_index"])
    if len(rows) != len(timeline) or {row["frame_index"] for row in rows} != set(timeline):
        raise ValueError("Saved common concentration timeline is incomplete for this run")
    # Only small frame receipts are read here. Raw float arrays stay unopened.
    for row in rows:
        asset = row["receipt"]
        if sha_file(asset["path"]) != asset["sha256"]:
            raise ValueError("Saved common concentration frame receipt changed")
        value = json.loads(Path(asset["path"]).read_text(encoding="utf-8-sig"))
        index = int(value["frame_index"])
        source, elapsed = timeline[index]
        if (value.get("calibration_freeze_sha256") != evidence.sha256 or value.get("complete") is not True
                or value.get("run") != run or value.get("source_sha256") != source
                or value.get("elapsed_h") != elapsed or value.get("frame_index") != row["frame_index"]):
            raise ValueError("Saved common concentration belongs to another source/time/freeze")
        row["verified_receipt"] = value
    return rows


@lru_cache(maxsize=16)
def _verified_png(path, expected, size, mtime):
    data = Path(path).read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Saved common concentration display changed")
    return data


def render_bytes(row, layer):
    receipt = row["verified_receipt"]
    descriptor = receipt["assets"][layer + "_display"]
    path = Path(row["frame_folder"]) / descriptor["path"]
    stat = path.stat()
    return _verified_png(str(path), descriptor["sha256"], stat.st_size, stat.st_mtime_ns)


def attach_saved_common_review(dialog, candidate, initial_frame):
    """Add an opt-in action; nothing is looked up or selected automatically."""
    if _value(candidate, "dataset") != _setting('dataset'):
        return
    from PySide6.QtCore import QThread, Signal, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
        QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

    class Lookup(QThread):
        loaded = Signal(object, str)
        def run(self):
            try:
                matches = matching_registrations(candidate)
                self.loaded.emit([(str(path), record, admit_registration(candidate, path, record)) for path, record in matches], "")
            except Exception as error:
                self.loaded.emit([], str(error))

    class Picture(QThread):
        loaded = Signal(object, str, int)
        def __init__(self, row, layer, generation, parent):
            super().__init__(parent); self.row, self.layer, self.generation = row, layer, generation
        def run(self):
            try:
                self.loaded.emit(render_bytes(self.row, self.layer), "", self.generation)
            except Exception as error:
                self.loaded.emit(b"", str(error), self.generation)

    class Viewer(QMainWindow):
        def __init__(self, matches):
            super().__init__(dialog)
            self.setWindowTitle("Saved common concentration · " + str(_value(candidate, "run")).upper())
            self.resize(1250, 780); self.matches, self.jobs, self.generation = matches, [], 0
            body = QWidget(); self.setCentralWidget(body); layout = QVBoxLayout(body)
            self.notice = QLabel("Read-only common frozen calibration · saved three-phase maps · no map rescaling. "
                "Main mass excludes the bottom strip and pixels filled below the photograph. Grey means unquantified.")
            self.notice.setWordWrap(True); layout.addWidget(self.notice)
            controls = QHBoxLayout(); layout.addLayout(controls)
            self.package = QComboBox()
            for path, record, rows in matches:
                self.package.addItem(Path(record["production_root"]).name)
            self.layer = QComboBox(); self.layer.addItem("Aqueous concentration (g/L) · fixed 0–2 scale", "c_aq_g_l")
            self.layer.addItem("Gas saturation · fixed 0–1 scale", "s_g")
            controls.addWidget(self.package); controls.addWidget(self.layer)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); self.image = QLabel(); self.image.setAlignment(Qt.AlignCenter)
            scroll.setWidget(self.image); layout.addWidget(scroll, 1)
            self.status = QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
            self.slider = QSlider(Qt.Horizontal); layout.addWidget(self.slider)
            self.layer.currentIndexChanged.connect(self.show_frame)
            self.package.currentIndexChanged.connect(self.change_package)
            self.slider.valueChanged.connect(self.show_frame)
            self.change_package()
        def change_package(self):
            self.rows = self.matches[self.package.currentIndex()][2]
            self.slider.blockSignals(True); self.slider.setRange(0, len(self.rows) - 1)
            indices = [row["frame_index"] for row in self.rows]
            self.slider.setValue(indices.index(initial_frame) if initial_frame in indices else 0)
            self.slider.blockSignals(False); self.show_frame()
        def show_frame(self):
            self.generation += 1
            row = self.rows[self.slider.value()]; receipt = row["verified_receipt"]
            self.status.setText(f"Frame {row['frame_index'] + 1} · camera time {row['elapsed_h']:.4f} h · loading saved display…")
            worker = Picture(row, self.layer.currentData(), self.generation, self)
            self.jobs.append(worker); worker.loaded.connect(self.picture_ready)
            worker.finished.connect(lambda w=worker: self.jobs.remove(w) if w in self.jobs else None)
            worker.start()
        def picture_ready(self, data, error, generation):
            if generation != self.generation:
                return
            if error:
                self.status.setText(error); return
            pixmap = QPixmap(); pixmap.loadFromData(data); self.image.setPixmap(pixmap)
            row = self.rows[self.slider.value()]; receipt = row["verified_receipt"]
            result = receipt["native_result"]
            self.status.setText(f"Frame {row['frame_index'] + 1} · camera time {row['elapsed_h']:.4f} h · "
                f"aqueous {result.get('aqueous_mass_g')} g · gas {result.get('gas_mass_g')} g · total {result.get('total_mass_g')} g\n"
                + receipt["run_caveat"] + "\n" + receipt["phase_caveat"])
        def closeEvent(self, event):
            self.hide(); event.ignore()

    button = QPushButton("Review saved common calibration")
    button.setToolTip("Explicitly select a saved common concentration layer matching this exact phase candidate. No fit is started.")
    dialog.centralWidget().layout().insertWidget(1, button)
    dialog._common_mass_lookups = []
    dialog._common_mass_viewers = []
    def ready(matches, error):
        button.setEnabled(True)
        if error or not matches:
            QMessageBox.information(dialog, "Saved common calibration", error or "No complete saved common calibration matches this phase candidate yet.")
            return
        viewer = Viewer(matches); dialog._common_mass_viewers.append(viewer); viewer.show(); viewer.raise_()
    def start_lookup():
        button.setEnabled(False); worker = Lookup(dialog); dialog._common_mass_lookups.append(worker)
        worker.loaded.connect(ready)
        worker.finished.connect(lambda w=worker: dialog._common_mass_lookups.remove(w) if w in dialog._common_mass_lookups else None)
        worker.start()
    button.clicked.connect(start_lookup)
    dialog.saved_common_calibration_button = button
