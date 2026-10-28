"""Explicit mass-fit holdout choices, independent of image and phase training inputs."""
from copy import deepcopy
from pathlib import Path
import tomllib

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


def _active_runs(document):
    table = document.get("runs", {})
    runs = table.get("ids", [])
    excluded = table.get("excluded", {})
    if (not isinstance(runs, list) or not all(isinstance(run, str) and run for run in runs)
            or len(runs) != len(set(runs)) or not isinstance(excluded, (dict, list))):
        raise ValueError("The series needs distinct run identifiers and an explicit excluded-run list.")
    omitted = set(excluded)
    return [run for run in runs if run not in omitted], [run for run in runs if run in omitted]


def explicit_mass_cohort(document):
    """Return a validated saved split, or None without inventing one."""
    mass = document.get("calibration", {}).get("mass", {})
    if not isinstance(mass, dict):
        raise ValueError("The mass-calibration choices must be a table.")
    if "fit_runs" not in mass:
        return None
    active, excluded = _active_runs(document)
    fit, heldout = mass["fit_runs"], mass.get("heldout_runs", [])
    for values in (fit, heldout):
        if (not isinstance(values, list) or not all(isinstance(run, str) and run for run in values)
                or len(values) != len(set(values))):
            raise ValueError("Fit and holdout runs must be distinct run lists.")
    if (not fit or set(fit) & set(heldout) or set(fit) | set(heldout) != set(active)):
        raise ValueError("Select at least one fit run; every active run must be fit or holdout.")
    return dict(fit=[run for run in active if run in fit],
                heldout=[run for run in active if run in heldout], descriptive=[], excluded=excluded)


def mass_cohort_roles(series_root, *, scope="shared"):
    """The operative roles; individual fits have no shared holdout cohort."""
    root = Path(series_root)
    document = tomllib.loads((root / "series.toml").read_text(encoding="utf-8-sig"))
    active, excluded = _active_runs(document)
    unassigned = []
    if scope == "individual":
        roles = dict(fit=active, heldout=[], descriptive=[], excluded=excluded)
    elif scope == "shared":
        roles = explicit_mass_cohort(document)
        if roles is None:
            from .calibration_review_runs import review_run_rows
            rows = review_run_rows(root, roles_only=True)
            names = {"fit": "fit", "heldout": "held-out", "descriptive": "descriptive", "excluded": "excluded"}
            roles = {key: [row.run for row in rows.rows if row.role == value and row.run in active]
                     for key, value in names.items()}
            roles["excluded"] = excluded
            # New series have no historical fit to inherit.
            if not any(roles[key] for key in ("fit", "heldout", "descriptive")):
                roles["fit"] = active
            else:
                assigned = {run for values in roles.values() for run in values}
                unassigned = [run for run in active if run not in assigned]
    else:
        raise ValueError("Choose shared or individual calibration.")
    summary = (f"{len(roles['fit'])} fit · {len(roles['heldout'])} holdout from mass fit"
               + (f" · {len(roles['descriptive'])} descriptive" if roles['descriptive'] else "")
               + f" · {len(roles['excluded'])} excluded.")
    if unassigned:
        summary += " Left out of mass calibration, without a role: " + ", ".join(unassigned) + "."
    return roles, summary


def read_mass_cohort(series_root):
    root = Path(series_root)
    raw = (root / "series.toml").read_bytes()
    document = tomllib.loads(raw.decode("utf-8-sig"))
    active, _excluded = _active_runs(document)
    roles, summary = mass_cohort_roles(root)
    from .series_method import calibration_ancestry_reference
    from .calibration_review_runs import _calibration_document
    reference = calibration_ancestry_reference(document.get("method", {}).get("phase_model", {}))
    historical_fit = []
    if reference:
        try:
            historical_fit = _calibration_document(root, reference)[1].get("calibration_runs", [])
        except (OSError, ValueError, TypeError):
            pass
    from .pipeline.current_calibration_selection import _saved_target
    image_rows = document.get("image_selection", {}).get("runs", {})
    eligibility = {run: ("" if run in historical_fit or any(
        _saved_target(row, image_rows.get(run, {})) is not None
        for row in image_rows.get(run, {}).get("selected", [])) else
        "No recorded eligible camera targets; choose calibration images before fitting this run.") for run in active}
    if (root / "series.toml").read_bytes() != raw:
        raise ValueError("The series changed while its calibration cohort was read.")
    return dict(root=root, raw=raw, active=active, roles=roles, summary=summary, eligibility=eligibility,
                explicit=explicit_mass_cohort(document) is not None)


def current_mass_cohort(series_root, roles):
    """Bind changed GUI roles while preserving the old accepted-source route."""
    if series_root is None:
        return None
    root = Path(series_root)
    document = tomllib.loads((root / "series.toml").read_text(encoding="utf-8-sig"))
    active, _excluded = _active_runs(document)
    if explicit_mass_cohort(document) is None and roles.get("fit") != active:
        return None
    from .mass.n1_pipeline import asset
    return dict(format="darsia-current-mass-cohort-v1", analysis_runs=active,
                roles=deepcopy(roles), series_manifest=asset(root / "series.toml"))


def save_mass_cohort(series_root, fit_runs, *, expected_bytes, context_current=None):
    from .series_method import ManifestChangedError, _write_manifest
    from .series_calibration_approval import require_unlocked
    root = Path(series_root)
    path = root / "series.toml"
    original = path.read_bytes()
    if original != expected_bytes:
        raise ManifestChangedError("The series changed; reload the calibration cohort before saving.")
    document = tomllib.loads(original.decode("utf-8-sig"))
    require_unlocked(document)
    active, _excluded = _active_runs(document)
    if (not isinstance(fit_runs, list) or len(fit_runs) != len(set(fit_runs))
            or not set(fit_runs) <= set(active)):
        raise ValueError("Select fit runs only from the active series.")
    updated = deepcopy(document)
    choices = updated.setdefault("calibration", {}).setdefault("mass", {})
    choices["fit_runs"] = [run for run in active if run in fit_runs]
    choices["heldout_runs"] = [run for run in active if run not in fit_runs]
    explicit_mass_cohort(updated)
    _write_manifest(path, original, updated, context_current=context_current)
    return read_mass_cohort(root)


class SharedCalibrationCohort(QWidget):
    """Choose the shared mass-fit cohort; saved image choices remain untouched."""
    saved = Signal(object)

    def __init__(self, panel):
        super().__init__(panel)
        self.panel, self.root, self.snapshot = panel, None, None
        self._generation = 0
        from .ui_metadata_preview import MetadataPreview
        self.preview = MetadataPreview(self)
        self.preview.completed.connect(self._read)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Runs for shared calibration")
        layout.addWidget(title)
        text = QLabel("Untick a run to hold it out from the mass fit. It remains available for later validation. "
            "This does not hold it out from phase-model training or colour-path preparation.")
        text.setWordWrap(True)
        layout.addWidget(text)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("Run", "Use in mass fit", "Role"))
        self.table.verticalHeader().hide()
        self.table.setMaximumHeight(220)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._changed)
        layout.addWidget(self.table)
        self.save_button = QPushButton("Save calibration run selection")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        panel.calibration_scope_combo.currentIndexChanged.connect(self._scope_changed)
        self._scope_changed()

    def _scope_changed(self, *_args):
        self.setVisible(self.root is not None and self.panel.calibration_scope_combo.currentData() == "shared")

    def set_series(self, root):
        self.root = Path(root) if root is not None else None
        self._generation += 1
        self.snapshot = None
        self.preview.cancel()
        self.table.setRowCount(0)
        self.save_button.setEnabled(False)
        self._scope_changed()
        if self.root is not None:
            root = self.root
            self.status.setText("Reading the saved mass-fit and holdout choices…")
            self.preview.start(lambda cancelled: None if cancelled() else read_mass_cohort(root))

    def _read(self, value, error):
        if error:
            self.status.setText(error)
            return
        if value is None or value["root"] != self.root:
            return
        self.snapshot = value
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(value["active"]))
            for index, run in enumerate(value["active"]):
                name = QTableWidgetItem(run)
                name.setToolTip(value["eligibility"][run])
                self.table.setItem(index, 0, name)
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                use.setCheckState(Qt.CheckState.Checked if run in value["roles"]["fit"] else Qt.CheckState.Unchecked)
                self.table.setItem(index, 1, use)
                role = ("Fit — select calibration images" if value["eligibility"][run] else "Fit") if run in value["roles"]["fit"] else "Holdout from mass fit"
                self.table.setItem(index, 2, QTableWidgetItem(role))
        self.table.resizeColumnsToContents()
        self.status.setText(value["summary"] + (" Saved selection." if value["explicit"] else " Inherited selection; save to make changes."))
        self.save_button.setEnabled(True)

    def _fit_runs(self):
        return [self.table.item(index, 0).text() for index in range(self.table.rowCount())
                if self.table.item(index, 1).checkState() == Qt.CheckState.Checked]

    def _changed(self, item):
        if item.column() == 1:
            fit = item.checkState() == Qt.CheckState.Checked
            run = self.table.item(item.row(), 0).text()
            missing = self.snapshot["eligibility"].get(run, "") if self.snapshot else ""
            self.table.item(item.row(), 2).setText(("Fit — select calibration images" if missing else "Fit") if fit else "Holdout from mass fit")
            self.status.setText("Unsaved calibration run selection.")

    def _save(self):
        if self.snapshot is None or not self.save_button.isEnabled():
            return
        root, generation, raw, fit = self.root, self._generation, self.snapshot["raw"], self._fit_runs()
        self.save_button.setEnabled(False)
        self.status.setText("Saving calibration run selection…")
        def save():
            try:
                value = save_mass_cohort(root, fit, expected_bytes=raw,
                    context_current=lambda: self.root == root and self._generation == generation)
                return dict(value=value, error="")
            except (OSError, ValueError, RuntimeError) as error:
                return dict(value=None, error=str(error))
        if not self.panel._call(save, self._saved):
            self.status.setText("Wait for the current calibration operation before saving the run selection.")
            self.save_button.setEnabled(True)

    def _saved(self, result):
        if result["error"]:
            self.status.setText("Calibration run selection was not saved: " + result["error"])
            self.save_button.setEnabled(True)
            return
        self._read(result["value"], "")
        self.status.setText("✓ Calibration run selection saved. " + result["value"]["summary"] + " No job started.")
        self.saved.emit(self.root)
