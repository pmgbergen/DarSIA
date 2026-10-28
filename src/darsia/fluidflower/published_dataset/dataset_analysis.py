"""New analysis of a published dataset opened as a series.

The page shows what the dataset's plan fixes (phase model, mass calibration,
frames, storage profile and stored resolution) and embeds the reproduction
launcher, pre-filled with the dataset and the user's results location. Its
machine settings therefore use the published memory measurement exactly as
the launcher does. Settings the user changes are listed as differences from
the published plan, on the page and in the report.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .layout import inside
from .reproduce_window import ReproduceWindow


def plan_facts(root, manifest: dict) -> list[str]:
    """What a dataset's published plan fixes, in words (read from the dataset only)."""
    root = Path(root)
    facts = []
    try:
        with (root / "series.toml").open("rb") as stream:
            method = (tomllib.load(stream).get("method") or {})
        model = (method.get("phase_model") or {}).get("model_version")
        if model:
            facts.append(f"Phase model: {model}")
    except (OSError, ValueError):
        pass
    try:
        plan = tomllib.loads(inside(root, manifest["plan"]["path"]).read_text(encoding="utf-8"))
        calibration = (plan.get("job") or {}).get("calibration_sha256")
        if calibration:
            facts.append(f"Mass calibration: SHA-256 {calibration[:12]}…")
    except (OSError, ValueError, KeyError):
        pass
    frames = sum(len(row["frames"]) for row in manifest["runs"])
    facts.append(f"Frames: {frames} in {len(manifest['runs'])} run(s), as published")
    facts.append(f"Storage profile: {manifest['plan'].get('storage_profile', 'full')}; precision: "
                 f"{manifest['plan'].get('precision')}")
    layer = (manifest.get("expected") or {}).get("layer") or {}
    if layer.get("resolution_mm") is not None:
        facts.append(f"Stored layer: {layer['resolution_mm']:g} mm cells")
    return facts


class DatasetAnalysisPage(QWidget):
    """New analysis for a published dataset: the plan's facts and the launcher, pre-filled."""

    def __init__(self, parent=None, *, window_factory=None):
        """An empty page; ``set_dataset`` fills it (``window_factory`` replaces the launcher in tests)."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel("New analysis of a published dataset")
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        self.facts_label = QLabel()
        self.facts_label.setWordWrap(True)
        layout.addWidget(self.facts_label)
        self.differences_label = QLabel()
        self.differences_label.setWordWrap(True)
        layout.addWidget(self.differences_label)
        self.launcher = (window_factory or ReproduceWindow)()
        self.launcher.setParent(self)
        layout.addWidget(self.launcher, 1)
        self.root = self.results = None
        for signal in (self.launcher.threads_spin.valueChanged, self.launcher.backend_combo.currentIndexChanged,
                       self.launcher.run_combo.currentIndexChanged, self.launcher.quick_spin.valueChanged,
                       self.launcher.override_check.toggled):
            signal.connect(self.refresh_differences)
        for button in self.launcher.scope_buttons.values():
            button.toggled.connect(self.refresh_differences)
        self.launcher.dataset_opened.connect(self.refresh_differences)

    def set_dataset(self, root, results) -> None:
        """Pre-fill the page for a dataset and its results location, then read the dataset."""
        from .dataset_series import fresh_results_folder

        self.root, self.results = Path(root), Path(results)
        self.launcher.dataset_edit.setText(str(self.root))
        self.launcher.results_edit.setText(str(fresh_results_folder(self.results)))
        self.facts_label.setText("The dataset is read-only. Results go to your results location.")
        self.launcher.open_dataset()

    def offer_fresh_results(self) -> None:
        """Point the results field at a fresh analysis folder when the current one holds files."""
        from .dataset_series import fresh_results_folder

        if self.results is None or self.busy():
            return
        current = Path(self.launcher.results_edit.text().strip() or ".")
        if current.is_dir() and any(current.iterdir()):
            self.launcher.results_edit.setText(str(fresh_results_folder(self.results)))
            self.launcher.refresh_settings()

    def refresh_differences(self, *_):
        """Show the published plan's facts and the settings that now differ from it."""
        from .reproduce import ReproductionRefused, plan_differences

        reproduction = self.launcher.reproduction
        if reproduction is None or reproduction.check is None:
            return
        manifest = reproduction.check.manifest
        self.facts_label.setText("Published plan (read-only): " + "; ".join(plan_facts(self.root, manifest)) + ".")
        try:
            lines = plan_differences(manifest, self.launcher.scope(), self.launcher.settings(),
                                     reproduction.check.code)
        except ReproductionRefused as error:
            lines = [str(error)]
        self.differences_label.setText("Changes from the published plan: " + ("; ".join(lines) if lines else "none")
                                       + ".")

    def busy(self) -> bool:
        """True while the embedded launcher works."""
        return self.launcher.busy()
