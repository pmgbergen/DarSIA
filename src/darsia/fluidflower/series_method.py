"""Per-series method choice and its small native UI panel.

Only an explicit save changes ``series.toml``. Legacy references remain opaque;
the native approval mode additionally binds existing files and locks the choice.
Neither mode fits or alters a calibration.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import os
import tomllib
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import toml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


METHODS = ("colour_path", "phase_model")
# Display names of the methods (N141). The keys above and in series.toml stay.
METHOD_NAMES = {"colour_path": "Direct colour method", "phase_model": "Phase-map method"}
METHOD_HEADERS = {"colour_path": "Direct colour method (DarSIA original)", "phase_model": "Phase-map method"}
METHOD_DESCRIPTIONS = {
    "colour_path": "Colour → concentration → mass; phases from colour-signal thresholds (DarSIA original).",
    "phase_model": "Trained phase map first; concentration and mass are computed inside the phases.",
}
MODEL_COMBINATIONS = {
    "colour_path": {
        "name": "Colour signal + SimpleFlash (DarSIA original)",
        "quantification": "Colour signal + SimpleFlash",
        "phase_guidance": "Not used; phases follow the original colour-signal rules.",
    },
    "phase_model": {
        "name": "Titration + dynamic phase maps",
        "quantification": "Titration",
        "phase_guidance": "Used as Expert Knowledge before concentration and mass are computed.",
    },
}


class ManifestChangedError(RuntimeError):
    """The manifest changed since this save's original bytes were read."""


@dataclass(frozen=True)
class MethodSelection:
    active: str = ""
    model_version: str = ""
    mass_calibration: str = ""
    colour_calibration: str = ""
    error: str = ""


def model_settings_description(selection: MethodSelection) -> dict[str, str]:
    """Describe the selected executable combination; do not infer readiness.

    The existing two route keys remain the actual backend choice. These are
    not independent switches and no model/calibration reference is rewritten.
    """
    return dict(MODEL_COMBINATIONS.get(selection.active, {
        "name": "No model combination selected",
        "quantification": "Choose a supported model combination.",
        "phase_guidance": "No combination selected.",
    }))


def missing_method_references(selection: MethodSelection) -> tuple[str, ...]:
    """Required references still missing; no path lookups or inference."""
    required = (
        (
            ("model version", selection.model_version),
            ("mass calibration", selection.mass_calibration),
        )
        if selection.active == "phase_model"
        else (
            (("colour-path calibration", selection.colour_calibration),)
            if selection.active == "colour_path"
            else ()
        )
    )
    return tuple(
        label
        for label, value in required
        if not isinstance(value, str) or not value.strip()
    )


def _reference(table: Mapping, key: str) -> str:
    value = table.get(key)
    return value if isinstance(value, str) else ""


def _selection(document: Mapping) -> MethodSelection:
    method = document.get("method", {})
    if not isinstance(method, Mapping):
        return MethodSelection(error="The method section must be a TOML table.")
    phase = method.get("phase_model", {})
    colour = method.get("colour_path", {})
    if not isinstance(phase, Mapping) or not isinstance(colour, Mapping):
        return MethodSelection(
            error="Method calibration references must be TOML tables."
        )
    phase = phase if isinstance(phase, Mapping) else {}
    colour = colour if isinstance(colour, Mapping) else {}
    active = method.get("active", "")
    return MethodSelection(
        active=active if isinstance(active, str) and active in METHODS else "",
        model_version=_reference(phase, "model_version"),
        mass_calibration=_reference(phase, "mass_calibration"),
        colour_calibration=_reference(colour, "calibration"),
    )


def read_series_method(series_root: str | Path) -> MethodSelection:
    """Read the saved choice without discovery, calibration or writes.

    An absent/unknown choice is not chosen. An unreadable or malformed manifest
    also remains not chosen and carries its error for the UI to show.
    """
    try:
        raw = (Path(series_root) / "series.toml").read_bytes()
        return _selection(tomllib.loads(raw.decode("utf-8-sig")))
    except (OSError, UnicodeError, ValueError) as error:
        return MethodSelection(error=str(error))


def calibration_ancestry_reference(method: Mapping) -> str | None:
    """Explicit retained selection/fit ancestry; never an active calibration fallback."""
    reference = method.get("mass_calibration")
    if isinstance(reference, str) and reference.strip():
        return reference
    previous = method.get("previous_mass_calibrations", [])
    if not isinstance(previous, list) or not all(isinstance(item, str) and item.strip() for item in previous):
        raise ValueError("Previous mass-calibration references must be a list of non-empty strings.")
    return previous[-1] if previous else None


def _same_values(first, second) -> bool:
    """Exact TOML semantics, including NaN and an offset's representation."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same_values(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, list):
        return len(first) == len(second) and all(
            _same_values(left, right) for left, right in zip(first, second)
        )
    if isinstance(first, float) and math.isnan(first):
        return math.isnan(second)
    if isinstance(first, (dt.datetime, dt.date, dt.time)):
        return first.isoformat() == second.isoformat()
    return first == second


def save_series_calibration_method(
    series_root: str | Path,
    active: str,
    *,
    expected_bytes: bytes | None = None,
    context_current=None,
) -> MethodSelection:
    """Record only the calculation method, before model or calibration exists."""
    if active not in METHODS:
        raise ValueError("Choose a supported calibration method before saving.")
    path = Path(series_root) / "series.toml"
    original = path.read_bytes()
    if expected_bytes is not None and original != expected_bytes:
        raise ManifestChangedError(f"Manifest changed since it was loaded: {path}")
    document = tomllib.loads(original.decode("utf-8-sig"))
    from .series_calibration_approval import require_unlocked

    require_unlocked(document)
    updated = copy.deepcopy(document)
    method = updated.setdefault("method", {})
    if not isinstance(method, dict):
        raise ValueError("The method section must be a TOML table.")
    method["active"] = active
    return _selection(_write_manifest(path, original, updated, context_current=context_current))


def save_series_method(
    series_root: str | Path,
    selection: MethodSelection,
    *,
    expected_bytes: bytes | None = None,
    prepared_model: bool = False,
    context_current=None,
) -> MethodSelection:
    """Atomically record one explicit method and its reference strings.

    Only the chosen method's reference fields are updated. The inactive method,
    unknown method fields and every unrelated TOML value are retained. Existing
    ``toml`` serialization may normalize comments/formatting; a parse-roundtrip
    must preserve all semantic values. Original bytes are checked immediately
    before replacement so a concurrent edit is rejected rather than overwritten.
    ``expected_bytes`` optionally also rejects changes since a caller's load.
    """
    if selection.active not in METHODS:
        raise ValueError("Choose colour_path or phase_model before saving.")
    for value in (
        selection.model_version,
        selection.mass_calibration,
        selection.colour_calibration,
    ):
        if not isinstance(value, str):
            raise ValueError("Calibration references must be strings.")
    missing = missing_method_references(selection)
    if prepared_model:
        if selection.active != "phase_model" or not selection.model_version.strip() or selection.mass_calibration:
            raise ValueError("A prepared model save needs a phase model and no active mass calibration.")
        missing = tuple(name for name in missing if name != "mass calibration")
    if missing:
        raise ValueError(
            "Provide " + " and ".join(missing) + " before saving this method."
        )
    path = Path(series_root) / "series.toml"
    original = path.read_bytes()
    if expected_bytes is not None and original != expected_bytes:
        raise ManifestChangedError(f"Manifest changed since it was loaded: {path}")
    document = tomllib.loads(original.decode("utf-8-sig"))
    from .series_calibration_approval import require_unlocked

    require_unlocked(document)
    updated = copy.deepcopy(document)
    method = updated.setdefault("method", {})
    if not isinstance(method, dict):
        raise ValueError("The method section must be a TOML table.")
    method["active"] = selection.active
    references = method.setdefault(selection.active, {})
    if not isinstance(references, dict):
        raise ValueError(f"method.{selection.active} must be a TOML table.")
    if selection.active == "phase_model":
        previous = references.get("mass_calibration")
        if prepared_model and previous:
            history = references.setdefault("previous_mass_calibrations", [])
            if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
                raise ValueError("The previous mass-calibration references must be a list of strings.")
            if previous not in history:
                history.append(previous)
        references.update(
            model_version=selection.model_version,
            mass_calibration=selection.mass_calibration,
        )
    else:
        references["calibration"] = selection.colour_calibration

    return _selection(_write_manifest(path, original, updated, context_current=context_current))


def _manifest_bytes(original, updated):
    """Encode exact save bytes with the original TOML semantic roundtrip."""
    text = toml.dumps(updated)
    # Preserve the manifest's line ending convention and UTF-8 BOM.
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    raw = b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    raw += text.encode("utf-8")
    parsed = tomllib.loads(raw.decode("utf-8-sig"))
    if not _same_values(updated, parsed):
        raise ValueError("Saving this choice would change other TOML values.")
    return raw, parsed


def _write_manifest(path, original, updated, *, context_current=None):
    """Original semantic roundtrip and compare-and-swap for explicit saves."""
    raw, parsed = _manifest_bytes(original, updated)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        if path.read_bytes() != original:
            raise ManifestChangedError(f"Manifest changed while saving: {path}")
        if context_current is not None and not context_current():
            raise ManifestChangedError("The selected series or model context changed while saving.")
        for delay in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, None):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows refuses the replace while a reader briefly holds the file.
                if delay is None:
                    raise
                time.sleep(delay)
                if path.read_bytes() != original:
                    raise ManifestChangedError(f"Manifest changed while saving: {path}")
    finally:
        # This is only the temporary file owned by this explicit save.
        if temporary.exists():
            temporary.unlink()
    return parsed


class SeriesMethodPanel(QWidget):
    """UI choice and references; saving belongs to the native window adapter."""

    use_requested = Signal(object)
    unlock_requested = Signal()

    def __init__(self, parent=None, *, require_bound_approval=False, method_choice_only=False):
        super().__init__(parent)
        self.method_choice_only = bool(method_choice_only)
        self._saved_active = ""
        self._require_bound_approval = require_bound_approval
        self._approval_root = None
        self._approval = None
        self._approval_error = ""
        self._approval_binding = None
        self._approval_manifest = None
        from .series_method_calls import SeriesMethodCalls

        self.calls = SeriesMethodCalls(self, verify_approval=not self.method_choice_only)
        self.calls.changed.connect(self._update_controls)
        self._available = False
        self._load_error = ""
        outer = QVBoxLayout(self)
        self.title = QLabel("Calibration Method" if self.method_choice_only else "Model settings")
        outer.addWidget(self.title)
        self.combination_note = QLabel(
            "Choose one supported combination. Quantification and dynamic phase guidance "
            "cannot currently be selected independently.")
        self.combination_note.setWordWrap(True)
        if self.method_choice_only:
            self.combination_note.setText(
                "Choose the calibration method for this series. Select or train the phase model "
                "under Phase model; calibration inputs and results belong to Mass calibration.")
        outer.addWidget(self.combination_note)
        self.colour_path_button = QRadioButton(MODEL_COMBINATIONS["colour_path"]["name"])
        self.phase_model_button = QRadioButton(MODEL_COMBINATIONS["phase_model"]["name"])
        self.colour_path_button.setToolTip("Original colour response and SimpleFlash quantification. "
                                         "Dynamic phase maps are not used.")
        self.phase_model_button.setToolTip("Titration quantification guided by the selected dynamic "
                                         "phase-map model as Expert Knowledge.")
        self.method_buttons = QButtonGroup(self)
        self.method_buttons.addButton(self.colour_path_button, 0)
        self.method_buttons.addButton(self.phase_model_button, 1)
        outer.addWidget(self.colour_path_button)
        self.colour_references = QWidget()
        colour_fields = QFormLayout(self.colour_references)
        colour_fields.setContentsMargins(0, 0, 0, 0)
        self.colour_calibration = QLineEdit()
        self.colour_calibration.setPlaceholderText("No calibration referenced")
        colour_fields.addRow("Colour-path calibration", self.colour_calibration)
        outer.addWidget(self.colour_references)
        outer.addWidget(self.phase_model_button)
        self.phase_references = QWidget()
        phase_fields = QFormLayout(self.phase_references)
        phase_fields.setContentsMargins(0, 0, 0, 0)
        self.model_version = QLineEdit()
        self.mass_calibration = QLineEdit()
        self.model_version.setPlaceholderText("No model version referenced")
        self.mass_calibration.setPlaceholderText("No mass calibration referenced")
        phase_fields.addRow("Phase model version", self.model_version)
        phase_fields.addRow("Mass calibration", self.mass_calibration)
        outer.addWidget(self.phase_references)
        self.combination_details = QWidget()
        selected = QFormLayout(self.combination_details)
        selected.setContentsMargins(0, 0, 0, 0)
        self.quantification_model = QLabel()
        self.phase_guidance = QLabel()
        for label in (self.quantification_model, self.phase_guidance):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
        selected.addRow("Chemistry / quantification model", self.quantification_model)
        selected.addRow("Dynamic phase maps (Expert Knowledge)", self.phase_guidance)
        outer.addWidget(self.combination_details)
        self.status = QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.approval_summary = QLabel()
        self.approval_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.approval_summary.setWordWrap(True)
        outer.addWidget(self.approval_summary)
        self.unlock_button = QPushButton("Unlock to change…")
        self.unlock_button.clicked.connect(self._request_unlock)
        outer.addWidget(self.unlock_button)
        self.use_button = QPushButton("Use for this series")
        self.use_button.clicked.connect(self._request_use)
        outer.addWidget(self.use_button)
        self.colour_path_button.toggled.connect(self._update_controls)
        self.phase_model_button.toggled.connect(self._update_controls)
        for field in (
            self.model_version,
            self.mass_calibration,
            self.colour_calibration,
        ):
            field.textChanged.connect(self._update_controls)
        if self.method_choice_only:
            self.colour_references.hide()
            self.phase_references.hide()
            self.combination_details.hide()
            self.approval_summary.hide()
        self.set_selection(MethodSelection(), available=False)

    def set_selection(
        self,
        saved: MethodSelection,
        *,
        phase_model_reference: str = "",
        mass_calibration_reference: str = "",
        colour_calibration_reference: str = "",
        available: bool = True,
    ) -> None:
        """Saved nonempty refs win; only supplied actual selections may seed."""
        self._available = available
        self._load_error = saved.error
        self._saved_active = saved.active
        self.method_buttons.setExclusive(False)
        self.colour_path_button.setChecked(saved.active == "colour_path")
        self.phase_model_button.setChecked(saved.active == "phase_model")
        self.method_buttons.setExclusive(True)
        self.model_version.setText(saved.model_version or phase_model_reference)
        self.mass_calibration.setText(
            saved.mass_calibration or mass_calibration_reference
        )
        self.colour_calibration.setText(
            saved.colour_calibration or colour_calibration_reference
        )
        self._update_controls()

    def set_approval_context(self, root, record=None, error="", *, manifest_bytes=None, force=False) -> None:
        """Current series only; passive metadata never records an approval."""
        self._approval_root = root
        self._approval = record
        self._approval_error = error
        self._approval_manifest = manifest_bytes
        self.calls.set_context(root, self.selection(), manifest_bytes, record, error, force=force)
        self._update_controls()

    def selection(self) -> MethodSelection:
        active = (
            "phase_model"
            if self.phase_model_button.isChecked()
            else "colour_path" if self.colour_path_button.isChecked() else ""
        )
        return MethodSelection(
            active=active,
            model_version=self.model_version.text(),
            mass_calibration=self.mass_calibration.text(),
            colour_calibration=self.colour_calibration.text(),
        )

    def _update_controls(self, *_args) -> None:
        locked = bool(self._approval and self._approval.get("locked"))
        action_busy = self.calls.thread is not None and self.calls.kind != "verify"
        available = self._available and not self._load_error and not locked and not action_busy
        self.colour_path_button.setEnabled(available)
        self.phase_model_button.setEnabled(available)
        self.colour_calibration.setEnabled(
            available and self.colour_path_button.isChecked()
        )
        for field in (self.model_version, self.mass_calibration):
            field.setEnabled(available and self.phase_model_button.isChecked())
        current = self.selection()
        if self.method_choice_only:
            self.calls.set_context(self._approval_root, current, self._approval_manifest,
                                   self._approval, self._approval_error)
            self._approval_binding = None
            saved = current.active in METHODS and current.active == self._saved_active
            self.use_button.setText("✓ Calibration method saved" if saved else "Save calibration method")
            self.use_button.setEnabled(available and self.calls.thread is None
                                       and current.active in METHODS and not saved
                                       and not self._approval_error)
            self.unlock_button.setVisible(locked)
            self.unlock_button.setEnabled(self._available and locked and self.calls.thread is None)
            self.status.setText(self._load_error or self._approval_error or self.calls.error or (
                "Choose a series on Start." if not self._available else
                "The saved calibration method is locked. Unlock it to change the method." if locked else
                "Saving the calibration method…" if action_busy else
                "Saved method: " + MODEL_COMBINATIONS[current.active]["name"] + "." if saved else
                "Choose a calibration method, then save it for this series."))
            return
        description = model_settings_description(current)
        self.quantification_model.setText(description["quantification"])
        self.phase_guidance.setText(description["phase_guidance"])
        missing = missing_method_references(current)
        if self._require_bound_approval:
            self.calls.set_context(self._approval_root, current, self._approval_manifest,
                                   self._approval, self._approval_error)
        self._approval_binding = self.calls.binding if self.calls.ready() else None
        binding_error = self._approval_error or self.calls.error
        bound = not self._require_bound_approval or current.active != "phase_model" or self.calls.ready()
        self.use_button.setEnabled(
            available and self.calls.thread is None and current.active in METHODS
            and not missing and bound and not binding_error
        )
        self.unlock_button.setVisible(locked)
        self.unlock_button.setEnabled(self._available and locked and self.calls.thread is None)
        binding = (self._approval or {}).get("binding") if locked else self._approval_binding
        if binding:
            identity = binding["phase_model"]["identity"]
            self.approval_summary.setText(
                ("Approved and locked. " if locked else "Ready to approve. ")
                + "Chemistry model: Titration. Phase model: " + identity["model_id"]
                + ". Mass calibration: " + binding["mass_calibration"]
                + (". " + binding_error if binding_error else ".")
            )
        else:
            self.approval_summary.setText(binding_error or (
                "Checking the selected saved calibration and model…"
                if self._require_bound_approval and current.active == "phase_model" and not missing
                else ""))

        self.status.setText(
            self._load_error
            or (
                "Method reference missing: " + " and ".join(missing) + "."
                if self._available and current.active in METHODS and missing
                else (
                    ("Unlock with a reason to choose a different calibration." if locked
                     else "Use approves and locks the selected calibration and model."
                     if self._require_bound_approval and current.active == "phase_model"
                     else "Save the method and its calibration references for this series.")
                    if self._available
                    else "Choose a series on Start."
                )
            )
        )

    def _request_use(self) -> None:
        if self.use_button.isEnabled():
            self.use_requested.emit(self.selection())

    def _request_unlock(self) -> None:
        if self.unlock_button.isEnabled():
            self.unlock_requested.emit()

    def can_close(self):
        return self.calls.can_close()


# Common UI terminology; callers using the original class/API remain valid.
ModelSettingsPanel = SeriesMethodPanel
