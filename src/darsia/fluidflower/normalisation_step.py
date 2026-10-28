"""Read saved v5 normalisation evidence; explicitly save only its reference choice.

No sandbox implementation is imported or executed. Approved source identities
name published evidence files, never their private filesystem location.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tomllib
import uuid

import toml
from PySide6.QtCore import QCoreApplication, QThread, Qt, Signal, Slot, QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


APPROVED_V5_SOURCES = {
    "v5_20260929/RESULTS.md": "f5ebc2e316956e1c95b83f60173dd1b23b9df8d2ae89e495c3fbbc608d76d9c9",
    "v5_20260929/HANDOFF_20260929.md": "669a237f329d2906ef1a450b2924eced1267ae8f5be5306921c5e6e89c0f88ee",
    "v5_20260929/protocol_rank.py": "21a546f72cfcefc2d52e80d95e54bda589f7c4bdd40cd08748e1de95bed004db",
    "v5_20260929/protocols_parsed.json": "3739cbcdc9891f22a9221bbfb2fa531bf0f2c536e15a610cae027845e1fcaa03",
    "v5_20260929/proto_parse.py": "9b22578a461df7fddf38c5ec4c9f01c865b0e7aa79fa4058f01707f3eda0e2ab",
    "v5_20260929/dye_curve_distance_per_run.json": "78a8c2e43e9bbe047dab124211961a962cee270ebec8f6b2e74d7775629adfe5",
    "v5_20260929/protocol_chemistry_ranking.json": "ec922f8c11c52c01b5e0e34a1c2377506a93175433983bc8b73d2ae1d78205d0",
    "two_step_20260928/v4_params.json": "46d6a13bd49ebd0a9309c3d87120ecfee202a711250454939751a2f111a36538",
    "two_step_20260928/v4.py": "466996d8e4b637456e122e83faaabd0330bcce2b470b21cf2cb0fc9aad552c13",
    "v5_20260929/v5.py": "fd2f32377f0f294356ee21de9c99ef7ff7ab4a0316cfa45d2f72ba6e57336a9b",
    "v5_20260929/scores_v5_RESULT_cdye.json": "911c83ae7b2a46b28115def797559b1b295591c42ab2f713b3133c697abefa37",
}
MAX_METADATA_BYTES = 2 * 1024 * 1024
RECORD_FORMAT = "series-saved-v5-normalisation-v1"


def _key(value):
    return re.sub(r"[ _-]", "", str(value)).casefold()


def _finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _bounded(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError(f"Saved metadata exceeds the size limit: {path}")
    return raw


@dataclass(frozen=True)
class SavedV5Snapshot:
    source_folder: Path
    series_id: str
    report_reference: str
    snapshot_date: str
    majority_class: str
    suggested_reference: str
    runs: tuple[str, ...]
    rows: tuple[dict, ...]
    sources: tuple[dict, ...]
    bundle_sha256: str


class PublishedV5Provider:
    """Read a location-independent, explicitly approved historical evidence set.

    The digest map also binds the saved ranking to its original parsed protocol
    snapshot and parser/ranker. It does not claim today's XLSX files were read.
    Alternate evidence providers must be explicit, e.g. synthetic test inputs.
    """

    def __init__(self, expected_sources=None):
        self.expected_sources = dict(
            APPROVED_V5_SOURCES if expected_sources is None else expected_sources
        )

    def __call__(self, folder, series_id, runs):
        folder = Path(folder).resolve()
        if folder.name == "v5_20260929":
            folder = folder.parent
        data, sources = {}, []
        if set(self.expected_sources) != set(APPROVED_V5_SOURCES):
            raise ValueError("The saved evidence source list is incomplete.")
        for relative, expected in self.expected_sources.items():
            path = folder / relative
            raw = _bounded(path)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected:
                raise ValueError(f"Saved evidence origin differs: {relative}")
            data[relative] = raw
            sources.append(dict(relative_path=relative, sha256=digest, size=len(raw)))
        results = data["v5_20260929/RESULTS.md"].decode("utf-8-sig")
        references = set(re.findall(r"\bReference run ([A-Za-z0-9_-]+)\.", results))
        if len(references) != 1:
            raise ValueError("The saved report has no unambiguous reference run.")
        reference = next(iter(references)).casefold()
        ranking_source = data["v5_20260929/protocol_rank.py"].decode("utf-8-sig")
        identity = re.search(r"Rank ([A-Za-z0-9_-]+) runs", ranking_source)
        if not identity or _key(identity.group(1)) != _key(series_id):
            raise ValueError("The saved report belongs to another experimental series.")
        handoff = data["v5_20260929/HANDOFF_20260929.md"].decode("utf-8-sig")
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", handoff)
        if not date_match:
            raise ValueError("The saved evidence date is missing.")
        snapshot_date = "-".join(date_match.groups())
        parsed = json.loads(data["v5_20260929/protocols_parsed.json"])
        ranking = json.loads(data["v5_20260929/protocol_chemistry_ranking.json"])
        params = json.loads(data["two_step_20260928/v4_params.json"])
        residuals = json.loads(data["v5_20260929/scores_v5_RESULT_cdye.json"])
        if not isinstance(ranking.get("rows"), dict) or not isinstance(
            ranking.get("rank"), list
        ):
            raise ValueError("The saved protocol ranking is incomplete.")
        ranks = ranking["rank"]
        if len(ranks) != len(set(ranks)) or set(ranks) != set(ranking["rows"]):
            raise ValueError("The saved ranking has missing or repeated runs.")
        residual_rows = residuals.get("rows")
        if not isinstance(residual_rows, list):
            raise ValueError("Saved v5 residuals are missing.")
        by_run = {}
        for row in residual_rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("run"), str)
                or row["run"] in by_run
            ):
                raise ValueError(
                    "Saved residuals have invalid or repeated run identities."
                )
            by_run[row["run"]] = row
        current = tuple(str(run).casefold() for run in runs)
        if not current or len(current) != len(set(current)):
            raise ValueError("The selected series has no valid distinct run list.")
        classes = params.get("state") or {}
        counts = Counter(classes.get(run) for run in current)
        majority = next(
            (
                state
                for state, count in counts.items()
                if state in {"same", "new"} and count > len(current) / 2
            ),
            "",
        )
        if not majority:
            raise ValueError("The selected series has no majority saved light class.")
        rows = []
        for position, run in enumerate(ranks, 1):
            if run not in current:
                continue
            protocol = parsed.get(run)
            rank_row = ranking["rows"].get(run)
            residual = (by_run.get(run) or {}).get("v5")
            S = (params.get("chart_stat") or {}).get(run)
            state = classes.get(run)
            if (
                not isinstance(protocol, dict)
                or protocol.get("run") != run
                or not isinstance(protocol.get("file"), str)
                or not isinstance(rank_row, dict)
                or not _finite(rank_row.get("score"))
                or not _finite(S)
                or state not in {"same", "new"}
                or not isinstance(residual, dict)
                or not all(_finite(residual.get(key)) for key in ("all", "G", "chroma"))
            ):
                raise ValueError(
                    f"Saved ranking, protocol, light class or residual is missing: {run}"
                )
            if (S <= 0.02) != (state == "same"):
                raise ValueError(f"Saved S and light class disagree: {run}")
            flags, why = rank_row.get("flags"), rank_row.get("why")
            if (
                not isinstance(flags, list)
                or not isinstance(why, list)
                or not all(isinstance(x, str) for x in flags + why)
            ):
                raise ValueError(f"Saved protocol reasons are missing: {run}")
            excluded = list(
                dict.fromkeys(
                    flags + [reason for reason in why if not reason.startswith("pH z ")]
                )
            )
            if state != majority:
                excluded.append("Not the majority light class")
            if not _finite(rank_row.get("BTB_gL")) or not _finite(
                rank_row.get("NaOH_gL")
            ):
                excluded.append("Recipe not reported")
            ph = rank_row.get("ph")
            if (
                not isinstance(ph, list)
                or len(ph) != 4
                or not all(_finite(x) for x in ph)
            ):
                excluded.append("Port pH not reported")
            rows.append(
                dict(
                    run=run,
                    raw_rank=position,
                    score=rank_row["score"],
                    S=S,
                    light_class=state,
                    residual_all=residual["all"],
                    residual_G=residual["G"],
                    residual_chroma=residual["chroma"],
                    eligible=not excluded,
                    exclusion="; ".join(excluded),
                    protocol_file=protocol["file"],
                    protocol_notes=str(protocol.get("notes") or ""),
                )
            )
        if {row["run"] for row in rows} != set(current) or reference not in current:
            raise ValueError(
                "Saved evidence does not cover every selected run and its report reference."
            )
        eligible = [row["run"] for row in rows if row["eligible"]]
        if not eligible:
            raise ValueError("No protocol-eligible run has the majority light class.")
        bundle = hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return SavedV5Snapshot(
            folder,
            series_id,
            reference,
            snapshot_date,
            majority,
            eligible[0],
            current,
            tuple(rows),
            tuple(sources),
            bundle,
        )


def read_series_context(series_root, series_id):
    raw = _bounded(Path(series_root) / "series.toml")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if _key(document.get("series_id", "")) != _key(series_id):
        raise ValueError(
            "The selected folder's series manifest identifies another series."
        )
    runs = (document.get("runs") or {}).get("ids")
    if (
        not isinstance(runs, list)
        or not runs
        or not all(isinstance(run, str) and run.strip() for run in runs)
        or len({_key(run) for run in runs}) != len(runs)
    ):
        raise ValueError("The series manifest needs a distinct nonempty run list.")
    return raw, document, tuple(run.casefold() for run in runs)


def record_for(snapshot, reference, override):
    if reference not in snapshot.runs:
        raise ValueError("Choose a run belonging to this experimental series.")
    if not override and reference != snapshot.suggested_reference:
        raise ValueError("The automatic choice must use the saved eligible ranking.")
    return dict(
        format=RECORD_FORMAT,
        series_id=snapshot.series_id,
        reference_run=reference,
        selection="override" if override else "automatic",
        suggested_reference=snapshot.suggested_reference,
        report_reference=snapshot.report_reference,
        source_folder=str(snapshot.source_folder),
        source_bundle_sha256=snapshot.bundle_sha256,
        snapshot_date=snapshot.snapshot_date,
        runs=list(snapshot.runs),
        ranking=[dict(row) for row in snapshot.rows],
        sources=[dict(row) for row in snapshot.sources],
    )


def saved_ready(record, snapshot):
    if not isinstance(record, dict) or record.get("selection") not in {
        "automatic",
        "override",
    }:
        return False
    try:
        expected = record_for(
            snapshot, record.get("reference_run"), record["selection"] == "override"
        )
    except ValueError:
        return False
    return (
        all(record.get(key) == value for key, value in expected.items())
        and isinstance(record.get("saved_utc"), str)
        and bool(record["saved_utc"])
        and record["reference_run"] == snapshot.report_reference
    )


class ManifestChangedError(RuntimeError):
    pass


def _same_values(left, right):
    # Reuse the UI manifest save's exact TOML round-trip contract.
    from darsia.fluidflower.series_method import _same_values as same

    return same(left, right)


def save_reference(
    series_root, series_id, snapshot, reference, override, *, expected_bytes, provider
):
    """One explicit save; source and manifest are rechecked before replacement."""
    path = Path(series_root) / "series.toml"
    original, document, runs = read_series_context(series_root, series_id)
    if original != expected_bytes:
        raise ManifestChangedError(
            "series.toml changed since this step was loaded; reload before saving."
        )
    fresh = provider(snapshot.source_folder, series_id, runs)
    if fresh != snapshot:
        raise ValueError("Saved normalisation evidence changed before saving.")
    updated = deepcopy(document)
    existing = updated.setdefault("phase_normalisation", {})
    if not isinstance(existing, dict):
        raise ValueError("phase_normalisation must be a TOML table.")
    existing.update(record_for(fresh, reference, override))
    existing["saved_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    text = toml.dumps(updated)
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    raw = (
        b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    ) + text.encode("utf-8")
    if not _same_values(updated, tomllib.loads(raw.decode("utf-8-sig"))):
        raise ValueError("Saving this reference would change unrelated TOML values.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        if path.read_bytes() != original:
            raise ManifestChangedError(
                "series.toml changed while saving; the existing manifest was preserved."
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return raw


def _read_normalisation_packet(series_root, series_id, source_folder, provider):
    """Finite metadata operation; no Qt objects or scientific payloads are used."""
    raw, document, runs = read_series_context(series_root, series_id)
    packet = dict(raw=raw, document=document, runs=runs, snapshot=None, imported=None,
                  source_folder=source_folder, error="")
    try:
        saved = document.get("phase_normalisation") or {}
        if not isinstance(saved, dict):
            raise ValueError("The saved normalisation section is not a table.")
        from .saved_v5_source_import import is_source_import, read_source_import
        if is_source_import(saved):
            packet["imported"] = read_source_import(saved, series_id, runs)
        else:
            if source_folder is None and isinstance(saved.get("source_folder"), str):
                source_folder = Path(saved["source_folder"])
            if source_folder is None:
                raise ValueError("Open the saved lighting evidence for this series.")
            packet["source_folder"] = source_folder
            packet["snapshot"] = provider(source_folder, series_id, runs)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        packet["error"] = str(error)
    return packet


class NormalisationStep(QWidget):
    readiness_changed = Signal(bool)

    def __init__(self, parent=None, *, provider=None, allow_save=True):
        super().__init__(parent)
        self.provider = provider or PublishedV5Provider()
        self.allow_save = allow_save
        self.series_root, self.series_id, self.source_folder = None, "", None
        self.snapshot, self.manifest_bytes, self.document = None, None, {}
        self.ready = False
        self.source_import = None
        self.thread = self.worker = None
        self._epoch, self._pending, self._closing = 0, None, False
        QCoreApplication.instance().aboutToQuit.connect(self.wait_for_workers)
        outer = QVBoxLayout(self)
        from .image_normalisation_step import ImageNormalisationChoice
        self.method_choice = ImageNormalisationChoice(allow_save=allow_save, owns_context=False)
        self.method_choice.choice_saved.connect(self.reload)
        self.method_choice.busy_changed.connect(self._choice_busy_changed)
        outer.addWidget(self.method_choice)
        from .normalisation_replay_step import NormalisationReplayStep
        self.reproduction = NormalisationReplayStep(allow_run=allow_save)
        self.reproduction.busy_changed.connect(self._choice_busy_changed)
        outer.addWidget(self.reproduction)
        self.status = QLabel("Missing — choose an experimental series.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.source_label = QLabel("No saved lighting evidence selected")
        self.source_label.setWordWrap(True)
        outer.addWidget(self.source_label)
        actions = QHBoxLayout()
        self.choose_source = QPushButton("Open saved lighting results…")
        self.choose_source.clicked.connect(self._choose_source)
        actions.addWidget(self.choose_source)
        self.reload_button = QPushButton("Reload saved status")
        self.reload_button.clicked.connect(self.reload)
        actions.addWidget(self.reload_button)
        outer.addLayout(actions)
        self.imported_fields = QTableWidget(0, 3)
        self.imported_fields.setHorizontalHeaderLabels(("Saved field", "Literal saved value", "Bound source pointer"))
        self.imported_fields.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.imported_fields.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.imported_fields.horizontalHeader().setStretchLastSection(True)
        self.imported_fields.hide()
        outer.addWidget(self.imported_fields)
        self.ranking_note = QLabel(
            "Historical chemistry ranking is evidence only; it is not a light-only reference proposal."
        )
        self.ranking_note.setWordWrap(True)
        outer.addWidget(self.ranking_note)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            (
                "Raw rank",
                "Run",
                "Score",
                "Light",
                "S",
                "Residual all / G / chroma",
                "Reference eligibility",
            )
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.table, 1)
        line = QHBoxLayout()
        self.override = QCheckBox("Choose another historical reference")
        self.override.toggled.connect(self._selection_changed)
        line.addWidget(self.override)
        self.reference = QComboBox()
        self.reference.currentIndexChanged.connect(self._selection_changed)
        line.addWidget(self.reference, 1)
        self.save_button = QPushButton("Save historical reference")
        self.save_button.clicked.connect(self._save)
        line.addWidget(self.save_button)
        outer.addLayout(line)
        self.provenance = QLabel(
            "Saved source snapshots are shown as historical evidence; current raw protocols are not re-read or re-ranked."
        )
        self.provenance.setWordWrap(True)
        outer.addWidget(self.provenance)
        from .native_normalisation_layout import install_normalisation_layout
        install_normalisation_layout(self)
        self._update_controls()

    def set_context(self, series_root, series_id):
        changed = (str(series_root), str(series_id)) != (
            str(self.series_root),
            self.series_id,
        )
        self.series_root = Path(series_root) if series_root else None
        self.series_id = str(series_id or "")
        if changed:
            self.source_folder = None
        self.reload()

    def set_source(self, folder):
        """Explicit source preview or capture-provider injection; never a save."""
        self.source_folder = Path(folder) if folder else None
        self.reload()

    def _choose_source(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Published saved lighting evidence folder"
        )
        if folder:
            self.set_source(folder)

    def _emit_ready(self, ready):
        self._legacy_ready = bool(ready)
        self._publish_ready()

    def prepared_completion(self, complete):
        """A completed Setup image preparation for the saved settings is ready too."""
        self._prepared_ready = bool(complete)
        self._publish_ready()

    def _publish_ready(self):
        self.ready = bool(getattr(self, "_legacy_ready", False) or getattr(self, "_prepared_ready", False))
        self.readiness_changed.emit(self.ready)

    def _context(self):
        return self._epoch, str(self.series_root or ""), self.series_id

    def reload(self):
        if self._closing:
            return
        self._epoch += 1
        self._pending = None
        self.snapshot, self.manifest_bytes, self.document = None, None, {}
        self.method_choice.set_context(self.series_root, self.series_id)
        self.reproduction.set_context(self.series_root, self.series_id)
        self.source_import = None
        self.provenance.setText("Saved source snapshots are shown as historical evidence; current raw protocols are not re-read or re-ranked.")
        self.ranking_note.setText("Historical chemistry ranking is evidence only; it is not a light-only reference proposal.")
        self.imported_fields.hide()
        self.imported_fields.setRowCount(0)
        self.table.show()
        self.override.show()
        self.reference.show()
        self.save_button.show()
        self.table.setRowCount(0)
        with QSignalBlocker(self.reference), QSignalBlocker(self.override):
            self.reference.clear()
            self.override.setChecked(False)
        self._emit_ready(False)
        if not self.series_root or not self.series_id:
            self.status.setText("Missing — choose an experimental series first.")
            self._update_controls()
            return
        root, series = str(self.series_root), self.series_id
        folder, provider = self.source_folder, self.provider
        self.status.setText("Reading the selected series lighting evidence…")
        self._start("read", self._context(), lambda: _read_normalisation_packet(root, series, folder, provider))

    def _start(self, operation, context, function):
        if self._closing:
            return
        if self.thread is not None or not self.method_choice.can_close() or not self.reproduction.can_close():
            self._pending = operation, context, function
            self._update_controls()
            return
        if operation == "read":
            from .ui_metadata_preview import MetadataPreview
            if not hasattr(self, "_metadata_preview"):
                self._metadata_preview = MetadataPreview(self)
                self._metadata_preview.completed.connect(self._metadata_preview_ready)
            self._metadata_preview_context = context
            self._preview_cancelled = False
            self._metadata_preview.start(lambda cancelled: None if cancelled() else function())
            self._update_controls()
            return
        self.cancel_preview()
        from .painting_gap_review_ui import _Read
        self.thread, self.worker = QThread(self), _Read(function)
        self.worker.key = operation, context
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self._retired)
        self._update_controls()
        self.thread.start()

    def _metadata_preview_ready(self, packet, error):
        if self._closing or self._metadata_preview_context != self._context():
            return
        if error:
            self.status.setText("Missing historical metadata - " + error)
            self._emit_ready(False)
        elif packet is not None:
            self._install_packet(packet)
        self._update_controls()

    def cancel_preview(self):
        preview = getattr(self, "_metadata_preview", None)
        if preview is not None:
            self._preview_cancelled = getattr(self, "_preview_cancelled", False) or preview.busy
            preview.cancel()

    def hideEvent(self, event):
        self.cancel_preview()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_preview_cancelled", False) and not self._closing:
            self._preview_cancelled = False
            self.reload()

    def _choice_busy_changed(self, busy):
        if not busy and self.thread is None and self.method_choice.can_close() and self.reproduction.can_close() and self._pending is not None and not self._closing:
            pending, self._pending = self._pending, None
            self._start(*pending)
        self._update_controls()
        if self._closing and self.can_close():
            self.close()

    @Slot()
    def _retired(self, expected=None):
        expected = self.sender() if expected is None else expected
        if self.thread is None or self.thread is not expected:
            return
        thread, worker = self.thread, self.worker
        thread.wait()
        self.thread = self.worker = None
        if not self._closing and worker.key[1] == self._context():
            if worker.error:
                self.status.setText("Missing — " + worker.error)
                self.method_choice.status.setText("Method metadata unavailable — " + worker.error)
                self._emit_ready(False)
            else:
                self._install_packet(worker.result)
        del worker
        thread.deleteLater()
        pending, self._pending = self._pending, None
        if pending is not None and not self._closing:
            self._start(*pending)
        self._update_controls()
        if self._closing:
            self.close()

    def _install_packet(self, packet):
        self.manifest_bytes, self.document = packet["raw"], packet["document"]
        runs = tuple(packet["runs"])
        self.method_choice.install_snapshot(self.series_root, self.series_id,
            (self.manifest_bytes, self.document, runs))
        self.reproduction.install_snapshot(runs, document=self.document)
        self.source_folder = packet["source_folder"]
        try:
            if packet["error"]:
                raise ValueError(packet["error"])
            saved = self.document.get("phase_normalisation") or {}
            if packet["imported"] is not None:
                self._show_source_import(saved, runs, packet["imported"])
                self._update_controls()
                return
            self.snapshot = packet["snapshot"]
            self.source_folder = self.snapshot.source_folder
            self.source_label.setText(
                f"Saved lighting evidence · {self.snapshot.snapshot_date}\n{self.source_folder}"
            )
            self.table.setRowCount(len(self.snapshot.rows))
            for index, row in enumerate(self.snapshot.rows):
                cells = (
                    row["raw_rank"],
                    row["run"].upper(),
                    row["score"],
                    "Same light" if row["light_class"] == "same" else "New lamps",
                    row["S"],
                    " / ".join(
                        str(row[key])
                        for key in ("residual_all", "residual_G", "residual_chroma")
                    ),
                    "Eligible" if row["eligible"] else row["exclusion"],
                )
                for column, value in enumerate(cells):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(
                        f"Saved protocol: {row['protocol_file']}\n{row['protocol_notes']}"
                    )
                    self.table.setItem(index, column, item)
            self.ranking_note.setText(
                f"Historical chemistry order and reference {self.snapshot.suggested_reference.upper()} are retained literally. "
                "They are not a current light-only proposal. The method choice above uses no chemistry ranking; "
                "a new automatic proposal requires current ColorChecker and light-field evidence."
            )
            chosen = (
                saved.get("reference_run")
                if saved.get("reference_run") in self.snapshot.runs
                else self.snapshot.suggested_reference
            )
            with QSignalBlocker(self.reference), QSignalBlocker(self.override):
                for run in self.snapshot.runs:
                    self.reference.addItem(run.upper(), run)
                self.reference.setCurrentIndex(self.reference.findData(chosen))
                self.override.setChecked(saved.get("selection") == "override")
            ready = saved_ready(saved, self.snapshot)
            if ready:
                self.status.setText(
                    f"Done — saved lighting evidence, reference {chosen.upper()} · saved {saved['saved_utc']}"
                )
            elif chosen != self.snapshot.report_reference:
                self.status.setText(
                    "Missing — saved results use another reference. Changing the choice does not recompute corrections."
                )
            else:
                self.status.setText(
                    "Not done — saved evidence is verified; explicitly save the reference to bind it to this series."
                )
            self._emit_ready(ready)
        except (AttributeError, ValueError, KeyError, TypeError) as error:
            self.snapshot = None
            self.source_label.setText(str(self.source_folder or "No saved lighting evidence selected"))
            self.status.setText("Missing — " + str(error))
            self._emit_ready(False)
        self._update_controls()

    def metadata_busy(self):
        return bool(getattr(getattr(self, "_metadata_preview", None), "busy", False))

    def can_close(self):
        return self.thread is None and self.method_choice.can_close() and self.reproduction.can_close()

    def wait_for_workers(self):
        self.cancel_preview()
        self._closing, self._pending = True, None
        self._epoch += 1
        self._update_controls()
        self.method_choice.wait_for_workers()
        self.reproduction.wait_for_workers()
        if self.thread is not None:
            thread = self.thread
            thread.wait()
            self._retired(thread)

    def closeEvent(self, event):
        self.cancel_preview()
        self._closing, self._pending = True, None
        self._epoch += 1
        self.reproduction.cancel_work()
        self._update_controls()
        if not self.can_close():
            self.status.setText("The lighting operation is finishing; close after it returns.")
            event.ignore()
            return
        super().closeEvent(event)

    def _show_source_import(self, saved, runs, imported):
        from darsia.fluidflower.saved_v5_source_import import import_label
        self.source_import = imported
        self.source_label.setText(import_label(imported))
        self.status.setText("Imported saved lighting parameters; ranking and numerical readiness are not available from the bound sources.")
        self.ranking_note.setText("Ranking, score, majority light class and automatic reference proposal: not available from the bound sources. No ranking or normalisation was recomputed.")
        rows = []
        for name, value in imported["v5_raw_photometry"].items():
            bound = imported["field_sources"]["v5_raw_photometry"]
            rows.append((name, value, bound["source"]["sha256"] + bound["pointer"] + "/" + name))
        for name, value in imported["materialization"].items():
            bound = imported["field_sources"]["materialization"]
            rows.append((name, value, bound["source"]["sha256"] + "/" + name))
        self.imported_fields.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, value in enumerate(row):
                text = json.dumps(value, ensure_ascii=False) if column == 1 else str(value)
                self.imported_fields.setItem(index, column, QTableWidgetItem(text))
        self.table.hide()
        self.override.hide()
        self.reference.hide()
        self.save_button.hide()
        self.imported_fields.show()
        self.provenance.setText("Read-only source import. Provisional, validation-only and no-physical-geometry flags are retained literally. Saved models and mass-calibration values remain inspectable; this is not a completed normalisation or training admission.")
        self._emit_ready(False)

    def _update_controls(self):
        busy = self.thread is not None or self.metadata_busy()
        self.method_choice.set_owner_busy(busy or self._closing or not self.reproduction.can_close())
        self.reproduction.set_owner_busy(busy or self._closing or not self.method_choice.can_close())
        idle = not busy and not self._closing and self.method_choice.can_close() and self.reproduction.can_close()
        self.choose_source.setEnabled(self.source_import is None and idle)
        self.reload_button.setEnabled(idle)
        self.choose_source.setToolTip(
            "This source import has no published ranking; the bound literal fields are shown below."
            if self.source_import is not None else ""
        )
        usable = self.snapshot is not None and self.manifest_bytes is not None and idle
        saved_input_locked = bool((self.document or {}).get("phase_normalisation", {}).get("v5_raw_photometry"))
        self.reference.setEnabled(usable and self.override.isChecked() and not saved_input_locked)
        self.override.setEnabled(usable and not saved_input_locked)
        self.save_button.setEnabled(usable and self.allow_save and not saved_input_locked)
        missing = ("Choose an experimental series first." if not self.series_root or not self.series_id
            else "Open saved lighting results for this series, then reload the saved status.")
        self.override.setToolTip("Choose a different saved reference for this series." if usable else missing)
        self.reference.setToolTip(missing if not usable else
            "Enable Choose another historical reference to inspect a different saved choice."
            if not self.override.isChecked() else "Choose a saved reference.")
        self.save_button.setToolTip("Save the selected reference for this series." if usable else missing)
        if not self.allow_save:
            self.save_button.setToolTip(
                "Read-only preview; no series manifest is changed."
            )
        if saved_input_locked:
            self.save_button.setToolTip("This historical result retains its bound reference. Choose the current method above and prepare new images before retraining.")

    def _selection_changed(self, *_args):
        if self.snapshot is not None and not self.override.isChecked():
            with QSignalBlocker(self.reference):
                self.reference.setCurrentIndex(
                    self.reference.findData(self.snapshot.suggested_reference)
                )
        self._update_controls()
        if self.snapshot is not None:
            self._emit_ready(False)
            self.status.setText(
                "Not done — reference choice changed; save explicitly. No corrections are recalculated here."
            )

    def _save(self):
        if (not self.allow_save or self.snapshot is None or self.manifest_bytes is None
                or bool((self.document or {}).get("phase_normalisation", {}).get("v5_raw_photometry")) or self.thread is not None
                or self._closing or not self.method_choice.can_close() or not self.reproduction.can_close()):
            return
        root, series = str(self.series_root), self.series_id
        snapshot, original, provider = self.snapshot, self.manifest_bytes, self.provider
        reference, override = self.reference.currentData(), self.override.isChecked()
        folder = self.source_folder
        def save():
            save_reference(root, series, snapshot, reference, override,
                expected_bytes=original, provider=provider)
            return _read_normalisation_packet(root, series, folder, provider)
        self._emit_ready(False)
        self.status.setText("Saving the historical reference…")
        self._start("save", self._context(), save)
