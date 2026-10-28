"""One passive Analysis line for explicit manifest report/table references.

Only supplied series.toml bytes are read. Referenced products are not inspected,
parsed, hashed or opened during navigation. Their saved SHA-256 values remain
explicit source bindings, not a claim of a new verification or calculation.
"""
from __future__ import annotations

from html import escape
from pathlib import Path, PureWindowsPath
import re
import tomllib

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel

MAX_MANIFEST_BYTES = 1024 * 1024


def manifest_references(root, raw):
    """Return only literal frozen-reference links, without following any file."""
    if root is None or raw is None:
        return None
    if not isinstance(raw, bytes) or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Series reference metadata exceeds the display limit.")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    results = document.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("The results section must be a table.")
    reference = results.get("reference")
    if reference is None:
        return None
    if not isinstance(reference, dict) or not isinstance(reference.get("description"), str):
        raise ValueError("The frozen reference needs a description and report/table links.")
    links = {}
    for key in ("report", "table"):
        item = reference.get(key)
        if not isinstance(item, dict):
            raise ValueError("The frozen reference needs explicit report and table descriptors.")
        value, sha = item.get("path"), item.get("sha256")
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("A frozen-reference file path is required.")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("Each frozen-reference link needs its saved SHA-256.")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
            raise ValueError("Frozen references must identify local or NAS file paths.")
        path = Path(value)
        windows = PureWindowsPath(value)
        if (windows.drive or windows.root) and not (path.is_absolute() or windows.is_absolute()):
            raise ValueError("A frozen-reference path must be absolute or series-relative.")
        if not (path.is_absolute() or windows.is_absolute()):
            parts = value.replace("\\", "/").split("/")
            if ".." in parts:
                raise ValueError("Relative frozen-reference paths must remain inside the series.")
            path = Path(root).joinpath(*parts)
        links[key] = {"path": str(path), "sha256": sha}
    return {"description": reference["description"], "links": links}


class FrozenReferenceLine(QLabel):
    """Read-only links; file opening happens only after an explicit user click."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._links = {}
        self._analysis_visible = False
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setOpenExternalLinks(False)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.linkActivated.connect(self._open_reference)
        self.hide()

    def set_reference(self, root, raw):
        self._links = {}
        self.clear()
        self.hide()
        self.setToolTip("")
        try:
            saved = manifest_references(root, raw)
        except (ValueError, UnicodeError, TypeError) as error:
            self.setText("Frozen reference unavailable: " + escape(str(error)))
            self.show_for_analysis(self._analysis_visible)
            return
        if saved is None:
            return
        self._links = saved["links"]
        links = [f'<a href="{key}">{label}</a> (SHA {item["sha256"][:8]}…)'
                 for key, label in (("report", "report"), ("table", "mass table"))
                 for item in (self._links[key],)]
        self.setText("Frozen reference: " + escape(saved["description"]) + " · " + " · ".join(links))
        self.show_for_analysis(self._analysis_visible)
        self.setToolTip("Saved SHA-256 bindings; files open only on an explicit click.\n" +
                        "\n".join(key + ": " + item["path"] + "\nSHA-256 " + item["sha256"]
                                  for key, item in self._links.items()))

    def show_for_analysis(self, visible):
        self._analysis_visible = bool(visible)
        self.setVisible(bool(self._analysis_visible and self.text()))

    def _open_reference(self, key):
        item = self._links.get(key)
        if item is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(item["path"]))
