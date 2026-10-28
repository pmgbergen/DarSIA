"""Keep unsaved native settings when opening a series reference.

The original renderer adds default fields that were not in a loaded TOML.
Remember its clean normalized display, rather than ignoring new user fields.
No configuration is written here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib

from PySide6.QtWidgets import QMessageBox

_UNREAD = object()


class ReferenceOpenGuard:
    def __init__(self, window):
        self.window = window
        self.path = ""
        self.config_object = None
        self.clean = None
        self.disk = None
        self.dirty = False
        self._recording = False
        factory = window.settings_factory
        original = factory._render_settings_tabs

        def render(*args, **kwargs):
            self._before_render()
            result = original(*args, **kwargs)
            self._after_render()
            return result

        render.__wrapped__ = original
        factory._render_settings_tabs = render
        original_save = factory.save_settings

        def save(*args, **kwargs):
            # A normal return alone is not proof of a save: bind its exact
            # serialized input to the actual disk file before resetting.
            import toml

            path = str(window.config_file or "")
            expected = toml.loads(toml.dumps(self._view_values()))
            result = original_save(*args, **kwargs)
            if path and path == str(window.config_file or ""):
                try:
                    saved = self._read(path)
                    if saved == expected:
                        self.path = path
                        self.config_object = window.config_dict
                        self.disk = deepcopy(saved)
                        self.clean = deepcopy(window.config_dict)
                        self.dirty = False
                except (OSError, ValueError, TypeError, RuntimeError):
                    pass  # A cancelled/failed/unverified save cannot clear edits.
            return result

        save.__wrapped__ = original_save
        factory.save_settings = save
        if window.config_file:
            self._after_render()

    @staticmethod
    def _read(path):
        with Path(path).open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Configuration exceeds the bounded metadata read.")
        return tomllib.loads(raw.decode("utf-8-sig"))

    def _view_values(self):
        """Read the original form serializer without changing the live config.

        Rendering may supply schema defaults. A read-only dirty check must not
        install those defaults in an unrelated explicit Depth save.
        """
        window = self.window
        actual = window.config_dict
        try:
            window.config_dict = deepcopy(actual)
            window.settings_factory._sync_settings_inputs_to_config_dict()
            return deepcopy(window.config_dict)
        finally:
            window.config_dict = actual

    def _before_render(self):
        if self._recording:
            return
        window = self.window
        same = (
            self.path == str(window.config_file or "")
            and self.config_object is window.config_dict
        )
        if same and self.clean is not None:
            self._recording = True
            try:
                self.dirty |= self._view_values() != self.clean
            finally:
                self._recording = False
        else:
            self.dirty = False  # An explicit successful load supplied a new dict.

    def _after_render(self):
        if self._recording or not self.window.config_file:
            return
        window = self.window
        new = (
            self.path != str(window.config_file)
            or self.config_object is not window.config_dict
        )
        self._recording = True
        try:
            current = self._view_values()
            if new:
                self.path = str(window.config_file)
                self.config_object = window.config_dict
                provider = getattr(window, '_prepared_guard_values', None)
                prepared = provider(self.path) if provider is not None else None
                self.disk = prepared if prepared is not None else self._read(self.path)
                self.dirty = False
            if not self.dirty:
                self.clean = current
        except (OSError, ValueError, TypeError, RuntimeError):
            self.dirty = True
        finally:
            self._recording = False

    def allowed(self, *, notify: bool = True, disk_values=_UNREAD) -> bool:
        window = self.window
        if not window.config_file:
            return True
        try:
            current = self._view_values()
            saved = self._read(window.config_file) if disk_values is _UNREAD else disk_values
            # An explicit user Save may have cleared an earlier dirty state.
            if current == saved:
                self.path = str(window.config_file)
                self.config_object = window.config_dict
                self.disk = deepcopy(saved)
                self.clean = current
                self.dirty = False
                return True
            if (
                self.path == str(window.config_file)
                and self.config_object is window.config_dict
                and not self.dirty
                and current == self.clean
                and saved == self.disk
            ):
                return True
            reason = "The settings shown here have unsaved changes or the file changed on disk."
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            reason = "The current settings could not be checked: " + str(error)
        message = (
            reason
            + " Save or reload the current configuration first. The series, run and reference were not changed."
        )
        if notify:
            window.print_log(message)
            QMessageBox.warning(window, "Save current configuration first", message)
        return False
