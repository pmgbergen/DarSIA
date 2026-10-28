"""Command-line entry point for the standalone phase-model studio."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--exit-after-ms", type=int, default=0)
    parser.add_argument("--select-model", type=Path, default=None,
                        help="Select an existing model without narrowing the model library")
    parser.add_argument("--configuration", type=Path, default=None,
                        help="Explicit content-bound parameter DATA for saved-model readers")
    parser.add_argument("--configuration-sha256", default=None,
                        help="SHA256 of --configuration; required together with its path")
    return parser


def select_initial_model(page, root: Path, completed=None):
    """Wait for fresh discovery instead of rejecting a model absent from cache."""
    from PySide6.QtCore import QTimer

    selected = str(root.expanduser().resolve())
    if page.select_validation_model(selected, series_context=True):
        page.tabs.setCurrentWidget(page.evidence_tab)
        if completed:
            completed()
        return None
    # Include the explicitly requested model in fresh discovery without
    # narrowing the library or registering an unverified model permanently.
    page._completed_model_roots[selected] = selected
    page._model_catalog_pending = True
    page._request_async_model_refresh()
    timer = QTimer(page)
    timer.setInterval(100)
    deadline = time.monotonic() + 120
    def attempt():
        if page.select_validation_model(selected, series_context=True):
            timer.stop()
            page.tabs.setCurrentWidget(page.evidence_tab)
            if completed:
                completed()
        elif (time.monotonic() >= deadline or
              (not page._review_sessions_loading and not page._model_catalog_inflight
               and not page._model_catalog_pending)):
            timer.stop()
            page.catalog_loading_status.setText(
                "Could not select the requested model from storage: " + selected +
                ". Refresh models after checking its availability.")
            page.catalog_loading_status.show()
            if completed:
                completed()
    timer.timeout.connect(attempt)
    timer.start()
    return timer




@contextmanager
def _configuration_context(args):
    """Adopt actual application DATA before Studio imports and async discovery.

    The ContextVar covers synchronous readers. The identical bound environment
    document covers newly created Qt/Python threads, which do not inherit that
    context. Prior values are restored when this application invocation ends.
    """
    if (args.configuration is None) != (args.configuration_sha256 is None):
        raise ValueError("--configuration and --configuration-sha256 are required together")
    if args.configuration is None:
        yield None
        return
    from .native.config import load_parameters, configure_parameters, reset_parameters
    path = args.configuration.expanduser().resolve()
    document = load_parameters(path, args.configuration_sha256)
    from .series_native_context import _bind_current_saved_readers
    document, current_code_bindings = _bind_current_saved_readers(document)
    token = configure_parameters(document)
    previous = {name: os.environ.get(name) for name in
                ("DARSIA_NATIVE_CONFIGURATION", "DARSIA_NATIVE_CONFIGURATION_SHA256")}
    try:
        os.environ["DARSIA_NATIVE_CONFIGURATION"] = str(path)
        os.environ["DARSIA_NATIVE_CONFIGURATION_SHA256"] = args.configuration_sha256
        yield dict(path=str(path), sha256=args.configuration_sha256,
                   size_bytes=path.stat().st_size,
                   current_code_bindings=list(current_code_bindings))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_parameters(token)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with _configuration_context(args) as configuration:
        return _run_application(args, configuration=configuration)


def _run_application(args, *, configuration):
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "The DarSIA Phase Model Studio requires PySide6. Install DarSIA "
            "with the 'fluidflower-app' optional dependency."
        ) from exc

    from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
    from darsia.fluidflower.ui.phase_model_studio import PhaseModelStudioWindow
    from darsia.fluidflower.ui.theme import application_icon

    app = QApplication(sys.argv[:1])
    app.setApplicationName("DarSIA Phase Model Studio")
    app.setOrganizationName("DarSIA")
    app.setStyle("Fusion")
    install_interaction_safety(app)
    from darsia.fluidflower.machine_paths import install_machine_path_map

    install_machine_path_map()
    if os.name == "nt":
        app.setFont(QFont("Segoe UI", 10))
    app.setWindowIcon(application_icon())
    window = PhaseModelStudioWindow()
    # Record the actual desktop startup, not a separately constructed test page.
    # This makes missing registrations and stale imported code diagnosable.
    from darsia.fluidflower.ui import model_training
    page = window.training_page
    startup = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "DarSIA" / "logs" / "phase-model-studio"
    startup.mkdir(parents=True, exist_ok=True)
    def record_runtime():
        (startup / f"runtime-{os.getpid()}.json").write_text(json.dumps({
            "pid": os.getpid(), "executable": sys.executable,
            "configuration": configuration,
            "training_page_source": model_training.__file__,
            "training_page_sha256": hashlib.sha256(Path(model_training.__file__).read_bytes()).hexdigest(),
            "settings_file": page._model_filter_settings.fileName(),
            "registered_roots": page._completed_model_roots,
            "discovered_roots": [entry["root"] for entry in page.model_history],
            "selected_model": page.validation_model.currentData(),
            "requested_model": str(args.select_model) if args.select_model else None,
        }, indent=2), encoding="utf-8")
    record_runtime()
    if args.select_model is not None:
        window._initial_selection_timer = select_initial_model(page, args.select_model, record_runtime)
    window.show()

    if args.screenshot is not None:
        target = args.screenshot.expanduser().resolve()

        def save_screenshot() -> None:
            pending = getattr(window, "_initial_selection_timer", None)
            if pending is not None and pending.isActive():
                QTimer.singleShot(200, save_screenshot)
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(target)):
                raise RuntimeError(f"Could not save screenshot to {target}")
            if not args.exit_after_ms:
                app.quit()

        QTimer.singleShot(800, save_screenshot)
    if args.exit_after_ms:
        QTimer.singleShot(max(1, args.exit_after_ms), app.quit)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
