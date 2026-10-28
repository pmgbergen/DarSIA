"""Command-line entry point for the Qt FluidFlower application."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--exit-after-ms", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "The FluidFlower app requires PySide6. Install DarSIA with the "
            "'fluidflower-app' optional dependency."
        ) from exc

    from darsia.fluidflower.project import ProjectRegistry
    from darsia.fluidflower.ui.interaction_safety import install_interaction_safety
    from darsia.fluidflower.ui.shell import FluidFlowerMainWindow
    from darsia.fluidflower.ui.theme import application_icon

    app = QApplication(sys.argv[:1])
    app.setApplicationName("DarSIA FluidFlower Experiments")
    app.setOrganizationName("DarSIA")
    app.setStyle("Fusion")
    install_interaction_safety(app)
    if os.name == "nt":
        app.setFont(QFont("Segoe UI", 10))
    app.setWindowIcon(application_icon())
    registry = ProjectRegistry(args.registry) if args.registry else ProjectRegistry()
    window = FluidFlowerMainWindow(
        registry=registry,
        initial_project=args.project,
    )
    window.show()

    if args.screenshot is not None:
        target = args.screenshot.expanduser().resolve()

        def save_screenshot() -> None:
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
