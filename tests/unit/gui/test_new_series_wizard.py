"""Native New series rendering regression, without creating a series."""

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import pytest
from PySide6.QtCore import QPoint, QSettings, QTimer, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QWizard, QWidget


def _painted_colors(image, widget, wizard):
    """Inspect actual paint under a button, rather than only its visible flag."""
    origin = widget.mapTo(wizard, QPoint(0, 0))
    rectangle = widget.rect().translated(origin)
    assert image.rect().contains(rectangle)
    return {
        image.pixelColor(x, y).rgba()
        for x in range(rectangle.left(), rectangle.right() + 1)
        for y in range(rectangle.top(), rectangle.bottom() + 1)
    }


def _modal_probe(theme, settings_root):
    """Exercise a real modal in a fresh app, isolated from other tests' slots."""
    from darsia.fluidflower.series_wizard import NewSeriesFolderWizard
    from darsia.gui.ui.theme import apply_theme

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, settings_root)
    app = QApplication([])
    app.setOrganizationName("DarSIA-new-series-tests")
    app.setApplicationName(theme)
    messages = []
    observed = {}
    previous_handler = None

    def record_message(kind, context, message):
        messages.append(message)
        if previous_handler is not None:
            previous_handler(kind, context, message)

    parent = QWidget()
    parent.show()
    wizard = NewSeriesFolderWizard(parent)

    def inspect():
        try:
            image = wizard.grab().toImage()
            observed["page_ids"] = wizard.pageIds()
            observed["title"] = wizard.currentPage().title()
            observed["visible"] = wizard.isVisible()
            for name, button_id in (
                ("next", QWizard.WizardButton.NextButton),
                ("cancel", QWizard.WizardButton.CancelButton),
            ):
                button = wizard.button(button_id)
                observed[name] = {
                    "visible": button.isVisible(),
                    "enabled": button.isEnabled(),
                    "colors": len(_painted_colors(image, button, wizard)),
                }
        except BaseException:
            observed["error"] = traceback.format_exc()
        finally:
            wizard.reject()

    try:
        apply_theme(app, theme)
        previous_handler = qInstallMessageHandler(record_message)
        QTimer.singleShot(50, inspect)
        wizard.exec()
        assert "error" not in observed, observed.get("error")
        assert observed["visible"]
        assert observed["page_ids"] == [0, 1, 2]
        assert observed["title"] == "Series"
        assert observed["next"]["visible"]
        assert not observed["next"]["enabled"]
        assert observed["cancel"]["visible"]
        assert observed["cancel"]["enabled"]
        assert observed["next"]["colors"] > 2
        assert observed["cancel"]["colors"] > 2
        assert not any("Failed to obtain HWND for wizard." in m for m in messages)
        print(json.dumps({"theme": theme, "observed": observed, "messages": messages}))
    finally:
        qInstallMessageHandler(previous_handler)
        wizard.close()
        parent.close()
        app.quit()


@pytest.mark.parametrize("theme", ["System", "Light", "Dark"])
def test_new_series_modal_paints_navigation_without_native_handle(theme, tmp_path):
    # A theme change broadcasts to every existing widget/slot. A fresh process
    # keeps this rendering probe independent of widgets owned by other tests.
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        QT_QPA_PLATFORM="offscreen",
        NUMBA_CACHE_DIR=str(tmp_path / "numba"),
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import runpy, sys; "
            "runpy.run_path(sys.argv[1])['_modal_probe'](sys.argv[2], sys.argv[3])",
            str(Path(__file__).resolve()),
            theme,
            str(tmp_path / "settings"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["theme"] == theme
    assert observed["observed"]["page_ids"] == [0, 1, 2]


def test_new_series_all_pages_work_without_finishing(qtbot, tmp_path):
    from PIL import Image

    from darsia.fluidflower.series_wizard import NewSeriesFolderWizard

    experiment = tmp_path / "AC52"
    experiment.mkdir()
    Image.new("RGB", (8, 8), "white").save(experiment / "baseline.jpg")
    wizard = NewSeriesFolderWizard(experiment_folder=tmp_path)
    qtbot.addWidget(wizard)
    wizard.series_page.name.setText("Validation series")
    wizard.series_page.series_id.setText("ValidationSeries")
    wizard.series_page.rig_class.setCurrentIndex(
        wizard.series_page.rig_class.findData("medium")
    )
    wizard.series_page.physical_rig.setCurrentIndex(
        wizard.series_page.physical_rig.findData("FF AC")
    )
    wizard.experiments_page.find_experiments(confirm=False, background=False)
    wizard.show()
    qtbot.waitUntil(wizard.isVisible)
    assert wizard.currentPage().title() == "Series"
    assert wizard.series_page.isComplete()
    wizard.next()
    assert wizard.currentPage().title() == "Experiments"
    assert wizard.experiments_page.isComplete()
    assert wizard.experiments_page.table.rowCount() == 1
    wizard.next()
    assert wizard.currentPage().title() == "Create"
    assert "ac52" in wizard.create_page.summary.text()
    assert wizard.create_page.file_choice.count() >= 4
    assert wizard.button(QWizard.WizardButton.FinishButton).isEnabled()
    assert wizard.button(QWizard.WizardButton.FinishButton).text() == "Create series"
    wizard.reject()
    assert wizard.created is None
    assert not (tmp_path / "DarSIA").exists()
