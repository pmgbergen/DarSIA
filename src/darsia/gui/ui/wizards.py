"""Registry of per-category guided wizards (see e.g. preprocessing_wizard.py).

Each entry maps a sidebar category ("preprocessing", eventually "setup",
"calibration", "analysis", ...) to the dialog class that implements its guided
wizard. Deferred (string) module paths, so registering a wizard here never
forces every other wizard's dependencies to load up front — only the one
actually opened.

To add a wizard for a new category: build the dialog (same shape as
PreprocessingWizardDialog), then add one entry below. The Wizard menu and the
toolbar/Run-menu "current category" wizard button both pick it up automatically
— no other wiring needed.
"""

import importlib

WIZARD_REGISTRY: dict[str, tuple[str, str, str]] = {
    "preprocessing": (
        "darsia.gui.ui.preprocessing_wizard",
        "PreprocessingWizardDialog",
        "Preprocessing",
    ),
}
"""category -> (module path, class name, display label) for that category's wizard."""


def has_wizard(category: str | None) -> bool:
    """Whether `category` currently has a registered wizard."""
    return category in WIZARD_REGISTRY


def load_wizard_dialog_class(category: str):
    """Import and return the wizard dialog class registered for `category`.

    Raises
    ------
    KeyError
        If no wizard is registered for `category`.
    """
    module_path, class_name, _label = WIZARD_REGISTRY[category]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
