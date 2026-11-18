"""Registry category of the sidebar: one item per shared registry.

The registries (data selections, export formats, regions of interest, color
embeddings) hold named entries that the setup, calibration and analysis steps
refer to. They are edited and saved here, not run: selecting an item shows only
that config section (see MainWindow._on_sidebar_selection).
"""

# checkbox_id -> (label, icon, config section)
REGISTRY_ITEMS = {
    "data": ("Data", "fa5s.database", "registry"),
    "format": ("Format", "fa5s.file-image", "format_registry"),
    "roi": ("ROI", "fa5s.vector-square", "roi"),
    "color": ("Color", "fa5s.palette", "color"),
}


class RegistryTab:
    """Tab manager of the Registry sidebar category (no workflow to run)."""

    def __init__(self, main_window):
        self.main_window = main_window

    @staticmethod
    def section_for(checkbox_id: str) -> str:
        """Config section shown for a registry item."""
        return REGISTRY_ITEMS[checkbox_id][2]

    def on_run_clicked(self):
        self.main_window.print_log("Registry entries are edited and saved, not run.")

    def on_abort_clicked(self):
        """Nothing to abort (Stop aborts running workflows globally)."""

    def sidebar_items(self):
        """Return sidebar data structure for the Registry category."""
        from .help_text import get_help_text

        return [
            (
                "Registries",
                [
                    (
                        label,
                        checkbox_id,
                        icon,
                        get_help_text("registry", checkbox_id, label),
                    )
                    for checkbox_id, (label, icon, _section) in REGISTRY_ITEMS.items()
                ],
            ),
        ]
