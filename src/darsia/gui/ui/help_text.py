"""Central registry of help text for sidebar items, organized by category."""

HELP_TEXT_SETUP = {
    "depth": "Setup depth map from baseline image. TBC",
    "segmentation": "Segment colored image. TBC",
    "facies": "Setup facies. TBC",
    "protocols": "Setup protocol CSV templates. TBC",
    "rig": "Setup and store rig object. TBC",
    "all": "Run all setup steps. TBC",
    "show_plots": "Display plots during execution. TBC",
}

HELP_TEXT_CALIBRATION = {
    "color": "Calibrate color paths. TBC",
    "mass": "Calibrate color to mass analysis. TBC",
    "default_mass": "Set default mass calibration. TBC",
    "delete": "Delete all calibrations (destructive). TBC",
    "reset": "Reset mass calibration settings. TBC",
    "show": "Display plots during calibration. TBC",
}

HELP_TEXT_ANALYSIS = {
    "cropping": "Crop image to region of interest. TBC",
    "segmentation": "Segment images using analysis. TBC",
    "fingers": "Analyze finger features. TBC",
    "mass": "Analyze mass distribution. TBC",
    "volume": "Analyze volume distribution. TBC",
    "thresholding": "Apply thresholding to images. TBC",
    "all": "Analyze all images. TBC",
    "show": "Display plots during analysis. TBC",
}

HELP_TEXT_HELPER = {
    "color": "Color embedding helper. TBC",
    "roi": "ROI helper tool. TBC",
    "roi_viewer": "ROI viewer helper. TBC",
    "results": "Results reader helper. TBC",
    "show": "Display plots during helper. TBC",
}

HELP_TEXT_UTILS = {
    "build_media": "Build protocol-time media (MP4/GIF). TBC",
    "download_data": "Download/cache data. TBC",
    "export_calibration": "Export calibration bundle. TBC",
    "import_calibration": "Import calibration from file. TBC",
}

# Mapping of action -> help text dictionary
_HELP_TEXT_BY_ACTION = {
    "setup": HELP_TEXT_SETUP,
    "calibration": HELP_TEXT_CALIBRATION,
    "analysis": HELP_TEXT_ANALYSIS,
    "helper": HELP_TEXT_HELPER,
    "utils": HELP_TEXT_UTILS,
}


def get_help_text(action: str, checkbox_id: str, label: str = "") -> str:
    """Get help text for a checkbox_id within a category, with fallback to placeholder.

    Args:
        action: The category/action (e.g. "setup", "calibration", "analysis")
        checkbox_id: The checkbox identifier key
        label: Fallback label if no text found in registry

    Returns:
        Help text string, or a placeholder if not found
    """
    help_dict = _HELP_TEXT_BY_ACTION.get(action, {})
    return help_dict.get(checkbox_id, f"Help for {label}. TBC")
