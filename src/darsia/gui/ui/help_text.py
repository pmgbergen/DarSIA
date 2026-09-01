"""Central registry of help text for sidebar items, organized by category."""

HELP_TEXT_SETUP = {
    "depth": "Analyze depth images to create depth maps from a baseline.",
    "segmentation": "Segment colored images to identify fluid phases and interfaces.",
    "facies": "Define and configure geological layers (facies) for analysis.",
    "protocols": "Configure imaging protocols, timing, and data collection schedules.",
    "rig": "Initialize experimental rig geometry and hardware parameters.",
    "crop": (
        "Interactively select the four corners of the FluidFlower region "
        "for crop correction using matplotlib point-picker."
    ),
    "all": "Execute all setup steps in sequence.",
    "show_plots": "Display visualization plots during setup execution.",
}

HELP_TEXT_CALIBRATION = {
    "color": "Calibrate color paths for RGB image analysis using color targets.",
    "mass": "Calibrate color intensity to CO₂ mass relationships.",
    "default_mass": "Set default mass calibration parameters for analysis.",
    "delete": "Remove all existing calibration data (cannot be undone).",
    "reset": "Reset mass calibration to default values.",
    "show": "Display calibration plots and verification figures.",
}

HELP_TEXT_ANALYSIS = {
    "cropping": "Crop images to region of interest for focused analysis.",
    "segmentation": "Segment images to identify and map fluid phases.",
    "fingers": "Analyze viscous fingering features at fluid interfaces.",
    "mass": "Quantify CO₂ mass distribution from segmented images.",
    "volume": "Calculate volume metrics from segmented phases.",
    "thresholding": "Apply intensity thresholding for phase separation.",
    "show": "Display analysis results and plots.",
}

HELP_TEXT_HELPER = {
    "color": "Helper tool for color embedding and analysis.",
    "roi": "Define and manage regions of interest for analysis.",
    "roi_viewer": "Visualize and inspect defined regions of interest.",
    "results": "Read and display analysis results.",
    "show": "Display helper tool plots.",
}

HELP_TEXT_UTILS = {
    "build_media": "Create time-series MP4/GIF media from image sequence.",
    "download_data": "Download and cache experimental data locally.",
    "export_calibration": "Export calibration settings to a shareable bundle.",
    "import_calibration": "Import calibration from an external bundle file.",
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
