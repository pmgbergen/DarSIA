"""Registry mapping GUI checkbox IDs to their required config sections.

Each leaf entry points to a workflow entry-point function decorated with
@required_sections(...). The decorator is the single source of truth for what
sections that function requires.

Composite entries are lists of other (action, checkbox_id) keys whose sections should
be unioned together.
"""

# Mapping: (action, checkbox_id) -> (module_path, function_name) | list[(action, checkbox_id)]
#
# Leaf entries: (module_path, function_name) — resolves a function decorated with
#   @required_sections(...)
# Composite entries: list of (action, checkbox_id) keys — union of those checkboxes'
#   sections (in order, deduplicated)
CHECKBOX_TO_SECTIONS = {
    # Setup leaf mappings — point to the functions the GUI's setup.py::run_setup() calls
    ("setup", "protocols"): (
        "darsia.presets.workflows.setup.setup_protocols",
        "setup_imaging_protocol",
    ),
    ("setup", "depth"): (
        "darsia.presets.workflows.setup.setup_depth",
        "setup_depth_map",
    ),
    ("setup", "segmentation"): (
        "darsia.presets.workflows.setup.setup_labeling",
        "segment_colored_image",
    ),
    ("setup", "facies"): (
        "darsia.presets.workflows.setup.setup_facies",
        "setup_facies",
    ),
    ("setup", "rig"): (
        "darsia.presets.workflows.setup.setup_rig",
        "setup_rig",
    ),
    # Setup composite mapping: "all" runs depth + segmentation + facies + rig (no protocol)
    # Mirrors setup.py:104-115 run_setup() logic exactly.
    ("setup", "all"): [
        ("setup", "depth"),
        ("setup", "segmentation"),
        ("setup", "facies"),
        ("setup", "rig"),
    ],
    # Calibration leaf mappings
    ("calibration", "color"): (
        "darsia.presets.workflows.calibration.calibration_color_paths",
        "calibration_color_paths_from_context",
    ),
    ("calibration", "mass"): (
        "darsia.presets.workflows.calibration.calibration_color_to_mass_analysis",
        "calibration_color_to_mass_analysis_from_context",
    ),
    # Analysis leaf mappings — all point to the same branching function
    ("analysis", "fingers"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "prepare_analysis_context",
    ),
    ("analysis", "mass"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "prepare_analysis_context",
    ),
    ("analysis", "segmentation"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "prepare_analysis_context",
    ),
}


def get_required_sections(action: str, checkbox_id: str) -> tuple[str, ...] | None:
    """Get the required config sections for a checkbox.

    Handles both leaf entries (decorated functions) and composite entries (recursively
    resolve a list of other checkbox IDs and union their sections).

    Args:
        action: Workflow action (setup, calibration, analysis)
        checkbox_id: Checkbox ID (e.g., depth, color, fingers)

    Returns:
        Tuple of section names, or None if checkbox is not registered.

    Raises:
        ValueError: If a registered function is missing the @required_sections
            decorator or if the decorator is misconfigured.
    """
    key = (action, checkbox_id)
    if key not in CHECKBOX_TO_SECTIONS:
        return None

    entry = CHECKBOX_TO_SECTIONS[key]

    # Composite entry: list of other (action, checkbox_id) keys
    if isinstance(entry, list):
        all_sections = []
        seen = set()
        for sub_action, sub_id in entry:
            sub_sections = get_required_sections(sub_action, sub_id)
            if sub_sections:
                for section in sub_sections:
                    if section not in seen:
                        all_sections.append(section)
                        seen.add(section)
        return tuple(all_sections) if all_sections else None

    # Leaf entry: (module_path, function_name) — get sections from decorated function
    module_path, function_name = entry
    try:
        import importlib

        from darsia.presets.workflows.config.sections import gui_display_sections

        module = importlib.import_module(module_path)
        func = getattr(module, function_name, None)
        if func is None:
            raise ValueError(f"Function {function_name} not found in {module_path}")
        sections = gui_display_sections(func)
        return sections
    except (ImportError, AttributeError, ValueError) as e:
        # Re-raise ValueError as-is (includes missing decorator), wrap others
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error loading sections for {action}.{checkbox_id}: {e}")
