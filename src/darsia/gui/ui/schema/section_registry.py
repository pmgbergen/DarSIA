"""Registry mapping GUI checkbox IDs to their required config sections.

Each leaf entry points to a workflow entry-point function decorated with
@required_sections(...). The decorator is the single source of truth for what
sections that function requires.

Composite entries are lists of other (action, checkbox_id) keys whose sections should
be unioned together.
"""

import logging

logger = logging.getLogger(__name__)

# Mapping: (action, checkbox_id) -> (module_path, function_name) | list[(action, checkbox_id)]
#
# Leaf entries: (module_path, function_name) — resolves a function decorated with
#   @required_sections(...)
# Composite entries: list of (action, checkbox_id) keys — union of those checkboxes'
#   sections (in order, deduplicated)
CHECKBOX_TO_SECTIONS = {
    # Setup leaf mappings — preparation steps (protocols, crop).
    ("setup", "protocols"): (
        "darsia.presets.workflows.setup.setup_protocols",
        "setup_imaging_protocol",
    ),
    ("setup", "crop"): (
        "darsia.presets.workflows.setup.setup_crop",
        "setup_crop_correction",
    ),
    # Setup composite mapping - full setup
    ("setup", "all"): [
        ("setup", "depth"),
        ("setup", "segmentation"),
        ("setup", "facies"),
        ("setup", "rig"),
    ],
    # Setup leaf mappings — individual steps
    ("setup", "rig"): (
        "darsia.presets.workflows.setup.setup_rig",
        "setup_rig",
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
    ("analysis", "cropping"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "prepare_analysis_context",
    ),
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
    ("analysis", "thresholding"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "prepare_analysis_context",
    ),
    ("analysis", "volume"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "prepare_analysis_context",
    ),
}

# Tab visibility override: limit which sections are shown for specific workflows.
# If a (action, checkbox_id) key is absent, all required sections are shown.
# Used for workflows where some required sections are always pre-populated by
# an earlier step (e.g., Setup) and would just add visual clutter here.
TAB_VISIBILITY = {
    # Setup > Preparation
    ("setup", "protocols"): ("data", "protocols"),
    ("setup", "crop"): ("rig", "corrections", "options"),
    # Setup > Full setup
    ("setup", "all"): (
        "rig",
        "depth",
        "labeling",
        "facies",
        "image_porosity",
        "corrections",
        "options",
    ),
    # Setup > Single steps
    ("setup", "depth"): ("depth", "options"),
    ("setup", "segmentation"): ("labeling", "options"),
    ("setup", "facies"): ("facies", "options"),
    ("setup", "rig"): ("rig", "corrections", "image_porosity", "options"),
    # Calibration > Color & Mass
    ("calibration", "color"): ("color", "calibration", "options"),
    ("calibration", "mass"): ("color", "calibration", "options"),
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


def _is_section_satisfied(config_dict: dict, section: str) -> bool:
    """Check whether a section already has a non-empty value in the GUI's
    TOML-parsed config dict.

    Args:
        config_dict: The GUI's raw config dict (main_window.config_dict).
        section: Section name (e.g., "rig", "calibration.color").

    Returns:
        True if the section key path resolves to a non-empty dict or value;
        False otherwise.
    """
    if not isinstance(config_dict, dict):
        return False
    try:
        value = config_dict
        for key in section.split("."):
            if not isinstance(value, dict) or key not in value:
                return False
            value = value[key]
        # Section is satisfied if the final value is non-empty dict or non-None
        return bool(value) if isinstance(value, dict) else value is not None
    except Exception as e:
        logger.debug(f"Error checking section {section}: {e}")
        return False


def filter_visible_sections(
    action: str,
    checkbox_id: str,
    required_sections: tuple[str, ...],
    config_dict: dict,
) -> tuple[str, ...]:
    """Filter visible sections based on customization, force-showing unmet ones.

    Ensures no required section is hidden if it's not yet satisfied in the
    config. When TAB_VISIBILITY is defined, its tuple order determines display
    order; only displays sections listed in TAB_VISIBILITY unless they are
    currently unsatisfied (missing/empty).

    Args:
        action: Workflow action (e.g., "calibration").
        checkbox_id: Checkbox ID (e.g., "color").
        required_sections: Tuple of required sections from the decorator.
        config_dict: The GUI's raw TOML-parsed config dict
            (main_window.config_dict).

    Returns:
        Filtered tuple of section names to display (always includes unmet ones).
        Order follows TAB_VISIBILITY's declared sequence (if set).
    """
    wanted = TAB_VISIBILITY.get((action, checkbox_id))

    # No customization for this action/checkbox; show all required sections
    if wanted is None:
        return required_sections

    def _include(section: str) -> bool:
        base = section.split(".")[0]
        return base in wanted or not _is_section_satisfied(config_dict, section)

    visible: list[str] = []
    seen: set[str] = set()

    # Pass 1: order by `wanted`'s declared sequence, matching sections by base
    for base in wanted:
        for section in required_sections:
            section_base = section.split(".")[0]
            if section_base == base and section not in seen and _include(section):
                visible.append(section)
                seen.add(section)

    # Pass 2: append any remaining sections not yet added (preserve original order)
    for section in required_sections:
        if section not in seen and _include(section):
            visible.append(section)
            seen.add(section)

    return tuple(visible)
