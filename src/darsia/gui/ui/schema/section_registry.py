"""Registry mapping GUI checkbox IDs to their required config sections.

Each entry points to either:
- A module and the REQUIRED_SECTIONS constant that lives next to that module's
  workflow entry point's config.check() call (leaf mapping).
- A composite list of other (action, checkbox_id) keys whose sections should be
  unioned together (composite mapping).
"""

# Mapping: (action, checkbox_id) -> (module_path, const_name) | list[(action, checkbox_id)]
#
# Leaf entries: (module_path, const_name) — resolves a REQUIRED_SECTIONS constant
# Composite entries: list of (action, checkbox_id) keys — union of those checkboxes'
#   sections (in order, deduplicated)
CHECKBOX_TO_SECTIONS = {
    # Setup leaf mappings
    ("setup", "protocol"): (
        "darsia.presets.workflows.setup.setup_protocols",
        "REQUIRED_SECTIONS",
    ),
    ("setup", "depth"): (
        "darsia.presets.workflows.setup.setup_depth",
        "REQUIRED_SECTIONS",
    ),
    ("setup", "segmentation"): (
        "darsia.presets.workflows.setup.setup_labeling",
        "REQUIRED_SECTIONS",
    ),
    ("setup", "facies"): (
        "darsia.presets.workflows.setup.setup_facies",
        "REQUIRED_SECTIONS",
    ),
    ("setup", "rig"): ("darsia.presets.workflows.setup.setup_rig", "REQUIRED_SECTIONS"),

    # Setup composite mapping: "all" runs depth + segmentation + facies + rig (no protocol)
    # Mirrors setup.py:104-115 run_setup() logic exactly.
    ("setup", "all"): [
        ("setup", "depth"),
        ("setup", "segmentation"),
        ("setup", "facies"),
        ("setup", "rig"),
    ],

    # Calibration leaf mappings (dormant for now, will be fixed separately)
    ("calibration", "color"): (
        "darsia.presets.workflows.calibration.calibration_color_paths",
        "REQUIRED_SECTIONS",
    ),
    ("calibration", "mass"): (
        "darsia.presets.workflows.calibration.calibration_color_to_mass_analysis",
        "REQUIRED_SECTIONS",
    ),

    # Analysis leaf mappings (dormant for now, will be fixed separately)
    ("analysis", "fingers"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "REQUIRED_SECTIONS",
    ),
    ("analysis", "mass"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "REQUIRED_SECTIONS",
    ),
    ("analysis", "segmentation"): (
        "darsia.presets.workflows.analysis.analysis_context",
        "REQUIRED_SECTIONS",
    ),
}


def get_required_sections(action: str, checkbox_id: str) -> tuple[str, ...] | None:
    """Get the required config sections for a checkbox.

    Handles both leaf entries (import a REQUIRED_SECTIONS constant from a workflow
    module) and composite entries (recursively resolve a list of other checkbox IDs
    and union their sections).

    Args:
        action: Workflow action (setup, calibration, analysis)
        checkbox_id: Checkbox ID (e.g., depth, color, fingers)

    Returns:
        Tuple of section names, or None if checkbox is not registered.
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

    # Leaf entry: (module_path, const_name)
    module_path, const_name = entry
    try:
        import importlib

        module = importlib.import_module(module_path)
        sections = getattr(module, const_name, None)
        return sections
    except (ImportError, AttributeError):
        return None
