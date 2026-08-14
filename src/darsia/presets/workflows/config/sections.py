"""Decorators and utilities for declaring GUI-required config sections in workflow.

This module provides a single source of truth for which FluidFlowerConfig sections each
workflow entry point requires. The decorator both:
1. Drives the entry point's own config.check(*sections) call (runtime enforcement)
2. Provides the GUI with static section info to populate the settings panel

So the declared sections and the enforced sections are the same object — they can't drift.
"""

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def required_sections(
    *sections: str,
    default: tuple[str, ...] | None = None,
    when: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> Callable[[F], F]:
    """Declare which FluidFlowerConfig sections a workflow entry point requires.

    Two forms:

    **Flat (unconditional):**
        @required_sections("depth", "rig")
        def setup_depth_map(...):

    **Branching (discriminated by one argument):**
        @required_sections(
            default=("protocols", "data", "rig"),
            when={"section": {
                "analysis": ("analysis", "protocols", "data", "rig"),
                "calibration": ("calibration", "protocols", "data", "rig"),
            }},
        )
        def prepare_analysis_context(..., section="analysis", ...):

    At runtime, use required_sections(func, **discriminants) to determine the sections
    for a specific call. For the GUI (which can't know the runtime discriminant), use
    gui_display_sections(func) to get the union of all branches.

    Args:
        *sections: Unconditional sections (for flat form only).
        default: Default sections for branching form.
        when: Nested dict {arg_name: {arg_value: (sections,), ...}, ...}.

    Returns:
        Decorator that stores metadata on func.__required_sections__.
    """
    if sections and (default is not None or when is not None):
        raise ValueError(
            "Cannot specify both positional sections and default/when. "
            "Use either flat form (*sections) or branching form (default=..., when=...)."
        )

    if (default is None) != (when is None):
        raise ValueError("Both 'default' and 'when' must be specified together.")

    # Flat form
    if sections:
        declaration = {"form": "flat", "sections": sections}
    # Branching form
    else:
        declaration = {"form": "branching", "default": default, "when": when}

    def decorator(func: F) -> F:
        func.__required_sections__ = declaration  # type: ignore
        return func

    return decorator


def list_required_sections(
    func: Callable[..., Any], **discriminants: Any
) -> tuple[str, ...]:
    """Get the required sections for one specific call to func.

    For branching declarations, pass the discriminant argument by name.

    Args:
        func: A function decorated with @required_sections.
        **discriminants: For branching declarations, {arg_name: arg_value, ...}.

    Returns:
        Tuple of required section names.

    Raises:
        ValueError: If the function is not decorated or if discriminants don't match.
    """
    if not hasattr(func, "__required_sections__"):
        raise ValueError(f"{func.__name__} is not decorated with @required_sections.")

    decl = func.__required_sections__

    if decl["form"] == "flat":
        if discriminants:
            raise ValueError(
                f"{func.__name__} has a flat declaration but discriminants were passed: "
                f"{discriminants}"
            )
        return decl["sections"]

    # Branching form
    if not discriminants:
        # No discriminants provided; use default
        return decl["default"]

    # Exactly one discriminant should be provided
    if len(discriminants) != 1:
        raise ValueError(
            f"{func.__name__} expects exactly one discriminant, got {len(discriminants)}: "
            f"{list(discriminants.keys())}"
        )

    arg_name, arg_value = next(iter(discriminants.items()))
    if arg_name not in decl["when"]:
        raise ValueError(
            f"Unknown discriminant '{arg_name}' for {func.__name__}. "
            f"Expected one of: {list(decl['when'].keys())}"
        )

    branches = decl["when"][arg_name]
    if arg_value not in branches:
        return decl["default"]  # Fall back to default if value not in branches

    return branches[arg_value]


def gui_display_sections(func: Callable[..., Any]) -> tuple[str, ...]:
    """Get all sections potentially required by func for GUI display.

    For flat declarations, returns the unconditional sections. For branching
    declarations, returns the union of default and all branch sections (in order,
    deduplicated) — a safe superset since the GUI doesn't know the runtime
    discriminant ahead of time.

    Args:
        func: A function decorated with @required_sections.

    Returns:
        Tuple of all potentially required section names.

    Raises:
        ValueError: If the function is not decorated.
    """
    if not hasattr(func, "__required_sections__"):
        raise ValueError(f"{func.__name__} is not decorated with @required_sections.")

    decl = func.__required_sections__

    if decl["form"] == "flat":
        return decl["sections"]

    # Branching form: union of all branches and default
    seen = set()
    result = []

    # Add default first
    for section in decl["default"]:
        if section not in seen:
            result.append(section)
            seen.add(section)

    # Add all branch sections
    for branches in decl["when"].values():
        for branch_sections in branches.values():
            for section in branch_sections:
                if section not in seen:
                    result.append(section)
                    seen.add(section)

    return tuple(result)
