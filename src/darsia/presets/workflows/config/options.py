"""Configuration for runtime options across different workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SetupOptions:
    """Options for setup workflows."""

    show_plots: bool = field(
        default=False,
        metadata={
            "name": "Show plots",
            "help": "Display plots during setup execution.",
        },
    )


@dataclass
class CalibrationOptions:
    """Options for calibration workflows."""

    show_plots: bool = field(
        default=False,
        metadata={
            "name": "Show plots",
            "help": "Display plots during calibration execution.",
        },
    )


@dataclass
class AnalysisOptions:
    """Options for analysis workflows."""

    show_plots: bool = field(
        default=False,
        metadata={
            "name": "Show plots",
            "help": "Display plots during analysis execution.",
        },
    )


@dataclass
class HelperOptions:
    """Options for helper workflows."""

    show_plots: bool = field(
        default=False,
        metadata={
            "name": "Show plots",
            "help": "Display plots during helper execution.",
        },
    )


@dataclass
class OptionsConfig:
    """Runtime options for all workflows, grouped by activity."""

    setup: SetupOptions = field(
        default_factory=SetupOptions,
        metadata={"name": "Setup"},
    )
    calibration: CalibrationOptions = field(
        default_factory=CalibrationOptions,
        metadata={"name": "Calibration"},
    )
    analysis: AnalysisOptions = field(
        default_factory=AnalysisOptions,
        metadata={"name": "Analysis"},
    )
    helper: HelperOptions = field(
        default_factory=HelperOptions,
        metadata={"name": "Helper"},
    )
