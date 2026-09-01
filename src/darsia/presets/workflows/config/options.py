"""Configuration for runtime options across different workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section, _get_section_from_toml


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

    def load(self, sec: dict) -> "SetupOptions":
        self.show_plots = _get_key(
            sec, "show_plots", required=False, default=False, type_=bool
        )
        return self


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

    def load(self, sec: dict) -> "CalibrationOptions":
        self.show_plots = _get_key(
            sec, "show_plots", required=False, default=False, type_=bool
        )
        return self


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
    random_traverse: bool = field(
        default=False,
        metadata={
            "name": "Random traverse",
            "help": "Process images in random order instead of chronological.",
        },
    )

    def load(self, sec: dict) -> "AnalysisOptions":
        self.show_plots = _get_key(
            sec, "show_plots", required=False, default=False, type_=bool
        )
        self.random_traverse = _get_key(
            sec, "random_traverse", required=False, default=False, type_=bool
        )
        return self


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

    def load(self, sec: dict) -> "HelperOptions":
        self.show_plots = _get_key(
            sec, "show_plots", required=False, default=False, type_=bool
        )
        return self


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

    def load(self, path: Path) -> "OptionsConfig":
        try:
            sec = _get_section_from_toml(path, "options")
        except KeyError:
            # If [options] section is missing, use empty dicts for all sub-sections
            sec = {}

        setup_sec = _get_section(sec, "setup") if "setup" in sec else {}
        self.setup = SetupOptions().load(setup_sec)

        calibration_sec = (
            _get_section(sec, "calibration") if "calibration" in sec else {}
        )
        self.calibration = CalibrationOptions().load(calibration_sec)

        analysis_sec = _get_section(sec, "analysis") if "analysis" in sec else {}
        self.analysis = AnalysisOptions().load(analysis_sec)

        helper_sec = _get_section(sec, "helper") if "helper" in sec else {}
        self.helper = HelperOptions().load(helper_sec)

        return self
