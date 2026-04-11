"""Configuration for workflow utility options."""

from dataclasses import dataclass
from pathlib import Path

from .utils import _get_key, _get_section_from_toml


@dataclass
class WorkflowUtilsConfig:
    """Configuration for utility workflows."""

    export_calibration_bundle: Path | None = None
    import_calibration_bundle: Path | None = None

    def load(self, path: Path | list[Path]) -> "WorkflowUtilsConfig":
        sec = _get_section_from_toml(path, "utils")
        calibration_sec = sec.get("calibration", {})
        if not isinstance(calibration_sec, dict):
            calibration_sec = {}
        self.export_calibration_bundle = _get_key(
            calibration_sec,
            "export_bundle",
            default=_get_key(
                sec, "export_calibration_bundle", default=None, required=False
            ),
            required=False,
            type_=Path,
        )
        self.import_calibration_bundle = _get_key(
            calibration_sec,
            "import_bundle",
            default=_get_key(
                sec, "import_calibration_bundle", default=None, required=False
            ),
            required=False,
            type_=Path,
        )
        if self.export_calibration_bundle is not None:
            self.export_calibration_bundle = Path(self.export_calibration_bundle)
        if self.import_calibration_bundle is not None:
            self.import_calibration_bundle = Path(self.import_calibration_bundle)
        return self
