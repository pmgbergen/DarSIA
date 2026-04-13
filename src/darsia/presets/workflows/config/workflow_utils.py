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
        flat_export = _get_key(
            sec, "export_calibration_bundle", default=None, required=False
        )
        if flat_export is not None:
            flat_export = Path(flat_export)
        flat_import = _get_key(
            sec, "import_calibration_bundle", default=None, required=False
        )
        if flat_import is not None:
            flat_import = Path(flat_import)
        self.export_calibration_bundle = _get_key(
            calibration_sec,
            "export_bundle",
            default=flat_export,
            required=False,
            type_=Path,
        )
        self.import_calibration_bundle = _get_key(
            calibration_sec,
            "import_bundle",
            default=flat_import,
            required=False,
            type_=Path,
        )
        return self
