"""Explicit corrected-grid opt-in around the original rig setup dispatcher.

Raw configurations use the original class and original dispatcher.  Corrected
configurations keep the entire original Rig.setup orchestration and select the
already existing False APIs through a local subclass, never a global patch.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
from pathlib import Path
import tomllib

import numpy as np

import darsia
from darsia.presets.workflows.config.utils import _get_section_from_toml
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_rig import setup_rig as _original_setup_rig


def configured_frame(path: Path | list[Path]) -> str:
    """Read the opt-in without modifying the original configuration loaders."""
    try:
        labeling = _get_section_from_toml(path, "labeling")
    except KeyError:
        return "raw"  # Original depth-only commands need neither map section.
    frame = labeling.get("frame", "raw")
    if frame not in ("raw", "rig-corrected"):
        raise ValueError(f"Unsupported labeling.frame: {frame!r}.")
    if frame == "raw":
        return frame
    # FaciesConfig.load interprets every key except props/path as a numeric ID.
    # Only labeling.frame carries this joint labels/facies opt-in.
    facies = _get_section_from_toml(path, "facies")
    if "frame" in facies:
        raise ValueError(
            "Use labeling.frame for the joint labels/facies frame; "
            "facies.frame is not supported by the original FaciesConfig loader."
        )
    if frame == "rig-corrected":
        if not str(labeling.get("labels") or "").strip():
            raise ValueError("Corrected-frame setup requires an explicit labels file.")
        if not str(facies.get("path") or "").strip():
            raise ValueError("Corrected-frame setup requires an explicit facies file.")
    return frame


def _require_exact_grid(path: Path, baseline, field: str) -> None:
    """Reject mismatched grids before the original False API can resize them.

    No coordinates, labels or values are changed here.  Both image reads use the
    existing reader; the second read belongs to the unchanged original method.
    Exact metadata comparisons have no geometric tolerance or inferred values.
    """
    image = darsia.imread(path)
    _require_exact_grid_image(image, baseline, path, field)


def _require_exact_grid_image(image, baseline, path: Path, field: str) -> None:
    """Compare the same grid properties for decoded images or archive headers."""
    # Generic Image archives may keep scalar=False without a range axis.
    # This contract requires one value per spatial voxel, independent of that flag.
    if image.series or image.img.ndim != image.space_dim:
        raise ValueError(f"Corrected {field} must be a single scalar image: {path}")
    for name in ("space_dim", "indexing"):
        if getattr(image, name) != getattr(baseline, name):
            raise ValueError(f"Corrected {field} grid mismatch ({name}): {path}")
    if tuple(image.img.shape) != tuple(baseline.img.shape[: baseline.space_dim]):
        raise ValueError(f"Corrected {field} grid mismatch (shape): {path}")
    for name in ("dimensions", "origin", "voxel_size"):
        if not np.array_equal(getattr(image, name), getattr(baseline, name)):
            raise ValueError(f"Corrected {field} grid mismatch ({name}): {path}")


def corrected_rig_class(cls: type[Rig]) -> type[Rig]:
    """Select existing False APIs without copying or replacing Rig.setup."""

    class CorrectedFrameRig(cls):
        def setup_labels(self, path, apply_corrections=False, log=None):
            from .rig_analysis_domain import setup_prepared_analysis_field
            if setup_prepared_analysis_field(self, "labels", path, log=log):
                return
            _require_exact_grid(path, self.baseline, "labels")
            return super().setup_labels(path=path, apply_corrections=False, log=log)

        def setup_facies(
            self, path, apply_corrections=False, log=None, show_plot=False
        ):
            from .rig_analysis_domain import setup_prepared_analysis_field
            if setup_prepared_analysis_field(self, "facies", path, log=log, show_plot=show_plot):
                return
            _require_exact_grid(path, self.baseline, "facies")
            return super().setup_facies(
                path=path,
                apply_corrections=False,
                log=log,
                show_plot=show_plot,
            )

    return CorrectedFrameRig


def _phase_map_build_policy(raw_build_request):
    """Read the selected method from the exact series bound to this Build."""
    if raw_build_request is None:
        return None
    from .rig_raw_build import read_build_request_metadata, _input_bytes, _verify_inputs
    request, _ = read_build_request_metadata(raw_build_request, artifact_kind="raw_rig_baseline")
    manifest = request["series_context"]["manifest"]
    _verify_inputs([manifest])
    payload = _input_bytes(manifest["path"])
    if len(payload) != manifest["size_bytes"] or hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("Build method's bound series manifest changed while reading")
    document = tomllib.loads(payload.decode("utf-8-sig"))
    if document.get("method", {}).get("active") != "phase_model":
        return None
    return dict(format="darsia-build-photometry-policy-v1", method="phase_model",
        series_manifest=deepcopy(manifest), build_colour_correction_applied=False,
        build_illumination_correction_applied=False,
        pixel_source="shape_corrected_baseline",
        photometry_stage="Calibration: original RAW full-photograph correction before crop")


def _phase_map_rig_class(cls: type[Rig], policy: dict) -> type[Rig]:
    """Keep geometric Build separate from the phase method's RAW photometry."""
    class PhaseMapRig(cls):
        def setup_color_corrections(self, corrections_config=None, log=None, show_plot=False):
            self.color_corrections = []
            self.baseline = self.shape_corrected_baseline.copy()
            self._build_photometry_policy = deepcopy(policy)
            logging.getLogger(__name__).info(
                "Phase-map Build retains the shape-corrected RAW baseline; "
                "colour and illumination are applied in Calibration before crop.")

    return PhaseMapRig


def setup_rig(cls: type[Rig], path: Path | list[Path], show: bool = False,
              *, raw_build_request: dict | None = None) -> None:
    """Keep frame dispatch and forward an explicit RAW request to the original setup."""
    frame = configured_frame(path)
    if frame == "rig-corrected":
        from darsia.fluidflower.corrected_setup import require_corrected_export_for_build

        require_corrected_export_for_build(path)
    selected_class = cls if frame == "raw" else corrected_rig_class(cls)
    policy = _phase_map_build_policy(raw_build_request)
    if policy is not None:
        selected_class = _phase_map_rig_class(selected_class, policy)
    return _original_setup_rig(selected_class, path, show,
        **({"raw_build_request": raw_build_request} if raw_build_request is not None else {}))


def main() -> None:
    """Native-only CLI adapter; upstream CLI and raw actions stay unchanged."""
    from darsia.presets.workflows.user_interface_setup import (
        build_parser_for_setup,
        preset_setup,
    )

    args = build_parser_for_setup().parse_args()
    paths = [Path(path) for path in args.config]
    if configured_frame(paths) == "raw":
        return preset_setup()
    # A corrected map must not be silently regenerated by --all/segmentation/
    # facies before Build.  Those actions remain unqualified for this opt-in.
    if args.rig and not any(
        (args.all, args.depth, args.segmentation, args.facies, args.delete, args.info)
    ):
        return setup_rig(Rig, paths, args.show)
    raise ValueError(
        "Corrected-frame geometry supports Build run model only. Full setup, "
        "segmentation and facies generation are not qualified for this frame."
    )


if __name__ == "__main__":
    main()
