
from darsia.fluidflower import preset_wasserstein_inputs as _preset_io
# TODO: Any connections to darsia.SimpleFluidFlower?
# TODO: integrate into darsia4porotwin.FluidFlower?
# TODO: Clean up concentration analyses. Can be removed?

import json
import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia
from darsia.presets.analysis.porosity import patched_porosity_analysis
from darsia.presets.workflows.config.corrections import (
    CorrectionsConfig,
    IlluminationCorrectionConfig,
    PatchwiseIlluminationCorrectionConfig,
)
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.facies_props import FaciesProps
from darsia.presets.workflows.setup.illustrations import save_scalar_map_illustration

if TYPE_CHECKING:
    from darsia.presets.workflows.config.roi_registry import RoiRegistry

logger = logging.getLogger(__name__)


def _resolve_named_roi_voxels(
    roi_key: str | None,
    roi_registry: "RoiRegistry | None",
    image: darsia.Image,
    as_landmarks: bool = False,
) -> np.ndarray | None:
    """Resolve a named ROI registry key to voxel coordinates for use as a
    shape/color correction's 'roi' config value, as an alternative to
    auto-detecting a color checker via :func:`darsia.find_colorchecker`.

    Parameters
    ----------
    roi_key : str, optional
        Registry key to resolve, or None to skip (caller falls back to
        auto-detection).
    roi_registry : RoiRegistry, optional
        Registry to resolve `roi_key` against.
    image : darsia.Image
        Image providing the coordinate system for the physical-to-voxel
        conversion (the registry stores ROIs in physical coordinates).
    as_landmarks : bool
        If True, return the box as 4 ordered corner voxels, anti-clockwise
        from the top-left (row-min, col-min) corner - the convention
        :class:`darsia.ColorCorrection` expects for its reference (brown
        swatch) corner. If False, return the box corners/polygon vertices
        directly, which is sufficient for :class:`darsia.DriftCorrection`
        (it collapses any input to a bounding box anyway).

    Returns
    -------
    np.ndarray or None
        Voxel coordinates (matrix-indexed), or None if no key/registry given.
    """
    if roi_key is None or roi_registry is None:
        return None
    roi_config = roi_registry.resolve_rois([roi_key])[roi_key]
    # RoiConfig.roi itself raises for shapes with no plain geometric
    # representation (label/facies/intersection/union) - re-raise with
    # context specific to this call site (which of those two config
    # fields was involved, and that those shapes have no well-defined
    # bounding box at this point in setup, before any labels/facies image
    # exists).
    try:
        geometry = roi_config.roi
    except ValueError as exc:
        raise ValueError(
            f"ROI '{roi_key}' has shape={roi_config.shape!r}, but "
            "corrections.color.roi/corrections.drift.roi only support "
            "shape='box'/'polygon' (label/facies/intersection/union ROIs "
            "have no well-defined bounding box at this point in setup)."
        ) from exc
    # 'box' gives 2 corners, 'polygon' gives >= 3 vertices - both reduce to a
    # bounding box the same way below, so no shape-specific branching needed.
    voxels = image.coordinatesystem.voxel(geometry)
    if not as_landmarks:
        return voxels
    return darsia.bounding_box_inverse(darsia.bounding_box(voxels))


def _preserve_native_setup(method):
    """Isolate replacement fields and restore the whole prior native Rig on failure.

    The same instance retains the original RAW output ownership. Its old mutable
    fields are unreachable through instance state during replacement setup.
    Failed output attempts remain owned and retained by their original context.
    """
    native_names = {"native_product", "analysis_exclusions"}
    native_positions = tuple(index for index, name in enumerate(
        method.__code__.co_varnames[1:method.__code__.co_argcount]) if name in native_names)

    @wraps(method)
    def setup(rig, *args, **kwargs):
        previous = rig.__dict__
        incoming = any(kwargs.get(name) is not None for name in native_names) or any(
            index < len(args) and args[index] is not None for index in native_positions)
        if previous.get("_analysis_domain") is None and not incoming:
            return method(rig, *args, **kwargs)
        rig.__dict__ = {}
        try:
            return method(rig, *args, **kwargs)
        except BaseException:
            rig.__dict__ = previous
            raise

    return setup

# TODO keep log? or use results?


class Rig:
    """Rig object for CO2 analysis."""

    def _read_optical_image(self, path: Path, **kwargs) -> darsia.Image:
        """Share the configured optical input basis between setup and reading."""
        binding = getattr(self, "raw_camera", None)
        if binding is not None:
            from darsia.fluidflower.rig_raw_input import read_raw_image

            if Path(path) == getattr(self, "baseline_path", None):
                kwargs["expected_sha256"] = getattr(self, "raw_baseline_sha256", None)
            return read_raw_image(path, binding, **kwargs)
        return darsia.imread(path, **kwargs)

    @property
    def corrections(self) -> list[darsia.BaseCorrection]:
        """Combined correction workflow in execution order."""
        return getattr(self, "shape_corrections", []) + getattr(
            self, "color_corrections", []
        )

    @staticmethod
    def _is_shape_correction(correction: object) -> bool:
        return isinstance(
            correction,
            (
                darsia.TypeCorrection,
                darsia.ResizeCorrection,
                darsia.DriftCorrection,
                darsia.CurvatureCorrection,
            ),
        )

    @staticmethod
    def _is_color_correction(correction: object) -> bool:
        return isinstance(
            correction,
            (
                darsia.ColorCorrection,
                darsia.RelativeColorCorrection,
                darsia.IlluminationCorrection,
                darsia.PatchwiseIlluminationCorrection,
            ),
        )

    def setup_reading(
        self,
        baseline_path: Path,
        experiment: darsia.ProtocolledExperiment,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
        log: Path | None = None,
        show_plot: bool = False,
    ):
        """Set up the original reading baseline and shape corrections within any owned RAW operation."""
        from darsia._opencv_runtime import scientific_thread, require_ipp_off, completed_frame
        ipp_before = scientific_thread("Rig.setup_reading:before")
        if getattr(self, "raw_camera", None) is not None:
            operation = getattr(self, "_raw_build", None)
            if operation is None:
                raise ValueError("RAW reading setup requires an exclusively owned build operation")
            operation.require_setup(self, log)
        # Cache experiment for protocol-based datetime lookup
        self.experiment = experiment

        # Setup (based on baseline image without any corrections applied)
        pre_baseline = self._read_optical_image(baseline_path)
        if getattr(self, "raw_camera", None) is not None:
            self._raw_setup_input_metadata = pre_baseline.metadata()

        if show_plot:
            # Allows to visually check the baseline image before any corrections
            # are applied, which is useful for debugging and understanding the
            # impact of corrections.
            plt.imshow(pre_baseline.img)
            plt.show()

        self.setup_shape_corrections(
            pre_baseline=pre_baseline,
            corrections_config=corrections_config,
            roi_registry=roi_registry,
        )

        # Define shape-corrected baseline.
        self.shape_corrected_baseline = self._read_optical_image(
            baseline_path, transformations=self.shape_corrections
        )
        self.baseline = self.shape_corrected_baseline.copy()

        # Plot corrected baseline - show and/or log.
        plt.figure("Corrected baseline")
        plt.imshow(self.baseline.img)
        plt.title("Corrected baseline")
        if show_plot:
            plt.show()
        if log:
            if getattr(self, "raw_camera", None) is not None:
                with self._raw_build.output.open(log / "corrected_baseline.png") as stream:
                    plt.savefig(stream, format="png", dpi=500)
            else:
                plt.savefig(
                    log / "corrected_baseline.png",
                    dpi=500,
                )
        plt.close()

        if getattr(self, "raw_camera", None) is not None:
            self._raw_reading_opencv = completed_frame(ipp_before, "Rig.setup_reading:after")
        else:
            require_ipp_off("Rig.setup_reading:after")
        logger.info("Reading setup completed.")

    def load_corrections(
        self,
        folder: Path,
        corrections_config: CorrectionsConfig | None = None,
    ) -> None:
        """Load persisted corrections from disk.

        Supports both split-format files (`shape_correction_*`, `color_correction_*`)
        and legacy mixed-format files (`correction_*`).
        """
        if corrections_config is None:
            corrections_config = CorrectionsConfig()

        self.shape_corrections = []
        self.color_corrections = []

        shape_paths = sorted(_preset_io.glob_paths(folder, "shape_correction_*.npz"))
        color_paths = sorted(_preset_io.glob_paths(folder, "color_correction_*.npz"))
        if shape_paths or color_paths:
            for correction_path in shape_paths:
                correction = darsia.read_correction(correction_path)
                if self._is_shape_correction(correction):
                    self.shape_corrections.append(correction)
                    logger.info(f"Loaded shape correction {type(correction).__name__}")
                else:
                    logger.warning(
                        f"Skipping non-shape correction in shape pipeline: "
                        f"{type(correction).__name__}"
                    )
            for correction_path in color_paths:
                correction = darsia.read_correction(correction_path)
                if self._is_color_correction(correction):
                    self.color_corrections.append(correction)
                    logger.info(f"Loaded color correction {type(correction).__name__}")
                else:
                    logger.warning(
                        f"Skipping non-color correction in color pipeline: "
                        f"{type(correction).__name__}"
                    )
            logger.info("Corrections loaded from split format.")
            return

        for correction_path in sorted(_preset_io.glob_paths(folder, "correction_*.npz")):
            correction = darsia.read_correction(correction_path)
            if self._is_shape_correction(correction):
                self.shape_corrections.append(correction)
                logger.info(f"Loaded shape correction {type(correction).__name__}")
            elif self._is_color_correction(correction):
                self.color_corrections.append(correction)
                logger.info(f"Loaded color correction {type(correction).__name__}")
            else:
                logger.warning(
                    f"Skipping unknown correction type {type(correction).__name__}"
                )
        logger.info("Corrections loaded from legacy format.")

    def setup_shape_corrections(
        self,
        pre_baseline: darsia.Image,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
    ) -> None:
        """Setup shape corrections that do not depend on labels/porosity."""
        if corrections_config is None:
            corrections_config = CorrectionsConfig()

        self.shape_corrections = []
        """Shape corrections applied prior to label-dependent corrections."""

        baseline_for_setup = pre_baseline

        if corrections_config.type:
            # Aux: needed for rescaling not leaving the range of color space
            self.type_converter = darsia.TypeCorrection(
                corrections_config.type.target_type
            )
            """Type correction to convert images to float32."""
            baseline_for_setup = self.type_converter(baseline_for_setup)

            # Update corrections workflow
            self.shape_corrections.append(self.type_converter)

        # Track the combined resize factor for downstream corrections (e.g.
        # curvature).
        self.resize_factor = 1.0

        if corrections_config.resize:
            # Apply user-configured resize if present.
            current_shape = baseline_for_setup.shape[: baseline_for_setup.space_dim]
            target_shape, self.resize_factor = corrections_config.resize.resolve(
                current_shape
            )

            self.rescale_correction = darsia.ResizeCorrection(shape=target_shape)
            """User-configured resize correction."""

            # Update baseline and corrections workflow
            baseline_for_setup = self.rescale_correction(baseline_for_setup)
            self.shape_corrections.append(self.rescale_correction)

        # Resize for consistency (drift, curvature). Free no-op when shapes
        # already match, via early return in Resize.resize_array.
        self.resize_correction = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim]
        )
        """Resize correction to baseline shape (before drift/curvature)."""

        # Update baseline and corrections workflow
        baseline_for_setup = self.resize_correction(baseline_for_setup)
        self.shape_corrections.append(self.resize_correction)

        self.resize_correction_inter_nearest = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim],
            interpolation="inter_nearest",
        )
        """Resize for int data (labels/facies) before drift/curvature."""

        if corrections_config.drift:
            # Define translation correction object, based on the configured
            # reference mode: a named ROI, or auto-detected color checker.
            if corrections_config.drift.mode == "roi":
                roi_voxels = _resolve_named_roi_voxels(
                    corrections_config.drift.roi, roi_registry, baseline_for_setup
                )
                self.drift_correction = darsia.DriftCorrection(
                    baseline_for_setup, config={"roi": roi_voxels,
                        "linear_input": getattr(self, "raw_camera", None) is not None}
                )
                """Drift correction based on a named ROI."""
            else:
                try:
                    _, cc_voxels = darsia.find_colorchecker(
                        baseline_for_setup, corrections_config.drift.colorchecker
                    )
                    self.drift_correction = darsia.DriftCorrection(
                        baseline_for_setup, config={"roi": cc_voxels,
                            "linear_input": getattr(self, "raw_camera", None) is not None}
                    )
                    """Drift correction based on color checker alignment."""
                except Exception as e:
                    warn(
                        f"Color checker not found. Drift correction not setup. "
                        f"Error: {e}",
                        UserWarning,
                    )
                    self.drift_correction = darsia.DriftCorrection(
                        baseline_for_setup,
                        config={"linear_input": getattr(self, "raw_camera", None) is not None},
                    )

            # Update baseline and corrections workflow
            ...  # baseline image does not need drift correction.
            self.shape_corrections.append(self.drift_correction)

        if corrections_config.curvature:
            # Define curvature correction as derived from analysis of laser grid
            # images. Pass resize_factor to rescale pixel-calibrated config values
            # (crop corners, bulge/stretch offsets) if images were resized.
            self.curvature_correction = darsia.CurvatureCorrection(
                config=corrections_config.curvature.to_dict(),
                resize_factor=self.resize_factor,
            )
            """Curvature correction based on laser grid analysis."""

            # Update baseline and corrections workflow
            baseline_for_setup = self.curvature_correction(baseline_for_setup)
            self.shape_corrections.append(self.curvature_correction)

        # Harmonize back to canonical baseline shape. Free no-op when shapes
        # already match, via early return in Resize.resize_array.
        self.resize_correction_final = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim]
        )
        """Final resize correction harmonizing output to baseline shape."""

        # Update baseline and corrections workflow
        baseline_for_setup = self.resize_correction_final(baseline_for_setup)
        self.shape_corrections.append(self.resize_correction_final)

        self.resize_correction_inter_nearest_final = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim],
            interpolation="inter_nearest",
        )
        """Final nearest-interpolation harmonizer for label/facies data."""

        logger.info("Shape corrections setup complete.")

    def setup_color_corrections(
        self,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup label-dependent color corrections after labels/porosity are available.

        Execution order is fixed:
        1) illumination, 2) relative color, 3) color correction.
        Relative color can either be loaded from a precomputed correction file or
        calibrated from configured images (interactive mode only).
        """
        if corrections_config is None:
            corrections_config = CorrectionsConfig()
        if not hasattr(self, "shape_corrected_baseline"):
            raise RuntimeError(
                "Shape-corrected baseline missing. Run setup_shape_corrections first."
            )

        self.color_corrections = []
        """Color corrections initialized after labels and porosity are available."""

        # 1) Illumination correction.
        if corrections_config.illumination:
            self.illumination_correction = self.setup_illumination_correction(
                corrections_config.illumination,
                log=log,
                show_plot=show_plot,
            )
            self.color_corrections.append(self.illumination_correction)

        # 1b) patchwise illumination correction.
        if corrections_config.patchwise_illumination:
            assert (
                not corrections_config.illumination
            ), "Can only use a single illumination correction method at a time."
            self.illumination_correction = self.setup_patchwise_illumination_correction(
                corrections_config.patchwise_illumination,
                show_plot=show_plot,
            )
            self.color_corrections.append(self.illumination_correction)

        # 2) Relative color correction (reserved in ordering; setup currently guarded).
        if corrections_config.relative_color:
            relative_config = corrections_config.relative_color
            if relative_config.path is not None:
                self.relative_color_correction = darsia.RelativeColorCorrection(
                    self.shape_corrected_baseline
                )
                self.relative_color_correction.load(relative_config.path)
                self.color_corrections.append(self.relative_color_correction)
            elif relative_config.images:
                if not relative_config.interactive:
                    raise ValueError(
                        "Interactive calibration is required when using "
                        "corrections.relative_color.images. Set "
                        "corrections.relative_color.interactive=true or provide "
                        "corrections.relative_color.path."
                    )
                relative_color_images = []
                for path in relative_config.images:
                    img = self._read_optical_image(path)
                    for correction in self.shape_corrections:
                        img = correction(img)
                    relative_color_images.append(img)
                self.relative_color_correction = darsia.RelativeColorCorrection(
                    self.shape_corrected_baseline,
                    relative_color_images,
                    relative_config.to_options_dict(),
                )
                self.color_corrections.append(self.relative_color_correction)

        # 3) Color correction.
        if corrections_config.color:
            # Define color correction, based on the configured reference
            # mode: a named ROI (as 4 ordered landmark corners), or
            # auto-detected color checker.
            if corrections_config.color.mode == "roi":
                roi_voxels = _resolve_named_roi_voxels(
                    corrections_config.color.roi,
                    roi_registry,
                    self.shape_corrected_baseline,
                    as_landmarks=True,
                )
                self.color_correction = darsia.ColorCorrection(
                    self.shape_corrected_baseline,
                    config={
                        "roi": roi_voxels,
                        "clip": False,
                    },
                )
                """Color correction based on a named ROI."""
            else:
                try:
                    _, cc_voxels = darsia.find_colorchecker(
                        self.shape_corrected_baseline,
                        corrections_config.color.colorchecker,
                    )
                    self.color_correction = darsia.ColorCorrection(
                        self.shape_corrected_baseline,
                        config={
                            "roi": cc_voxels,
                            "clip": False,
                        },
                    )
                    """Color correction based on color checker alignment."""
                except Exception as e:
                    raise ValueError(
                        "The configured ColorChecker could not be located; "
                        "colour correction was not initialized."
                    ) from e
            self.color_corrections.append(self.color_correction)

        # Apply the configured color correction pipeline to the shape-corrected baseline.
        # At this point, setup has already been run for each configured correction.
        self.baseline = self.shape_corrected_baseline.copy()
        for correction in self.color_corrections:
            self.baseline = correction(self.baseline)

        logger.info("Color corrections setup complete.")

    # ! ---- GEOMETRY ----

    def setup_depth(
        self,
        path: Path,
        log: Path | None = None,
    ) -> None:
        """Setup depth map for the rig object.

        Parameters
        ----------
        path : Path | None
            Path to the precomputed depth map file.
            If provided, it will be loaded and used as the depth map.
        log : Path | None
            Path to the log folder where geometry images will be saved.
        """
        # Load depth map from file and reshape to baseline shape
        assert _preset_io.exists(path), f"Path to depth map {path} does not exist."
        pre_depth = darsia.imread(path)
        self.depth: darsia.Image = darsia.resize(pre_depth, ref_image=self.baseline)
        """Depth map for the rig object."""

        # Log results
        if log:
            # Plot depth map
            plt.figure()
            plt.imshow(self.depth.img)
            plt.colorbar()
            plt.title("Depth map")
            plt.savefig(log / "depth_map.png", dpi=500)
            plt.close()

        logging.info("Depth map setup completed.")

    def setup_geometry(self) -> None:
        """Setup geometry for volumetric integration of images."""
        shape_meta = self.baseline.shape_metadata()
        self.geometry = darsia.ExtrudedPorousGeometry(
            depth=self.depth, porosity=self.porosity, **shape_meta
        )
        if getattr(self, "_analysis_domain", None) is not None:
            from darsia.fluidflower.rig_analysis_domain import wrap_analysis_geometry

            self.geometry = wrap_analysis_geometry(self, self.geometry)
        """Geometry for volumetric integration of images."""
        logging.info("Geometry setup completed.")

    # ! ---- LABELS -----

    def setup_labels(
        self, path: Path, apply_corrections: bool = False, log: Path | None = None
    ) -> None:
        """Setup labels for the rig object.

        This method loads labels from a specified path and applies corrections if needed.

        Parameters
        ----------
        path : Path
            Path to the labels file. If the file exists, it will be loaded.
        apply_corrections : bool
            If True, applies corrections to the labels based on
            the baseline image.
        log : Path | None
            Path to the log folder where label images will be saved.
        """
        assert _preset_io.exists(path), f"Labels file {path} does not exist."
        if apply_corrections:
            # Assume that the labels are based on non-corrected baseline image.
            # Thus, need to apply the relevant corrections from the read routine.
            plain_labels = darsia.imread(path)
            labels = plain_labels
            if hasattr(self, "resize_correction_inter_nearest"):
                labels = self.resize_correction_inter_nearest(labels)
            if hasattr(self, "curvature_correction"):
                labels = self.curvature_correction(labels)
            labels = self.resize_correction_inter_nearest_final(labels)
            self.labels = labels

        else:
            # Assume the labels are aligned with the corrected baseline.
            pre_labels = darsia.imread(path)
            self.labels = darsia.resize(
                pre_labels,
                ref_image=self.baseline,
                interpolation="inter_nearest",
            )

        if log:
            # Plot labels
            plt.figure()
            plt.imshow(self.labels.img)
            plt.colorbar()
            plt.title("Labels")
            plt.savefig(Path(log) / "labels.png", dpi=500)
            plt.close()

        logger.info("Labels setup completed.")

    def setup_inner_labels(self, log: Path | None = None) -> None:
        """Define boolean mask which excludes the label boundaries."""
        # Identify the boundary pixels.
        boundary_mask = skimage.segmentation.find_boundaries(
            self.labels.img, mode="outer", connectivity=1
        )

        # Extend the boundary pixels by a certain number of pixels (e.g., 2)
        # to create a buffer zone.
        # TODO: Allow for configuration of the buffer size and structuring element.
        buffer_zone = skimage.morphology.binary_dilation(
            boundary_mask, footprint=skimage.morphology.disk(2)
        )

        # Create a boolean mask where the inner labels (excluding the boundary)
        # are True and the rest is False.
        inner_labels = ~buffer_zone

        self.inner_labels = darsia.Image(
            inner_labels.astype(bool), **self.labels.metadata()
        )
        """Boolean mask for inner labels, excluding boundaries."""

    def setup_facies(
        self,
        path: Path,
        apply_corrections: bool = False,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup facies.

        This method loads facies from a specified path and applies corrections if needed.

        Parameters
        ----------
        path : Path
            Path to the facies file. If the file exists, it will be loaded.
        apply_corrections : bool
            If True, applies corrections to the facies based on
            the baseline image.
        log : Path | None
            Path to the log folder where facies images will be saved.
        """
        assert _preset_io.exists(path), f"Facies file {path} does not exist."
        if apply_corrections:
            # Assume that the facies are based on non-corrected baseline image.
            # Thus, need to apply the relevant corrections from the read routine.
            plain_facies = darsia.imread(path)
            facies = plain_facies
            if hasattr(self, "resize_correction_inter_nearest"):
                facies = self.resize_correction_inter_nearest(facies)
            if hasattr(self, "curvature_correction"):
                facies = self.curvature_correction(facies)
            facies = self.resize_correction_inter_nearest_final(facies)
            self.facies = facies

        else:
            # Assume the facies are aligned with the corrected baseline.
            pre_facies = darsia.imread(path)
            self.facies = darsia.resize(
                pre_facies,
                ref_image=self.baseline,
                interpolation="inter_nearest",
            )

        if log:
            # Plot facies
            plt.figure()
            plt.imshow(self.facies.img)
            plt.colorbar()
            plt.title("Facies")
            plt.savefig(log / "facies.png", dpi=500)
            plt.close()

        if show_plot:
            # Display facies on top of baseline for visual inspection.
            plt.figure()
            plt.imshow(self.baseline.img)
            plt.imshow(self.facies.img, alpha=0.5)
            plt.title("Facies overlay on baseline")
            plt.show()

        logger.info("Facies setup completed.")

    def setup_facies_props(
        self,
        props_path: Path | None = None,
        porosity: Path | None = None,
        permeability: Path | None = None,
    ) -> None:
        """Define facies properties like porosity.

        Parameters
        ----------
        props_path : Path | None
            Path to the facies properties CSV file.
            If provided, it will load the facies properties from this path.
        porosity : Path | None
            Path to the porosity image file. If provided,
            it will load the porosity image from this path.
        """
        if props_path:
            facies_props = FaciesProps.load(facies=self.facies, path=props_path)
            self.porosity = facies_props.porosity
            """Porosity for the rig object."""
            self.permeability = facies_props.permeability
            """Permeability for the rig object."""
            if getattr(self, "_analysis_domain", None) is not None:
                from darsia.fluidflower.rig_analysis_domain import apply_property_support

                apply_property_support(self, facies_props)
        else:
            if not (porosity and permeability):
                raise FileNotFoundError("No facies properties provided.")
            self.porosity = darsia.imread(porosity)
            self.permeability = darsia.imread(permeability)

    # ! ---- ILLUMINATION CORRECTION ----
    def setup_illumination_correction(
        self,
        config: IlluminationCorrectionConfig | None,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> darsia.IlluminationCorrection:
        """Setup and return illumination correction.

        Parameters
        ----------
        config : IlluminationCorrectionConfig | None
            Configuration for the illumination
            correction. If provided, it will set up the illumination correction based
            on this configuration.
        log : Path | None
            Path to the log folder where diagnostic plots will be saved.
        show_plot : bool
            Whether to show diagnostic plots during setup (default: False).

        Notes
        -----
            Illumination calibration in Rig intentionally uses the shape-corrected
            baseline as setup input.
        """
        illumination_correction = darsia.IlluminationCorrection()

        # Fetch samples for illumination correction based on labels and baseline.
        if config is not None:
            sample_groups = []
            if config.samples:
                # Explicit windows from the configuration form a single group, e.g.
                # hand-picked sand windows; labels/width/num_samples/seed are ignored.
                sample_groups.append(
                    [
                        (slice(row_start, row_stop), slice(col_start, col_stop))
                        for row_start, row_stop, col_start, col_stop in config.samples
                    ]
                )
            elif not config.labels:
                # If no labels specified, use random samples from the whole image.
                samples = illumination_correction.select_random_samples(
                    mask=darsia.ones_like(self.shape_corrected_baseline, dtype=bool),
                    config=config,
                )
                sample_groups.append(samples)
            else:
                for label in config.labels:
                    assert (
                        label in self.labels.img
                    ), f"Label {label} not found in labels image."
                    mask = self.labels.img == label
                    samples = illumination_correction.select_random_samples(
                        mask=mask, config=config
                    )
                    sample_groups.append(samples)

            # Reference sample for the closed-form method: an index, or the sample
            # whose centre has the smallest row + column ("top_left").
            reference_sample = getattr(config, "reference_sample", 0)
            if isinstance(reference_sample, str):
                first_group = sample_groups[0] if sample_groups else []
                if not first_group:
                    raise ValueError(
                        "reference_sample='top_left' needs at least one sample window."
                    )
                centres = [
                    (s[0].start + s[0].stop) // 2 + (s[1].start + s[1].stop) // 2
                    for s in first_group
                ]
                reference_sample = int(np.argmin(centres))

            # Determine illumination correction based on inputs
            illumination_correction.setup(
                # Use shape-corrected baseline as explicit setup input.
                base=self.shape_corrected_baseline,
                sample_groups=sample_groups,
                mask=(
                    self.boolean_porosity
                    if getattr(config, "use_porosity_mask", True)
                    else None
                ),
                outliers=config.outliers,
                filter=lambda x: skimage.filters.gaussian(x, sigma=config.sigma),
                colorspace=config.colorspace,
                interpolation=config.interpolation,
                show_plot=show_plot,
                log=log,
                method=getattr(config, "method", "group_average"),
                reference_sample=reference_sample,
            )

        return illumination_correction

    # ! ---- ILLUMINATION CORRECTION ----
    def setup_patchwise_illumination_correction(
        self,
        config: PatchwiseIlluminationCorrectionConfig | None,
        show_plot: bool = False,
    ) -> darsia.PatchwiseIlluminationCorrection:
        """Setup and return patchwise illumination correction.

        Parameters
        ----------
        config : PatchwiseIlluminationCorrectionConfig | None
            Configuration for
            the patchwise illumination correction. If provided, it will set up the
            correction based on this configuration.

        Notes
        -----
            Patchwise illumination calibration uses the labels image to enable
            per-label reference color computation within each patch, ensuring
            heterogeneous regions contribute equally by material.
        """
        baseline_images = [self.read_image(path) for path in config.baseline_paths]

        illumination_correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=baseline_images,
            labels=self.labels,
            nw=config.nw,
            eps=config.eps,
            show_images=show_plot,
        )
        return illumination_correction

    # ! ---- POROSITY ----

    def setup_image_porosity(
        self,
        path: Path | None = None,
        log: Path | None = None,
        config: ImagePorosityConfig | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup image porosity based on the baseline image.

        Behaviour is controlled by *config*:

        * ``mode="full"`` (default): constant porosity of ``1`` over the full domain.
        * ``mode="from_image"``: porosity derived from the baseline image via
          :func:`~darsia.patched_porosity_analysis` using the parameters stored in
          *config* (``patches``, ``num_clusters``, ``sample_width``,
          ``tol_color_distance``, ``tol_color_gradient``).

        When *path* is provided the image is always loaded from disk regardless of
        *config*, which is useful for restoring a previously saved rig.

        Parameters
        ----------
        path : Path | None
            Path to a previously saved porosity ``.npz`` file.
            When given, the file is loaded and *config* is not used.
        log : Path | None
            Folder for diagnostic output.  When given, a JPG
            illustration is stored to ``log/image_porosity/image_porosity.jpg``.
        config : ImagePorosityConfig | None
            Porosity configuration.  Defaults to
            ``ImagePorosityConfig()`` (i.e. ``mode="full"``) when not provided.
        show_plot : bool
            When ``True`` the image porosity is displayed interactively.
            Pass ``True`` when calling from a GUI or user-interface workflow.
        """
        if config is None:
            config = ImagePorosityConfig()
        self._image_porosity_config = config

        if path is not None:
            self.image_porosity = darsia.imread(path)
        elif config.mode == "from_image":
            self.image_porosity = patched_porosity_analysis(
                baseline=self.baseline,
                patches=config.patches,
                labels=self.labels,
                num_clusters=config.num_clusters,
                sample_width=config.sample_width,
                tol_color_distance=config.tol_color_distance,
                tol_color_gradient=config.tol_color_gradient,
            )
        else:
            # mode == "full"
            self.image_porosity = darsia.ones_like(
                self.baseline, mode="voxels", dtype=np.float32
            )
        """Image porosity for the rig object."""

        if log:
            out_dir = Path(log)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_scalar_map_illustration(
                self.image_porosity.img,
                out_dir / "image_porosity.jpg",
                title="Image porosity",
                colorbar_label="Porosity",
            )

        if show_plot:
            self.image_porosity.show(title="Image porosity")

        logger.info("Porosity setup completed.")

    def setup_boolean_image_porosity(
        self,
        threshold: float | None = None,
        log: Path | None = None,
        config: ImagePorosityConfig | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup boolean porosity based on the defined threshold.

        In ``mode="full"`` the boolean mask is always all-``True`` (full image domain),
        regardless of *threshold* / *tol*.  In ``mode="from_image"`` the mask is
        derived by thresholding ``self.image_porosity`` with the effective tolerance.

        The effective tolerance is resolved in order of precedence:

        1. *threshold* argument (when explicitly passed).
        2. ``config.tol`` (when *config* is given).
        3. ``self._image_porosity_config.tol`` (stored by :meth:`setup_image_porosity`).
        4. ``0.9`` (hard-coded default).

        Parameters
        ----------
        threshold : float | None
            Override tolerance value.  Deprecated in favour of
            ``config.tol``; kept for backward compatibility.
        log : Path | None
            Folder for diagnostic output.  When given, a JPG
            illustration is stored to
            ``log/image_porosity/boolean_porosity.jpg``.
        config : ImagePorosityConfig | None
            Porosity configuration.  Falls back to
            the config stored by the last call to :meth:`setup_image_porosity`, and
            finally to ``ImagePorosityConfig()`` (``mode="full"``).
        show_plot : bool
            When ``True`` the boolean porosity is displayed
            interactively.  Pass ``True`` when calling from a GUI or
            user-interface workflow.
        """
        if config is None:
            config = getattr(self, "_image_porosity_config", ImagePorosityConfig())

        # Resolve effective threshold: explicit argument wins over config.
        tol = threshold if threshold is not None else config.tol

        if config.mode == "full":
            # Always full boolean mask regardless of tol.
            self.boolean_porosity = darsia.ones_like(
                self.baseline, mode="voxels", dtype=bool
            )
        else:
            # from_image: threshold the continuous porosity map.
            self.boolean_porosity = self.image_porosity > tol

        if log:
            out_dir = Path(log)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_scalar_map_illustration(
                self.boolean_porosity.img.astype(float),
                out_dir / "boolean_porosity.jpg",
                title="Boolean porosity",
                colorbar_label="Porosity (boolean)",
            )

        if show_plot:
            self.boolean_porosity.show(title="Boolean porosity")

        logger.info("Boolean porosity setup completed.")

    @_preserve_native_setup
    def setup(
        self,
        experiment: darsia.ProtocolledExperiment,
        baseline_path: Path,
        depth_map_path: Path,
        labels_path: Path,
        facies_path: Path | None = None,
        facies_props_path: Path | None = None,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
        image_porosity_config: ImagePorosityConfig | None = None,
        # ref_colorchecker_path: Path,
        log: Path | None = None,
        show_plot: bool = False,
        raw_camera: dict | None = None,
        raw_baseline_sha256: str | None = None,
        raw_build_id: str | None = None,
        raw_build=None,
        native_product: dict | None = None,
        analysis_exclusions: dict | None = None,
    ) -> None:
        """Fast setup."""
        from darsia.fluidflower.rig_raw_input import camera_binding, require_raw_rebuild

        selected_camera = camera_binding(raw_camera)
        corrected_native = native_product is not None or analysis_exclusions is not None
        if corrected_native and selected_camera is None:
            raise ValueError("Native analysis inputs require the original RAW reading context")
        if selected_camera is not None:
            require_raw_rebuild(corrections_config)
            if (raw_build is None or raw_build.camera != selected_camera
                    or raw_build.raw_source["sha256"] != raw_baseline_sha256
                    or raw_build.operation_id != raw_build_id):
                raise ValueError("RAW rig setup requires its exact reserved build context")
        if getattr(self, "_analysis_domain", None) is not None:
            self._analysis_domain = None
        self.raw_camera = selected_camera
        if self.raw_camera is not None:
            raw_build.begin_setup(self, log)
        # Create log directory if it doesn't exist
        if log:
            assert isinstance(log, Path), "Log path must be a Path object."
            log.mkdir(parents=True, exist_ok=True)

        # Cache baseline path - TODO: Really needed?
        self.baseline_path = baseline_path

        # Cache reference date
        self.reference_date = experiment.experiment_start

        # Initialize the rig - responsible for reading/preprocessing photographs
        self.setup_reading(
            baseline_path,
            experiment,
            corrections_config=corrections_config,
            roi_registry=roi_registry,
            log=log,
            show_plot=show_plot,
        )

        if self.raw_camera is not None:
            # Only the corrected-baseline diagnostic has an owned, bounded
            # writer. Later optional plots include nested writers and repeated
            # filenames, so RAW setup keeps their scientific work/display but
            # suppresses their disk diagnostics until those writers are owned.
            log = None

        # Fetch depth map
        self.setup_depth(
            depth_map_path,
            log=log,
        )

        if corrected_native:
            from darsia.fluidflower.rig_analysis_domain import prepare_analysis_domain

            prepare_analysis_domain(
                self,
                analysis_exclusions,
                labels_path=labels_path,
                facies_path=facies_path,
                native_product=native_product,
            )

        # Add labels
        self.setup_labels(
            path=labels_path,
            apply_corrections=not corrected_native,
            log=log,
        )

        # Setup inner labels
        self.setup_inner_labels(
            log=log,
        )

        # Add facies
        if facies_path is not None:
            self.setup_facies(
                path=facies_path,
                apply_corrections=not corrected_native,
                log=log,
                show_plot=show_plot,
            )
        else:
            self.facies = self.labels.copy()

        # Setup facies props
        self.setup_facies_props(facies_props_path)

        # Define geometry for integration
        self.setup_geometry()

        # Pre-setup image and boolean porosity based on baseline (redo after color correction)
        self.setup_image_porosity(
            log=log, config=image_porosity_config, show_plot=show_plot
        )
        self.setup_boolean_image_porosity(log=log, show_plot=show_plot)

        # Setup color corrections (wait until here to use label and porosity information)
        self.setup_color_corrections(
            corrections_config=corrections_config,
            roi_registry=roi_registry,
            log=log,
            show_plot=show_plot,
        )

        # Redo the porosity setup after color corrections, as the latter can change the image.
        if len(self.color_corrections) > 0:
            self.setup_image_porosity(
                log=log, config=image_porosity_config, show_plot=show_plot
            )
            self.setup_boolean_image_porosity(log=log, show_plot=show_plot)

        # TODO Setup concentration analysis and transformations?
        # self.co2_analysis = ...
        # self.co2_g_analysis = ...

        # Setup co2 mass analysis.
        # TODO Add?
        # self.setup_mass_analysis(
        #    atmospheric_pressure=kwargs.get("atmospheric_pressure", 1.010),
        #    temperature=kwargs.get("temperature", 23.0),
        # )

        logger.info("Rig setup completed.")

        # ! ---- AVERAGING ----

        # Use porosity-based averaging for restoration
        restoration = darsia.porosity_based_averaging(
            self.labels, self.image_porosity, self.baseline
        )

        # Cache
        self.restoration = restoration
        """Restoration model based on porosity-based averaging."""

        clipping = darsia.ClipModel(min_value=0.0)
        self.upscaling = darsia.CombinedModel([clipping] + 2 * [restoration])
        logger.info("Upscaling setup completed.")

        # ! ---- COMBINED MODELS ----

        # self.concentration_analysis_aq = darsia.CombinedModel(
        #    [self.co2_analysis, self.upscaling]
        # )
        # self.concentration_analysis_g = darsia.CombinedModel(
        #    [self.co2_g_analysis, self.upscaling]
        # )
        logger.info("Concentration analysis setup completed.")

    def setup_mass_analysis(
        self,
        atmospheric_pressure,
        atmospheric_temperature,
    ) -> None:
        # Define plain mass analysis (combining gas and aqueous phase)
        self.co2_mass_analysis = darsia.CO2MassAnalysis(
            self.baseline,
            atmospheric_pressure=atmospheric_pressure,
            atmospheric_temperature=atmospheric_temperature,
        )
        logger.info("Mass analysis setup completed.")

    # ! ---- ANALYSIS ----

    def mass_analysis(self, img: darsia.Image) -> darsia.MassAnalysisResults:
        """Mass analysis of the image."""
        raise NotImplementedError

    def threshold_analysis(
        self, mass_analysis_result: darsia.MassAnalysisResults
    ) -> darsia.ThresholdAnalysisResults:
        """Threshold analysis of the mass analysis result."""
        raise NotImplementedError

    # ! ---- I/O ----

    def save(self, folder: Path) -> None:
        """Save rig object to file."""

        owned = None
        if getattr(self, "raw_camera", None) is not None:
            operation = getattr(self, "_raw_build", None)
            if operation is None:
                raise ValueError("RAW rig save requires its original exclusive build context")
            owned = operation.output
            owned.begin_save(self, folder)

        def save_image(image, path):
            if owned is None:
                image.save(path)
            else:
                owned.image(image, path)

        def save_correction(correction, path):
            if owned is None:
                correction.save(path)
            else:
                owned.correction(correction, path)

        # Create folder if not exists
        folder.mkdir(parents=True, exist_ok=True)

        # Dump meta data to json file
        # TODO: Needed for what?
        meta_data = {
            "baseline_path": str(self.baseline_path),
        }
        if getattr(self, "raw_camera", None) is not None:
            meta_data["raw_camera"] = dict(self.raw_camera)
            if getattr(self, "raw_baseline_sha256", None) is not None:
                meta_data["raw_baseline_sha256"] = self.raw_baseline_sha256
            if getattr(self, "_raw_build_id", None) is not None:
                meta_data["raw_build_id"] = self._raw_build_id
        if getattr(self, "_analysis_domain", None) is not None:
            from darsia.fluidflower.rig_analysis_domain import save_analysis_state

            save_analysis_state(self, folder, owned, meta_data)
        if owned is None:
            with open(folder / "meta_data.json", "w") as f:
                json.dump(meta_data, f)
        else:
            with owned.open(folder / "meta_data.json") as stream:
                stream.write(json.dumps(meta_data).encode("utf-8"))

        # Save reading information
        save_image(self.baseline, folder / "baseline.npz")
        if hasattr(self, "shape_corrected_baseline"):
            save_image(self.shape_corrected_baseline, folder / "shape_corrected_baseline.npz")

        # Save split correction pipelines.
        for i, correction in enumerate(self.shape_corrections):
            correction_name = type(correction).__name__.lower()
            save_correction(correction, folder / f"shape_correction_{i}_{correction_name}.npz")
        for i, correction in enumerate(self.color_corrections):
            correction_name = type(correction).__name__.lower()
            save_correction(correction, folder / f"color_correction_{i}_{correction_name}.npz")

        # Save geometry information
        try:
            assert isinstance(self.depth, darsia.Image)
            save_image(self.depth, folder / "depth.npz")
        except Exception:
            warn("Depth not available for saving.", UserWarning)

        # Save labels information
        try:
            save_image(self.labels, folder / "labels.npz")
        except Exception:
            warn("Labels not available for saving.", UserWarning)

        # Save facies information
        try:
            save_image(self.facies, folder / "facies.npz")
        except Exception:
            warn("Facies not available for saving.", UserWarning)

        # Save facies properties
        try:
            save_image(self.porosity, folder / "porosity.npz")
        except Exception:
            warn("Porosity not available for saving.", UserWarning)

        try:
            save_image(self.permeability, folder / "permeability.npz")
        except Exception:
            warn("Permeability not available for saving.", UserWarning)

        # Save porosity information
        try:
            save_image(self.image_porosity, folder / "image_porosity.npz")
        except Exception:
            warn("Image porosity not available for saving.", UserWarning)

        if getattr(self, "_analysis_domain", None) is not None:
            from darsia.fluidflower.rig_analysis_domain import verify_saved_analysis_state

            verify_saved_analysis_state(self, folder, meta_data)

        if owned is not None:
            owned.finish_save()

        # Use green color and hyperlink
        folder_uri = Path(folder).absolute().as_uri()
        logger.info(
            """\033[92mRig object saved to \033]8;;"""
            f"""{folder_uri}\033\\{folder}\033]8;;\033\\\033[0m"""
        )

    @classmethod
    def load(
        cls, folder: Path, corrections_config: CorrectionsConfig | None = None
    ) -> "Rig":
        """Load rig object from file.

        Mimick the save method.

        Parameters
        ----------
        folder : Path
            Path to the folder where the rig object is saved.
        """
        # Create rig object
        rig = cls()

        # TODO: move into single function self.setup_basics() or so.
        # Load meta data
        with _preset_io.open_read(folder / "meta_data.json", "r") as f:
            meta_data = json.load(f)
        rig.baseline_path = Path(meta_data["baseline_path"])
        if "raw_camera" in meta_data:
            from darsia.fluidflower.rig_raw_input import camera_binding

            rig.raw_camera = camera_binding(meta_data["raw_camera"])
            rig.raw_baseline_sha256 = meta_data.get("raw_baseline_sha256")

        # Load data for reading images
        rig.baseline = darsia.imread(folder / "baseline.npz")
        if _preset_io.exists(folder / "shape_corrected_baseline.npz", 'exists'):
            rig.shape_corrected_baseline = darsia.imread(
                folder / "shape_corrected_baseline.npz"
            )
        else:
            rig.shape_corrected_baseline = rig.baseline.copy()
        logger.info("Baseline setup complete.")

        # Load corrections
        rig.load_corrections(folder, corrections_config=corrections_config)

        # Load depth map
        rig.setup_depth(path=folder / "depth.npz")

        # Load labels information - corrections not needed assuming labels are aligned with
        # the corrected baseline.
        rig.setup_labels(path=folder / "labels.npz", apply_corrections=False)

        # Setup inner labels information
        rig.setup_inner_labels()

        # Load facies information - corrections not needed assuming facies are aligned with
        rig.setup_facies(path=folder / "facies.npz", apply_corrections=False)

        # Load facies properties
        rig.setup_facies_props(
            porosity=folder / "porosity.npz",
            permeability=folder / "permeability.npz",
        )

        if "analysis_domain" in meta_data:
            from darsia.fluidflower.rig_analysis_domain import restore_analysis_state

            restore_analysis_state(rig, folder, meta_data)

        # Setup geometry information
        rig.setup_geometry()

        # Load image porosity information
        rig.setup_image_porosity(path=folder / "image_porosity.npz")
        rig.setup_boolean_image_porosity()

        # TODO setup color_analysis, and pw_transformation.

        logger.info("Rig object loaded.")
        return rig

    # ! ---- I/O ----

    def import_from_csv(
        self,
        path: Path,
        *,
        delimiter: str = ",",
        date=None,
        reference_date=None,
        time=None,
        name: str | None = None,
        is_extensive: bool = False,
    ) -> darsia.ScalarImage | darsia.ExtensiveImage:
        """Import scalar result data from CSV."""
        if not path.exists():
            raise FileNotFoundError(f"CSV file {path} does not exist.")

        # Read csv
        try:
            data = np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            data = np.loadtxt(path, delimiter=delimiter, skiprows=1)

        # Apply lex sort
        # Sort, such that 0th column changes fasters, then 1st column.
        indices = np.lexsort((data[:, 0], data[:, 1]))
        data = data[indices]

        # Extract coordinates (x, y) and values if in coordinate format,
        # otherwise treat as array data
        coordinates_x = data[:, 0]
        coordinates_y = data[:, 1]
        values = data[:, 2]

        # Determine the shape = frequency of x_coordinates
        # (fastest changing) and y_coordinates (slowest changing)
        unique_x = np.unique(coordinates_x)
        unique_y = np.unique(coordinates_y)

        row = len(unique_y)
        col = len(unique_x)
        shape = (row, col)

        # Determine the cell size in dx and dy direction
        dx = np.min(np.diff(unique_x))
        dy = np.min(np.diff(unique_y))

        # Determine origin
        origin = (unique_x[0] - dx / 2, unique_y[-1] + dy / 2)

        # Determine the dimensions
        dimensions = (
            np.max(coordinates_y) - np.min(coordinates_y) + dy,
            np.max(coordinates_x) - np.min(coordinates_x) + dx,
        )

        # Reshape values to the determined shape, remember that the
        # values are ordered wrt. Euclidean coordinates, with x changing
        # fastest, so we need to reshape accordingly. Also, we need to
        # switch to row-col order, with origin at the top-left corner,
        # so we need to flip the y-coordinates and reshape accordingly.
        values_reshaped = values.reshape(
            shape, order="C"
        )  # Reshape to (row, col) with C order (x changes fastest)
        values_reshaped = np.flip(values_reshaped, axis=0)  # Flip

        # Collect the metadata
        metadata = {
            "origin": origin,
            "cell_size": (dx, dy),
            "dimensions": dimensions,
            "name": name,
            "time": time,
            "date": date,
            "reference_date": reference_date,
            "series": False,
            "scalar": True,
        }

        if is_extensive:
            return darsia.ExtensiveImage(values_reshaped, **metadata)
        else:
            return darsia.ScalarImage(values_reshaped, **metadata)

    def read_image(self, path: Path, dtype: Any = float) -> darsia.Image:
        """Read image from file and apply corrections.

        The respective date is extracted from the path using the imaging protocol.

        Parameters
        ----------
        path : Path
            Path to the image file.
        dtype : numpy dtype, optional
            Floating-point type used when reading optical input.

        Returns
        -------
        darsia.Image
            Image object with applied corrections.
        """
        from darsia._opencv_runtime import scientific_thread, require_ipp_off
        scientific_thread("Rig.read_image:before")
        assert hasattr(
            self, "experiment"
        ), "Experiment not defined. Run load_experiment() first."
        # Convert date from path
        date = self.experiment.get_datetime(path)

        # Read image from file and apply corrections
        img = self._read_optical_image(
            path,
            transformations=self.corrections,
            date=date,
            reference_date=self.reference_date,
            name=path.name,
            dtype=dtype,
        )

        require_ipp_off("Rig.read_image:after")
        return img

    # ! ---- EXPERIMENT AND PROTOCOLS ----

    def load_experiment(self, experiment: darsia.ProtocolledExperiment) -> None:
        """Load experiment and associated protocols.

        This is required to read images and compute the injected mass correctly.

        Parameters
        ----------
        experiment : darsia.ProtocolledExperiment
            Experiment object containing imaging,
            injection, and pressure/temperature protocols.
        """
        self.experiment = experiment
        self.injection_protocol = experiment.injection_protocol
        self.pressure_temperature_protocol = experiment.pressure_temperature_protocol
        self.reference_date = experiment.experiment_start
        logger.info("Experiment and protocols loaded.")

    def update(self, path: Path) -> None:
        """Update the state of the rig based on the image path.

        This method updates the current date, time, pressure, and temperature
        of the rig object based on the provided image path.

        Parameters
        ----------
        path : Path
            Path to the image file.
        """
        # Convert date from path
        date = self.experiment.get_datetime(path)
        self.current_date = date
        self.current_time = (date - self.reference_date).total_seconds() / 3600.0
        state = self.pressure_temperature_protocol.get_state(date)
        self.current_pressure = state.pressure
        self.current_temperature = state.temperature
        self.setup_mass_analysis(
            atmospheric_pressure=self.current_pressure,
            atmospheric_temperature=self.current_temperature,
        )
        logger.info(f"State updated to {self.current_date}.")
