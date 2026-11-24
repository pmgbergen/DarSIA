# TODO: Any connections to darsia.SimpleFluidFlower?
# TODO: integrate into darsia4porotwin.FluidFlower?
# TODO: Clean up concentration analyses. Can be removed?

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, get_args
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage

import darsia
from darsia.presets.workflows.config.corrections import (
    CorrectionsConfig,
    IlluminationCorrectionConfig,
    PatchwiseIlluminationCorrectionConfig,
)
from darsia.presets.workflows.config.facies import RegionName
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.facies_props import FaciesProps

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


# TODO keep log? or use results?


class Rig:
    """Rig object for CO2 analysis."""

    # ! ---- MEMBERS LOADED ON FIRST USE ----

    def _defer(self, names: tuple[str, ...], loader: Callable[[], None]) -> None:
        """Register a loader which sets the members `names` on their first access.

        Large members which only some workflows need (e.g. the permeability for
        Wasserstein distances) are then read when a workflow uses them, and not by
        every workflow loading the rig. Afterwards they are ordinary attributes.

        Parameters
        ----------
        names : tuple of str
            Members set by the loader.
        loader : callable
            Function without arguments setting all members `names`.

        """
        deferred = self.__dict__.setdefault("_deferred", {})
        for name in names:
            deferred[name] = (names, loader)

    def _defer_file(
        self,
        name: str,
        path: Path,
        read: Callable[[Path], darsia.Image] = darsia.imread,
    ) -> None:
        """Register the image file `path` as member `name`, read on first access.

        A missing file raises here, as reading it right away would.

        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self._defer((name,), lambda: setattr(self, name, read(path)))

    def __getattr__(self, name: str):
        # Called only for attributes which are not set: load deferred members.
        deferred = self.__dict__.get("_deferred")
        if deferred is None or name not in deferred:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        names, loader = deferred[name]
        with darsia.timed(f"load {name}"):
            loader()
        for loaded in names:
            deferred.pop(loaded, None)
        return self.__dict__[name]

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
        # Cache experiment for protocol-based datetime lookup
        self.experiment = experiment

        # Setup (based on baseline image without any corrections applied)
        pre_baseline = darsia.imread(baseline_path)

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
        self.shape_corrected_baseline = darsia.imread(
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
            plt.savefig(
                log / "corrected_baseline.png",
                dpi=500,
            )
        plt.close()

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

        shape_paths = sorted(folder.glob("shape_correction_*.npz"))
        color_paths = sorted(folder.glob("color_correction_*.npz"))
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

        for correction_path in sorted(folder.glob("correction_*.npz")):
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
                    baseline_for_setup, config={"roi": roi_voxels}
                )
                """Drift correction based on a named ROI."""
            else:
                try:
                    _, cc_voxels = darsia.find_colorchecker(
                        baseline_for_setup, corrections_config.drift.colorchecker
                    )
                    self.drift_correction = darsia.DriftCorrection(
                        baseline_for_setup, config={"roi": cc_voxels}
                    )
                    """Drift correction based on color checker alignment."""
                except Exception as e:
                    warn(
                        f"Color checker not found. Drift correction not setup. "
                        f"Error: {e}",
                        UserWarning,
                    )
                    self.drift_correction = darsia.DriftCorrection(baseline_for_setup)

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
        image_porosity_config: ImagePorosityConfig | None = None,
    ) -> None:
        """Setup label-dependent color corrections after labels are available.

        Execution order is fixed:
        1) illumination, 2) relative color, 3) color correction.
        Relative color can either be loaded from a precomputed correction file or
        calibrated from configured images (interactive mode only).

        With `[corrections.illumination] mask = "boolean_porosity"`, the illumination
        correction is fitted on the boolean image porosity of the current baseline
        (`[image_porosity]` method > `tol`), computed here and not kept; the stored image
        porosity is a setup step of its own, after the rig.
        """
        if corrections_config is None:
            corrections_config = CorrectionsConfig()
        if not hasattr(self, "shape_corrected_baseline"):
            raise RuntimeError(
                "Shape-corrected baseline missing. Run setup_shape_corrections first."
            )

        self.color_corrections = []
        """Color corrections initialized after labels are available."""

        # 1) Illumination correction.
        if corrections_config.illumination:
            self.illumination_correction = self.setup_illumination_correction(
                corrections_config.illumination,
                log=log,
                show_plot=show_plot,
                image_porosity_config=image_porosity_config,
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
                    img = darsia.imread(path)
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
                    warn(
                        f"Color checker not found. Color correction not setup. "
                        f"Error: {e}",
                        UserWarning,
                    )
                    self.color_correction = darsia.ColorCorrection(
                        self.shape_corrected_baseline
                    )
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
        # Load depth map from file, on the grid of the baseline
        assert path.exists(), f"Path to depth map {path} does not exist."
        self.depth: darsia.Image = self.read_field(path)
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
        assert path.exists(), f"Labels file {path} does not exist."
        # Drawn on the uncorrected baseline photo, or aligned with the corrected one
        self.labels = self.read_field(path, on_photo=apply_corrections)

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
        buffer_zone = scipy.ndimage.binary_dilation(
            boundary_mask, structure=skimage.morphology.disk(2)
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
        assert path.exists(), f"Facies file {path} does not exist."
        # Drawn on the uncorrected baseline photo, or aligned with the corrected one
        self.facies = self.read_field(path, on_photo=apply_corrections)

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
        else:
            if not (porosity and permeability):
                raise FileNotFoundError("No facies properties provided.")
            self.porosity = self.read_field(porosity)
            self.permeability = self.read_field(permeability)

    def regions(self, name: RegionName) -> darsia.Image:
        """Regions to group pixels by, on the grid of the labels.

        Parameters
        ----------
        name : RegionName
            "labels" (segments), "facies" (groups of labels; the labels without a
            facies file) or "global" (one region 0, the whole image).

        Returns
        -------
        darsia.Image
            The labels or facies (not copies; not to be modified), or a new zero image.

        """
        if name == "labels":
            return self.labels
        if name == "facies":
            return self.facies
        if name == "global":
            return darsia.zeros_like(self.labels, dtype=np.uint8)
        raise ValueError(
            f"Unknown regions {name!r}, use one of {get_args(RegionName)}."
        )

    # ! ---- ILLUMINATION CORRECTION ----
    def setup_illumination_correction(
        self,
        config: IlluminationCorrectionConfig | None,
        log: Path | None = None,
        show_plot: bool = False,
        image_porosity_config: ImagePorosityConfig | None = None,
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
        image_porosity_config : ImagePorosityConfig | None
            `[image_porosity]`: with `config.mask = "boolean_porosity"`, the correction is
            fitted on the boolean image porosity of the current baseline (method >
            `tol`, before cutoff and boundary band), computed here and not kept. None
            means the defaults.

        Notes
        -----
            Illumination calibration in Rig intentionally uses the shape-corrected
            baseline as setup input.
        """
        illumination_correction = darsia.IlluminationCorrection()

        # Fetch samples for illumination correction per region ID (`config.labels`).
        if config is not None:
            sample_groups = []
            if not config.labels:
                # If no IDs specified, use random samples from the whole image.
                samples = illumination_correction.select_random_samples(
                    mask=darsia.ones_like(self.shape_corrected_baseline, dtype=bool),
                    config=config,
                )
                sample_groups.append(samples)
            else:
                regions = self.regions(config.regions)
                for label in config.labels:
                    assert label in regions.img, (
                        f"Region {label} not found in the {config.regions} "
                        "([corrections.illumination] regions)."
                    )
                    mask = regions.img == label
                    samples = illumination_correction.select_random_samples(
                        mask=mask, config=config
                    )
                    sample_groups.append(samples)
                del regions

            # Fit on the boolean image porosity of the current baseline, if asked for
            if config.mask == "boolean_porosity":
                from darsia.presets.workflows.processing.image_porosity import (
                    boolean_image_porosity,
                    detect_image_porosity,
                )

                porosity_config = image_porosity_config or ImagePorosityConfig()
                mask = boolean_image_porosity(
                    detect_image_porosity(
                        self.baseline,
                        self.regions(porosity_config.regions),
                        porosity_config,
                    ),
                    porosity_config,
                )
            else:
                mask = darsia.ones_like(
                    self.shape_corrected_baseline, mode="voxels", dtype=bool
                )

            # Determine illumination correction based on inputs
            illumination_correction.setup(
                # Use shape-corrected baseline as explicit setup input.
                base=self.shape_corrected_baseline,
                sample_groups=sample_groups,
                mask=mask,
                outliers=config.outliers,
                filter=lambda x: skimage.filters.gaussian(x, sigma=config.sigma),
                colorspace=config.colorspace,
                interpolation=config.interpolation,
                show_plot=show_plot,
                log=log,
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
        baseline_images = [self.read(path) for path in config.baseline_paths]

        illumination_correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=baseline_images,
            labels=self.labels,
            nw=config.nw,
            eps=config.eps,
            show_images=show_plot,
        )
        return illumination_correction

    # ! ---- POROSITY ----

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
    ) -> None:
        """Fast setup."""
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

        # Fetch depth map
        self.setup_depth(
            depth_map_path,
            log=log,
        )

        # Add labels
        self.setup_labels(
            path=labels_path,
            apply_corrections=True,
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
                apply_corrections=True,
                log=log,
                show_plot=show_plot,
            )
        else:
            self.facies = self.labels.copy()

        # Setup facies props
        self.setup_facies_props(facies_props_path)

        # Define geometry for integration
        self.setup_geometry()

        # Setup color corrections (wait until here to use label information; the
        # illumination correction uses the image porosity configuration for its mask).
        # The image porosity itself is a setup step of its own, after the rig.
        self.setup_color_corrections(
            corrections_config=corrections_config,
            roi_registry=roi_registry,
            log=log,
            show_plot=show_plot,
            image_porosity_config=image_porosity_config,
        )

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

        # Create folder if not exists
        folder.mkdir(parents=True, exist_ok=True)

        # Dump meta data to json file
        # TODO: Needed for what?
        meta_data = {
            "baseline_path": str(self.baseline_path),
        }
        with open(folder / "meta_data.json", "w") as f:
            json.dump(meta_data, f)

        # Save reading information
        self.baseline.save(folder / "baseline.npz")
        if hasattr(self, "shape_corrected_baseline"):
            self.shape_corrected_baseline.save(folder / "shape_corrected_baseline.npz")

        # Save split correction pipelines.
        for i, correction in enumerate(self.shape_corrections):
            correction_name = type(correction).__name__.lower()
            correction.save(folder / f"shape_correction_{i}_{correction_name}.npz")
        for i, correction in enumerate(self.color_corrections):
            correction_name = type(correction).__name__.lower()
            correction.save(folder / f"color_correction_{i}_{correction_name}.npz")

        # Save geometry information
        try:
            assert isinstance(self.depth, darsia.Image)
            self.depth.save(folder / "depth.npz")
        except Exception:
            warn("Depth not available for saving.", UserWarning)

        # Save labels information
        try:
            self.labels.save(folder / "labels.npz")
        except Exception:
            warn("Labels not available for saving.", UserWarning)

        # Save facies information
        try:
            self.facies.save(folder / "facies.npz")
        except Exception:
            warn("Facies not available for saving.", UserWarning)

        # Save facies properties
        try:
            self.porosity.save(folder / "porosity.npz")
        except Exception:
            warn("Porosity not available for saving.", UserWarning)

        try:
            self.permeability.save(folder / "permeability.npz")
        except Exception:
            warn("Permeability not available for saving.", UserWarning)

        # Use green color and hyperlink
        folder_uri = Path(folder).absolute().as_uri()
        logger.info(
            """\033[92mRig object saved to \033]8;;"""
            f"""{folder_uri}\033\\{folder}\033]8;;\033\\\033[0m"""
        )

    @classmethod
    def load(
        cls,
        folder: Path,
        corrections_config: CorrectionsConfig | None = None,
    ) -> "Rig":
        """Load rig object from file.

        Mimick the save method.

        Parameters
        ----------
        folder : Path
            Path to the folder where the rig object is saved.
        corrections_config : CorrectionsConfig | None
            Corrections of the config (``[corrections]``).

        The image porosity is not part of the rig: see the setup step "Image porosity"
        (`processing/image_porosity`).
        """
        # Create rig object
        rig = cls()

        # TODO: move into single function self.setup_basics() or so.
        # Load meta data
        with open(folder / "meta_data.json", "r") as f:
            meta_data = json.load(f)
        rig.baseline_path = Path(meta_data["baseline_path"])

        # Load data for reading images
        with darsia.timed("baseline"):
            rig.baseline = darsia.imread(folder / "baseline.npz")

        # Only needed to set up color corrections or to save the rig: read on first use
        shape_corrected_path = folder / "shape_corrected_baseline.npz"
        if shape_corrected_path.exists():
            rig._defer_file("shape_corrected_baseline", shape_corrected_path)
        else:
            rig._defer(
                ("shape_corrected_baseline",),
                lambda: setattr(rig, "shape_corrected_baseline", rig.baseline.copy()),
            )
        logger.info("Baseline setup complete.")

        # Load corrections
        with darsia.timed("corrections"):
            rig.load_corrections(folder, corrections_config=corrections_config)

        # Load depth map
        with darsia.timed("depth"):
            rig.setup_depth(path=folder / "depth.npz")

        # Load labels information - corrections not needed assuming labels are aligned with
        # the corrected baseline.
        with darsia.timed("labels"):
            rig.setup_labels(path=folder / "labels.npz", apply_corrections=False)

        # Inner labels (only for restorations ignoring label boundaries): computed on
        # first use
        rig._defer(("inner_labels",), rig.setup_inner_labels)

        # Load facies information - corrections not needed assuming facies are aligned with
        with darsia.timed("facies"):
            rig.setup_facies(path=folder / "facies.npz", apply_corrections=False)

        # Load facies properties; the permeability (e.g. for Wasserstein distances) is
        # read on first use.
        with darsia.timed("facies properties"):
            rig.porosity = rig.read_field(folder / "porosity.npz")
            rig._defer_file(
                "permeability", folder / "permeability.npz", read=rig.read_field
            )

        # Setup geometry information
        with darsia.timed("geometry"):
            rig.setup_geometry()

        # TODO setup color_analysis, and pw_transformation.

        logger.info("Rig object loaded.")
        return rig

    # ! ---- I/O ----

    @darsia.timed("read image")
    def read(
        self,
        source: Path | bytes,
        *,
        date: datetime | None = None,
        native: bool = False,
    ) -> darsia.Image:
        """Read an image, given as a file or as the bytes of a file, and apply the
        corrections of the rig.

        The date of the image is, in this order: the given `date`; for a file, the date
        of the imaging protocol if an experiment is loaded (see `load_experiment`; a
        file missing from the protocol raises); otherwise none. The time of the image
        counts from the start of the loaded experiment (the first injection); without
        an experiment, from the date of the baseline. Without a date or such a
        reference, the image has no time. Analyses which need the date or the time check
        for it themselves.

        Parameters
        ----------
        source : Path or bytes
            Image file, or the bytes of an image file (e.g. received over a network).
        date : datetime, optional
            Date of the image, instead of the one of the imaging protocol.
        native : bool
            Keep the data type resulting from the corrections (the integers of the
            file without a type correction) instead of converting to float in the
            end, see `darsia.imread_from_optical`. Saves memory when many images are
            held; the user of the image has to support the type (`Image.as_float`).

        Returns
        -------
        darsia.Image
            Image with the corrections of the rig applied.

        """
        experiment = getattr(self, "experiment", None)
        if date is None and experiment is not None and not isinstance(source, bytes):
            date = experiment.get_datetime(source)
        if experiment is not None:
            reference_date = self.reference_date
        else:
            baseline = getattr(self, "baseline", None)
            reference_date = None if baseline is None else baseline.date

        return darsia.imread(
            source,
            transformations=self.corrections,
            date=date,
            reference_date=reference_date,
            native=native,
            **({} if isinstance(source, bytes) else {"name": source.name}),
        )

    def read_field(self, path: Path, *, on_photo: bool = False) -> darsia.Image:
        """Read a field of the rig (labels, facies, depth, porosity, ...) onto the
        grid of the baseline.

        Parameters
        ----------
        path : Path
            File of the field.
        on_photo : bool
            Whether the field was drawn on the uncorrected baseline photo (e.g. labels
            from the labeling step). Then the shape corrections for fields follow:
            nearest neighbor resize, curvature correction, final nearest neighbor
            resize. Drift and type corrections do not apply to fields drawn on the
            baseline photo itself. Otherwise (default, e.g. saved rigs) the field lies
            on the corrected baseline and is transferred to its grid
            (`darsia.transfer`): center sampling for integer fields, area averaging
            for float fields; the identity on the grid of the baseline.

        Returns
        -------
        darsia.Image
            The field on the grid of the baseline.

        Raises
        ------
        ValueError
            If a field on the corrected baseline covers another domain than the
            baseline.

        """
        field = darsia.imread(path)
        if not on_photo:
            return darsia.transfer(field, self.baseline)
        if hasattr(self, "resize_correction_inter_nearest"):
            field = self.resize_correction_inter_nearest(field)
        if hasattr(self, "curvature_correction"):
            field = self.curvature_correction(field)
        return self.resize_correction_inter_nearest_final(field)

    # ! ---- EXPERIMENT AND PROTOCOLS ----

    def load_experiment(self, experiment: darsia.ProtocolledExperiment) -> None:
        """Load experiment and associated protocols.

        This gives images read from files their date (`read`) and is required to
        compute the injected mass correctly.

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
