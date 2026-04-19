# TODO: Any connections to darsia.SimpleFluidFlower?
# TODO: integrate into darsia4porotwin.FluidFlower?
# TODO: Clean up concentration analyses. Can be removed?

import json
import logging
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia
from darsia.presets.analysis.porosity import patched_porosity_analysis
from darsia.presets.workflows.config.corrections import (
    CorrectionsConfig,
    IlluminationCorrectionConfig,
)
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.facies_props import FaciesProps
from darsia.presets.workflows.setup.illustrations import save_scalar_map_illustration

logger = logging.getLogger(__name__)

# TODO keep log? or use results?


class Rig:
    """Rig object for CO2 analysis."""

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
                darsia.Resize,
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
            ),
        )

    def setup_reading(
        self,
        baseline_path: Path,
        experiment: darsia.ProtocolledExperiment,
        corrections_config: CorrectionsConfig | None = None,
        log: Path | None = None,
        show_plot: bool = False,
    ):
        # Cache experiment for protocol-based datetime lookup
        self.experiment = experiment

        # Setup (based on baseline image without any corrections applied)
        pre_baseline = darsia.imread(baseline_path)
        self.setup_shape_corrections(
            pre_baseline=pre_baseline,
            corrections_config=corrections_config,
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

        if True:  # corrections_config.resize:
            # Define resize correction that resizes to the shape of the baseline image.
            # This is needed to ensure that later curvature corrections or concentration
            # analysis work correctly.
            self.resize_correction = darsia.Resize(
                shape=baseline_for_setup.shape[: baseline_for_setup.space_dim]
            )
            """Resize correction to baseline shape."""

            # TODO: Allow for config options for resizing, e.g. scaling or target shape.
            # This is in part covered by the curvature correction.
            if corrections_config.resize:
                raise NotImplementedError("Custom resize options not implemented yet.")
                self.rescale_correction = darsia.Resize(
                    fx=corrections_config.resize.scale,
                    fy=corrections_config.resize.scale,
                    shape=corrections_config.resize.target_shape,
                )

            self.resize_correction_inter_nearest = darsia.Resize(
                shape=baseline_for_setup.shape[: baseline_for_setup.space_dim],
                interpolation="inter_nearest",
            )
            """Resize for int data."""

            # Update corrections workflow
            self.shape_corrections.append(self.resize_correction)

        if corrections_config.drift:
            # Define translation correction object based on color checker
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
                    f"Color checker not found. Drift correction not setup. Error: {e}",
                    UserWarning,
                )
                self.drift_correction = darsia.DriftCorrection(baseline_for_setup)

            # Update corrections workflow
            self.shape_corrections.append(self.drift_correction)

        if corrections_config.curvature:
            # Define curvature correction as derived from analysis of laser grid images
            self.curvature_correction = darsia.CurvatureCorrection(
                config=corrections_config.curvature.config
            )
            """Curvature correction based on laser grid analysis."""
            baseline_for_setup = self.curvature_correction(baseline_for_setup)

            # Update corrections workflow
            self.shape_corrections.append(self.curvature_correction)

        logger.info("Shape corrections setup complete.")

    def setup_color_corrections(
        self,
        corrections_config: CorrectionsConfig | None = None,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup label-dependent color corrections after labels/porosity are available.

        Execution order is fixed:
        1) illumination, 2) relative color, 3) color correction.
        Note: relative color setup is currently guarded/unsupported in Rig and only
        kept as an explicit reserved stage in this ordering.
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

        # 2) Relative color correction (reserved in ordering; setup currently guarded).
        if corrections_config.relative_color:
            warn(
                "relative_color requested but automated setup in Rig is not implemented; "
                "skipping relative color correction.",
                UserWarning,
            )

        # 3) Color correction.
        if corrections_config.color:
            # Define color correction based on color checker on shape-corrected baseline.
            try:
                _, cc_voxels = darsia.find_colorchecker(
                    self.shape_corrected_baseline, corrections_config.color.colorchecker
                )
                self.color_correction = darsia.ColorCorrection(
                    self.shape_corrected_baseline,
                    config={
                        "roi": cc_voxels,
                        "clip": False,
                    },
                )
            except Exception as e:
                warn(
                    f"Color checker not found. Color correction not setup. Error: {e}",
                    UserWarning,
                )
                self.color_correction = darsia.ColorCorrection(
                    self.shape_corrected_baseline
                )
            """Color correction based on color checker alignment."""
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

        Args:
            path (Path | None): Path to the precomputed depth map file.
                If provided, it will be loaded and used as the depth map.
            log (Path | None): Path to the log folder where geometry images will be saved.

        """
        # Load depth map from file and reshape to baseline shape
        assert path.exists(), f"Path to depth map {path} does not exist."
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
        """Geometry for volumetric integration of images."""
        logging.info("Geometry setup completed.")

    # ! ---- LABELS -----

    def setup_labels(
        self, path: Path, apply_corrections: bool = False, log: Path | None = None
    ) -> None:
        """Setup labels for the rig object.

        This method loads labels from a specified path and applies corrections if needed.

        Args:
            path (Path): Path to the labels file. If the file exists, it will be loaded.
            apply_correction (bool): If True, applies corrections to the labels based on
                the baseline image.
            log (Path | None): Path to the log folder where label images will be saved.

        """
        assert path.exists(), f"Labels file {path} does not exist."
        if apply_corrections:
            # Assume that the labels are based on non-corrected baseline image.
            # Thus, need to apply the relevant corrections from the read routine.
            plain_labels = darsia.imread(path)
            labels = plain_labels
            if hasattr(self, "resize_correction_inter_nearest"):
                labels = self.resize_correction_inter_nearest(labels)
            if hasattr(self, "curvature_correction"):
                labels = self.curvature_correction(labels)
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

    def setup_facies(
        self, path: Path, apply_corrections: bool = False, log: Path | None = None
    ) -> None:
        """Setup facies.

        This method loads facies from a specified path and applies corrections if needed.

        Args:
            path (Path): Path to the facies file. If the file exists, it will be loaded.
            apply_correction (bool): If True, applies corrections to the facies based on
                the baseline image.
            log (Path | None): Path to the log folder where facies images will be saved.

        """
        assert path.exists(), f"Facies file {path} does not exist."
        if apply_corrections:
            # Assume that the facies are based on non-corrected baseline image.
            # Thus, need to apply the relevant corrections from the read routine.
            plain_facies = darsia.imread(path)
            facies = plain_facies
            if hasattr(self, "resize_correction_inter_nearest"):
                facies = self.resize_correction_inter_nearest(facies)
            if hasattr(self, "curvature_correction"):
                facies = self.curvature_correction(facies)
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

        logger.info("Facies setup completed.")

    def setup_facies_props(
        self,
        props_path: Path | None = None,
        porosity: Path | None = None,
        permeability: Path | None = None,
    ) -> None:
        """Define facies properties like porosity.

        Args:
            props_path (Path | None): Path to the facies properties CSV file.
                If provided, it will load the facies properties from this path.
            porosity (Path | None): Path to the porosity image file. If provided,
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

        Args:
            config (IlluminationCorrectionConfig | None): Configuration for the illumination
                correction. If provided, it will set up the illumination correction based
                on this configuration.
            log (Path | None): Path to the log folder where diagnostic plots will be saved.
            show_plot (bool): Whether to show diagnostic plots during setup (default: False).

        Notes:
            Illumination calibration in Rig intentionally uses the shape-corrected
            baseline as setup input.

        """
        illumination_correction = darsia.IlluminationCorrection()

        # Fetch samples for illumination correction based on labels and baseline.
        if config is not None:
            sample_groups = []
            if not config.labels:
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

            # Determine illumination correction based on inputs
            illumination_correction.setup(
                # Use shape-corrected baseline as explicit setup input.
                base=self.shape_corrected_baseline,
                sample_groups=sample_groups,
                mask=self.boolean_porosity,
                outliers=config.outliers,
                filter=lambda x: skimage.filters.gaussian(x, sigma=config.sigma),
                colorspace=config.colorspace,
                interpolation=config.interpolation,
                show_plot=show_plot,
                log=log,
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

        Args:
            path (Path | None): Path to a previously saved porosity ``.npz`` file.
                When given, the file is loaded and *config* is not used.
            log (Path | None): Folder for diagnostic output.  When given, a JPG
                illustration is stored to ``log/image_porosity/image_porosity.jpg``.
            config (ImagePorosityConfig | None): Porosity configuration.  Defaults to
                ``ImagePorosityConfig()`` (i.e. ``mode="full"``) when not provided.
            show_plot (bool): When ``True`` the image porosity is displayed interactively.
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
            out_dir = Path(log) / "image_porosity"
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

        Args:
            threshold (float | None): Override tolerance value.  Deprecated in favour of
                ``config.tol``; kept for backward compatibility.
            log (Path | None): Folder for diagnostic output.  When given, a JPG
                illustration is stored to
                ``log/image_porosity/boolean_porosity.jpg``.
            config (ImagePorosityConfig | None): Porosity configuration.  Falls back to
                the config stored by the last call to :meth:`setup_image_porosity`, and
                finally to ``ImagePorosityConfig()`` (``mode="full"``).
            show_plot (bool): When ``True`` the boolean porosity is displayed
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
            out_dir = Path(log) / "image_porosity"
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

    def setup(
        self,
        experiment: darsia.ProtocolledExperiment,
        baseline_path: Path,
        depth_map_path: Path,
        labels_path: Path,
        facies_path: Path | None = None,
        facies_props_path: Path | None = None,
        corrections_config: CorrectionsConfig | None = None,
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
            log=log,
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

        # Add facies
        if facies_path is not None:
            self.setup_facies(
                path=facies_path,
                apply_corrections=True,
                log=log,
            )
        else:
            self.facies = self.labels.copy()

        # Setup facies props
        self.setup_facies_props(facies_props_path)

        # Define geometry for integration
        self.setup_geometry()

        # Setup porosity based on baseline
        self.setup_image_porosity(
            log=log, config=image_porosity_config, show_plot=show_plot
        )

        # Define boolean image porosity
        self.setup_boolean_image_porosity(log=log, show_plot=show_plot)

        # Setup color corrections (wait until here to use label and porosity information)
        self.setup_color_corrections(
            corrections_config=corrections_config,
            log=log,
            show_plot=show_plot,
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

        # Save porosity information
        try:
            self.image_porosity.save(folder / "image_porosity.npz")
        except Exception:
            warn("Image porosity not available for saving.", UserWarning)

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

        Args:
            folder (Path): Path to the folder where the rig object is saved.

        """
        # Create rig object
        rig = cls()

        # TODO: move into single function self.setup_basics() or so.
        # Load meta data
        with open(folder / "meta_data.json", "r") as f:
            meta_data = json.load(f)
        rig.baseline_path = Path(meta_data["baseline_path"])

        # Load data for reading images
        rig.baseline = darsia.imread(folder / "baseline.npz")
        if (folder / "shape_corrected_baseline.npz").exists():
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

        # Load facies information - corrections not needed assuming facies are aligned with
        rig.setup_facies(path=folder / "facies.npz", apply_corrections=False)

        # Load facies properties
        rig.setup_facies_props(
            porosity=folder / "porosity.npz",
            permeability=folder / "permeability.npz",
        )

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

        # Extract coordinates (x, y) and values if in coordinate format, otherwise treat as array data
        coordinates_x = data[:, 0]
        coordinates_y = data[:, 1]
        values = data[:, 2]

        # Determine the shape = frequency of x_coordinates (fastest changing) and y_coordinates (slowest changing)
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

        # Reshape values to the determined shape, remember that the values are ordered wrt. Euclidean coordinates,
        # with x changing fastest, so we need to reshape accordingly. Also, we need to switch to row-col order,
        # with origin at the top-left corner, so we need to flip the y-coordinates and reshape accordingly.
        values_reshaped = values.reshape(
            shape, order="F"
        )  # Reshape to (row, col) with Fortran order (y changes fastest)
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

    def read_image(self, path: Path) -> darsia.Image:
        """Read image from file and apply corrections.

        The respective date is extracted from the path using the imaging protocol.

        Args:
            path (Path): Path to the image file.

        Returns:
            darsia.Image: Image object with applied corrections.

        """
        assert hasattr(
            self, "experiment"
        ), "Experiment not defined. Run load_experiment() first."
        # Convert date from path
        date = self.experiment.get_datetime(path)

        # Read image from file and apply corrections
        img = darsia.imread(
            path,
            transformations=self.corrections,
            date=date,
            reference_date=self.reference_date,
            name=path.name,
        )

        return img

    # ! ---- EXPERIMENT AND PROTOCOLS ----

    def load_experiment(self, experiment: darsia.ProtocolledExperiment) -> None:
        """Load experiment and associated protocols.

        This is required to read images and compute the injected mass correctly.

        Args:
            experiment (darsia.ProtocolledExperiment): Experiment object containing imaging,
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

        Args:
            path (Path): Path to the image file.

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
