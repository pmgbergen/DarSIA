# TODO: Any connections to darsia.SimpleFluidFlower?
# TODO: integrate into darsia4porotwin.FluidFlower?

import json
import logging
from pathlib import Path
from warnings import warn

import darsia
import matplotlib.pyplot as plt
import numpy as np
from darsia.presets.analysis.porosity import patched_porosity_analysis

logger = logging.getLogger(__name__)


class Rig:
    """Rig object for CO2 analysis."""

    def __init__(self) -> None:
        self.define_specs()

    def define_specs(self) -> None:
        """Define some basic specs."""
        # TODO read from config file
        self.porosity = 0.44

    def setup_reading(
        self,
        baseline_path: Path,
        imaging_protocol: darsia.ImagingProtocol,
        correction_config: Path | None = None,
        log: Path | None = None,
    ):
        # Cache imaging protocol
        self.imaging_protocol = imaging_protocol

        # Setup (based on baseline image without any corrections applied)
        pre_baseline = darsia.imread(baseline_path)
        self.setup_corrections(pre_baseline=pre_baseline, config=correction_config)

        # Define reference baseline (with corrections applied)
        self.baseline = self.read_image(baseline_path)

        # Log corrected baseline image
        if log:
            # Create folder
            (Path(log) / "corrections").mkdir(parents=True, exist_ok=True)

            # Plot corrected baseline
            plt.figure()
            plt.imshow(self.baseline.img)
            plt.title("Corrected baseline")
            plt.savefig(Path(log) / "corrections" / "corrected_baseline.png", dpi=500)
            plt.close()

        logger.info("Reading setup completed.")

    def setup_corrections(
        self,
        path: Path | None = None,
        pre_baseline: darsia.Image | None = None,
        config: Path | None = None,
    ) -> None:
        """Setup corrections for the museum rig.

        Prioritize loading existing corrections from the specified path.

        Args:
            path (Path | None): Path to the folder containing correction files.
                If provided, it will load existing corrections from this path.
            pre_baseline (darsia.Image | None): Pre-baseline image used for defining
                corrections. If `path` is provided and exists, this argument is ignored.
            curvature_correction (Path | None): Path to curvature correction config.

        """
        if path and path.exists():
            self.corrections = []
            for correction_path in sorted(path.glob("correction_*.npz")):
                correction = darsia.read_correction(correction_path)
                self.corrections.append(correction)
            logger.info("Corrections setup complete.")
            return

        # ! ---- CORRECTIONS ----
        assert pre_baseline is not None, "Pre-baseline image is not set."

        # Aux: needed for rescaling not leaving the range of color space
        self.type_converter = darsia.TypeCorrection(np.float32)
        pre_baseline = self.type_converter(pre_baseline)

        # Define resize correction that resizes to the shape of the baseline image.
        # This is needed to ensure that later curvature corrrections or concentration
        # analysis work correctly.
        self.resize_correction = darsia.Resize(
            shape=pre_baseline.shape[: pre_baseline.space_dim]
        )
        """Resize correction to baseline shape."""

        self.resize_correction_inter_nearest = darsia.Resize(
            shape=pre_baseline.shape[: pre_baseline.space_dim],
            interpolation="inter_nearest",
        )
        """Resize for int data."""

        # Define translation correction object based on color checker
        try:
            _, cc_voxels = darsia.find_colorchecker(pre_baseline, "upper_left")
            self.drift_correction = darsia.DriftCorrection(
                pre_baseline, config={"roi": cc_voxels}
            )
        except Exception as e:
            warn(
                f"Color checker not found. Drift correction not setup. Error: {e}",
                UserWarning,
            )
            self.drift_correction = darsia.DriftCorrection(pre_baseline)
        """Drift correction based on color checker alignment."""

        # Define curvature correction as derived from analysis of laser grid images
        self.curvature_correction = darsia.CurvatureCorrection(config=config)
        """Curvature correction based on laser grid analysis."""

        # Define workflow of corrections
        self.corrections = [
            self.type_converter,
            self.resize_correction,
            self.drift_correction,
            self.curvature_correction,
        ]
        """Workflow of corrections applied to the baseline image."""

        logger.info("Corrections setup complete.")

    # ! ---- GEOMETRY ----

    def setup_depth(
        self,
        path: Path,
        log: Path | None = None,
    ) -> None:
        """Setup depth map for the museum rig object.

        Args:
            path (Path | None): Path to the precomputed depth map file.
                If provided, it will be loaded and used as the depth map.
            log (Path | None): Path to the log folder where geometry images will be saved.

        """
        # Load depth map from file and reshape to baseline shape
        assert path.exists(), f"Path to depth map {path} does not exist."
        pre_depth = darsia.imread(path)
        self.depth: darsia.Image = darsia.resize(pre_depth, ref_image=self.baseline)
        """Depth map for the museum rig object."""

        # Log results
        if log:
            # Create folder for geometry
            geometry_folder = Path(log) / "geometry"
            geometry_folder.mkdir(parents=True, exist_ok=True)

            # Plot depth map
            plt.figure()
            plt.imshow(self.depth.img)
            plt.colorbar()
            plt.title("Depth map")
            plt.savefig(geometry_folder / "depth_map.png", dpi=500)
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
        """Setup labels for the museum rig object.

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
            resized_labels = self.resize_correction_inter_nearest(plain_labels)
            self.labels = self.curvature_correction(resized_labels)

        else:
            # Assume the labels are aligned with the corrected baseline.
            pre_labels = darsia.imread(path)
            self.labels = darsia.resize(
                pre_labels,
                ref_image=self.baseline,
                interpolation="inter_nearest",
            )

        if log:
            # Create folder for label images
            (Path(log) / "labels").mkdir(parents=True, exist_ok=True)

            # Plot labels
            plt.figure()
            plt.imshow(self.labels.img)
            plt.colorbar()
            plt.title("Labels")
            plt.savefig(Path(log) / "labels" / "labels.png", dpi=500)
            plt.close()

        logger.info("Labels setup completed.")

    # ! ---- ILLUMINATION CORRECTION ----
    def setup_illumination_correction(self, log: Path | None = None) -> None:
        """Setup illumination correction (empty in Rig)"""
        pass

    # ! ---- POROSITY ----

    def setup_image_porosity(
        self, path: Path | None = None, log: Path | None = None, **kwargs
    ) -> None:
        """Setup image porosity based on the baseline image.

        If `path` is provided, it will load the porosity image from the specified path.
        If `path` is None, it will compute the porosity based on the baseline image
        and labels using the patched porosity analysis.

        Args:
            path (Path | None): Path to the porosity image file. If provided,
                it will load the image from this path.
            log (Path | None): Path to the log folder where porosity images will be saved.
            **kwargs: Additional keyword arguments for the patched porosity analysis:
                - patch: Tuple defining the size of the patches (default is (32, 64)).
                - gamma: Gamma value for the Gaussian kernel (default is 10).
                - sample_width: Width of the samples (default is 50).
                - num_clusters: Number of clusters for clustering (default is 5).
                - tol_color_distance: Tolerance for color distance (default is 0.1).
                - tol_color_gradient: Tolerance for color gradient (default is 0.02).

        """
        self.image_porosity = darsia.ones_like(
            self.baseline, mode="voxels", dtype=np.float32
        )
        """Image porosity for the museum rig object."""
        logger.info("Porosity setup completed.")

    def setup_boolean_image_porosity(
        self, threshold: float = 0.9, log: Path | None = None
    ) -> None:
        """Setup boolean porosity based on the defined threshold.

        Args:
            threshold (float): Threshold for defining boolean porosity.
                Default is 0.9, meaning that porosity values above 0.9 are considered
                as porous (True), and below or equal to 0.9 as non-porous (False).
            log (Path | None): Path to the log folder where boolean porosity images will be saved.

        """
        self.boolean_porosity = self.image_porosity > threshold

        if log:
            # Create folder for porosity images
            (Path(log) / "porosity").mkdir(parents=True, exist_ok=True)

            # Plot boolean porosity
            plt.figure()
            plt.imshow(self.boolean_porosity.img)
            plt.colorbar()
            plt.title("Boolean porosity")
            plt.savefig(Path(log) / "porosity" / "boolean_porosity.png")
            plt.close()

        logger.info("Porosity setup completed.")

    def setup(
        self,
        experiment: darsia.ProtocolledExperiment,
        baseline_path: Path,
        depth_map_path: Path,
        labels_path: Path,
        correction_config_path: Path | None = None,
        # ref_colorchecker_path: Path,
        log: Path | None = None,
    ) -> None:
        """Fast setup."""

        # Cache baseline path - TODO: Really needed?
        self.baseline_path = baseline_path

        # Cache reference date
        self.reference_date = experiment.experiment_start

        # Initialize the museum rig - responsible for reading/preprocessing photographs
        self.setup_reading(
            baseline_path,
            experiment.imaging_protocol,
            correction_config_path,
            log=log,
        )

        # Fetch depth map
        self.setup_depth(
            depth_map_path,
            log=log,
        )

        # Define geometry for integration
        self.setup_geometry()

        # Add labels
        self.setup_labels(
            path=labels_path,
            apply_corrections=True,
            log=log,
        )

        # Setup illumination correction
        self.setup_illumination_correction(
            log=log,
        )

        # Setup porosity based on baseline
        self.setup_image_porosity(log=log)

        # Define boolean image porosity
        self.setup_boolean_image_porosity(log=log)

        # TODO Setup concentration analysis and transformations?
        # self.co2_analysis = ...
        # self.co2_g_analysis = ...

        # Setup co2 mass analysis.
        # TODO Add?
        # self.setup_mass_analysis(
        #    atmospheric_pressure=kwargs.get("atmospheric_pressure", 1.010),
        #    temperature=kwargs.get("temperature", 23.0),
        # )

        logger.info("Museum rig setup completed.")

        # ! ---- AVERAGING ----

        # Use porosity-based averaging for restoration
        restoration = darsia.porosity_based_averaging(
            self.labels, self.image_porosity, self.baseline
        )

        # Cache
        self.restoration = restoration
        """Restoration model based on porosity-based averaging."""

        clipping = darsia.ClipModel()
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
        temperature,
    ) -> None:
        # Define plain mass analysis (combining gas and aqueous phase)
        self.co2_mass_analysis = darsia.CO2MassAnalysis(
            self.baseline,
            atmospheric_pressure=atmospheric_pressure,
            temperature=temperature,
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
        # TODO? At the moment, on needs to load the experiment to get the imaging protocol.
        # self.imaging_protocol.save(folder / "imaging_protocol.json")
        for i, correction in enumerate(self.corrections):
            correction.save(folder / f"correction_{i}.npz")

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

        # Save porosity information
        try:
            self.image_porosity.save(folder / "porosity.npz")
        except Exception:
            warn("Porosity not available for saving.", UserWarning)

        logger.info(f"Rig object saved to {folder}.")

    def load(self, folder: Path) -> None:
        """Load museum rig object from file.

        Mimick the save method.

        Args:
            folder (Path): Path to the folder where the museum rig object is saved.

        """
        # TODO: move into single function self.setup_basics() or so.
        # Load meta data
        with open(folder / "meta_data.json", "r") as f:
            meta_data = json.load(f)
        self.baseline_path = Path(meta_data["baseline_path"])

        # Load data for reading images
        self.baseline = darsia.imread(folder / "baseline.npz")
        logger.info("Baseline setup complete.")

        # Load corrections
        self.setup_corrections(folder)

        # Load depth map
        self.setup_depth(path=folder / "depth.npz")

        # Setup geometry information
        self.setup_geometry()

        # Load labels information - corrections not needed assuming labels are aligned with
        # the corrected baseline.
        self.setup_labels(path=folder / "labels.npz", apply_corrections=False)

        # Load image porosity information
        self.setup_image_porosity(path=folder / "porosity.npz")
        self.setup_boolean_image_porosity()

        # TODO setup color_analysis, and pw_transformation.

        logger.info("Museum rig object loaded.")

    # ! ---- I/O ----

    def read_image(self, path: Path) -> darsia.Image:
        """Read image from file and apply corrections.

        The respective date is extracted from the path using the imaging protocol.

        Args:
            path (Path): Path to the image file.

        Returns:
            darsia.Image: Image object with applied corrections.

        """
        assert hasattr(self, "imaging_protocol"), (
            "Imaging protocol not defined. Run load_experiment() first."
        )
        # Convert date from path
        date = self.imaging_protocol.get_datetime(path)

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
        self.imaging_protocol = experiment.imaging_protocol
        self.injection_protocol = experiment.injection_protocol
        self.pressure_temperature_protocol = experiment.pressure_temperature_protocol
        self.reference_date = experiment.experiment_start
        logger.info("Experiment and protocols loaded.")

    def update(self, path: Path) -> None:
        """Update the state of the museum rig based on the image path.

        This method updates the current date, time, pressure, and temperature
        of the museum rig object based on the provided image path.

        Args:
            path (Path): Path to the image file.

        """
        # Convert date from path
        date = self.imaging_protocol.get_datetime(path)
        self.current_date = date
        self.current_time = (date - self.reference_date).total_seconds() / 3600.0
        state = self.pressure_temperature_protocol.get_state(date)
        self.current_pressure = state.pressure
        self.current_temperature = state.temperature
        self.setup_mass_analysis(
            atmospheric_pressure=self.current_pressure,
            temperature=self.current_temperature,
        )
        logger.info(f"State updated to {self.current_date}.")
