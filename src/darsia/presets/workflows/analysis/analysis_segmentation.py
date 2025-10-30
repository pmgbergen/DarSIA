"""Template for segmentation analysis."""

import logging
from pathlib import Path
import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)
from darsia.presets.workflows.mass_computation import MassComputation
from darsia.utils.augmented_plotting import plot_contour_on_image

logger = logging.getLogger(__name__)


class SegmentationAnalysis:
    """Threshold based segmentation analysis."""

    def __init__(
        self,
        thresholds: dict[str, list[float, float]],
        colors: dict[str, str] | None = None,
        linewidth: int = 2,
    ):
        self.thresholds = thresholds
        # convert None to +/- inf
        for label, (lower, upper) in self.thresholds.items():
            if lower is None:
                lower = -float("inf")
            if upper is None:
                upper = float("inf")
            self.thresholds[label] = (lower, upper)
        self.colors = colors if colors is not None else {}
        self.linewidth = linewidth

    def extract_mask(self, img: darsia.ScalarImage, label: str) -> darsia.ScalarImage:
        """Extract phase based on thresholding.

        Args:
            img: Signal to segment.
            label: Label to extract.

        Returns:
            darsia.Image: Segmented phase (boolean) image.

        """
        assert label in self.thresholds, f"Label {label} not found in thresholds."
        lower, upper = self.thresholds[label]
        mask = (img.img >= lower) & (img.img <= upper)
        return darsia.ScalarImage(img=mask, **img.metadata())

    def add_contours(
        self, img: darsia.Image, masks: dict[str, darsia.ScalarImage]
    ) -> darsia.Image:
        """Add contours to image based on segmentation of mass.

        Args:
            img: Image to add contours to.
            masks: Mask as basis for contour extraction.

        Returns:
            Image with contours added.
        """
        contour_image = plot_contour_on_image(
            img=img,
            mask=list(masks.values()),
            color=[self.colors[label] for label in masks],
            thickness=self.linewidth,
            return_image=True,
        )
        return contour_image


def analysis_segmentation(
    cls,
    path: Path,
    show: bool = False,
    save_jpg: bool = False,
    save_npz: bool = False,
):
    # ! ---- LOAD RUN AND RIG ----
    config = FluidFlowerConfig(path)
    config.check(
        "analysis",
        "analysis.segmentation",
        "protocol",
        "data",
        "color_paths",
        "color_signal",
        "mass",
        "rig",
    )

    # Mypy type checking
    for c in [
        config.color_signal,
        config.color_paths,
        config.data,
        config.protocol,
        config.analysis,
        config.mass,
        config.rig,
    ]:
        assert c is not None

    # ! ---- LOAD RIG AND RUN ----

    fluidflower = cls()
    fluidflower.load(config.rig.path)

    # Load experiment
    experiment = darsia.ProtocolledExperiment(
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )
    fluidflower.load_experiment(experiment)

    # Plotting
    plot_folder = config.data.results / "segmentation"
    plot_folder.mkdir(parents=True, exist_ok=True)

    # ! ---- CONCENTRATION ANALYSIS ---- ! #

    color_signal_analysis = HeterogeneousColorAnalysis(
        baseline=fluidflower.baseline,
        labels=fluidflower.labels,
        # restoration=fluidflower.restoration,
        ignore_labels=config.color_paths.ignore_labels,
    )
    color_signal_analysis.load(config.color_signal.calibration_file)

    # ! ---- MASS ----

    experiment_start = experiment.experiment_start
    flash = darsia.SimpleFlash()
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=experiment.pressure_temperature_protocol.get_state(
            experiment_start
        ).pressure,
        temperature=experiment.pressure_temperature_protocol.get_state(
            experiment_start
        ).temperature,
    )
    fluidflower.co2_mass_analysis = co2_mass_analysis
    mass_computation = MassComputation(
        baseline=fluidflower.baseline,
        geometry=fluidflower.geometry,
        flash=flash,
        co2_mass_analysis=fluidflower.co2_mass_analysis,
    )
    mass_computation.load(config.mass.calibration_file)

    # ! ---- SEGMENTATION ---- ! #

    segmentation_analysis = SegmentationAnalysis(
        thresholds=config.analysis.segmentation.thresholds,
        colors=config.analysis.segmentation.colors,
    )

    # ! ---- ANALYSIS ----

    if len(config.analysis.image_paths) > 0:
        image_paths = [config.data.folder / p for p in config.analysis.image_paths]
    else:
        image_times = config.analysis.image_times
        image_datetimes = [
            experiment.experiment_start + darsia.timedelta(hours=t) for t in image_times
        ]
        image_paths = experiment.imaging_protocol.find_images_for_datetimes(
            paths=config.data.data,
            datetimes=image_datetimes,
        )

    # Loop over images and analyze
    for path in image_paths:
        # Extract color signal
        img = fluidflower.read_image(path)
        color_signal = color_signal_analysis(img)
        mass_analysis_result = mass_computation(color_signal)
        masks = {}
        masks["CO2(g)"] = segmentation_analysis.extract_mask(
            mass_analysis_result.saturation_g, label="CO2(g)"
        )
        masks["CO2(aq)"] = segmentation_analysis.extract_mask(
            mass_analysis_result.concentration_aq, label="CO2(aq)"
        )
        contours = segmentation_analysis.add_contours(img, masks)

        if show:
            contours.show(
                title=f"Contours for {path.stem} | {img.time} seconds", delay=False
            )

        if save_npz:
            path = plot_folder / f"{path.stem}.npz"
            raise NotImplementedError

        if save_jpg:
            path = plot_folder / f"{path.stem}.jpg"
            contours.write(path, quality=80)
