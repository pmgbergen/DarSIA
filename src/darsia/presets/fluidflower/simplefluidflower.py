"""Preset for a standard FluidFlower analysis pipeline.

This preset is designed for a simple tabletop setup with a dominating sand layer.
It includes the following steps:
1. Shape correction: drift and curvature correction
2. Segmentation: segmentation of the image
3. Color correction: illumination and color correction

The setup method is used to define the tabletop based on characteristic input images.
The read_image method is used to read an image and apply the corrections.
The save and load methods are used to save and load the tabletop.

Modifications to the pipeline can be made by changing the setup method or by adding
additional methods. The expert_knowledge method can be used to apply expert knowledge
to the image after preprocessing.

"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import skimage

import darsia


class SimpleFluidFlower:
    def __init__(
        self,
        baseline: Path,
        debug: bool = False,
    ) -> None:
        """Constructor for SimpleFluidFlower.

        The correction routines assume a simple setup (e.g., no strong curvature) and
        at least one dominating sand layer spreading the domain.

        Args:
            baseline (Path): path to baseline image
            debug (bool): flag for debugging

        """
        self.raw_baseline = darsia.imread(baseline)
        """Baseline image for the tabletop without any corrections."""

        self.reference_date = self.raw_baseline.date
        """Reference date of experiment."""

        self.corrections = []
        """List of corrections to be applied to the images."""

        self.drift_config = {}
        """Configuration for drift correction."""

        self.curvature_config = {}
        """Configuration for curvature correction."""

        self.debug = debug
        """Flag for debugging."""

    def setup(
        self,
        roi: Path,
        segmentation: Path,
        color_checker_position: darsia.ColorCheckerPosition,
        specs: dict,
        **kwargs,
    ) -> None:
        """Setup Table top based on characteristic input image (preferably the baseline).

        Args:
            roi (Path): path to image with ROI
            segmentation (Path): path to segmentation image
            color_checker_position (ColorCheckerPosition): position of the color checker
            specs (dict): specifications of the tabletop

        """

        # Specs of ROI
        self.width = specs.get("width", 0.92)
        self.height = specs.get("height", 0.55)
        self.depth = specs.get("depth", 0.012)
        self.porosity = specs.get("porosity", 0.44)

        # ! ---- SETUP SHAPE CORRECTION ----

        roi_mode = kwargs.get("roi_mode", "interactive")
        roi_color = kwargs.get("roi_color")
        self.drift_correction, self.curvature_correction = self.setup_shape_correction(
            roi, roi_mode, roi_color, color_checker_position
        )
        self.corrections = [self.drift_correction, self.curvature_correction]

        # ! ---- SETUP BASELINE ----

        self.baseline = self.raw_baseline.copy()
        for correction in self.corrections[1:]:
            self.baseline = correction(self.baseline)

        # ! ---- SETUP SEGMENTATION ----
        self.labels = self.setup_segmentation(
            segmentation
        )

        # ! ---- SETUP COLOR CORRECTION ----
        self.illumination_correction, self.color_correction = (
            self.setup_color_correction(color_checker_position)
        )
        self.corrections.extend([self.illumination_correction, self.color_correction])

        # ! ---- BASELINE ----

        self.baseline = self.raw_baseline.copy()
        for correction in self.corrections:
            self.baseline = correction(self.baseline)

        # ! ---- GEOMETRY ----

        shape_meta = self.baseline.shape_metadata()
        self.geometry = darsia.ExtrudedPorousGeometry(
            depth=self.depth, porosity=self.porosity, **shape_meta
        )

    def setup_shape_correction(
        self,
        roi: Path,
        roi_mode: Literal["interactive", "automatic"],
        roi_color: Optional[list[float]],
        color_checker_position: darsia.ColorCheckerPosition,
    ) -> tuple:
        """Setup shape correction based on provided images.

        Args:
            roi (Path): path to image with ROI
            color_checker_position (ColorCheckerPosition): position of the color checker

        Returns:
            tuple: drift_correction, curvature_correction

        """

        # Define translation correction object based on color checker
        _, cc_voxels = darsia.find_colorchecker(
            self.raw_baseline, color_checker_position
        )
        self.drift_config = {"roi": cc_voxels}
        drift_correction = darsia.DriftCorrection(
            self.raw_baseline, config=self.drift_config
        )

        # Read auxiliary images for calibration - make sure they are of the same size
        roi_image = darsia.resize(
            darsia.imread(roi),
            ref_image=self.raw_baseline,
        )

        # Define  Restrict to region with frame
        crop_assistant = darsia.CropAssistant(roi_image)
        if roi_mode == "interactive":
            # Generate curvature config from image using interactive mode
            self.curvature_config = crop_assistant()
        elif roi_mode == "automatic":
            # Generate curvature config from marked image
            self.curvature_config = crop_assistant.from_image(
                color=roi_color, width=self.width, height=self.height
            )
        else:
            raise ValueError(f"Unknown roi_mode: {roi_mode}")
        curvature_correction = darsia.CurvatureCorrection(config=self.curvature_config)
        return drift_correction, curvature_correction

    def set_corrections(self) -> None:
        if self.drift_config:
            self.drift_correction = darsia.DriftCorrection(
                self.raw_baseline, config=self.drift_config
            )

        if self.curvature_config:
            self.curvature_correction = darsia.CurvatureCorrection(
                config=self.curvature_config
            )

        if self.color_config:
            self.color_correction = darsia.ColorCorrection(config=self.color_config)

    def setup_color_correction(
        self,
        color_checker_position: darsia.ColorCheckerPosition,
    ) -> tuple:
        """Setup color correction based on color checker.

        Args:
            color_checker_position (ColorCheckerPosition): position of the color checker

        Returns:
            tuple: illumination_correction, color_correction

        """
        # Define illumination gradient correction by estimating the lightness on distributed
        # samples. Use random samples in main reservoir.
        # Define main reservoir as the label with largest count
        largest_label = np.argmax(np.bincount(self.labels.img.flatten()))
        mask = self.labels.img == largest_label

        # Find random patches, restricted to the masked regions
        width = 50
        num_patches = 10
        samples = darsia.random_patches(mask, width = width, num_patches=num_patches)

        # Find sample in the center
        # TODO

        # Determine illumination correction based on inputs
        illumination_correction = darsia.IlluminationCorrection()
        illumination_correction.setup(
            self.baseline,
            samples,
            ref_sample=-1,
            filter=lambda x: skimage.filters.gaussian(x, sigma=200),
            colorspace="hsl-scalar",
            interpolation="illumination",
            show_plot=False,
        )

        # Define color correction object - target here the same colors as in the original
        # image (modulo curvature correction)
        colorchecker, cc_aligned_voxels = darsia.find_colorchecker(
            self.baseline, color_checker_position
        )
        self.color_config = {
            "colorchecker": colorchecker,
            "roi": cc_aligned_voxels,
            "clip": False,
        }
        color_correction = darsia.ColorCorrection(config=self.color_config)

        return illumination_correction, color_correction

    def setup_segmentation(self, segmentation: Path) -> darsia.Image:
        """Setup segmentation based on provided image.

        Args:
            segmentation (Path): path to segmentation image

        Returns:
            darsia.Labels: labels object

        """
        segmentation_image = darsia.resize(
            darsia.imread(segmentation),
            ref_image=self.raw_baseline,
            interpolation="inter_nearest",
        )
        segmentation_image = self.curvature_correction(segmentation_image)

        # Define geometric segmentation using assistant
        assistant = darsia.LabelsAssistant(
            background=segmentation_image, verbosity=self.debug
        )
        labels = assistant()

        return labels

    def expert_knowledge(self, img: darsia.Image) -> None:
        """Possibility to apply expert knowledge to the image after preprocessing.

        Args:
            img (np.ndarray): image array

        """
        ...

    def read_image(self, path: Path) -> darsia.Image:
        """Read image and apply corrections.

        Args:
            path (Path): path to image

        Returns:
            darsia.Image: image object

        """

        # Read image from file and apply corrections
        img = darsia.imread(
            path, transformations=self.corrections, reference_date=self.reference_date
        )

        # Deactivate water zone
        self.expert_knowledge(img)

        return img

    # ! ---- I/O ----

    def save(self, folder: Path) -> None:
        """Save the tabletop to a folder.

        Args:
            folder (Path): path to folder

        """
        # Make sure folder exists
        folder.mkdir(parents=True, exist_ok=True)

        # Save baseline
        self.baseline.save(folder / Path("baseline.npz"))

        # Save corrections
        self.drift_correction.save(folder / Path("drift.npz"))
        self.curvature_correction.save(folder / Path("curvature.npz"))
        self.illumination_correction.save(folder / Path("illumination.npz"))
        self.color_correction.save(folder / Path("color.npz"))

        # Save segmentation
        self.labels.save(folder / Path("labels.npz"))

        # Save specs
        specs = {
           "width": self.width,
           "height": self.height,
           "depth": self.depth,
           "porosity": self.porosity,
        }
        np.savez(folder / Path("specs.npz"), specs=specs)
        print(f"Specs saved to {folder / Path("specs.npz")}.")

        print(f"Tabletop saved to {folder}.")

    def load(self, folder: Path) -> None:

        # Load baseline
        self.baseline = darsia.imread(folder / Path("baseline.npz"))
        self.reference_date = self.baseline.date

        # Load specs
        specs = np.load(folder / Path("specs.npz"), allow_pickle=True)["specs"].item()
        self.width = specs["width"]
        self.height = specs["height"]
        self.depth = specs["depth"]
        self.porosity = specs["porosity"]

        # Load corrections
        self.drift_correction = darsia.DriftCorrection(self.raw_baseline)
        self.drift_correction.load(folder / Path("drift.npz"))

        self.curvature_correction = darsia.CurvatureCorrection()
        self.curvature_correction.load(folder / Path("curvature.npz"))

        self.illumination_correction = darsia.IlluminationCorrection()
        self.illumination_correction.load(folder / Path("illumination.npz"))

        self.color_correction = darsia.ColorCorrection()
        self.color_correction.load(folder / Path("color.npz"))

        self.corrections = [
            self.drift_correction,
            self.curvature_correction,
            self.illumination_correction,
            self.color_correction,
        ]

        # Load segmentation
        self.labels = darsia.imread(folder / Path("labels.npz"))

        # Load geometry
        shape_meta = self.baseline.shape_metadata()
        self.geometry = darsia.ExtrudedPorousGeometry(
            depth=self.depth, porosity=self.porosity, **shape_meta
        )
