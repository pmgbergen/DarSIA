"""Capabilities to analyze concentrations/saturation profiles based on image comparison.

"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class ConcentrationAnalysis:
    """Class providing the capabilities to determine concentration/saturation
    profiles based on image comparison, and tuning of concentration-intensity
    maps.

    """

    # ! ---- Setter methods

    def __init__(
        self,
        base: Optional[
            Union[
                darsia.Image,
                list[darsia.Image],
            ]
        ] = None,
        signal_reduction: darsia.SignalReduction = None,
        balancing: Optional[darsia.Model] = None,
        restoration: Optional[darsia.TVD] = None,
        model: Optional[darsia.Model] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Constructor of ConcentrationAnalysis.

        Args:
            base (Image or list of such): baseline image(s); if multiple provided,
                these are used to define a cleaning filter.
            signal_reduction (darsia.SignalReduction): reduction from multi-dimensional
                to 1-dimensional data; default value (None) denotes an identity
                operation.
            balancing (darsia.Model, optional): operator balancing the signal, e.g., in
                different facies; the default value (None) denotes an identity
                operation.
            restoration (darsia.TVD, optional): regularizer; the default value (`None`)
                denotes an identity.
            model (darsia.Model, optional): Conversion of signals to actual physical
                data; default value (None) denotes an identity operation.
            labels (array, optional): labeled image of domain; the default value (None)
                denotes the presence of a homogeneous medium.
            kwargs (keyword arguments): interface to all tuning parameters.
                - 'diff option': option for defining differences of images
                    (options: 'positive', 'negative', 'absolute', 'plain')
                - 'restoration -> model': option for defining order of routines;
                    if True, restoration is applied before model conversion.
                - 'verbosity':
                    - 0: no intermediate results are displayed
                    - 1: only final result is displayed
                    - 2: intermediate results are displayed

        """
        self.base: Optional[darsia.Image] = None
        """Baseline image."""
        self._base_collection: list[darsia.Image] = []
        """Collection of baseline images."""
        if base is not None:
            if not isinstance(base, list):
                base = [base]
            # Make sure that the image is converted to float for substraction
            if any(
                [img.img.dtype not in [float, np.float32, np.float64] for img in base]
            ):
                base = [img.img_as(float) for img in base]
                warn(
                    "The baseline image needed to be converted to float for substraction."
                )
            self.base = base[0].copy()
            self._base_collection = base
            if self.base.space_dim != 2:
                raise NotImplementedError

        self.signal_reduction = signal_reduction
        """Reduction to scalar signal."""

        self.balancing = balancing
        """Balancing for heterogeneous signals."""

        self.model = model
        """Signal to data conversion model."""

        self.restoration = restoration
        """Restoration model."""

        self.labels = labels
        """Indicator for heterogeneous image."""

        self._diff_option = kwargs.get("diff option", "absolute")
        """Option for defining differences of images."""

        self.first_restoration_then_model = kwargs.get("restoration -> model", True)
        """Option for defining order of routines."""

        # Define a cleaning filter based on remaining images.
        self.find_cleaning_filter()

        self.mask: Optional[np.ndarray] = (
            None if self.base is None else np.ones(self.base.img.shape[:2], dtype=bool)
        )
        """Mask."""

        self.verbosity: int = kwargs.get("verbosity", 0)
        """Fetch verbosity. With increasing number, more intermediate results
        are displayed. Useful for parameter tuning."""

    def update(
        self,
        base: Optional[darsia.Image] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Update of the baseline image or parameters.

        Args:
            base (Image, optional): image array
            mask (np.ndarray, optional): boolean mask, detecting which pixels
                will be considered, all other will be ignored in the analysis.

        """
        if base is not None:
            self.base = base.copy()
            # Make sure that the image is converted to float for substraction
            if any(
                [img.img.dtype not in [float, np.float32, np.float64] for img in base]
            ):
                base = [img.img_as(float) for img in base]
                warn(
                    "The baseline image needed to be converted to float for substraction."
                )
        if mask is not None:
            self.mask = mask

    # ! ---- Cleaning filter methods

    def find_cleaning_filter(
        self,
        baseline_images: Optional[list[darsia.Image]] = None,
        reset: bool = False,
    ) -> None:
        """Determine structural noise by studying a series of baseline images.
        The resulting cleaning filter will be used prior to the conversion
        of signal to concentration. The cleaning filter should be understood
        as thresholding mask.

        Args:
            baseline_images (list of images): series of baseline_images; default: use
                internally available baseline images.
            reset (bool): flag whether the cleaning filter shall be reset.

        """

        if baseline_images is None and self.base is not None:
            # Use internal baseline images, if available.
            baseline_images = self._base_collection[1:]
            if len(baseline_images) == 0:
                baseline_images = None

        self.threshold_cleaning_filter = None
        """Cleaning filter."""

        # Learn structural noise from collection of images
        if baseline_images is not None:
            self.threshold_cleaning_filter = np.zeros(
                self.base.img.shape[:2], dtype=float
            )

            # Combine the results of a series of images
            for img in baseline_images:
                probe_img = img.copy()

                # Take (unsigned) difference
                diff = self._subtract_background(probe_img)

                # Extract scalar version
                monochromatic_diff = self._reduce_signal(diff)

                # Consider elementwise max
                self.threshold_cleaning_filter = np.maximum(
                    self.threshold_cleaning_filter, monochromatic_diff
                )

    def read_cleaning_filter_from_file(self, path: Union[str, Path]) -> None:
        """Read cleaning filter from file.

        Args:
            path (str or Path): path to cleaning filter array.

        """
        # Fetch the threshold mask from file
        self.threshold_cleaning_filter = np.load(path)

        # Resize threshold mask if unmatching the size of the base image
        if self.base is not None:
            base_shape = self.base.img.shape[:2]
            if self.threshold_cleaning_filter.shape[:2] != base_shape:
                self.threshold_cleaning_filter = cv2.resize(
                    self.threshold_cleaning_filter, tuple(reversed(base_shape))
                )

    def write_cleaning_filter_to_file(self, path_to_filter: Union[str, Path]) -> None:
        """Store cleaning filter to file.

        Args:
            path_to_filter (str or Path): path for storage of the cleaning filter.

        """
        path_to_filter = Path(path_to_filter)
        path_to_filter.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(path_to_filter, self.threshold_cleaning_filter)

    # ! ---- Main method

    def __call__(self, img: darsia.Image) -> darsia.Image:
        """Extract concentration based on a reference image and rescaling.

        Args:
            img (darsia.Image): probing image

        Returns:
            darsia.Image: concentration

        """
        # Make sure that the image is converted to float for substraction
        if img.img.dtype not in [float, np.float32, np.float64]:
            probe_img = copy.deepcopy(img).img_as(float)
            warn(
                "The input for concentration analysis needed to be converted to float."
            )
        else:
            probe_img = copy.deepcopy(img)

        # Remove background image
        diff = self._subtract_background(probe_img)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_diff(diff)

        # Extract monochromatic version and take difference wrt the baseline image
        signal = self._reduce_signal(diff)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_scalar_signal(signal)

        # Clean signal
        clean_signal = self._clean_signal(signal)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_clean_signal(clean_signal)

        # Balance signal (take into account possible heterogeneous effects)
        balanced_signal = self._balance_signal(clean_signal)

        # Regularize/upscale signal to Darcy scale and convert from signal to concentration
        # or other way around.
        if self.first_restoration_then_model:
            smooth_signal = self._restore_signal(balanced_signal)
            concentration = self._convert_signal(smooth_signal, diff)
        else:
            nonsmooth_concentration = self._convert_signal(balanced_signal, diff)
            concentration = self._restore_signal(nonsmooth_concentration)

        # Invoke plot
        if self.verbosity >= 1:
            plt.show()

        metadata = img.metadata()
        is_scalar = len(concentration.shape) == len(img.shape) - 1
        if is_scalar:
            return darsia.ScalarImage(concentration, **metadata)
        else:
            return type(img)(concentration, **metadata)

    # ! ---- Inspection routines
    def _inspect_diff(self, img: np.ndarray) -> None:
        """Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image

        """
        if self.verbosity >= 2:
            plt.figure("Difference")
            plt.imshow(img)

    def _inspect_scalar_signal(self, img: np.ndarray) -> None:
        """Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image

        """
        if self.verbosity >= 2:
            plt.figure("Scalar signal")
            plt.imshow(img)

    def _inspect_clean_signal(self, img: np.ndarray) -> None:
        """Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image

        """
        if self.verbosity >= 2:
            plt.figure("Clean signal")
            plt.imshow(img)

    # ! ---- Pre- and post-processing methods
    def _subtract_background(self, img: darsia.Image) -> darsia.Image:
        """Take difference between input image and baseline image, based
        on cached option.

        Args:
            img (darsia.Image): test image.

        Returns:
            darsia.Image: difference with background image

        """

        if self.base is None:
            if self._diff_option == "positive":
                diff = np.clip(img.img, 0, None)
            elif self._diff_option == "negative":
                diff = np.clip(-img.img, 0, None)
            elif self._diff_option == "absolute":
                diff = np.absolute(img.img)
            elif self._diff_option == "plain":
                diff = img.img
            else:
                raise ValueError(f"Diff option {self._diff_option} not supported")
        else:
            if self._diff_option == "positive":
                diff = np.clip(img.img - self.base.img, 0, None)
            elif self._diff_option == "negative":
                diff = np.clip(self.base.img - img.img, 0, None)
            elif self._diff_option == "absolute":
                diff = skimage.util.compare_images(
                    img.img, self.base.img, method="diff"
                )
            elif self._diff_option == "plain":
                diff = img.img - self.base.img
            else:
                raise ValueError(f"Diff option {self._diff_option} not supported")

        return diff

    def _reduce_signal(self, img: np.ndarray) -> np.ndarray:
        """Make a scalar image from potentially multi-colored image.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: monochromatic reduction of the array

        """
        if self.signal_reduction is None:
            return img
        else:
            return self.signal_reduction(img)

    def _clean_signal(self, img: np.ndarray) -> np.ndarray:
        """Apply cleaning thresholds.

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: cleaned image

        """
        return (
            img
            if self.threshold_cleaning_filter is None
            else np.clip(img - self.threshold_cleaning_filter, 0, None)
        )

    def _balance_signal(self, img: np.ndarray) -> np.ndarray:
        """Routine responsible for rescaling wrt segments.

        Here, it is assumed that only one segment is present.
        Thus, no rescaling is performed.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: balanced image

        """
        return img if self.balancing is None else self.balancing(img)

    def _restore_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply restoration.

        Args:
            signal (np.ndarray): input signal

        Return:
            np.ndarray: smooth signal
        """
        return signal if self.restoration is None else self.restoration(signal)

    def _convert_signal(self, signal: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """Postprocessing routine, essentially converting a continuous
        signal into physical data (binary, continuous concentration etc.)

        Args:
            signal (np.ndarray): clean continous signal with values
                in the range between 0 and 1.
            diff (np.ndarray): original difference of images, allowing
                to extract new information besides the signal.

        Returns:
            np.ndarray: physical data

        """
        return signal if self.model is None else self.model(signal)


class PriorPosteriorConcentrationAnalysis(ConcentrationAnalysis):
    """Special case of the ConcentrationAnalysis performing a
    prior-posterior analysis, i.e., allowing to review the
    conversion performed through a prior model.

    """

    def __init__(
        self,
        base: Union[
            darsia.Image,
            list[darsia.Image],
        ],
        signal_reduction: darsia.SignalReduction,
        balancing: Optional[darsia.Model],
        restoration: darsia.TVD,
        prior_model: darsia.Model,
        posterior_model: darsia.Model,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # Cache the posterior model
        self.posterior_model = posterior_model

        # Define the concentration analysis (note the prior_model is stored under self.model)
        super().__init__(
            base,
            signal_reduction,
            balancing,
            restoration,
            prior_model,
            labels,
            **kwargs,
        )

    def _convert_signal(self, signal: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """Postprocessing routine, essentially converting a continuous
        signal into physical data (binary, continuous concentration etc.)
        Use a prior-posterior approach, allowing to review the prior choice.

        Args:
            signal (np.ndarray): mooth signal
            diff (np.ndarray): original difference of images, allowing
                to extract new information besides the signal.

        Returns:
            np.ndarray: physical data
        """
        # Determine prior
        prior = self.model(signal, self.mask)

        # Determine posterior
        posterior = self.posterior_model(signal, prior, diff)

        if self.verbosity >= 2:
            plt.figure("Prior")
            plt.imshow(prior)
            plt.figure("Posterior")
            plt.imshow(posterior)

        return posterior
