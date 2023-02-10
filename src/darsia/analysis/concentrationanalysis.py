"""
Module that contains a class which provides the capabilities to
analyze concentrations/saturation profiles based on image comparison.
"""

import abc
import copy
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import skimage
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor

import darsia


class ConcentrationAnalysis:
    """
    Class providing the capabilities to determine concentration/saturation
    profiles based on image comparison, and tuning of concentration-intensity
    maps.
    """

    # ! ---- Setter methods

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        signal_reduction: darsia.SignalReduction,
        restoration: Optional[darsia.TVD] = None,
        model: darsia.Model = darsia.Identity,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Constructor of ConcentrationAnalysis.

        Args:
            base (darsia.Image or list of such): baseline image(s); if multiple provided,
                these are used to define a cleaning filter.
            labels (array, optional): labeled image of domain
            kwargs (keyword arguments): interface to all tuning parameters
        """
        ########################################################################

        # Fix single baseline image and reference time
        if not isinstance(base, list):
            base = [base]
        self.base: darsia.Image = base[0].copy()
        self.base_time = self.base.timestamp

        # Define scalar space.
        self.signal_reduction = signal_reduction

        # Initialize the threshold values.
        self.model = model

        # TVD parameters for pre and post smoothing
        self.apply_restoration = restoration is not None
        self.restoration = restoration

        # Cache heterogeneous distribution
        self.labels = labels

        # Option for defining differences of images.
        self._diff_option = kwargs.get("diff option", "absolute")

        # Define a cleaning filter based on remaining images.
        self.find_cleaning_filter(base[1:])

        # Mask
        self.mask: np.ndarray = np.ones(self.base.img.shape[:2], dtype=bool)

        # Fetch verbosity. With increasing number, more intermediate results
        # are displayed. Useful for parameter tuning.
        self.verbosity: int = kwargs.get("verbosity", 0)

    def update(
        self,
        base: Optional[darsia.Image] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update of the baseline image or parameters.

        Args:
            base (darsia.Image, optional): image array
            mask (np.ndarray, optional): boolean mask, detecting which pixels
                will be considered, all other will be ignored in the analysis.
        """
        if base is not None:
            self.base = base.copy()
        if mask is not None:
            self.mask = mask

    # ! ---- Cleaning filter methods

    def find_cleaning_filter(
        self, baseline_images: list[darsia.Image], reset: bool = False
    ) -> None:
        """
        Determine natural noise by studying a series of baseline images.
        The resulting cleaning filter will be used prior to the conversion
        of signal to concentration. The cleaning filter should be understood
        as thresholding mask.

        Args:
            baseline_images (list of darsia.Image): series of baseline_images.
            reset (bool): flag whether the cleaning filter shall be reset.
        """
        # Initialize cleaning filter
        self.threshold_cleaning_filter = np.zeros(self.base.img.shape[:2], dtype=float)

        # Combine the results of a series of images
        for img in baseline_images:

            probe_img = img.copy()

            # Take (unsigned) difference
            diff = self._subtract_background(probe_img)

            # Extract scalar version
            monochromatic_diff = self._extract_scalar_information(diff)

            # Consider elementwise max
            self.threshold_cleaning_filter = np.maximum(
                self.threshold_cleaning_filter, monochromatic_diff
            )

    def read_cleaning_filter_from_file(self, path: Union[str, Path]) -> None:
        """
        Read cleaning filter from file.

        Args:
            path (str or Path): path to cleaning filter array.
        """
        # Fetch the threshold mask from file
        self.threshold_cleaning_filter = np.load(path)

        # Resize threshold mask if unmatching the size of the base image
        base_shape = self.base.img.shape[:2]
        if self.threshold_cleaning_filter.shape[:2] != base_shape:
            self.threshold_cleaning_filter = cv2.resize(
                self.threshold_cleaning_filter, tuple(reversed(base_shape))
            )

    def write_cleaning_filter_to_file(self, path_to_filter: Union[str, Path]) -> None:
        """
        Store cleaning filter to file.

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
        probe_img = copy.deepcopy(img)

        # Remove background image
        diff = self._subtract_background(probe_img)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_diff(diff)

        # Extract monochromatic version and take difference wrt the baseline image
        signal = self._extract_scalar_information(diff)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_scalar_signal(signal)

        # Clean signal
        clean_signal = self._clean_signal(signal)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_clean_signal(clean_signal)

        # Homogenize signal (take into account possible heterogeneous effects)
        homogenized_signal = self._homogenize_signal(clean_signal)

        # Regularize/upscale signal to Darcy scale
        smooth_signal = self._prepare_signal(homogenized_signal)

        # Convert from signal to concentration
        concentration = self._convert_signal(smooth_signal, diff)

        # Invoke plot
        if self.verbosity >= 1:
            plt.show()

        return darsia.Image(concentration, img.metadata)

    # ! ---- Inspection routines
    def _inspect_diff(self, img: np.ndarray) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image
        """
        if self.verbosity >= 2:
            plt.figure("Difference")
            plt.imshow(img)

    def _inspect_scalar_signal(self, img: np.ndarray) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image
        """
        if self.verbosity >= 2:
            plt.figure("Scalar signal")
            plt.imshow(img)

    def _inspect_clean_signal(self, img: np.ndarray) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image
        """
        if self.verbosity >= 2:
            plt.figure("Clean signal")
            plt.imshow(img)

    # ! ---- Pre- and post-processing methods
    def _subtract_background(self, img: darsia.Image) -> darsia.Image:
        """
        Take difference between input image and baseline image, based
        on cached option.

        Args:
            img (darsia.Image): test image.

        Returns:
            darsia.Image: difference with background image
        """

        if self._diff_option == "positive":
            diff = np.clip(img.img - self.base.img, 0, None)
        elif self._diff_option == "negative":
            diff = np.clip(self.base.img - img.img, 0, None)
        elif self._diff_option == "absolute":
            diff = skimage.util.compare_images(img.img, self.base.img, method="diff")
        else:
            raise ValueError(f"Diff option {self._diff_option} not supported")

        return diff

    def _extract_scalar_information(self, img: np.ndarray) -> np.ndarray:
        """
        Make a scalar image from potentially multi-colored image.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: monochromatic reduction of the array
        """
        return self.signal_reduction(img)

    def _clean_signal(self, img: np.ndarray) -> np.ndarray:
        """
        Apply cleaning thresholds.

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

    def _homogenize_signal(self, img: np.ndarray) -> np.ndarray:
        """
        Routine responsible for rescaling wrt segments.

        Here, it is assumed that only one segment is present.
        Thus, no rescaling is performed.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: homogenized image
        """
        return img

    def _prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply restoration.

        Args:
            signal (np.ndarray): input signal

        Return:
            np.ndarray: smooth signal
        """
        # Apply restoration
        if self.apply_restoration:
            signal = self.restoration(signal)

        return signal

    def _convert_signal(self, signal: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """
        Postprocessing routine, essentially converting a continuous
        signal into physical data (binary, continuous concentration etc.)

        Args:
            signal (np.ndarray): clean continous signal with values
                in the range between 0 and 1.
            diff (np.ndarray): original difference of images, allowing
                to extract new information besides the signal.

        Returns:
            np.ndarray: physical data
        """
        # Obtain data from model
        data = self.model(signal)

        return data

    # ! ---- Calibration tools for signal to concentration conversion

    def calibrate_model(
        self,
        images: list[darsia.Image],
        geometry: darsia.Geometry,
        options: dict,
    ) -> bool:
        """
        Utility for calibrating the model used in the concentration analysis.

        NOTE: Require to combine ConcentrationAnalysis with a calibration model.

        Args:
            images (list of darsia.Image): calibration images
            geometry (darsia.Geometry): geometry for integration
            injection_rate (float): known injection rate in ml/hr
            options (dict): container holding tuning information for the numerical
                calibration routine

        Returns:
            bool: success of the calibration study.

        """
        # Apply the same steps as in __call__ to all images.

        # Prepare calibration and determine fixed data
        images_diff = [self._subtract_background(img) for img in images]

        # Extract monochromatic version and take difference wrt the baseline image
        images_signal = [self._extract_scalar_information(diff) for diff in images_diff]

        # Clean signal
        images_clean_signal = [self._clean_signal(signal) for signal in images_signal]

        # Homogenize signal (take into account possible heterogeneous effects)
        images_homogenized_signal = [
            self._homogenize_signal(clean_signal)
            for clean_signal in images_clean_signal
        ]

        # Smoothen the signals
        images_smooth_signal = [
            self._prepare_signal(homogenized_signal)
            for homogenized_signal in images_homogenized_signal
        ]

        # NOTE: The only step missing from __call__ is the conversion of the signal
        # applying the provided model. This step will be used to tune the
        # model -> calibration.

        # Fetch calibration options
        initial_guess = options.get("initial_guess")
        tol = options.get("tol")
        maxiter = options.get("maxiter")

        # Define reference time (not important which image serves as basis)
        SECONDS_TO_HOURS = 1.0 / 3600.0
        relative_times = [
            (img.timestamp - self.base_time).total_seconds() * SECONDS_TO_HOURS
            for img in images
        ]

        # Double check an objective has been provided for calibration
        if not hasattr(self, "define_objective_function"):
            raise NotImplementedError(
                "The concentration analysis is not equipped with a calibration model."
            )
        else:
            calibration_objective = self.define_objective_function(
                images_smooth_signal, images_diff, relative_times, geometry, options
            )

        # Perform optimization step
        opt_result = optimize.minimize(
            calibration_objective,
            initial_guess,
            tol=tol,
            options={"maxiter": maxiter, "disp": True},
        )
        if opt_result.success:
            print(
                f"Calibration successful with obtained model parameters {opt_result.x}."
            )
        else:
            print("Calibration not successful.")

        # Update model
        self.model.update_model_parameters(opt_result.x)

        return opt_result.success


# Old approach using bisection only...
#        def _scaling_vs_deviation(scaling: float) -> float:
#            return _deviation([scaling, 0.], input_images, images_diff, relative_times)
#
#        st_time = time.time()
#        # Perform bisection
#        initial_guess = [1,10]
#        xtol = 1e-1
#        maxiter = 10
#        calibrated_scaling = bisect(
#            _scaling_vs_deviation,
#            *initial_guess,
#            xtol=xtol,
#            maxiter=maxiter
#        )
#        print(calibrated_scaling)
#        print("bisection", time.time() - st_time)
#        self.model.update(scaling = calibrated_scaling)


class PriorPosteriorConcentrationAnalysis(ConcentrationAnalysis):
    """
    Special case of the ConcentrationAnalysis performing a
    prior-posterior analysis, i.e., allowing to review the
    conversion performed through a prior model.

    """

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        signal_reduction: darsia.SignalReduction,
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
            base, signal_reduction, restoration, prior_model, labels, **kwargs
        )

    def _convert_signal(self, signal: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """
        Postprocessing routine, essentially converting a continuous
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


###################################################################
# Calibration Models
# TODO add other models as direct comparison with absolute volumes.
###################################################################
class CalibrationModel:
    @abc.abstractmethod
    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        relative_times: list[float],
        geometry: darsia.Geometry,
        options: dict,
    ):
        pass


class InjectionRateObjectiveMixin(CalibrationModel):
    """
    Calibration model based on matching injection rates.
    Has to be combined with ConcentrationAnalysis.

    """

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        relative_times: list[float],
        geometry: darsia.Geometry,
        options: dict,
    ):
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            relative_times (list of float): times
            geometry (darsia.Geometry): geometry object for integration
            options (dict): dictionary with objective value, here the injection rate

        Returns:
            callable: objetive function

        """

        # Fetch the injection rate
        injection_rate = options.get("injection_rate")

        # Define the objective function
        def objective_function(params: np.ndarray) -> float:
            """
            Compute the deviation between anticipated and expected injection rate.

            Args:
                params (np.ndarray): model parameters
                args: concentration analysis based arguments.

            """

            # Set the stage
            self.model.update_model_parameters(params)

            # For each image, compute the total concentration, based on the currently
            # set tuning parameters, and compute the relative time.
            M3_TO_ML = 1e6
            volumes = [
                geometry.integrate(self._convert_signal(img, diff)) * M3_TO_ML
                for img, diff in zip(input_images, images_diff)
            ]

            # Determine slope in time by linear regression
            ransac = RANSACRegressor()
            ransac.fit(np.array(relative_times).reshape(-1, 1), np.array(volumes))

            # Extract the slope and convert to
            effective_injection_rate = ransac.estimator_.coef_[0]

            # Measure deffect
            defect = effective_injection_rate - injection_rate
            return defect**2

        return objective_function


class AbsoluteVolumeObjectiveMixin(CalibrationModel):
    """
    Calibration model based on matching injection rates.
    Has to be combined with ConcentrationAnalysis.

    """

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        relative_times: list[float],
        geometry: darsia.Geometry,
        options: dict,
    ):
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            relative_times (list of float): times
            geometry (darsia.Geometry): geometry object for integration
            options (dict): dictionary with objective value, here the injection rate

        Returns:
            callable: objetive function

        """

        # Fetch input data
        input_times = np.array(options.get("times"))
        input_volumes = np.array(options.get("volumes"))
        input_data = interpolate.interp1d(input_times, input_volumes)

        # Sample data on time interval of interest
        time_interval = np.array(options.get("time_interval"))
        total_time = np.max(time_interval) - np.min(time_interval)
        dt_min = np.min(np.diff(np.unique(input_times)))
        num_samples = int(total_time / dt_min)
        sampled_times = np.min(time_interval) + np.arange(num_samples) * dt_min
        sampled_input_volumes = input_data(sampled_times)

        # Define the objective function
        def objective_function(params: np.ndarray) -> float:
            """
            Compute the deviation between anticipated and expected evolution
            of the volumes in the L2 sense.

            Args:
                params (np.ndarray): model parameters
                args: concentration analysis based arguments.

            """
            # Set the stage
            self.model.update_model_parameters(params)

            # For each image, compute the total concentration, based on the currently
            # set tuning parameters, and compute the relative time.
            M3_TO_ML = 1e6
            volumes = [
                geometry.integrate(self._convert_signal(img, diff)) * M3_TO_ML
                for img, diff in zip(input_images, images_diff)
            ]

            # Create interpolation
            estimated_data = interpolate.interp1d(relative_times, volumes)

            # Sample data
            sampled_estimated_volumes = estimated_data(sampled_times)

            # Measure defect - Compare the 1d functions
            defect = sampled_input_volumes - sampled_estimated_volumes
            return np.sum(defect**2) * dt_min

        return objective_function
