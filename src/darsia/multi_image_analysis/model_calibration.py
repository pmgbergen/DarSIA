"""Calibration tools.

In particular objective functions for calibration in
ConcentrationAnalysis.calibrate_model()

"""
from __future__ import annotations

import abc
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
from scipy import interpolate
from sklearn.linear_model import LinearRegression, RANSACRegressor

import darsia


class AbstractModelObjective:
    """
    Abstract class for defining an objective function
    to be called in ConcentrationAnalysis.calibrate_model().

    """

    @abc.abstractmethod
    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        times: list[float],
        options: dict,
    ):
        """
        Abstract method to define an objective function.

        Returns:
            callable: objective function.

        """
        pass

    def update_model_for_calibration(
        self, parameters: np.ndarray, options: dict
    ) -> None:
        """
        Wrapper for updating the model, depending on
        whether it is a single model or a combined model.

        Args:
            parameters (np.ndarray): model parameters,
            options (dict): further tuning parameters and extra info.

        """
        # Check whether the model is part of a combined model,
        # and possibly determine position of the model
        if isinstance(self.model, darsia.CombinedModel):
            pos_model = options.get("model_position")
            self.model.update_model_parameters(parameters, pos_model)
        else:
            self.model.update_model_parameters(parameters)

    def calibrate_model(
        self,
        images: Union[darsia.Image, list[darsia.Image]],
        options: dict,
        plot_result: bool = False,
    ) -> bool:
        """
        Utility for calibrating the model used in darsia.ConcentrationAnalysis.

        NOTE: Require to combine darsia.ConcentrationAnalysis with a calibration
        model mixin via multiple inheritance.

        Args:
            images (list of darsia.Image): calibration images
            options (dict): container holding tuning information for the numerical
                calibration routine
            plot_result (bool): flag controlling whether the calibration is displayed
                in a plot.

        Returns:
            bool: success of the calibration study.

        """
        # Convert to list of images, if space time image provided.
        if not isinstance(images, list):
            assert images.series
            image_series = images.copy()
            images = [image_series.time_slice(i) for i in range(image_series.time_num)]

        # Apply the same steps as in __call__ to all images.

        # Prepare calibration and determine fixed data
        images_diff = [self._subtract_background(img) for img in images]

        # Extract monochromatic version and take difference wrt the baseline image
        images_signal = [self._extract_scalar_information(diff) for diff in images_diff]

        # Clean signal
        images_clean_signal = [self._clean_signal(signal) for signal in images_signal]

        # Balance signal (take into account possible heterogeneous effects)
        images_balanced_signal = [
            self._balance_signal(clean_signal) for clean_signal in images_clean_signal
        ]

        # Smoothen the signals
        images_smooth_signal = [
            self._prepare_signal(balanced_signal)
            for balanced_signal in images_balanced_signal
        ]

        # NOTE: The only step missing from __call__ is the conversion of the signal
        # applying the provided model. This step will be used to tune the
        # model -> calibration.

        # Fetch calibration options - default option chosen if None provided.
        initial_guess = options["initial_guess"]
        tol = options.get("tol")
        maxiter = options.get("maxiter")
        method = options.get("method")

        # Define reference time (not important which image serves as basis)
        times = [img.time for img in images]
        if any([time is None for time in times]):
            raise ValueError("Provide images with well-defined reference time.")

        # Double check an objective has been provided for calibration
        if not hasattr(self, "define_objective_function"):
            raise NotImplementedError(
                """The defined concentration analysis is not equipped with the
                functionality to calibrate a model."""
            )
        calibration_objective = self.define_objective_function(
            images_smooth_signal, images_diff, times, options
        )

        # Perform optimization step
        opt_result = optimize.minimize(
            calibration_objective,
            initial_guess,
            tol=tol,
            options={"maxiter": maxiter, "disp": True},
            method=method,
        )
        if opt_result.success:
            print(
                f"Calibration successful with obtained model parameters {opt_result.x}."
            )
        else:
            print("Calibration not successful.")

        # Update model (use functionality from calibration)
        self.update_model_for_calibration(opt_result.x, options)

        # Allow for post-analysis. Plot the result.
        if plot_result:
            self._visualize_model_calibration(
                images_smooth_signal, images_diff, times, options
            )

        return opt_result.success


# Old approach using bisection only...
#        def _scaling_vs_deviation(scaling: float) -> float:
#            return _deviation([scaling, 0.], input_images, images_diff, times)
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


class InjectionRateModelObjectiveMixin(AbstractModelObjective):
    """
    Calibration model based on matching injection rates.
    Has to be combined with ConcentrationAnalysis.

    """

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        times: list[float],
        options: dict,
    ):
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            times (list of float): times (units assumed to be compatible)
            options (dict): dictionary with objective value, here the injection rate

        Returns:
            callable: objective function

        """

        # Fetch the injection rate and geometry
        injection_rate = options["injection_rate"]  # in same units as geometry
        geometry = options["geometry"]
        regression_type = options.get("regression_type", "ransac").lower()
        assert regression_type in ["ransac", "linear"]

        # Define the objective function
        def objective_function(params: np.ndarray) -> float:
            """
            Compute the deviation between anticipated and expected injection rate.

            Args:
                params (np.ndarray): model parameters
                args: concentration analysis based arguments.

            """

            # Set the stage
            self.update_model_for_calibration(params, options)

            # For each image, compute the total concentration, based on the currently
            # set tuning parameters, and compute the relative time.
            volumes = [
                geometry.integrate(self._convert_signal(img, diff))
                for img, diff in zip(input_images, images_diff)
            ]

            # Determine slope in time by linear regression. Allow for different
            # regression types.
            if regression_type == "ransac":

                ransac = RANSACRegressor()
                ransac.fit(np.array(times).reshape(-1, 1), np.array(volumes))
                regression = ransac.estimator_

            elif regression_type == "linear":

                regression = LinearRegression()
                regression.fit(np.array(times).reshape(-1, 1), np.array(volumes))

            # Extract the slope and convert to
            effective_injection_rate = regression.coef_[0]

            # Cache result for possible post-analysis
            self._slope = regression.coef_[0]
            self._reference_slope = injection_rate
            self._intercept = regression.intercept_

            # Measure relative defect
            defect = effective_injection_rate - injection_rate
            if abs(injection_rate) > 1e-15:
                defect /= injection_rate
            return defect**2

        return objective_function

    def _visualize_model_calibration(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        times: list[float],
        options: dict,
    ) -> None:
        """
        Illustrate result of calibration.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            times (list of float): times (unit assumed to be compatible)
            options (dict): dictionary with objective value, here the injection rate

        """
        # Fetch the injection rate and geometry
        geometry = options["geometry"]

        # For each image, compute the total concentration, based on the currently
        # set tuning parameters, and compute the relative time.
        volumes = [
            geometry.integrate(self._convert_signal(img, diff))
            for img, diff in zip(input_images, images_diff)
        ]

        ## ! ---- Find intercept through RANSAC analysis

        # Plot the evolution of the rescaled signal over time, and in addtion a dashed
        # line corresponding to the reference injection rate.
        plt.figure("Model calibration")
        plt.plot(times, volumes)
        plt.plot(
            times,
            [self._intercept + self._slope * time for time in times],
            color="black",
            linestyle=(0, (5, 5)),
        )
        plt.plot(
            times,
            [self._intercept + self._reference_slope * time for time in times],
            color="green",
            linestyle=(0, (5, 5)),
        )
        plt.xlabel("time [s]")
        plt.ylabel("volume [m**3]")
        plt.legend(["rescaled signal", "regression", "reference"])
        plt.show()

    def model_calibration_postanalysis(self) -> float:
        """Interpret calibration result.

        Returns:
            float: time (in seconds) at which the signal is zero (based on calibration)

        """

        # Interpret the results of the regression
        print(
            f"""The effective injection rate for the calibration images is """
            f"""{self._slope} (in volumetric units per time compatible with images)."""
        )
        print(
            f"""The signal at absolute time 0 is {self._intercept} (in volumetric """
            """units compatible with images)."""
        )

        # Determine the time of zero signal
        time_zero_signal = -self._intercept / self._slope
        print(
            f"""The time of zero signal is deducted to be {time_zero_signal} """
            """(in time units compatible with images)."""
        )

        return time_zero_signal


class AbsoluteVolumeModelObjectiveMixin(AbstractModelObjective):
    """
    Calibration model based on matching injection rates.
    Has to be combined with ConcentrationAnalysis.

    """

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        times: list[float],
        options: dict,
    ):
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            times (list of float): times
            options (dict): dictionary with objective value, here the injection rate

        Returns:
            callable: objetive function

        """

        # Fetch the geometry for integration
        geometry = options.get("geometry")

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
            self.update_model_for_calibration(params, options)

            # For each image, compute the total concentration, based on the currently
            # set tuning parameters, and compute the relative time.
            M3_TO_ML = 1e6
            volumes = [
                geometry.integrate(self._convert_signal(img, diff)) * M3_TO_ML
                for img, diff in zip(input_images, images_diff)
            ]

            # Create interpolation
            estimated_data = interpolate.interp1d(times, volumes)

            # Sample data
            sampled_estimated_volumes = estimated_data(sampled_times)

            # Measure defect - Compare the 1d functions
            defect = sampled_input_volumes - sampled_estimated_volumes
            return np.sum(defect**2) * dt_min

        return objective_function
