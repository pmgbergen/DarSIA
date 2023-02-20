"""
Module collecting several calibration tools,
and in particular objective functions for calibration
in ConcentrationAnalysis.calibrate_model()

"""

import abc

import numpy as np
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor

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
        relative_times: list[float],
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
            pos_model = options.get("model position")
            self.model.update_model_parameters(parameters, pos_model)
        else:
            self.model.update_model_parameters(parameters)


class InjectionRateModelObjectiveMixin(AbstractModelObjective):
    """
    Calibration model based on matching injection rates.
    Has to be combined with ConcentrationAnalysis.

    """

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        relative_times: list[float],
        options: dict,
    ):
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            relative_times (list of float): times in hrs
            options (dict): dictionary with objective value, here the injection rate

        Returns:
            callable: objetive function

        """

        # Fetch the injection rate and geometry
        injection_rate = options.get("injection_rate")  # in ml/hrs
        geometry = options.get("geometry")

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


class AbsoluteVolumeModelObjectiveMixin(AbstractModelObjective):
    """
    Calibration model based on matching injection rates.
    Has to be combined with ConcentrationAnalysis.

    """

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        relative_times: list[float],
        options: dict,
    ):
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            relative_times (list of float): times
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
            estimated_data = interpolate.interp1d(relative_times, volumes)

            # Sample data
            sampled_estimated_volumes = estimated_data(sampled_times)

            # Measure defect - Compare the 1d functions
            defect = sampled_input_volumes - sampled_estimated_volumes
            return np.sum(defect**2) * dt_min

        return objective_function
