"""
Organizer of various thresholding methods.

"""

import darsia
from typing import Optional
import numpy as np


class ThresholdingModel:
    def __init__(self, labels: Optional[np.ndarray] = None, **kwargs) -> None:

        apply_dynamic_thresholding: bool = kwargs.pop("threshold dynamic", True)

        if apply_dynamic_thresholding:

            # Determine threshold strategy and lower and upper bounds.
            threshold_method = kwargs.get("threshold method", "tailored global min")
            threshold_value_lower_bound: Union[float, list] = kwargs.pop(
                "threshold value min", 0.0
            )
            threshold_value_upper_bound: Optional[Union[float, list]] = kwargs.pop(
                "threshold value max", None
            )

            # Define final thresholding model.
            self.model = darsia.DynamicThresholdModel(
                threshold_method,
                threshold_value_lower_bound,
                threshold_value_upper_bound,
                labels,
                **kwargs,
            )

        else:

            # Determine specs
            threshold_value: Union[float, list] = kwargs.get("threshold value")

            # Create thresholding model
            self.model = darsia.StaticThresholdModel(
                threshold_low=threshold_value,
                labels=labels,
            )

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert signal according to the considered model.

        Args:
            img (np.ndarray): signal
            mask (np.ndarray, optional): mask (region of interest)

        Returns:
            np.ndarray: converted signal
        """
        return self.model(img, mask)
