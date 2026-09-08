"""
Organizer of various thresholding methods.

"""

from typing import Optional, Union

import numpy as np

import darsia


class ThresholdModel:
    """Manager of the available thresholding models.

    Examples
    --------
    .. code-block:: python

        options = {
            "example threshold dynamic": False,
            "example threshold value": 0.2,
        }
        model = darsia.ThresholdModel(key="example ", **options)
        mask = model(img)
    """

    def __init__(
        self, labels: Optional[np.ndarray] = None, key: str = "", **kwargs
    ) -> None:

        apply_dynamic_thresholding: bool = kwargs.get(key + "threshold dynamic", False)

        if apply_dynamic_thresholding:

            # Determine threshold strategy and lower and upper bounds.
            threshold_method = kwargs.get(
                key + "threshold method", "tailored global min"
            )
            threshold_value_lower_bound: Union[float, list] = kwargs.get(
                key + "threshold value min", 0.0
            )
            threshold_value_upper_bound: Optional[Union[float, list]] = kwargs.get(
                key + "threshold value max", None
            )

            # Define final thresholding model.
            self.model = darsia.DynamicThresholdModel(
                threshold_method,
                threshold_value_lower_bound,
                threshold_value_upper_bound,
                labels,
                key,
                **kwargs,
            )

        else:

            # Determine specs
            threshold_value: Union[float, list] = kwargs.get(key + "threshold value")

            # Create thresholding model
            self.model = darsia.StaticThresholdModel(
                threshold_lower=threshold_value,
                labels=labels,
            )

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert signal according to the considered model.

        Parameters
        ----------
        img : np.ndarray
            Signal.
        mask : np.ndarray, optional
            Mask (region of interest).

        Returns
        -------
        np.ndarray
            Converted signal.
        """
        return self.model(img, mask)
