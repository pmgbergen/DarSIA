"""
Module containing abstract Model.
Models convert signals to data.

"""
import abc
from typing import Literal, Optional, Union

import numpy as np


class Model:
    @abc.abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Translation of signal to data.
        """
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        Calibration routine of model.
        """
        pass

    @abc.abstractmethod
    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[Union[list[tuple[int, str]], Literal["all"]]] = None,
    ) -> None:
        """
        Update routine of model parameters.
        """
        pass
