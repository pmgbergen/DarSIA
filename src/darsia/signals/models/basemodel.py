"""
Module containing abstract Model.
Models convert signals to data.

"""
import abc
import copy
from typing import Literal, Optional, Union

import numpy as np

import darsia


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


class HeterogeneousModel(Model):
    def __init__(self, obj: Union[Model, list[Model]], labels: darsia.Image):

        self.masks = darsia.Masks(labels)
        self.obj = {}
        for i in range(self.masks.size):
            self.obj[i] = copy.copy(obj)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        output = np.zeros(signal.shape[:2])  # TODO shape?
        for i, mask in enumerate(self.masks):
            output[mask.img] = self[i](signal[mask.img])
        return output

    def __getitem__(self, key):
        return self.obj[key]
