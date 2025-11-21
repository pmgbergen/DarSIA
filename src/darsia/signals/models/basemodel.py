"""
Module containing abstract Model.
Models convert signals to data.

"""

import abc
import copy
from typing import Literal, Optional, Union, overload

import numpy as np

import darsia


class Model:
    @abc.abstractmethod
    @overload
    def __call__(self, signal: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    @overload
    def __call__(self, signal: darsia.Image) -> darsia.Image: ...

    @abc.abstractmethod
    def __call__(self, img: np.ndarray | darsia.Image) -> np.ndarray | darsia.Image:
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
    def __init__(
        self,
        obj: Model | list[Model] | dict[int, Model],
        labels: darsia.Image,
        ignore_labels: list[int] | None = None,
    ) -> None:
        self.masks = darsia.Masks(labels)
        """Masks for each label in the image."""
        self.ignore_labels = ignore_labels or []
        """Labels to ignore for signals."""
        self.obj = {}
        """Dictionary of models for each label."""
        for j, label in enumerate(self.masks.unique_labels):
            if isinstance(obj, list):
                assert len(obj) == len(self.masks.unique_labels), (
                    "Length of model list must match number of unique labels."
                )
                self.obj[label] = copy.copy(obj[j])
            elif isinstance(obj, dict):
                assert label in obj, f"Label {label} not found in model dictionary."
                self.obj[label] = copy.copy(obj[label])
            else:
                self.obj[label] = copy.copy(obj)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        output = np.zeros(signal.shape[:2])  # TODO shape?
        for i, mask in enumerate(self.masks):
            label = self.masks.unique_labels[i]
            if label in self.ignore_labels:
                continue
            output[mask.img] = self[label](signal[mask.img])
        return output

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, value):
        self.obj[key] = value
