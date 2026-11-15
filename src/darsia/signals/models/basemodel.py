"""
Module containing abstract Model.
Models convert signals to data.

"""

import abc
import copy
from typing import Literal, Optional, Union, overload

import numpy as np

import darsia
import darsia.signals.color.color_path_kernels as kernels


class Model:
    """Base class for signal-to-quantity models.

    A model is a callable mapping a scalar signal (array or :class:`~darsia.Image`)
    to a physical quantity such as a concentration or saturation. Subclasses
    implement ``__call__`` and, where calibratable, expose ``num_parameters``.
    """

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
    """Apply a different :class:`Model` per region of a labelled image."""

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
                assert len(obj) == len(
                    self.masks.unique_labels
                ), "Length of model list must match number of unique labels."
                self.obj[label] = copy.copy(obj[j])
            elif isinstance(obj, dict):
                assert label in obj, f"Label {label} not found in model dictionary."
                self.obj[label] = copy.copy(obj[label])
            else:
                self.obj[label] = copy.copy(obj)

    def _color_path_labels(self, signal: np.ndarray) -> list | None:
        """Labels to evaluate in one compiled pass, if the model allows it.

        This requires an image of colors, integer labels and a color path
        interpolation for each label which is not ignored.

        """
        labels = self.masks.labels.img
        if not (
            kernels.NUMBA_AVAILABLE
            and signal.ndim == 3
            and signal.shape[2] == 3
            and signal.dtype == np.float64
            and signal.shape[:2] == labels.shape
            and np.issubdtype(labels.dtype, np.integer)
        ):
            return None
        active = [
            label
            for label in self.masks.unique_labels
            if label not in self.ignore_labels
        ]
        if len(active) == 0 or not all(
            type(self[label]) is darsia.ColorPathInterpolation for label in active
        ):
            return None
        return active

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        # Color paths of all labels in a single pass over the pixels, instead of
        # extracting and fitting the pixels of each label separately.
        active = self._color_path_labels(signal)
        if active is not None:
            tables = [self[label].kernel_tables() for label in active]
            return kernels.evaluate_labeled_image(
                signal,
                self.masks.labels.img,
                active,
                *[list(table) for table in zip(*tables)],
            )

        output = np.zeros(signal.shape[:2])  # TODO shape?
        for label in self.masks.unique_labels:
            if label in self.ignore_labels:
                continue
            mask = self.masks.mask_for_label(label)
            output[mask] = self[label](signal[mask])
        return output

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, value):
        self.obj[key] = value

    def keys(self):
        return self.obj.keys()
