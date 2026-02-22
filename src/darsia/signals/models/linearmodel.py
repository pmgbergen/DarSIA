"""Module containing a linear (affine) conversion from signals to data.

"""

from __future__ import annotations

from typing import Literal, Optional, Union

import cv2
import numpy as np

import darsia


class ScalingModel(darsia.Model):
    """Linear model (plain scaling)."""

    def __init__(
        self,
        key: str = "",
        **kwargs,
    ) -> None:
        self._scaling = kwargs.get(key + "scaling", 1.0)
        self.num_parameters = 1

        self.volumes = None

    def update(
        self,
        scaling: Optional[float] = None,
    ) -> None:
        """
        Update of internal parameters.

        Args:
            scaling (float, optional): slope

        """

        if scaling is not None:
            self._scaling = scaling

    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[Union[list[Literal["scaling"]], Literal["all"]]] = None,
    ) -> None:
        """
        Short cut to update scaling and offset parameters using a
        general function signature.

        The main use is the model calibration. Do not update the offset.

        Args:
            parameters (np.ndarray): 2-array containing scaling and offset values.

        """
        if dofs is None or dofs == "all" or set(dofs) == set(["scaling"]):
            self.update(scaling=parameters[0])
        else:
            raise ValueError(f"Unknown dof {dofs}.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Application of linear model.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: converted signal

        """
        if np.isclose(self._scaling, 1.0):
            return img
        else:
            return self._scaling * img


class LinearModel(darsia.Model):
    """Linear model, applying an affine conversion for signals to data."""

    def __init__(
        self,
        key: str = "",
        **kwargs,
    ) -> None:
        self._scaling = kwargs.get(key + "scaling", 1.0)
        self._offset = kwargs.get(key + "offset", 0.0)
        self.num_parameters = 2

        self.volumes = None

    def update(
        self,
        scaling: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> None:
        """
        Update of internal parameters.

        Args:
            scaling (float, optional): slope
            offset (float, optional): offset

        """

        if scaling is not None:
            self._scaling = scaling

        if offset is not None:
            self._offset = offset

    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[
            Union[list[Literal["scaling", "offset"]], Literal["all"]]
        ] = None,
    ) -> None:
        """
        Short cut to update scaling and offset parameters using a
        general function signature.

        The main use is the model calibration. Do not update the offset.

        Args:
            parameters (np.ndarray): 2-array containing scaling and offset values.

        """
        if dofs is None or dofs == ["all"] or set(dofs) == set(["scaling", "offset"]):
            self.update(scaling=parameters[0], offset=parameters[1])
        elif set(dofs) == set(["scaling"]):
            self.update(scaling=parameters[0])
        elif set(dofs) == set(["offset"]):
            self.update(offset=parameters[0])
        else:
            raise ValueError(f"Unknown dof {dofs}.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Application of linear model.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: converted signal

        """
        return self._scaling * img + self._offset


class HeterogeneousLinearModel(darsia.Model):
    """
    Linear model, applying an affine conversion for signals to data.

    """

    def __init__(
        self,
        labels: np.ndarray,
        key: str = "",
        **kwargs,
    ) -> None:
        # Organize label info
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.num_labels = len(self.unique_labels)
        self.cached_labels = self.labels.copy()

        # Read model parameters from options
        self._scaling = kwargs.get(
            key + "scaling", np.ones(self.num_labels, dtype=float)
        )
        self._offset = kwargs.get(
            key + "offset", np.zeros(self.num_labels, dtype=float)
        )
        self.num_parameters = 2 * self.num_labels

        # Convert to arrays
        if isinstance(self._scaling, list):
            self._scaling = np.array(self._scaling)
        if isinstance(self._offset, list):
            self._offset = np.array(self._offset)

        # Check compatibility
        self._compatibility()

    def _compatibility(self) -> None:
        # Convert from scalar to vectors
        if isinstance(self._scaling, float):
            self._scaling = self._scaling * np.ones(self.num_labels, dtype=float)
        if isinstance(self._offset, float):
            self._offset = self._offset * np.ones(self.num_labels, dtype=float)

        # Compatibility checks
        assert (
            isinstance(self._scaling, np.ndarray)
            and len(self._scaling) == self.num_labels
        )
        assert (
            isinstance(self._offset, np.ndarray)
            and len(self._offset) == self.num_labels
        )

        self.volumes = None

    def update(
        self,
        scaling: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update of internal parameters.

        Args:
            scaling (np.ndarray, optional): slope
            offset (np.ndarray, optional): offset

        """

        if scaling is not None:
            self._scaling = scaling

        if offset is not None:
            self._offset = offset

        self._compatibility()

    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[
            Union[list[Literal["scaling", "offset"]], Literal["all"]]
        ] = None,
    ) -> None:
        """
        Short cut to update scaling and offset parameters using a
        general function signature.

        Args:
            parameters (np.ndarray): 2-array containing scaling and offset values.

        """
        if dofs is None or dofs == "all" or set(dofs) == set(["scaling", "offset"]):
            self.update(
                scaling=parameters[: self.num_labels],
                offset=parameters[self.num_labels :],
            )
        elif set(dofs) == set(["scaling"]):
            self.update(scaling=parameters[: self.num_labels])
        elif set(dofs) == set(["offset"]):
            self.update(offset=parameters[: self.num_labels])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Application of linear model.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: converted signal

        """
        # Potentially need to resize labels.
        if img.shape != self.cached_labels:
            self.cached_labels = cv2.resize(
                self.labels,
                tuple(reversed(img.shape[:2])),
                interpolation=cv2.INTER_NEAREST,
            )

        # Initialize result
        result = np.zeros_like(img, dtype=img.dtype)
        for l_counter, label in enumerate(self.unique_labels):
            tmp = self._scaling[l_counter] * img + self._offset[l_counter]
            mask = self.cached_labels == label
            result[mask] = tmp[mask]

        return result
