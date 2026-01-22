"""Module containing clipping operations."""

from typing import Literal, Optional

import numpy as np

import darsia


class ClipModel(darsia.Model):
    """Model clipping away signal at some min and max values."""

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Clipping model initialization.

        Args:
            min_value (float | None): lower clip value
            max_value (float | None): upper clip value
            key (str): additional key (prefix) for kwargs
            kwargs (keyword arguments):
                key + '_min_value': lower clip value
                key + '_max_value': upper clip value

        """
        if key is None:
            self._min_value = min_value
            self._max_value = max_value
        else:
            self._min_value = kwargs.get(key + "_min_value", None)
            self._max_value = kwargs.get(key + "_max_value", None)
        if self._min_value is None and self._max_value is None:
            raise ValueError("at least one of min_value or max_value must be provided")
        self.num_parameters = 2

    def update(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        """
        Update of internal parameters.

        Args:
            min_value (float, optional): lower clip value
            max_value (float, optional): upper clip value

        """
        if min_value is not None:
            self._min_value = min_value

        if max_value is not None:
            self._max_value = max_value

    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[list[Literal["min_value", "max_value"]] | Literal["all"]] = None,
    ) -> None:
        """
        Short cut to update scaling and offset parameters using a
        general function signature.

        Args:
            parameters (np.ndarray): 2-array containing min and max values.

        """
        if (
            dofs is None
            or dofs == "all"
            or set(dofs) == set(["min_value", "max_value"])
        ):
            self.update(min_value=parameters[0], max_value=parameters[1])
        elif set(dofs) == set(["min_value"]):
            self.update(min_value=parameters[0])
        elif set(dofs) == set(["max_value"]):
            self.update(max_value=parameters[0])
        else:
            raise ValueError("invalid list of degrees of freedom")

    def __call__(self, img: np.ndarray | darsia.Image) -> np.ndarray | darsia.Image:
        """
        Application of clipping.

        Args:
            img (np.ndarray | Image): image

        Returns:
            np.ndarray | Image: converted signal; output type is the same as input type

        """

        if isinstance(img, np.ndarray):
            return np.clip(img, self._min_value, self._max_value)
        elif isinstance(img, darsia.Image):
            result = img.copy()
            result.img = np.clip(result.img, self._min_value, self._max_value)
            return result
        else:
            raise ValueError("invalid input type")
