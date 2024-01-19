"""
Module containing clipping operations.

"""
from typing import Literal, Optional, Union

import numpy as np

import darsia


class ClipModel(darsia.Model):
    """
    Model clipping away signal at some
    min and max values.

    """

    def __init__(self, key: str = "", **kwargs) -> None:
        """
        Constructor.

        Args:
            key (str): additional key
            kwargs (keyword arguments):
                'min value': lower clip value
                'max value': upper clip value

        """
        self._min_value = kwargs.get(key + "min value", 0)
        self._max_value = kwargs.get(key + "max value", None)
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
        dofs: Optional[
            Union[list[Literal["min_value", "max_value"]], Literal["all"]]
        ] = None,
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

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Application of clipping.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: converted signal

        """

        return np.clip(img, self._min_value, self._max_value)
