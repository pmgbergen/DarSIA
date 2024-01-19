"""Combination of models.

NOTE: Combining models is experimental and the responsibility
lies with the user.

"""
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

import darsia


class CombinedModel(darsia.Model):
    def __init__(self, models: list[darsia.Model]) -> None:
        # Cache models
        self.models = models

        # Determine the number of parameters needed for calibration
        self.num_parameters = sum(
            [
                model.num_parameters if hasattr(model, "num_parameters") else 0
                for model in self.models
            ]
        )

    def __call__(self, img: np.ndarray, *args) -> np.ndarray:
        """
        concatenate the application of the models

        args:
            img (np.ndarray): input image

        returns:
            np.ndarray: combined model response

        """
        result = img.copy()
        for model in self.models:
            # determine the number of arguments in the signature of
            # the model (__call__) and pass only a suitable amount
            # of arguments. note: there is no guarantee that the
            # the models use the same positional arguments!
            all_args = model.__call__.__code__.co_argcount
            if all_args == 2:
                result = model(result)
            else:
                result = model(result, *args[: all_args - 2])

        return result

    def update_model_parameters(
        self,
        parameters: np.ndarray,
        dofs: Optional[Union[list[tuple[int, str]], Literal["all"]]] = None,
    ) -> None:
        """
        Wrapper of update routines of single models.

        Args:
            parameters (np.ndarray): parameter array
            pos_model (int): position index addressing a single model.

        """
        # Cache a copy of the parameters
        parameters_cache = parameters.copy()

        # Update the parameters of the model
        if dofs in [None, "all"]:
            # If no degrees of freedom are specified, update all parameters
            for pos_model, model in enumerate(self.models):
                model.update_model_parameters(parameters_cache)
                # Remove the updated parameters from the cache
                parameters_cache = parameters_cache[model.num_parameters :]
        else:
            # Analogously when only a subset of parameters is to be updated
            for pos_model, pos_parameter in dofs:
                model = self.models[pos_model]
                model.update_model_parameters(parameters_cache, pos_parameter)
                parameters_cache = parameters_cache[model.num_parameters :]

    def __getitem__(self, pos_model: int) -> darsia.Model:
        """Access single models.

        Args:
            pos_model (int): position index addressing a single model.

        Returns:
            darsia.Model: single model

        """
        return self.models[pos_model]
