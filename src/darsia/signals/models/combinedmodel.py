"""
Module containing combination of models.

NOTE: Combining models is experimental and the responsibility
lies with the user.

"""

import numpy as np

import darsia


class CombinedModel(darsia.Model):
    def __init__(self, models: list[darsia.Model]) -> None:

        self.models = models

    def __call__(self, img: np.ndarray, *args) -> np.ndarray:
        """
        Concatenate the application of the models

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: combined model response

        """
        result = img.copy()
        for model in self.models:
            # Determine the number of arguments in the signature of
            # the model (__call__) and pass only a suitable amount
            # of arguments. NOTE: There is no guarantee that the
            # the models use the same positional arguments!
            all_args = model.__call__.__code__.co_argcount
            if all_args == 2:
                result = model(result)
            else:
                result = model(result, *args[: all_args - 2])

        return result
