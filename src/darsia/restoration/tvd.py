"""
Module wrapping TV denoising algoriths from skimage
into the realm of DarSIA. These can be directly integrated
in darsia.ConcentrationAnalysis, in particular as part of
the definition of a restoration object.

"""

import numpy as np
import skimage

import darsia

from .heterogeneous_tvd import heterogeneous_tv_denoising


class TVD:
    """
    Total variation denoising interface to skimage.restoration
    as well as darsia.restoration.

    """

    def __init__(self, key: str = "", **kwargs) -> None:
        """
        Constructor.

        Args:
            key (str): Prefix for kwargs arguments.

        """
        self.method = kwargs.get(key + "smoothing method", "chambolle")
        if self.method == "heterogeneous anisotropic bregman":

            # Internal TV method
            self.weight = kwargs.get("weight", 0.1)
            self.omega = kwargs.get("omega", 1)
            self.penalty = kwargs.get("penalty", 1.0)
            self.tvd_stopping_criterion = kwargs.get(
                "tvd stopping criterion", darsia.StoppingCriterion(1e-4, 100)
            )
            self.cg_stopping_criterion = kwargs.get(
                "cg stopping criterion", darsia.StoppingCriterion(1e-2, 100)
            )

        else:

            # Skimage type methods
            self.weight = kwargs.get(key + "smoothing weight", 0.1)
            self.eps = kwargs.get(key + "smoothing eps", 2e-4)
            self.max_num_iter = kwargs.get(key + "smoothing max_num_iter", 200)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Application of anisotropic resizing and tv denoising.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: upscaled image

        """

        # Apply TVD
        if self.method == "chambolle":
            img = skimage.restoration.denoise_tv_chambolle(
                img,
                weight=self.weight,
                eps=self.eps,
                max_num_iter=self.max_num_iter,
            )

        elif self.method == "anisotropic bregman":
            img = skimage.restoration.denoise_tv_bregman(
                img,
                weight=self.weight,
                eps=self.eps,
                max_num_iter=self.max_num_iter,
                isotropic=False,
            )

        elif self.method == "isotropic bregman":
            img = skimage.restoration.denoise_tv_bregman(
                img,
                weight=self.weight,
                eps=self.eps,
                max_num_iter=self.max_num_iter,
                isotropic=True,
            )

        elif self.method == "heterogeneous anisotropic bregman":
            img = heterogeneous_tv_denoising(
                img,
                weight=self.weight,
                omega=self.omega,
                penalty=self.penalty,
                tvd_stopping_criterion=self.tvd_stopping_criterion,
                cg_stopping_criterion=self.cg_stopping_criterion,
            )

        else:
            raise ValueError(f"Method {self.method} not supported.")

        return img
