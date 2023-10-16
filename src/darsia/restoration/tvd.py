"""Module wrapping TV denoising algoriths from skimage into the realm of DarSIA. These
can be directly integrated in darsia.ConcentrationAnalysis, in particular as part of the
definition of a restoration object.

"""

from typing import Union

import numpy as np
import skimage

import darsia


class TVD:
    """Total variation denoising interface.

    Connects to skimage.restoration.denoise_tv_* methods as well as to the DarSIA
    implementation of the split Bregman method (:mod:'split_bregman_tvd.py').

    """

    def __init__(self, key: str = "", **kwargs) -> None:
        """Constructor.

        Args:
            key (str): Prefix for kwargs arguments.

        """
        # Determine method
        self.method = kwargs.pop(key + "method", "chambolle").lower()

        # Method-specific parameters
        if self.method == "heterogeneous bregman":
            # DarSIA type methods
            self.omega = kwargs.pop("omega", 1)
            self.regularization = kwargs.get("regularization", 1.0)

        # General parameters
        self.weight = kwargs.pop(key + "weight", 0.1)
        self.max_num_iter = kwargs.pop(key + "max_num_iter", 200)
        self.eps = kwargs.pop(key + "eps", 2e-4)
        self.kwargs = kwargs

    def __call__(
        self, img: Union[np.ndarray, darsia.Image]
    ) -> Union[np.ndarray, darsia.Image]:
        """Application of TV denoising.

        Args:
            img (np.ndarray or Image): image

        Returns:
            np.ndarray or Image: upscaled image (same type as input)

        """

        if isinstance(img, np.ndarray):
            return self._tvd_array(img)
        elif isinstance(img, darsia.Image):
            return self._tvd_image(img)
        else:
            raise TypeError(f"Input type {type(img)} not supported.")

    def _tvd_array(self, img: np.ndarray) -> np.ndarray:
        """Application of anisotropic resizing and tv denoising to numpy array.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: upscaled image

        """

        # Apply TVD
        if self.method == "chambolle":
            return skimage.restoration.denoise_tv_chambolle(
                img,
                weight=self.weight,
                max_num_iter=self.max_num_iter,
                eps=self.eps,
            )

        elif self.method == "anisotropic bregman":
            return skimage.restoration.denoise_tv_bregman(
                img,
                weight=self.weight,
                max_num_iter=self.max_num_iter,
                eps=self.eps,
                isotropic=False,
            )

        elif self.method == "isotropic bregman":
            return skimage.restoration.denoise_tv_bregman(
                img,
                weight=self.weight,
                max_num_iter=self.max_num_iter,
                eps=self.eps,
                isotropic=True,
            )

        elif self.method == "heterogeneous bregman":
            return darsia.split_bregman_tvd(
                img,
                mu=self.weight,
                omega=self.omega,
                ell=self.regularization,
                max_num_iter=self.max_num_iter,
                eps=self.eps,
                **self.kwargs,
            )

        else:
            raise ValueError(f"Method {self.method} not supported.")

    def _tvd_image(self, img: darsia.Image) -> darsia.Image:
        """Application of anisotropic resizing and tv denoising to darsia.Image.

        Args:
            img (darsia.Image): image

        Returns:
            darsia.Image: upscaled image

        """
        img_copy = img.copy()
        img_copy.img = self._tvd_array(img.img)
        return img_copy


def tvd(
    img: Union[np.ndarray, darsia.Image],
    method: str = "chambolle",
    weight: Union[float, np.ndarray] = 1.0,
    max_num_iter: int = 200,
    eps: float = 2e-4,
    **kwargs,
) -> Union[np.ndarray, darsia.Image]:
    """Inline application of TVD.

    Args:
        img (np.ndarray or Image): image
        method (str): TVD method
        weight (float or np.ndarray): weight
        max_num_iter (int): maximum number of iterations
        eps (float): tolerance
        **kwargs: additional arguments
            - omega (array or float): data fidelity weight for heterogeneous bregman
            - regularization (float): regularization parameter for heterogeneous bregman

    Returns:
        np.ndarray or Image: upscaled image (same type as input)

    """
    tvd = TVD(
        method=method,
        weight=weight,
        max_num_iter=max_num_iter,
        eps=eps,
        **kwargs,
    )
    return tvd(img)
