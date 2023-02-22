"""
Module containing functionality to determine the
Earth Mover's distance between images.

"""

from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt

# import ot
import numpy as np

import darsia


class EMD:
    """
    Class to determine the EMD through cv2.

    """

    def __init__(
        self, preprocess: Optional[callable] = None, **kwargs
    ) -> None:
        """
        Args:
            preprocess (callable, optional): preprocessing routine

        """
        # Cache
        self.preprocess = preprocess

    def __call__(self, img_1: darsia.Image, img_2: darsia.Image) -> float:
        """
        Earth mover's distance between images with same total sum.

        Args:
            img_1 (darsia.Image or array): image 1
            img_2 (darsia.Image or array): image 2

        Returns:
            float: distance between img_1 and img_2.

        """
        # Preprocess images
        preprocessed_img_1 = self._preprocess(img_1)
        preprocessed_img_2 = self._preprocess(img_2)

        # Pixel dimensions
        dx_1 = (preprocessed_img_1.dy, preprocessed_img_1.dx)
        dx_2 = (preprocessed_img_2.dy, preprocessed_img_2.dx)
        assert np.isclose(dx_1[0], dx_2[0])
        assert np.isclose(dx_1[1], dx_2[1])

        # Compatibilty check
        self._compatibility_check(preprocessed_img_1, preprocessed_img_2)

        # Normalization
        integral_1, normalized_img_1 = self._normalize(preprocessed_img_1)
        integral_2, normalized_img_2 = self._normalize(preprocessed_img_2)

        # Convert format to a signature
        sig_1 = self._img_to_sig(normalized_img_1, dx=dx_1)
        sig_2 = self._img_to_sig(normalized_img_2, dx=dx_2)

        # Compute EMD distance
        dist, _, flow = cv2.EMD(sig_1, sig_2, cv2.DIST_L2)

        return dist * integral_1

    def _preprocess(self, img: darsia.Image) -> darsia.Image:
        """
        Preprocessing routine, incl. extraction of array.

        Args:
            img (darsia.Image): image

        Returns:
            darsia.Image: image array under provided preprocessing
        """
        preprocessed_img = img.copy()
        if self.preprocess is not None:
            preprocessed_img = self.preprocess(preprocessed_img)
        return preprocessed_img

    def _compatibility_check(self, img_1: darsia.Image, img_2: darsia.Image) -> bool:
        """
        Compatibility check.

        Args:
            img_1 (darsia.Image): image 1
            img_2 (darsia.Image): image 2

        Returns:
            bool: flag whether images 1 and 2 can be compared.

        """
        # Compatible distributions
        assert np.isclose(np.sum(img_1.img), np.sum(img_2.img))

        # 2D and scalar-valued
        assert np.prod(img_1.img.shape) == np.prod(img_1.img.shape[:2])
        assert np.prod(img_2.img.shape) == np.prod(img_2.img.shape[:2])

    def _normalize(self, img: darsia.Image) -> tuple[float, np.ndarray]:
        """
        Normalization of images to images with sum 1.

        Args:
            img (darsia.Image): image

        Returns:
            float: original sum
            np.ndarray: normalized image

        """
        integral = np.sum(img.img)
        normalized_img = img.img / integral
        return integral, normalized_img

    def _img_to_sig(
        self, img: np.ndarray, dx: Union[float, tuple[float, float]] = 1
    ) -> np.ndarray:
        """Convert a 2D array to a signature for cv2.EMD.

        Args:
            img (array): image
            dx: (float or tuple): distance from one pixel to another

        Returns:
            np.ndarray: signature

        """
        if isinstance(dx, float):
            dx = (dx, dx)
        del_y, del_x = dx

        sig = np.empty((img.size, 3), dtype=np.float32)
        count = 0
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                sig[count] = np.array([img[row, col], col * del_x, row * del_y]).astype(
                    np.float32
                )
                count += 1

        return sig

    def distance_matrix(self, images: list[darsia.Image]) -> np.ndarray:
        """
        Compute the distance between each iteam of a list.

        Args:
            images (list of images): N images

        Returns:
            np.ndarray: N x N matrix with distances between images.

        """
        num_images = len(images)

        distance_matrix = np.zeros((num_images, num_images), dtype=float)

        # Matrix symmetric, so only compute one side of the diagonal.
        for i, img_i in enumerate(images):
            for j, img_j in enumerate(images):
                # Each image has distance 0 to itself.
                if i >= j:
                    continue
                distance_matrix[i, j] = self.__call__(img_i, img_j)

        # Fill in remaining entries
        for i, img_i in enumerate(images):
            for j, img_j in enumerate(images):
                # Each image has distance 0 to itself.
                if i <= j:
                    continue
                distance_matrix[i, j] = distance_matrix[j, i]

        return distance_matrix


## Determine EMD using ot
# if True:
#    # OT takes 1d arrays as inputs
#    a_flat = a.flatten(order = "F")
#    b_flat = b.flatten(order = "F")
#
#    # Cell centers of all cells - x and y coordinates.
#    cc_x = np.zeros((Nx,Ny), dtype=float).flatten("F")
#    cc_y = np.zeros((Nx,Ny), dtype=float).flatten("F")
#
#    cc_x, cc_y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
#
#    cc_x_flat = cc_x.flatten("F")
#    cc_y_flat = cc_y.flatten("F")
#
#    cc = np.vstack((cc_x_flat, cc_y_flat)).T
#
#    # Distance matrix
#    # NOTE the definition of this distance matrix is memory consuming and
#    # does not allow for too large distributions.
#    M = ot.dist(cc, cc, metric="euclidean")
#
#    dist_ot = ot.emd2(a_flat,b_flat,M)
#    print(dist_ot)
