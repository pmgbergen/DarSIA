"""
Module containing functionality to determine the
Earth Mover's distance between images.

"""

from typing import Optional, Union

import cv2

# import ot
import numpy as np

import darsia


class EMD:
    """
    Class to determine the EMD through cv2.

    """

    def __init__(self, preprocess: Optional[callable] = None, **kwargs) -> None:
        """
        Args:
            preprocess (callable, optional): preprocessing routine

        """
        # Cache
        self.preprocess = preprocess

    def __call__(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
    ) -> float:
        """
        Earth mover's distance between images with same total sum.

        Args:
            img_1 (darsia.Image): image 1
            img_2 (darsia.Image): image 2

        Returns:
            float or array: distance between img_1 and img_2.

        """
        # Preprocess images
        preprocessed_img_1 = self._preprocess(img_1)
        preprocessed_img_2 = self._preprocess(img_2)

        # Compatibilty check
        self._compatibility_check(preprocessed_img_1, preprocessed_img_2)

        # Normalization
        integral = self._sum(preprocessed_img_1)
        normalized_img_1 = self._normalize(preprocessed_img_1)
        normalized_img_2 = self._normalize(preprocessed_img_2)

        # Convert format to a signature
        dx = tuple(preprocessed_img_1.voxel_size)
        time_num = preprocessed_img_1.time_num
        sig_1 = self._img_to_sig(normalized_img_1, dx=dx, time_num=time_num)
        sig_2 = self._img_to_sig(normalized_img_2, dx=dx, time_num=time_num)

        # Compute EMD distance
        dist = np.zeros(time_num, dtype=float)
        for i in range(time_num):
            dist[i], _, flow = cv2.EMD(sig_1[i], sig_2[i], cv2.DIST_L2)

        rescaled_distance = np.multiply(dist, integral)

        return float(rescaled_distance) if time_num == 1 else rescaled_distance

    def _preprocess(self, img: darsia.Image) -> darsia.Image:
        """
        Preprocessing routine, incl. extraction of array.

        Args:
            img (Image): image

        Returns:
            Image: image array under provided preprocessing
        """
        preprocessed_img = img.copy()
        if self.preprocess is not None:
            preprocessed_img = self.preprocess(preprocessed_img)
        return preprocessed_img

    def _compatibility_check(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
    ) -> bool:
        """
        Compatibility check.

        Args:
            img_1 (Image): image 1
            img_2 (Image): image 2

        Returns:
            bool: flag whether images 1 and 2 can be compared.

        """
        # Scalar valued
        assert img_1.scalar and img_2.scalar

        # Series
        assert img_1.time_num == img_2.time_num

        # Two-dimensional
        assert img_1.space_dim == 2 and img_2.space_dim == 2

        # Check whether the coordinate system is compatible
        assert darsia.check_equal_coordinatesystems(
            img_1.coordinatesystem, img_2.coordinatesystem
        )
        assert np.allclose(img_1.voxel_size, img_2.voxel_size)

        # Compatible distributions - comparing sums is sufficient since it is implicitly
        # assumed that the coordinate systems are equivalent. Check each time step
        # separately.
        assert np.allclose(self._sum(img_1), self._sum(img_2))

    def _sum(self, img: darsia.Image) -> Union[float, np.ndarray]:
        """Sum over spatial entries.

        Args:
            img (darsia.Image): image

        Returns:
            float or array: integration over the space

        """
        sum_over_time = img.img.copy()
        for i in range(img.space_dim):
            sum_over_time = np.sum(sum_over_time, axis=0)
        return sum_over_time

    def _normalize(self, img: darsia.Image) -> np.ndarray:
        """
        Normalization of images to images with sum 1.

        Args:
            img (Image): image

        Returns:
            float: original sum
            np.ndarray: normalized image

        """
        integral = self._sum(img)
        normalized_img = np.divide(img.img, integral)
        return normalized_img

    def _img_to_sig(
        self,
        img: np.ndarray,
        dx: Union[float, tuple[float, float]] = 1.0,
        time_num: int = 1,
    ) -> np.ndarray:
        """Convert a 2D array to a signature for cv2.EMD.

        Args:
            img (array): image
            dx (float or tuple): distance from one pixel to another
            time_num (int): number of time steps

        Returns:
            np.ndarray: signature

        """
        if isinstance(dx, float):
            dx = (dx, dx)
        del_y, del_x = dx

        def _signature_for_slice(time_index: int = 0):
            """Auxiliary function. Create signature for single time slice."""
            sig = np.empty((int(img.size / time_num), 3), dtype=np.float32)
            count = 0
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    sig[count] = np.array(
                        [
                            np.atleast_3d(img)[row, col, time_index],
                            col * del_x,
                            row * del_y,
                        ]
                    ).astype(np.float32)
                    count += 1
            return sig

        sig = np.empty((time_num, int(img.size / time_num), 3), dtype=np.float32)
        for i in range(time_num):
            sig[i] = _signature_for_slice(i)

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
