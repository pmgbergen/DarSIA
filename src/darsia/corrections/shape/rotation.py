"""Module containing objects useful for correcting images wrt rotations

"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

import darsia


class RotationCorrection(darsia.BaseCorrection):
    """Class for rotation correction of nd images, up to 4d.

    Rotations are defined as combination of multiple basic rotations. In 2d, a single
    basic rotation is sufficient. In 3d, although three are available, two are
    sufficient.

    Attributes:
        dim (int): ambient dimension
        anchor (array or list): voxel coordinates of anchor
        rotation (array): rotation matrix
        rotation_inv (array): inverted rotation matrix

    """

    def __init__(
        self,
        anchor: Union[list[int], np.ndarray],
        **kwargs,
    ) -> None:
        # Cache anchor of rotation
        self.anchor = np.array(anchor)

        # Cache dimension (obtained from anchor)
        dim = len(self.anchor)
        self.dim = dim

        rotation_from_isometry = kwargs.get("rotation_from_isometry", False)
        if rotation_from_isometry:
            pts_src = kwargs.get("pts_src")
            pts_dst = kwargs.get("pts_dst")
            affine_map = darsia.AffineTransformation(
                pts_src,
                pts_dst,
            )
        else:
            rotations = kwargs.get("rotations")
            if rotations is None:
                raise ValueError("No means provided to determine rotations.")

        # Define rotation as combination of basic rotations
        if dim == 2:
            if rotation_from_isometry:
                self.rotation = affine_map.rotation
                self.rotation_inv = np.linalg.inv(affine_map.rotation)
            else:
                degree = rotations[0]
                vector = np.array([0, 0, 1])
                rotation = Rotation.from_rotvec(degree * vector)
                self.rotation = rotation.as_matrix()[:2, :2]
                rotation_inv = Rotation.from_rotvec(-degree * vector)
                self.rotation_inv = rotation_inv.as_matrix()[:2, :2]
        elif dim == 3:
            if rotation_from_isometry:
                self.rotation = affine_map.rotation
                self.rotation_inv = np.linalg.inv(affine_map.rotation)
            else:
                self.rotation = np.eye(dim)
                self.rotation_inv = np.eye(dim)
                for degree, cartesian_axis in rotations:
                    indexing = "xyz"[:dim]
                    matrix_axis, reverted = darsia.interpret_indexing(
                        cartesian_axis, indexing
                    )
                    vector = np.eye(dim)[matrix_axis]
                    scaling = -1 if reverted else 1
                    rotation = Rotation.from_rotvec(scaling * degree * vector)
                    self.rotation = np.matmul(self.rotation, rotation.as_matrix())
                    rotation_inv = Rotation.from_rotvec(-degree * vector)
                    self.rotation_inv = np.matmul(
                        self.rotation_inv, rotation_inv.as_matrix()
                    )

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Application of inherent rotation to provided image.

        Args:
            img (array): image

        Returns:
            array: rotated image

        """
        # Warp entire array by mapping target voxels to destination
        # voxels by applying the inverse rotations.

        # Implicitly assume the mapped image is of same size as the input image
        shape = img.shape
        num_voxels = np.prod(img.shape[: self.dim])

        # Collect all voxels in dim x num_voxels format
        if self.dim == 2:
            target_voxels = list(itertools.product(*[range(shape[0]), range(shape[1])]))
        elif self.dim == 3:
            target_voxels = list(
                itertools.product(*[range(shape[0]), range(shape[1]), range(shape[2])])
            )
        target_voxels = np.transpose(np.array(target_voxels))

        # Find corresponding voxels in the original image
        anchor_extruded = np.outer(self.anchor, np.ones(num_voxels))
        src_voxels = anchor_extruded + self.rotation_inv.dot(
            target_voxels - anchor_extruded
        )
        src_voxels = np.clip(
            src_voxels.astype(int),
            0,
            np.outer(np.array(shape) - 1, np.ones(num_voxels)),
        ).astype(int)
        rotated_img = np.zeros(shape)
        rotated_img[tuple(target_voxels[j] for j in range(self.dim))] = img[
            tuple(src_voxels[j] for j in range(self.dim))
        ]
        return rotated_img

        # print(target_voxels[:,34], src_voxels[:,34], img[src_voxels[:,34].astype(int)])

        print(self.anchor + self.rotation_inv.dot(np.array([3, 4]) - self.anchor))
        # Determine which columns lie within the image
        mask = np.logical_and(
            np.all(src_voxels > -1e-3, axis=0),
            np.all(
                np.less(src_voxels, np.outer(np.array(shape) - 1, np.ones(num_voxels))),
                axis=0,
            ),
        )

        # Deactivate voxels outside range - continue only with valid ones
        target_voxels = target_voxels[:, mask]
        src_voxels = src_voxels[:, mask]
        num_active_voxels = src_voxels.shape[1]

        # Find related voxels in two opposite corners defining a voxel,
        # containing src_voxels. Need to check whether the voxel is
        # contained in the image.
        base_corner = np.floor(src_voxels).astype(int)
        opposite_corner = base_corner + 1
        base_corner[
            np.less(
                opposite_corner,
                np.outer(np.array(shape) - 1, np.ones(num_active_voxels)),
            )
        ] -= 1
        opposite_corner[
            np.less(
                opposite_corner,
                np.outer(np.array(shape) - 1, np.ones(num_active_voxels)),
            )
        ] -= 1

        # Collect all corners of the marked voxels.
        corners = [
            np.vstack(comb)
            for comb in list(
                itertools.product(
                    *[[base_corner[i], opposite_corner[i]] for i in range(self.dim)]
                )
            )
        ]

        # Evaluate standard Q1 basis functions, and scale them such that they sum to 1.
        basis = [
            np.prod(1 - np.absolute(corner - src_voxels), axis=0) for corner in corners
        ]
        basis_sum = sum(basis)
        basis = [b / basis_sum for b in basis]

        # Bilinear / trilinear interpolation: sum(data in voxels * basis)
        q1_interpolation = sum(
            [
                np.multiply(
                    basis[i],
                    img[
                        tuple(
                            np.array(corners[i][j]).astype(int) for j in range(self.dim)
                        )
                    ],
                )
                for i in range(len(basis))
            ]
        )

        # Finally assign interpolated values to the associated target voxels.
        rotated_img = np.zeros(shape)
        rotated_img[
            tuple(np.floor(target_voxels[j]).astype(int) for j in range(self.dim))
        ] = q1_interpolation

        return rotated_img

    # ! ---- I/O ----

    def save(self, path: Path) -> None:
        raise NotImplementedError("Not implemented yet.")

    def load(self, path: Path) -> None:
        raise NotImplementedError("Not implemented yet.")
