"""Perspective map combined with so-called bulge and stretch routines.

The class offers warping as well as calibration routines.
In addition, quick-access is provided for routine-based use of DarSIA.

"""

from typing import Optional, Union, overload

import numpy as np
import scipy.optimize as optimize

import darsia


class GeneralizedPerspectiveTransformation:
    """Combination of perspective map, bulge and stretch for 2d images.

    Following the paradigm of phyisical image, generalized perspective transforms
    use physical parameters, and map from physical coordinates to coordinates. Thus,
    coordinate systems attached to images are paramount in translating to the associated
    voxel space.

    """

    def __init__(self) -> None:
        """Initialize generalized perspective transformation.

        The default parameters are set to the identity transformation.

        """
        self.A: np.array = np.array([1, 0, 0, 1], dtype=float)
        """Scaling matrix for perspective transformation"""
        self.b: np.array = np.zeros(2, dtype=float)
        """Translation vector for perspective transformation"""
        self.c: np.array = np.zeros(2, dtype=float)
        """Scaling vector for perspective transformation"""
        self.stretch_factor: np.array = np.zeros(2, dtype=float)
        """Stretch vector for stretch transformation"""
        self.stretch_center_off: np.array = np.zeros(2, dtype=float)
        """Offsett from center for stretch transformation"""
        self.bulge_factor: np.array = np.zeros(2, dtype=float)
        """Bulge vector for bulge transformation"""
        self.bulge_center_off: np.array = np.zeros(2, dtype=float)
        """Offsett from center for bulge transformation"""
        self.input_dtype = np.ndarray
        """Type of input for generalized perspective transformation"""
        self.output_dtype = np.ndarray
        """Type of output for generalized perspective transformation"""
        self.input_array_dtype = np.ndarray
        """Type of input arrays for generalized perspective transformation"""
        self.output_array_dtype = np.ndarray
        """Type of output arrays for generalized perspective transformation"""
        self.set_array_dtype()

        # Collect all input parameters in single vector to define a default state
        self.default_parameters = np.concatenate(
            (
                self.A.flatten(),
                self.b,
                self.c,
                self.stretch_factor,
                self.stretch_center_off,
                self.bulge_factor,
                self.bulge_center_off,
            )
        )
        """Default parameters for generalized perspective transformation"""

    def set_parameters_as_vector(self, parameters: np.array) -> None:
        """Set parameters of generalized perspective transformation.

        Args:
            parameters (np.array): parameters of generalized perspective transformation

        """
        assert len(parameters) <= len(self.default_parameters)
        self.A = parameters[:4].reshape((2, 2))
        self.b = parameters[4:6]
        self.c = parameters[6:8]
        if len(parameters) > 8:
            self.stretch_factor = parameters[8:10]
            self.stretch_center_off = parameters[10:12]
        if len(parameters) > 12:
            self.bulge_factor = parameters[12:14]
            self.bulge_center_off = parameters[14:16]

    def set_array_dtype(self) -> None:
        """Set array input and output type for generalized perspective transformation."""
        if self.input_dtype == darsia.Coordinate:
            self.input_array_dtype = darsia.CoordinateArray
        elif self.input_dtype == darsia.Voxel:
            self.input_array_dtype = darsia.VoxelArray
        else:
            self.input_array_dtype = np.ndarray

        if self.output_dtype == darsia.Coordinate:
            self.output_array_dtype = darsia.CoordinateArray
        elif self.output_dtype == darsia.Voxel:
            self.output_array_dtype = darsia.VoxelArray
        else:
            self.output_array_dtype = np.ndarray

    def __call__(
        self,
        x: Union[
            np.ndarray,
            darsia.Coordinate,
            darsia.CoordinateArray,
            darsia.Voxel,
            darsia.VoxelArray,
        ],
    ):
        """Application of generalized perspective to coordinate array.

        Args:
            x (np.ndarray, Coordinate, Voxel or corresponding Array of such):
                point to be transformed, type must match predefined input type

        Returns:
            output_dtype or output_array_dtype: warped point in predefined output type

        """
        # Be strict on input type (implicitly allow for arrays of coordinates or voxels)
        assert (
            type(x) is self.input_dtype or type(x[0]) is self.input_dtype
        ), """
            Input type must be of type {self.input_dtype} or {self.input_dtype} array
        """

        # For now, convert to plain numpy array
        x_arr = np.asarray(x)

        # For dimensionality reasons, collection of points require different treatment
        # than single points; the below code is written for arrays with columns as
        # points
        x_arr = np.atleast_2d(x_arr)
        array_input = x_arr.shape == x.shape
        x_arr = x_arr.T
        assert x_arr.shape[0] == 2, "Input must be 2d"

        # Initialize output and apply perspective transform onto x
        out = self.A @ x_arr
        out[0] += self.b[0]
        out[1] += self.b[1]
        scaling_factor = (self.c @ x_arr) + 1
        out[0] = np.divide(out[0], scaling_factor)
        out[1] = np.divide(out[1], scaling_factor)

        # Apply bulge transform onto out
        bulge_correction = np.zeros_like(out)
        relative_out = out.copy()
        relative_out[0] -= self.center[0] - self.bulge_center_off[0]
        relative_out[1] -= self.center[1] - self.bulge_center_off[1]
        relative_max = self.max_coordinate - self.center - self.bulge_center_off
        relative_min = self.min_coordinate - self.center - self.bulge_center_off
        bulge_correction[0] = (
            self.bulge_factor[0]
            * relative_out[0]
            * (relative_max[0] - relative_out[0])
            * (relative_out[0] - relative_min[0])
        )
        bulge_correction[1] = (
            self.bulge_factor[1]
            * relative_out[1]
            * (relative_max[1] - relative_out[1])
            * (relative_out[1] - relative_min[1])
        )
        out += bulge_correction

        # Apply stretch transform onto out
        stretch_correction = np.zeros_like(out)
        relative_out = out.copy()
        relative_out[0] -= self.center[0] * self.stretch_center_off[0]
        relative_out[1] -= self.center[1] * self.stretch_center_off[1]
        relative_max = self.max_coordinate - self.center - self.stretch_center_off
        relative_min = self.min_coordinate - self.center - self.stretch_center_off
        stretch_correction[0] = (
            self.stretch_factor[0]
            * relative_out[0]
            * (relative_max[1] - relative_out[1])
            * (relative_out[1] - relative_min[1])
        )
        stretch_correction[1] = (
            self.stretch_factor[1]
            * relative_out[1]
            * (relative_max[0] - relative_out[0])
            * (relative_out[0] - relative_min[0])
        )
        out += stretch_correction

        # Convert to right output type
        if array_input:
            return self.output_array_dtype(out.T)
        else:
            return self.output_dtype(out[0])

    def fit(
        self,
        pts_src: Union[np.ndarray, darsia.VoxelArray, darsia.CoordinateArray],
        pts_dst: Union[np.ndarray, darsia.VoxelArray, darsia.CoordinateArray],
        img_dst: Optional[darsia.Image] = None,
        maxiter: int = 100,
        tol: float = 1e-5,
        strategy: list = ["all"],
    ):
        """
        Fit generalized perspective transformation to given points.

        Also fixes types of input and output. Later evaluation of the transformation
        will require the same type of input and output.

        Args:
            pts_src (array, VoxelArray, or CoordinateArray): source points
            pts_dst (array, VoxelArray, or CoordinateArray): target points
            img_dst (darsia.Image): target image, defines coordinate system etc.
            maxiter (int): maximum number of iterations
            tol (float): tolerance for optimization
            strategy (list): list of strategies to use for optimization

        Returns:
            scipy.optimize.OptimizeResult: optimization result

        """
        # Assert (implicitly) pts_src and pts_dst are lists of coordinates or voxels.
        assert pts_src.shape == pts_dst.shape, "source and target points must match"
        # Update input and output type
        self.input_dtype = type(pts_src[0])
        self.output_dtype = type(pts_dst[0])
        self.set_array_dtype()

        # Retrieve some fixed parameters from a reference image
        assert img_dst is not None
        self.max_coordinate = (
            np.array(img_dst.num_voxels)
            if self.output_dtype == darsia.Voxel
            else np.max(np.vstack((img_dst.origin, img_dst.opposite_corner)), axis=0)
        )
        """Maximum coordinate of image for stretch and bulge transformation"""
        self.min_coordinate = (
            np.zeros(2, dtype=float)
            if self.output_dtype == darsia.Voxel
            else np.min(np.vstack((img_dst.origin, img_dst.opposite_corner)), axis=0)
        )
        """Minimum coordinate of image for stretch and bulge transformation"""
        self.center = 0.5 * (self.max_coordinate + self.min_coordinate)
        """Center of image for stretch and bulge transformation"""
        self.reference_shape = img_dst.num_voxels
        """Target shape of image for generalized perspective transformation"""
        self.reference_metadata = img_dst.metadata()
        """Reference metadata for generalized perspective transformation"""
        self.reference_coordinatesystem = img_dst.coordinatesystem
        """Reference coordinate system"""

        # Define initial parameters
        self.initial_parameters = self.default_parameters.copy()

        # Define the generalized perspective transformation for optimization
        def objective_function(params: np.ndarray):
            """LS objective function for generalized perspective transformation."""
            self.set_parameters_as_vector(params)
            pts_src_warped = self.__call__(pts_src)
            regularization = np.sum(
                (params - self.initial_parameters[: len(params)]) ** 2
            )
            return (
                np.linalg.norm(pts_src_warped - pts_dst, "fro") ** 2
                + 1e-8 * regularization
            )

        for item in strategy:
            if item == "perspective":
                ids = np.arange(8)
            elif item == "perspective+bulge":
                ids = np.arange(12)
            elif item == "all":
                ids = np.arange(len(self.initial_parameters))
            else:
                raise ValueError(f"Unknown strategy {item}")
            opt_result = optimize.minimize(
                objective_function,
                self.initial_parameters[ids],
                method="Powell",
                tol=tol,
                options={"maxiter": maxiter, "disp": True},
            )
            self.initial_parameters[ids] = opt_result.x[ids].copy()

        # Set parameters
        self.set_parameters_as_vector(opt_result.x)

        # Return optimization result
        return opt_result

    def warp(self, img: darsia.Image):
        """Warping of image according to generalized perspective.

        Args:
            img (darsia.Image): image to warp

        Returns:
            darsia.Image: warped image

        """
        assert len(img.img.shape) == 3, "currently only support for 2d images"

        # Define coordinates (of the input image)
        voxels = img.coordinatesystem.voxels
        coordinates = img.coordinatesystem.coordinates

        # Apply generalized perspective (wrt reference image), which is defined in
        # physical coordinates and maps to physical coordinates
        warped_coordinates = self.__call__(coordinates)
        warped_voxels = self.reference_coordinatesystem.voxel(warped_coordinates)

        # Identify valid coordinates
        valid_coordinates = np.all(
            np.logical_and(
                warped_coordinates >= self.min_coordinate,
                warped_coordinates < self.max_coordinate,
            ),
            axis=1,
        )

        # Restrict to valid coordinates
        voxels = voxels[valid_coordinates]
        warped_voxels = warped_voxels[valid_coordinates]

        # TODO use resultion of input image

        # Initialize data array
        warped_image_array = np.zeros(
            (*self.reference_shape, img.img.shape[2]), dtype=img.img.dtype
        )

        # Assign output to valid coordinates
        warped_image_array[warped_voxels[:, 0], warped_voxels[:, 1]] = img.img[
            voxels[:, 0], voxels[:, 1]
        ]

        # Convert data to Image
        warped_image = type(img)(warped_image_array, **self.reference_metadata)
        return warped_image


# Quick-access
def make_generalizedperspective():
    return None
