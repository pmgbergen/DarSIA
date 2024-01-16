"""Perspective map combined with so-called bulge and stretch routines.

The class offers warping as well as calibration routines.
In addition, quick-access is provided for routine-based use of DarSIA.

"""

from typing import Union

import numpy as np
import scipy.optimize as optimize

import darsia


class GeneralizedPerspectiveTransformation(darsia.BaseTransformation):
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
        super().__init__()

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

    def call_array(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Application of generalized perspective to coordinate array.

        Args:
            x (np.ndarray, Coordinate, Voxel or corresponding Array of such):
                point to be transformed, type must match predefined input type

        Returns:
            output_dtype or output_array_dtype: warped point in predefined output type

        """
        raise NotImplementedError("Evaluation not implemented")

    def inverse_array(self, x: np.ndarray) -> np.ndarray:
        """Evaluation of inverse generalized perspective to coordinate array.

        Args:
            x (np.ndarray, Coordinate, Voxel or corresponding Array of such):
                point to be transformed, type must match predefined input type

        Returns:
            output_dtype or output_array_dtype: warped point in predefined output type

        """
        # Convert to transpose format
        x_arr = x.T

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

        # Convert back to correct format
        return out.T

    def fit(
        self,
        pts_src: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        pts_dst: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        fit_options: dict = {},
    ):
        """
        Fit inverse generalized perspective transformation to given src and dst points.

        NOTE: Currently, the forward evaluation is not implemented.

        Also fixes types of input and output. Later evaluation of the transformation
        will require the same type of input and output.

        Args:
            pts_src (CoordinateArray, VoxelArray, VoxelCenterArray): source points
            pts_dst (CoordinateArray, VoxelArray, VoxelCenterArray): target points
            fit_options (dict): options for fitting
                coordinatesystem_dst (darsia.CoordinateSystem): coordinate system of
                    target points
                maxiter (int): maximum number of iterations
                tol (float): tolerance for optimization
                strategy (list): list of strategies to use for optimization

        Returns:
            scipy.optimize.OptimizeResult: optimization result

        """
        coordinatesystem_dst = fit_options.get("coordinatesystem_dst")
        assert coordinatesystem_dst is not None, "Need coordinatesystem_dst"
        maxiter = fit_options.get("maxiter", 100)
        tol = fit_options.get("tol", 1e-5)
        strategy = fit_options.get("strategy", ["all"])

        # Update domain and range of transformation
        self.set_dtype(pts_src, pts_dst)

        # Retrieve some fixed parameters from a reference image
        self.max_coordinate = (
            np.array(coordinatesystem_dst.shape)
            if self.output_dtype == darsia.Voxel
            else coordinatesystem_dst.max_coordinate
        )
        """Maximum coordinate of image for stretch and bulge transformation"""
        self.min_coordinate = (
            np.zeros(2, dtype=float)
            if self.output_dtype == darsia.Voxel
            else coordinatesystem_dst.min_coordinate
        )
        """Minimum coordinate of image for stretch and bulge transformation"""
        self.center = 0.5 * (self.max_coordinate + self.min_coordinate)
        """Center of image for stretch and bulge transformation"""

        # Define initial parameters
        self.initial_parameters = self.default_parameters.copy()

        # Define the (inverse) generalized perspective transformation for optimization
        def objective_function(params: np.ndarray):
            """LS objective function for generalized perspective transformation."""
            self.set_parameters_as_vector(params)
            pts_dst_warped = self.inverse_array(pts_dst)
            regularization = np.sum(
                (params - self.initial_parameters[: len(params)]) ** 2
            )
            return (
                np.linalg.norm(pts_dst_warped - pts_src, "fro") ** 2
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


class GeneralizedPerspectiveCorrection(darsia.TransformationCorrection):
    """Class for applying generalized perspective transformation to images."""

    def __init__(
        self,
        coordinatesystem_src: darsia.CoordinateSystem,
        coordinatesystem_dst: darsia.CoordinateSystem,
        pts_src: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        pts_dst: Union[
            darsia.CoordinateArray, darsia.VoxelArray, darsia.VoxelCenterArray
        ],
        fit_options: dict = {},
    ):
        # Setup transformation
        fit_options["coordinatesystem_dst"] = coordinatesystem_dst
        transformation = GeneralizedPerspectiveTransformation()
        transformation.fit(
            pts_src,
            pts_dst,
            fit_options,
        )
        super().__init__(coordinatesystem_src, coordinatesystem_dst, transformation)

        # Cache reference metadata
        self.dst_dimensions = coordinatesystem_dst.dimensions
        self.dst_origin = coordinatesystem_dst._coordinate_of_origin_voxel

    def correct_metadata(self, metadata: dict = {}) -> dict:
        """Extract metadata from the config file.

        Args:
            metadata (dict, optional): metadata dictionary to be updated. Defaults to {}.

        Returns:
            dict: metadata

        """
        return {
            "dimensions": self.dst_dimensions,
            "origin": self.dst_origin,
        }
