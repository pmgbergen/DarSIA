"""Perspective map combined with so-called bulge and stretch routines.

The class offers warping as well as calibration routines.
In addition, quick-access is provided for routine-based use of DarSIA.

"""

import numpy as np
import scipy.optimize as optimize

import darsia


class GeneralizedPerspectiveTransformation:
    """Combination of perspective map, bulge and stretch for 2d images."""

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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Application of generalized perspective to coordinate array.

        Args:
            x (np.ndarray): coordinate (array)

        Returns:
            np.ndarray: warped coordinate (array)

        """
        # For general definition of the method, convert to coordinate array
        single_coordinate_input = len(x.shape) == 1
        concatenated_input = len(x.shape) == 2 and x.shape[0] == 2
        coordinate_array_input = len(x.shape) == 3 and x.shape[0] == 2
        shape = x.shape
        if single_coordinate_input:
            x = x.reshape((2, 1))
        elif concatenated_input:
            pass
        elif coordinate_array_input:
            # Concatenate input to shape (2, n)
            x = x.reshape((2, -1), order="F")
        else:
            assert False, f"Unexpected shape of input: {x.shape}"

        # Initialize output
        out = np.zeros_like(x)

        # Apply perspective transform onto x
        out = self.A @ x
        out[0] += self.b[0]
        out[1] += self.b[1]
        scaling_factor = (self.c @ x) + 1
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

        if single_coordinate_input:
            return out.reshape((2,), order="F")
        elif concatenated_input:
            return out
        elif coordinate_array_input:
            return out.reshape((2, *shape[1:]), order="F")

    def fit(
        self,
        pts_src,
        pts_dst,
        img_src=None,
        img_dst=None,
        maxiter=100,
        tol=1e-5,
        strategy=["all"],
    ):
        """
        Fit generalized perspective transformation to given points.

        Args:
            pts_src (np.ndarray): source points
            pts_dst (np.ndarray): target points
            img_src (darsia.Image): source image
            img_dst (darsia.Image): target image
            maxiter (int): maximum number of iterations
            tol (float): tolerance for optimization
            strategy (list): list of strategies to use for optimization

        Returns:
            scipy.optimize.OptimizeResult: optimization result

        """
        # Convert input to arrays of shape (2, n)
        pts_src = np.array(pts_src).T
        pts_dst = np.array(pts_dst).T

        if img_dst is not None:
            # Retrieve some fixed parameters from a reference image - set default if not
            # given
            self.center = 0.5 * np.array(img_dst.num_voxels)
            """Center of image for stretch and bulge transformation"""
            self.max_coordinate = np.array(img_dst.num_voxels)
            """Maximum coordinate of image for stretch and bulge transformation"""
            self.min_coordinate = np.zeros(2, dtype=float)
            """Minimum coordinate of image for stretch and bulge transformation"""
            self.shape = img_dst.num_voxels
            """Target shape of image for generalized perspective transformation"""
            self.reference_metadata = img_dst.metadata()
            """Reference metadata for generalized perspective transformation"""

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
            return np.sum((pts_src_warped - pts_dst) ** 2) + 1e-8 * regularization

        for item in strategy:
            if item == "perspective":
                opt_result = optimize.minimize(
                    objective_function,
                    self.initial_parameters[:8],
                    method="Powell",
                    tol=tol,
                    options={"maxiter": maxiter, "disp": True},
                )
                self.initial_parameters[:8] = opt_result.x[:8].copy()

            elif item == "all":
                opt_result = optimize.minimize(
                    objective_function,
                    self.initial_parameters,
                    method="Powell",
                    tol=tol,
                    options={"maxiter": maxiter, "disp": True},
                )
                self.initial_parameters = opt_result.x.copy()

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

        assert len(img.img.shape) == 3, "TODO"

        # Define coordinates (of the input image)
        coordinates = np.indices(img.num_voxels, dtype=float).reshape(
            (2, -1), order="F"
        )

        # Apply generalized perspective (wrt reference image)
        warped_coordinates = self.__call__(coordinates)

        # Identify valid coordinates
        # warped_coordinates = warped_coordinates.reshape((2, -1), order="F")
        valid_warped_coordinate_ids = np.all(
            np.logical_and(
                warped_coordinates >= 0,
                warped_coordinates < np.array(self.shape).reshape((2, 1)),
            ),
            axis=0,
        )

        # Restrict to valid coordinates
        coordinates = coordinates[:, valid_warped_coordinate_ids]
        warped_coordinates = warped_coordinates[:, valid_warped_coordinate_ids]

        # Initialize data array
        warped_image_array = np.zeros(
            (*self.shape, img.img.shape[2]), dtype=img.img.dtype
        )

        # Assign output to valid coordinates
        for i in range(img.img.shape[2]):
            warped_image_array[
                warped_coordinates[0].astype(int), warped_coordinates[1].astype(int), i
            ] = img.img[coordinates[0].astype(int), coordinates[1].astype(int), i]

        # Convert data to Image
        warped_image = type(img)(warped_image_array, **self.reference_metadata)
        return warped_image


# Quick-access
def make_generalizedperspective():
    return None
