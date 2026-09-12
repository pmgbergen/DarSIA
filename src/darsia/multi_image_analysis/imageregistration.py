"""Module containing a diffeomorphic image registration tool.

"""

from __future__ import annotations

import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class DiffeomorphicImageRegistration:
    """
    Class to detect the deformation between different images.

    After all, DiffeomorphicImageRegistration is a wrapper using TranslationAnalysis.
    """

    def __init__(self, img_dst: darsia.Image, **kwargs) -> None:
        """Constructor for DiffeomorphicImageRegistration.

        Parameters
        ----------
        img_dst : darsia.Image
            Reference image, held fixed in the analysis (the destination).
        **kwargs
            Registration parameters::

                N_patches      list of two int -- patches in x and y
                rel_overlap    float -- relative patch overlap per direction
                max_features   int -- max features in the feature detection
                tol            float -- tolerance
                mask           ndarray -- ROI in which features are considered
        """
        # Create translation estimator
        max_features = kwargs.get("max_features", 200)
        tol = kwargs.get("tol", 0.05)
        self.translation_estimator = darsia.TranslationEstimator(max_features, tol)

        # Create translation analysis tool, and use the baseline image as reference point
        self.N_patches = kwargs.get("N_patches", [1, 1])
        self.rel_overlap = kwargs.get("rel_overlap", 0.0)
        mask_dst: Optional[darsia.Image] = kwargs.get("mask_dst", None)
        self.translation_analysis = darsia.TranslationAnalysis(
            img_dst,
            N_patches=self.N_patches,
            rel_overlap=self.rel_overlap,
            translation_estimator=self.translation_estimator,
            mask=mask_dst,
        )

    def update_dst(self, img_dst: darsia.Image) -> None:
        """
        Update of dst image.

        Parameters
        ----------
        img_dst : np.ndarray
            Image array.
        """
        self.translation_analysis.update_base(img_dst)

    def deduct(
        self, diffeomorphic_image_registration
    ) -> None:  #: DiffeomorphicImageRegistration
        """
        Effectviely copy from external DiffeomorphicImageRegistration.

        Parameters
        ----------
        diffeomorphic_image_registration : darsia.DiffeomorphicImageRegistration
            Diffeomorphic image registration object holding a translation analysis.
        """
        # The displacement is stored in the translation analysis as callable.
        # Thus, the current translation analysis has to be updated.
        self.translation_analysis.deduct_translation_analysis(
            diffeomorphic_image_registration.translation_analysis
        )

    def add(
        self, diffeomorphic_image_registration
    ) -> None:  #: DiffeomorphicImageRegistration
        """
        Update the store translation by adding the translation
        of an external diffeomorphic image registration.

        Parameters
        ----------
        diffeomorphic_image_registration : darsia.DiffeomorphicImageRegistration
            Diffeomorphic image registraton object holding a translation analysis.
        """
        # The displacement is stored in the translation analysis as callable.
        # Thus, the current translation analysis has to be updated.
        self.translation_analysis.add_translation_analysis(
            diffeomorphic_image_registration.translation_analysis
        )

    def __call__(
        self,
        img: darsia.Image,
        mask: Optional[np.ndarray] = None,
        return_transformed_dst: bool = False,
    ) -> darsia.Image:
        """
        Image registration routine.

        Parameters
        ----------
        img : darsia.Image
            Test image.
        mask : np.ndaray
            Active mask.
        return_transformed_dst : bool
            Flag whether the transform is also applied
            to dst (inversely).

        Returns
        -------
        darsia.Image
            Transformed test image.
        darsia.Image
            Transformed reference image #TODO?
        """
        transformed_img = self.translation_analysis(img, mask=mask)

        # Return results
        if return_transformed_dst:
            transformed_dst = self.apply(self.dst, reverse=True)
            return transformed_img, transformed_dst
        else:
            return transformed_img

    def call_with_output(
        self,
        img: darsia.Image,
        plot_patch_translation: bool = False,
        return_patch_translation: bool = False,
        mask: Optional[darsia.Image] = None,
    ) -> darsia.Image:
        """
        Determine the deformation pattern and apply diffeomorphism to the image
        aiming at matching the reference/destination (dst) image.

        This in the end only a wrapper for the translation analysis.

        Parameters
        ----------
        img : darsia.Image
            Test image.
        plot_patch_translation : bool
            Flag controlling whether the displacement is also
            visualized as vector field.
        return_patch_translation : bool
            Flag controlling whether the displacement
            in the patch centers is returned in the sense of img to dst,
            complying to the plot; default is False.
        mask : darsia.Image, optional
            Active mask.
        """
        transformed_img = self.translation_analysis(img, mask=mask)

        if return_patch_translation:
            patch_translation = self.translation_analysis.return_patch_translation(
                False
            )

        if plot_patch_translation:
            self.translation_analysis.plot_translation(False)

        if return_patch_translation:
            return transformed_img, patch_translation
        else:
            return transformed_img

    def evaluate(
        self,
        coords: Union[np.ndarray, darsia.Patches],
        reverse: bool = False,
        units: str = "metric",
    ) -> np.ndarray:
        """
        Evaluate diffeormorphism in arbitrary points.

        Parameters
        ----------
        coords : np.ndarray, or darsia.Patches
            Coordinate array with shape num_pts x 2,
            or alternatively num_rows_pts x num_cols_pts x 2, identifying points in a
            mesh/patched image, or equivalently patch.
        reverse : bool
            Flag whether the translation is understood as from the
            test image to the baseline image, or reversed. The default is the
            former latter.
        units : str
            Input and output units; "metric" default; otherwise assumed
            to be "pixel".

        Returns
        -------
        np.ndarray
            Deformation vectors for all coordinates.
        """

        # Reshape coords using a num_pts x 2 format.
        if isinstance(coords, darsia.Patches):
            coords = coords.global_centers_cartesian

        assert coords.shape[-1] == 2
        coords_shape = coords.shape
        coords = coords.reshape(-1, 2)

        # Convert coordinates to pixels with reverse matrix indexing,
        # if provided in metric units.
        if units == "metric":
            base = self.translation_analysis.base
            pixel_coords = base.coordinatesystem.voxel(coords)
            pixel_coords = np.fliplr(pixel_coords)  # Reverse matrix ind.

        elif units == "pixel":
            pixel_coords = coords
        else:
            raise ValueError

        # Interpolate at provided values - expect reverse matrix indexing
        translation = self.translation_analysis.translation(pixel_coords)

        # Flip, if required
        if reverse:
            translation *= -1.0

        # Collect results, use ordering of components consistent with matrix
        # indexing (i.e. flip of components needed)
        displacement = np.transpose(np.vstack((translation[1], translation[0])))

        # Convert to metric units if required; for pixels, use matrix indexing.
        if units == "metric":
            displacement = base.coordinatesystem.coordinate_vector(displacement)

        # Reshape to format used at input
        return displacement.reshape(coords_shape)

    def apply(self, img: darsia.Image, reverse: bool = False) -> darsia.Image:
        """
        Apply computed transformation onto arbitrary image.

        Parameters
        ----------
        img : np.ndarray or darsia.Image
            Image.
        reverse : bool
            Flag whether the translation is understood as from the
            test image to the baseline image, or reversed. The default is the
            latter.

        Returns
        -------
        np.ndarray, optional
            Transformed image, if input is array; no output otherwise.
        """
        # Load the image into translation_analysis
        self.translation_analysis.load_image(img)

        # Apply the transformation, stored in translation_analysis
        return self.translation_analysis.translate_image(reverse)

    def plot(
        self,
        scaling: float = 1.0,
        mask: Optional[darsia.Image] = None,
    ) -> None:
        """
        Plots diffeomorphism.

        Parameters
        ----------
        scaling : float
            Scaling for vectors.
        mask : darsia.Image, optional
            Active set.
        """
        # Wrapper for translation_analysis.
        self.translation_analysis.plot_translation(
            reverse=False, scaling=scaling, mask=mask
        )

    def displacement(self) -> np.ndarray:
        """
        Return displacement in metric units on all pixels.

        """
        # Define coordinates for each pixel
        Ny, Nx = self.translation_analysis.base.img.shape[:2]
        # Nz = 1
        x = np.arange(Nx)
        y = np.arange(Ny)
        X_pixel, Y_pixel = np.meshgrid(x, y)

        # Transform coordinates into the right format (vector)
        pixel_vector = np.transpose(np.vstack((np.ravel(X_pixel), np.ravel(Y_pixel))))
        # Reshape coords using a num_pts x 2 format.
        pixel_coords = pixel_vector.reshape(-1, 2)

        # Interpolate at provided values - expect reverse matrix indexing
        tic = time.time()
        translation = self.translation_analysis.translation(pixel_coords)
        print(f"translation evaluation: {time.time() - tic}")

        #        # Flip, if required
        #        if reverse:
        #            translation *= -1.0

        # results, use ardering of components consistent with matrix indexing
        displacement = np.transpose(np.vstack((translation[1], translation[0])))

        # Convert to metric units
        tic = time.time()
        displacement = (
            self.translation_analysis.base.coordinatesystem.coordinate_vector(
                displacement
            )
        )
        print(f"coordinate system: {time.time() - tic}")

        # Cache
        self.displacement = displacement

        return displacement


class MultiscaleDiffeomorphicImageRegistration:
    """Multiscale diffeomorphic image registration, for larger deformations."""

    def __init__(
        self,
        img_dst: darsia.Image,
        config: Union[dict, list[dict]],
        mask_dst: Optional[np.ndarray] = None,
        total_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Set up the multiscale registration.

        Parameters
        ----------
        img_dst : darsia.Image
            Reference image, held fixed in the analysis (the destination).
        config : dict or list of dict
            Hierarchy of per-level configuration dictionaries.
        mask_dst : numpy.ndarray, optional
            Active mask.
        total_config : dict, optional
            Parameters for the final, overall registration.
        """
        # Cache inputs
        self.img_dst = img_dst
        self.mask_dst = mask_dst

        assert isinstance(config, list) and all(
            [isinstance(config[i], dict) for i in range(len(config))]
        )
        self.config = config
        self.num_levels = len(self.config)

        # Make sure to have a fine config for the final image registration
        self.total_config = self.config[-1] if total_config is None else total_config

        # Cache verbosity
        self.verbosity = kwargs.get("verbosity", 0)

    def __call__(
        self,
        img: darsia.Image,
        mask: Optional[np.ndarray] = None,
        return_transformed_dst: bool = False,
    ) -> darsia.Image:  # TODO typing.
        """
        Image registration routine.

        Parameters
        ----------
        img : Image
            Test image.
        mask : np.ndaray
            Active mask.
        return_transformed_dst : bool
            Flag whether the transform is also applied
            to dst (inversely).

        Returns
        -------
        Image
            Transformed test image.
        """
        # Store inputs
        transformed_img = img.copy()
        transformed_mask = (
            np.ones(img.img.shape[:2], dtype=bool) if mask is None else mask.copy()
        )

        # Initialize combined image registration
        self.combined_image_registration = DiffeomorphicImageRegistration(
            transformed_img, **self.total_config
        )

        # Multi level approach succesively updating the combined image registration
        for level in range(self.num_levels):
            # Determine deformation for current level
            _, image_registration = self._single_level_iteration(
                transformed_img, transformed_mask, self.config[level]
            )

            # Update inputs
            if level == 0:
                self.combined_image_registration.deduct(image_registration)
            else:
                self.combined_image_registration.add(image_registration)
            transformed_mask = image_registration.apply(transformed_mask)
            transformed_img = self.combined_image_registration.apply(img)

        # Return results
        if return_transformed_dst:
            transformed_dst = self.combined_image_registration.apply(
                self.dst, reverse=True
            )
            return transformed_img, transformed_dst
        else:
            return transformed_img

    def _single_level_iteration(
        self,
        img: darsia.Image,
        mask: np.ndarray,
        config: dict,
    ) -> tuple[darsia.Image, DiffeomorphicImageRegistration]:  # TODO typing
        """One iteration of multiscale image registration.

        Parameters
        ----------
        img : darsia.Image
            Test image.
        mask : np.ndarray
            Active mask.
        config : dict
            Parameters for image registration.

        Returns
        -------
        darsia.Image
            Transformed image.
        darsia.DiffeomorphicImageRegistration
            Resulting image registration.
        """
        # Find image registration
        image_registration = DiffeomorphicImageRegistration(
            self.img_dst, mask=self.mask_dst, **config
        )

        plot = self.verbosity >= 2
        transformed_img, patch_translation = image_registration.call_with_output(
            img, plot_patch_translation=plot, return_patch_translation=True, mask=mask
        )

        if self.verbosity >= 2:
            plt.figure("comparison")
            plt.imshow(
                skimage.util.compare_images(
                    self.img_dst.img, transformed_img.img, method="blend"
                )
            )
            plt.show()

        return transformed_img, image_registration

    def apply(self, img: darsia.Image, reverse: bool = False) -> darsia.Image:
        """
        Apply computed transformation onto arbitrary image.

        Parameters
        ----------
        img : np.ndarray or darsia.Image
            Image.
        reverse : bool
            Flag whether the translation is understood as from the
            test image to the baseline image, or reversed. The default is the
            latter.

        Returns
        -------
        np.ndarray, optional
            Transformed image, if input is array; no output otherwise.
        """
        if not hasattr(self, "combined_image_registration"):
            raise ValueError("Construct the deformation first.")
        return self.combined_image_registration.apply(img, reverse)

    def plot(
        self,
        scaling: float,
        mask: Optional[darsia.Image] = None,
    ) -> None:
        """
        Plot the dislacement stored in the current image registration.

        Parameters
        ----------
        scaling : float
            Scaling parameter to controll the length of the arrows.
        mask : np.ndarray
            Active mask.
        """
        if not hasattr(self, "combined_image_registration"):
            raise ValueError("Construct the deformation first.")
        self.combined_image_registration.plot(scaling=scaling, mask=mask)

    def evaluate(
        self,
        coords: Union[np.ndarray, darsia.Patches],
        reverse: bool = False,
        units: str = "metric",
    ) -> np.ndarray:
        """See evaluate in DiffeomorphicImageRegistration."""
        return self.combined_image_registration.evaluate(coords, reverse, units)


# ! ---- Administrator of Image Registration algorithms


class ImageRegistration:
    """Patch-wise registration of an image onto a fixed destination image.

    Estimates a displacement field (e.g. from sand settling or compaction) that
    can be evaluated at arbitrary points or applied to another image.
    """

    def __init__(
        self,
        img_dst: darsia.Image,
        method: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Set up the image registration.

        Parameters
        ----------
        img_dst : darsia.Image
            Reference image, held fixed in the analysis (the destination).
        method : str, optional
            Registration method.
        **kwargs
            Multiscale parameters, each a value or a list (one entry per
            level)::

                N_patches      list of two int -- patches in x and y
                rel_overlap    float -- relative patch overlap per direction
                max_features   int -- max features in the feature detection
                tol            float -- tolerance
                mask           ndarray -- ROI in which features are considered
        """
        assert method in [None, "multilevel", "onelevel"]

        # Fetch keyword arguments
        N_patches = kwargs.get("N_patches", [1, 1])
        max_features = kwargs.get("max_features", 200)
        tol = kwargs.get("tol", 0.05)
        rel_overlap = kwargs.get("rel_overlap", 0.0)
        mask_dst: Optional[darsia.Image] = kwargs.get("mask_dst", None)
        verbosity = kwargs.get("verbosity", 0)

        # Method provided through N_patches
        if method is None:
            method = (
                "multilevel"
                if isinstance(N_patches, list)
                and all([isinstance(N_patches[i], list) for i in range(len(N_patches))])
                else "onelevel"
            )

        if method == "multilevel":
            if not isinstance(N_patches, list):
                N_patches = [N_patches]
            num_levels = len(N_patches)

            # Check compatibility
            compatibility = True
            if isinstance(max_features, int):
                max_features = num_levels * [max_features]
            else:
                compatibility = (
                    compatibility
                    and isinstance(max_features, list)
                    and len(max_features) == num_levels
                )

            if isinstance(tol, float):
                tol = num_levels * [tol]
            else:
                compatibility = (
                    compatibility and isinstance(tol, list) and len(tol) == num_levels
                )

            if isinstance(rel_overlap, float):
                rel_overlap = num_levels * [rel_overlap]
            else:
                compatibility = (
                    compatibility
                    and isinstance(rel_overlap, list)
                    and len(rel_overlap) == num_levels
                )

            if not compatibility:
                raise ValueError(
                    "Input for the multilevel image registration is not compatible."
                )

            # Prepare hierarchy of config dictionaries
            config = [
                {
                    "N_patches": N_patches[i],
                    "max_features": max_features[i],
                    "tol": tol[i],
                    "rel_overlap": rel_overlap[i],
                }
                for i in range(num_levels)
            ]

            self.image_registration = MultiscaleDiffeomorphicImageRegistration(
                img_dst, config, mask_dst, verbosity=verbosity
            )

        elif method == "onelevel":

            # Extract possibly the first value from list, if list is provided
            if isinstance(N_patches, list) and all(
                [isinstance(N_patches[i], list) for i in range(len(N_patches))]
            ):
                N_patches = N_patches[0]

            if isinstance(max_features, list):
                max_features = max_features[0]

            if isinstance(tol, list):
                tol = tol[0]

            if isinstance(rel_overlap, list):
                rel_overlap = rel_overlap[0]

            # Construct feasible config
            config = {
                "N_patches": N_patches,
                "max_features": max_features,
                "tol": tol,
                "rel_overlap": rel_overlap,
                "mask": mask_dst,
                "verbosity": verbosity,
            }

            self.image_registration = DiffeomorphicImageRegistration(img_dst, **config)

        else:
            raise NotImplementedError(
                f"Method {method} is not implemented for ImageRegistration."
            )

    def __call__(
        self,
        img: darsia.Image,
        mask: Optional[np.ndarray] = None,
        return_transformed_dst: bool = False,
    ) -> darsia.Image:
        return self.image_registration(img, mask, return_transformed_dst)

    def apply(self, img: darsia.Image, reverse: bool = False) -> darsia.Image:
        """
        Apply computed transformation onto arbitrary image.

        Parameters
        ----------
        img : np.ndarray or darsia.Image
            Image.
        reverse : bool
            Flag whether the translation is understood as from the
            test image to the baseline image, or reversed. The default is the
            latter.

        Returns
        -------
        np.ndarray, optional
            Transformed image, if input is array; no output otherwise.
        """
        return self.image_registration.apply(img, reverse)

    def plot(self, scaling: float, mask: Optional[darsia.Image] = None) -> None:
        """
        Plot the dislacement stored in the current image registration.

        Parameters
        ----------
        scaling : float
            Scaling parameter to controll the length of the arrows.
        mask : np.ndarray
            Active mask.
        """
        self.image_registration.plot(scaling=scaling, mask=mask)

    def evaluate(
        self,
        coords: Union[np.ndarray, darsia.Patches],
        reverse: bool = False,
        units: str = "metric",
    ) -> np.ndarray:
        """See evaluate in DiffeomorphicImageRegistration."""
        return self.image_registration.evaluate(coords, reverse, units)
