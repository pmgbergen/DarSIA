from __future__ import annotations
from os import remove

import numpy as np
from matplotlib.cm import get_cmap

import daria as da
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2



class CompareSegmentations:


    def __init__(self, number_of_segmented_images: int, **kwargs) -> None:
        """
        Constructor of compare segmentations class.
        
        Args:
            number_of_segmented_images (int): Number of segmentations to be compared
            Optional keyword arguments (kwargs):
                segmentation_names (list): list of names for the different segmented 
                    images. So far only used in legends and color dictionary.
                components (list): list of the different components that are
                    considered in the segmented images. So far only two are allowed
                    to be provided.
                component_names (list):  list of names for the different components.
                    So far only used in legends, and color dictionary.
                non_active_component (int): value of nonactive component in the segmented
                    images.
                gray_colors (np.ndarray): array of three different scales of
                    gray (in RGB format), one for each of the different combinations of
                    components in the segmentations.
                colors (np.ndarray): Array of different colors that should
                    indicate unique components in each segmentation.
                light_scaling (float): Indicate how much lighter the second
                    component should be scaled in its unique color.
        """

        self.number_of_segmented_images = number_of_segmented_images
        self.segmentation_names: list = kwargs.pop("segmentation_names", [f"Segmentation {i}" for i in range(self.number_of_segmented_images)])

        # Define components
        self.components: list = kwargs.pop("components", [1, 2])
        self.non_active_component: int = kwargs.pop("non_active_component", 0)
        self.component_names: list = kwargs.pop("component_names", ["Component 0", "Component 1"])


        # Define gray colors
        self.gray_colors: np.ndarray = kwargs.pop(
            "gray_colors",
            np.array([[180, 180, 180], [220, 220, 220], [200, 200, 200]], dtype=np.uint8),
        )

        self.gray_base: np.ndarray = np.fix(self.gray_colors / self.number_of_segmented_images).astype(
                np.uint8
            )

        # Define unique colors
        self.light_scaling: float = kwargs.pop("light_scaling", 1.5)
        if "colors" not in kwargs:
            colormap = get_cmap("Spectral")
            self.colors: np.ndarray = np.zeros(
                (self.number_of_segmented_images, 2, 3), dtype=np.uint8
            )
            for i in range(self.number_of_segmented_images):
                rgba = 255 * np.array(
                    colormap(1 / self.number_of_segmented_images * (i + 0.5))[0:3]
                )
                rgbalight = np.trunc(self.light_scaling * rgba)
                self.colors[i, 0] = rgba.astype(np.uint8)
                self.colors[i, 1] = rgbalight.astype(np.uint8)
        else:
            colors_pre = kwargs.pop("colors")
            colors_light: np.ndarray = np.trunc(self.light_scaling * colors_pre)
            np.clip(colors_light, 0, 255, out=colors_light)
            self.colors: np.ndarray = np.hstack((colors_pre, colors_light))

        # Assert that there are a sufficient amount of colors
        # and that all of the segmentations are of equal size
        assert self.colors.shape[0] == self.number_of_segmented_images

        self.color_dictionary: dict = {}
        for i in range(self.number_of_segmented_images):
            self.color_dictionary[f"Unique apperance of {self.component_names[0]} in {self.segmentation_names[i]}"] = self.colors[i,0]
            self.color_dictionary[f"Unique apperance of {self.component_names[1]} in {self.segmentation_names[i]}"] = self.colors[i,1]

        for i in range(self.number_of_segmented_images-1):    
            self.color_dictionary[f"{i+2} segmentations overlaps in {self.component_names[0]}"] = self.gray_base[0]*(i+1)
            self.color_dictionary[f"{i+2} segmentations overlaps in {self.component_names[1]}"] = self.gray_base[1]*(i+1)
            self.color_dictionary[f"Mixture {self.component_names[0]} and {self.component_names[1]} for {i+2} segmentations"] = self.gray_base[2]*(i+1)
            
        

    def __call__(self, *segmentations, remove_background: bool = True, plot_result: bool = False, **kwargs) -> np.ndarray:
        """
        Comparison of segmentations.

        Args:
            segmentations (asterisk argument): Allows to provide an arbitraty
                number of segmented numpy arrays of integers to be compared
            Optional keyword arguments (kwargs):
                remove_background (bool): "removes" the image background
                plot_result (bool): plots the result with matplotlib if True, 
                    default is False.
                roi (tuple): roi where the segmentations should be compared.
                

        """

        # Define number of segmentations
        assert self.number_of_segmented_images == len(segmentations)


        if "roi" in kwargs:
            roi = kwargs["roi"]
            return_image: np.ndarray = np.zeros((roi[0].stop-roi[0].start, roi[1].stop-roi[1].start) + (3,), dtype=np.uint8)

        else:
            nx = min([seg.shape[0] for seg in segmentations])
            ny = min([seg.shape[1] for seg in segmentations])
            roi = (slice(0, nx), slice(0, ny))
            return_image: np.ndarray = np.zeros((nx, ny) + (3,), dtype=np.uint8)

        # Enter gray everywhere there are ovelaps of different segmentations
        for k in range(self.number_of_segmented_images):
            for i in range(k + 1, self.number_of_segmented_images):

                # Overlap of components
                for c_num, c in enumerate(self.components):
                    return_image[
                        np.logical_and(
                            segmentations[k][roi] == c,
                            segmentations[i][roi] == c,
                        )
                    ] += self.gray_base[c_num]

                # Overlap of different components
                return_image[
                    np.logical_or(
                        np.logical_and(
                            segmentations[k][roi] == self.components[0],
                            segmentations[i][roi] == self.components[1],
                        ),
                        np.logical_and(
                            segmentations[k][roi] == self.components[1],
                            segmentations[i][roi] == self.components[0],
                        ),
                    )
                ] += self.gray_base[2]

        # Determine locations (and make modifications to return image) of unique components
        for c_num, c in enumerate(self.components):
            for k in range(self.number_of_segmented_images):
                only_tmp: np.ndarray = segmentations[k][roi] == c
                for j in filter(lambda j: j != k, range(self.number_of_segmented_images)):
                    only_tmp = np.logical_and(
                        only_tmp, segmentations[j][roi] == self.non_active_component
                    )

                return_image[only_tmp] = self.colors[k, c_num]

        if plot_result:
            self.plot(return_image, "Comparison")

        return return_image

    def plot(self, im, fig_name = "Comparison", legend_anchor = (0.7, 1)):

        plt.figure(fig_name)
        plt.imshow(im)


        patches = self._get_legend_patches(im)
        plt.legend(handles=patches, bbox_to_anchor=legend_anchor, loc=2, borderaxespad=0. )
        plt.show()

    def _get_legend_patches(self, im):
        colors = self._get_unique_colors(im)

        # function to return key for any value
        def get_key(val, dictionary):
            for key, value in dictionary.items():
                if np.array_equal(val, value):
                    return key

            return f"{key} doesn't exist in dictionary"

        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=c/255, label=get_key(c,self.color_dictionary)) for c in colors ]
        return patches

    def _get_unique_colors(self, im) -> np.ndarray:
        nx, ny, _ = im.shape
        flat_im = im.reshape((nx*ny, 3))
        unique =  np.unique(flat_im, axis = 0)
        return unique[1:]

    def _remove_background(self, im, alpha_colors = 1.) -> np.ndarray:
        tmp = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        alpha = alpha_colors*alpha
        print(alpha.dtype)
        alpha[self._get_transitions(im)] = 1
        print(alpha.dtype)
        b, g, r = cv2.split(im)
        print(b.dtype)
        rgba = [b,g,r, alpha]
        return cv2.merge(rgba,4)

    def _get_transitions(self, im) -> np.ndarray:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        diff_im = da.forward_diff_x(im) + da.forward_diff_y(im)
        return diff_im != 0

    def plot_overlay_segmentation(self, comparison_image: np.ndarray, base_image: np.ndarray, figure_name: str = "Comparison", alpha_colors: float = 0.6, legend_anchor: tuple[float] = (1.,1.)) -> None:
        """
        Plots a comparison image overlayed a base image using matplotlib.

        Args:
            comparison_image (np.ndarray): The image containing comparison of segmentations.
            base_image (np.ndarray): The base image that is to be overlayed.
            figure_name (str): Figure name.
            comparison_alpha (float): Tha alpha value to determine transparency of the 
        """
        no_background_comparison_image = self._remove_background(comparison_image, alpha_colors)
        # b, g, r = cv2.split(base_image)
        # rgba = [b, g, r , np.ones_like(b)]
        # base_image_alpha = cv2.merge(rgba, 4)
        plt.figure(figure_name)
        plt.imshow(base_image)
        plt.imshow(no_background_comparison_image)
        patches = self._get_legend_patches(comparison_image)
        plt.legend(handles=patches, bbox_to_anchor=legend_anchor, loc=2, borderaxespad=0. )
        plt.show()

    def _white_background(self, im) -> np.ndarray:
        pass
