"""
Module containing the general setup for a fluidflower rig
with segmented geometry.

"""
from pathlib import Path
from typing import Union

import numpy as np

import darsia


class FluidFlowerRig(darsia.AnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        darsia.AnalysisBase.__init__(self, baseline, config, update_setup)

        # Segment the baseline image; identidy water and esf layer.
        self._segment_geometry(update_setup=update_setup)

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self, update_setup: bool = False) -> None:
        """
        Use watershed segmentation and some cleaning to segment
        the geometry. Note that not all sand layers are detected
        by this approach.

        Args:
            update_setup (bool): flag controlling whether the segmentation
                is performed even if a reference file exists; default is False.
        """

        # Fetch or generate and store labels
        if (
            Path(self.config["segmentation"]["labels_path"]).exists()
            and not update_setup
        ):
            labels = np.load(self.config["segmentation"]["labels_path"])
        else:
            labels = darsia.segment(
                self.base.img,
                markers_method="supervised",
                edges_method="scharr",
                **self.config["segmentation"]
            )
            labels_path = Path(self.config["segmentation"]["labels_path"])
            labels_path.parents[0].mkdir(parents=True, exist_ok=True)
            np.save(labels_path, labels)

        # Cache the labeled image
        self.labels = labels

    def _labels_to_mask(self, ids: list[int]) -> np.ndarray:
        """
        Helper routine to connect labels with facies.

        Args:
            ids (list of int): ids in self.labels

        Returns:
            np.ndarray: corresponding mask

        """
        ids = ids if isinstance(ids, list) else [ids]
        mask = np.zeros(self.labels.shape[:2], dtype=bool)
        for i in ids:
            mask[self.labels == i] = True
        return mask
