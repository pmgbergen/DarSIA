"""Module containing object converting labels to masks."""

import numpy as np

import darsia


class Masks:
    """Iterable object converting a label imaged to masks."""

    def __init__(self, labels: darsia.Image) -> None:
        """Constructor.

        Args:
            labels (Image): labeled image

        """
        self.labels: darsia.Image = labels
        """Label image."""
        self.unique_labels: np.ndarray = np.unique(self.labels.img)
        """Labels."""
        self.num_labels: int = len(self.unique_labels)
        """Number of labels."""

    @property
    def size(self) -> int:
        """Return routine for total number of labels.

        Returns:
            int: number of labels
        """
        return self.num_labels

    def __iter__(self):
        """Iterator."""
        self.counter = 0
        return self

    def __next__(self) -> darsia.Image:
        """Next iterations.

        Returns:
            Image: next mask
        """
        if self.counter < self.num_labels:
            mask = self[self.counter]
            self.counter += 1
            return mask
        else:
            raise StopIteration

    def __getitem__(self, key) -> darsia.Image:
        """Access to specific mask.

        Args:
            key (int): counter (not the label!)

        Returns:
            Image: mask associated to counter

        """
        mask = self.labels.img == self.unique_labels[key]
        return darsia.Image(img=mask, **self.labels.metadata())
