"""Module containing object converting labels to masks."""

import numpy as np

import darsia


class Masks:
    """Iterable object converting a label imaged to masks."""

    def __init__(self, labels: darsia.Image, return_label: bool = False) -> None:
        """Constructor.

        Parameters
        ----------
        labels : Image
            Labeled image.
        return_label : bool
            Flag controlling whether iterator also returns labels.
        """
        self.labels: darsia.Image = labels
        """Label image."""
        self.unique_labels: np.ndarray = np.unique(self.labels.img)
        """Labels."""
        self.num_labels: int = len(self.unique_labels)
        """Number of labels."""
        self.return_label: bool = return_label
        self._masks_by_label: dict = {
            label: self.labels.img == label for label in self.unique_labels
        }
        """Per-label boolean masks, computed once; immutable after construction."""

    @property
    def size(self) -> int:
        """Return routine for total number of labels.

        Returns
        -------
        int
            Number of labels.
        """
        return self.num_labels

    def mask_for_label(self, label: int) -> np.ndarray:
        """Boolean mask for `label` (precomputed at construction, read-only).

        Parameters
        ----------
        label : int
            A value from `unique_labels`.

        Returns
        -------
        np.ndarray
            Boolean mask selecting the pixels belonging to `label`.
        """
        return self._masks_by_label[label]

    def __iter__(self) -> "MasksIterator":
        """Return a fresh, independent iterator.

        Each call returns a new `MasksIterator` holding its own counter,
        so concurrent iteration of the same `Masks` instance from
        multiple threads does not share or race on loop state.
        """
        return MasksIterator(self)

    def __getitem__(self, key) -> darsia.Image:
        """Access to specific mask.

        Parameters
        ----------
        key : int
            Counter (not the label!).

        Returns
        -------
        Image
            Mask associated to counter.
        """
        label = self.unique_labels[key]
        return darsia.Image(img=self.mask_for_label(label), **self.labels.metadata())


class MasksIterator:
    """Stateful iterator over a `Masks` object; one instance per iteration."""

    def __init__(self, masks: Masks) -> None:
        self._masks = masks
        self._counter = 0

    def __next__(self) -> darsia.Image:
        """Next iteration.

        Returns
        -------
        Image
            Next mask.
        """
        if self._counter < self._masks.num_labels:
            mask = self._masks[self._counter]
            label = self._masks.unique_labels[self._counter]
            self._counter += 1
            if self._masks.return_label:
                return mask, label
            else:
                return mask
        else:
            raise StopIteration
