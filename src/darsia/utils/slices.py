"""Module containing auxiliary functions for working with slices."""


def add_slices(slice1, slice2):
    """Auxiliary function to add slices."""
    return slice(slice1.start + slice2.start, slice1.stop + slice2.start)


def add_slice_pairs(slice_pair1, slice_pair2):
    """Auxiliary function to add slice pairs."""
    return (
        add_slices(slice_pair1[0], slice_pair2[0]),
        add_slices(slice_pair1[1], slice_pair2[1]),
    )


def subtract_slices(slice1, slice2):
    """Auxiliary function to subtract slices."""
    return slice(slice1.start - slice2.start, slice1.stop - slice2.start)


def subtract_slice_pairs(slice_pair1, slice_pair2):
    """Auxiliary function to subtract slice pairs."""
    return (
        subtract_slices(slice_pair1[0], slice_pair2[0]),
        subtract_slices(slice_pair1[1], slice_pair2[1]),
    )
