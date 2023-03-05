"""
Module containing auxiliary interpreters
for orientation of axes used for images.

Conventions. For unified use in the code base,
we introduce conventions of coordinate systems.

In 2d: The Cartesian coordinate system is
defined as x-y-coordinate system. E.g.
for an image with ij-matrix indexing the first
component corresponds to the y-axis with reversed
orientation, while the second axis corresponds
to the standard x-axis.

In 3d: The Cartesian coordinate system is
defines as xyz coordinate system. E.g.
for an image with ijk-matrix indexing, the first
component corresponds to the z axis with
reversed orientation, the second component corresponds
to the y-axis, and the third component corresponds to
the x-axis.

"""


def to_matrix_indexing(axis: Union[str, int]) -> str:
    """Conversion of single axis in Cartesian to matrix indexing.

    Args:
        axis (str or int): input axis in Cartesian indexing sense.

    Returns:
        str: converted axis in matrix indexing sense.

    """
    # Convert numeric axis description
    if isinstance(axis, int):
        axis = "xyz"[axis]

    if axis == "x":
        return "j"
    elif axis == "y":
        return "i"
    elif axis == "z":
        return "k"
    else:
        raise ValueError


def to_cartesian_indexing(axis: Union[str, int]) -> str:
    """Conversion of single axis in matrix indexing to Cartesian indexing.

    Args:
        axis (str or int): input axis in matrix indexing sense.

    Returns:
        str: converted axis in Cartesian indexing sense.

    """
    # Convert numeric axis description
    if isinstance(axis, int):
        axis = "ijk"[axis]

    if axis == "i":
        return "y"
    elif axis == "j":
        return "x"
    elif axis == "k":
        return "z"
    else:
        raise ValueError


def interpret_orientation(axis: str, orientation: str) -> tuple[int, bool]:
    """Interpretation of axes and their orientation.

    Args:
        axis (str): target axis, e.g., "x"
        orientation (str): orientation of an image, e.g., "ijk"

    Returns:
        int: component corresponding to the axis. Covered: "x", "y", "z",
            "i", "j", "k".
        bool: flag controlling whether the axis has to be reverted
            when converting. Covered: "xyz", "ijk".

    """

    # Consider all possible combinations.

    # ! ---- 2D Cartesian indexing

    if orientation == "xy":
        if axis == "x":
            return 0, False
        elif axis == "y":
            return 1, False
        elif axis == "i":
            return 1, True
        elif axis == "j":
            return 0, False

    # ! ---- 2D matrix indexing

    elif orientation == "ij":
        if axis == "x":
            return 1, False
        elif axis == "y":
            return 0, True
        elif axis == "i":
            return 0, False
        elif axis == "j":
            return 1, False

    # ! ---- 3D Cartesian indexing

    if orientation == "xyz":
        if axis == "x":
            return 0, False
        elif axis == "y":
            return 1, False
        elif axis == "z":
            return 2, False
        elif axis == "i":
            return 2, True
        elif axis == "j":
            return 0, False
        elif axis == "k":
            return 1, False

    # ! ---- 3D atrix indexing

    elif orientation == "ijk":
        if axis == "x":
            return 1, False
        elif axis == "y":
            return 2, False
        elif axis == "z":
            return 0, True
        elif axis == "i":
            return 0, False
        elif axis == "j":
            return 1, False
        elif axis == "k":
            return 2, False

    # If the method reaches this point, something went wrong.
    # This fact is used as safety check.
    raise ValueError
