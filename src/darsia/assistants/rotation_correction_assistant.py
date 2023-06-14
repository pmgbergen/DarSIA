"""Module containing an interactive tool to tune rotation corrections."""

from warnings import warn

import numpy as np

import darsia


class RotationCorrectionAssistant(darsia.BaseAssistant):
    """Class defining an assistant to set up :mod:`darsia.RotationCorrection`."""

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        """Constructor.

        Args:
            img (Image): image to be corrected
            kwargs (keyword arguments): optional arguments
                threshold (float): threshold for active voxels
                relative (bool): flag controlling whether the threshold is relative
                    or absolute
                scaling (float): scaling of the signal strength
                verbosity (bool): flag controlling verbosity, default is False

        """
        # Make sure to use pixel (currently the assistant is constructed to work
        # merely with pixel coordinates) FIXME.
        super().__init__(img, use_coordinates=False, **kwargs)

        # Initialize containers
        self.rotation_corrections = []
        self._reset()

        # Determine the center of the image
        self.center = 0.5 * (self.img.origin + self.img.opposite_corner)

        # Set name for titles in plots
        self.name = "Rotation correction assistant"

    def __call__(self) -> list[darsia.RotationCorrection]:
        """Call the assistant."""

        self._reset()
        super().__call__()
        return self.rotation_corrections

    def _reset(self) -> None:
        """Reset anchor and src/dst points."""
        self.ax_id = None
        self.anchor = None
        self.src = []
        self.dst = []
        self.finalized = False
        warn("Resetting subplot selection, anchor, and src/dst points.")

    def _print_current_selection(self) -> None:
        """Print current selection."""
        if self.verbosity:
            print(f"Current selection for subplot {self.ax_id}:")
            print("anchor: {}".format(self.anchor))
            print("src: {}".format(self.src))
            print("dst: {}".format(self.dst))

    def _print_instructions(self) -> None:
        """Print instructions."""

        print(
            """\n------------------------------------------------------------------"""
            """---"""
        )
        print("Welcome to the rotation correction assistant.")
        print("Consider merely one of the three subplots at a time.")
        print("'Right mouse click' to add anchor point.")
        print(
            "'Left mouse click' to add src points. Repeat left click to add dst points."
        )
        print(
            """TIP: Make sure to choose one anchor point, and at least two src and """
            """dst points."""
        )
        print("Press 'enter' to finalize the selection.")
        print("Press 'q' to quit the assistant (possibly before finalizing).")
        print("NOTE: Do not close the figure yourself.")
        print(
            """--------------------------------------------------------------------"""
            """--\n"""
        )

    def _setup_event_handler(self) -> None:
        """Define events."""
        super()._setup_event_handler()
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)

    def _on_mouse_click(self, event):
        """Add points to anchor and src/dst points."""

        # Only continue if no mode is active
        state = self.fig.canvas.toolbar.mode
        if state == "":
            # Determine which subplot has been clicked
            for ax_id in range(3):
                if event.inaxes == self.ax[ax_id]:
                    first_axis = "xxy"[ax_id]
                    second_axis = "yzz"[ax_id]
                    missing_axis = "zyx"[ax_id]
                    break

            first_index = "xyz".find(first_axis)
            second_index = "xyz".find(second_axis)
            missing_index = "xyz".find(missing_axis)

            # Fetch the physical coordinates in 2d plane and interpret 2d point in
            # three dimensions
            x_3d = np.zeros(3)
            x_3d[first_index] = event.xdata
            x_3d[second_index] = event.ydata
            x_3d[missing_index] = self.center[missing_index]

            # Determine action:
            # Left click: src and dst consecutively
            # Right click: anchor

            if (
                event.button in [1, 3]
                and self.ax_id is not None
                and self.ax_id != ax_id
            ):
                warn(
                    """Cannot add points in different subplots - work on one at the """
                    """time and finalize the selection first."""
                )
            else:
                # Fix subplot
                self.ax_id = ax_id

                # Add point
                if event.button == 1:
                    # Left click.
                    if len(self.src) <= len(self.dst):
                        # Identify point as src
                        self.src.append(x_3d)
                    elif len(self.src) > len(self.dst):
                        # Identify point as dst
                        self.dst.append(x_3d)
                        # Draw a line between src and dst
                        self.ax[ax_id].plot(
                            [
                                self.src[-1][first_index],
                                self.dst[-1][first_index],
                            ],
                            [
                                self.src[-1][second_index],
                                self.dst[-1][second_index],
                            ],
                            "r-",
                        )
                        self.fig.canvas.draw()
                    else:
                        warn("Cannot add more src points than dst points.")
                elif event.button == 3:
                    # Right click.

                    if self.anchor is not None:
                        warn("Anchor point already set. Use 'reset' option to update.")

                    # Identify point as anchor
                    self.anchor = x_3d
                    # Draw a circle at the anchor
                    self.ax[ax_id].plot(
                        self.anchor[first_index],
                        self.anchor[second_index],
                        "go",
                        markersize=10,
                    )
                    self.fig.canvas.draw()

            # Print current selection
            self._print_current_selection()

        else:
            warn(
                "Cannot add points while in {} mode. Deactivate it first.".format(state)
            )

    def _finalize(self) -> None:
        """Finalize selection and setup rotation correction."""
        if (
            self.ax_id is None
            or self.anchor is None
            or len(self.src) < 2
            or len(self.dst) < 2
        ):
            warn("Cannot finalize selection - not enough points set.")
            self._print_current_selection()

        else:
            # Convert anchor and src/dst points to voxel coordinates
            anchor = self.img.coordinatesystem.voxel(self.anchor)
            pts_src = self.img.coordinatesystem.voxel(self.src)
            pts_dst = self.img.coordinatesystem.voxel(self.dst)

            # Setup rotation correction
            setup_arguments = {
                "anchor": anchor,
                "rotation_from_isometry": True,
                "pts_src": pts_src,
                "pts_dst": pts_dst,
            }
            rotation_correction = darsia.RotationCorrection(**setup_arguments)

            # Print setup arguments to screen, so that one can also hardcode the
            # definition.
            if self.verbosity:
                print("The determined rotation correction can be setup with the input:")
                print(setup_arguments)

            # Store rotation correction and update image for potential further
            # corrections
            self.rotation_corrections.append(rotation_correction)
            self.img = rotation_correction(self.img)
            self.finalized = True

            # Next round.
            self.__call__()
