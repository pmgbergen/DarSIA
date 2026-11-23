"""Configuration of the image porosity (setup step "Image porosity").

The image porosity marks pixels of the baseline that show the usual (pore space)
colour of their facies (1) and pixels that do not (0: grains, dark spots,
reflections, label boundaries). It is not the physical porosity of the facies
properties. It is computed by the setup step `setup/setup_image_porosity.py` with the
methods of `presets/workflows/processing/image_porosity`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .utils import _get_section_from_toml

POROSITY_MASKS = ("none", "boolean_porosity")
"""Boolean masks: all pixels, or the boolean porosity (image porosity > `tol`)."""

POROSITY_WEIGHTS = ("none", "image_porosity", "boolean_porosity")
"""Weighting masks: none, the image porosity, or the boolean porosity."""


def load_porosity_mask(
    sec: dict,
    default: str,
    context: str,
    options: tuple[str, ...] = POROSITY_MASKS,
) -> str:
    """Read and check the key `mask` of a consumer section."""
    mask = sec.get("mask", default)
    if mask not in options:
        raise ValueError(f"{context} mask must be one of {options}, got {mask!r}")
    return mask


IMAGE_POROSITY_METHODS = ("none", "kmeans")
"""Methods; `"none"` gives ones (no detection), the others have an options sub-table
`[image_porosity.<method>]`."""


@dataclass
class KMeansPorosityConfig:
    """Options of the k-means detection (`[image_porosity.kmeans]`).

    Per patch and label, the colours are clustered (k-means). The dominant colour counts
    as porous (1), the brightest or darkest cluster colour, whichever lies farther from
    it, as not porous (0). Labels whose colours spread little count as homogeneous
    (all porous). Each pixel is interpolated from these colours with a Gaussian kernel.
    """

    patches: tuple[int, int] = field(
        default=(1, 1),
        metadata={
            "name": "Patches",
            "help": (
                "Number of patches (rows, cols), each analysed on its own, "
                "e.g. [10, 10]."
            ),
        },
    )
    """Number of patches ``(rows, cols)``."""
    num_clusters: int = field(
        default=5,
        metadata={
            "name": "Number of clusters",
            "help": "Number of k-means clusters per patch and label.",
        },
    )
    """Number of k-means clusters."""
    tol_color_distance: float = field(
        default=0.1,
        metadata={
            "name": "Tolerance for color distance",
            "help": (
                "A label counts as homogeneous (all porous) if its brightest and "
                "darkest cluster colours lie closer than this to the dominant one."
            ),
        },
    )
    """Colour distance below which a label counts as homogeneous."""
    tol_color_gradient: float = field(
        default=0.02,
        metadata={
            "name": "Tolerance for color gradient",
            "help": (
                "A label counts as homogeneous (all porous) if its mean colour "
                "gradient per pixel is below this value."
            ),
        },
    )
    """Mean colour gradient below which a label counts as homogeneous."""

    def load(self, sec: dict) -> "KMeansPorosityConfig":
        """Populate from the `[image_porosity.kmeans]` table."""
        patches = sec.get("patches", list(self.patches))
        if len(patches) != 2:
            raise ValueError(
                "[image_porosity.kmeans] patches must be a list of 2 integers, "
                f"got {patches!r}"
            )
        self.patches = (int(patches[0]), int(patches[1]))
        self.num_clusters = int(sec.get("num_clusters", self.num_clusters))
        self.tol_color_distance = float(
            sec.get("tol_color_distance", self.tol_color_distance)
        )
        self.tol_color_gradient = float(
            sec.get("tol_color_gradient", self.tol_color_gradient)
        )
        return self


_REPLACED_KEYS = {
    "mode",
    "patches",
    "num_clusters",
    "sample_width",
    "tol_color_distance",
    "tol_color_gradient",
}
"""Keys of the former flat section; using one is an error with a hint."""


@dataclass
class ImagePorosityConfig:
    """Configuration of the image porosity.

    Config section (TOML)::

        [image_porosity]
        method = "kmeans"      # "none" (default): ones, no detection
        tol = 0.9              # boolean porosity: image porosity > tol
        cutoff = 0.3           # values below become 0
        boundary_band = 5      # pixels around label boundaries set to 0

        [image_porosity.kmeans]
        patches = [10, 10]

    Only how the image porosity is computed; whether it is used is decided by each
    consumer. A missing section means the defaults.
    """

    method: Literal["none", "kmeans"] = field(
        default="none",
        metadata={
            "name": "Method",
            "help": (
                "Detection method. 'none': no detection, the image porosity is 1 "
                "except for the boundary band. Other methods have their options in "
                "their own group below."
            ),
            "options": list(IMAGE_POROSITY_METHODS),
        },
    )
    """Detection method, one of `IMAGE_POROSITY_METHODS`."""
    tol: float = field(
        default=0.9,
        metadata={
            "name": "Boolean threshold",
            "help": (
                "The boolean porosity (e.g. the mask of the colour calibration, if it "
                "uses the image porosity) is the image porosity above this value. "
                "Applied when the porosity is read, so no new setup is needed after "
                "a change."
            ),
        },
    )
    """Threshold of the boolean porosity."""
    cutoff: float = field(
        default=0.3,
        metadata={
            "name": "Cutoff",
            "help": "Image porosity values below this become 0.",
        },
    )
    """Values below this become 0."""
    boundary_band: int = field(
        default=5,
        metadata={
            "name": "Boundary band (pixels)",
            "help": (
                "Set the image porosity to 0 within this many pixels of every label "
                "boundary, on both sides: pixels there mix the colours of two facies. "
                "0 switches the band off. Applies with every method."
            ),
        },
    )
    """Radius in pixels of the zeroed band around label boundaries; 0: none."""
    kmeans: KMeansPorosityConfig = field(
        default_factory=KMeansPorosityConfig,
        metadata={
            "name": "k-means method",
            "depends_on": {"field": "method", "value": "kmeans"},
        },
    )
    """Options of the k-means method."""

    def load(self, path: Path | list[Path]) -> "ImagePorosityConfig":
        """Populate from a TOML config file (section `[image_porosity]`).

        Raises
        ------
        KeyError
            If the ``[image_porosity]`` section is absent.

        """
        sec = _get_section_from_toml(path, "image_porosity")
        return self._load_dict(sec)

    def _load_dict(self, sec: dict) -> "ImagePorosityConfig":
        """Populate from a plain dictionary (the parsed section).

        Raises
        ------
        ValueError
            For keys of the former section, an unknown method, `tol` outside
            (0, 1], or a negative boundary band.

        """
        replaced = sorted(_REPLACED_KEYS & set(sec))
        if replaced:
            raise ValueError(
                f"[image_porosity] keys {replaced} are no longer supported: use "
                'method = "none" | "kmeans" (no detection is method = "none"), the '
                "method options in [image_porosity.kmeans] (sample_width was "
                "dropped), and the mask options of the consumers."
            )

        method = sec.get("method", self.method)
        if method not in IMAGE_POROSITY_METHODS:
            raise ValueError(
                f"[image_porosity] method must be one of {IMAGE_POROSITY_METHODS}, "
                f"got {method!r}"
            )
        self.method = method

        tol = float(sec.get("tol", self.tol))
        if not (0.0 < tol <= 1.0):
            raise ValueError(f"[image_porosity] tol must be in (0, 1], got {tol!r}")
        self.tol = tol

        self.cutoff = float(sec.get("cutoff", self.cutoff))

        boundary_band = int(sec.get("boundary_band", self.boundary_band))
        if boundary_band < 0:
            raise ValueError(
                f"[image_porosity] boundary_band must be >= 0, got {boundary_band!r}"
            )
        self.boundary_band = boundary_band

        self.kmeans = KMeansPorosityConfig().load(sec.get("kmeans", {}))
        return self

    def method_options(self) -> KMeansPorosityConfig | None:
        """Options of the selected method; None for `"none"`."""
        return getattr(self, self.method, None) if self.method != "none" else None
