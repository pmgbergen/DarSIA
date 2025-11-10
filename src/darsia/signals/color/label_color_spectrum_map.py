"""Module for managing a mapping between integer labels and their corresponding color spectrum."""

import logging
import darsia

logger = logging.getLogger(__name__)


class LabelColorSpectrumMap(dict[int, darsia.ColorSpectrum]):
    """Mapping between integer labels and their corresponding color spectra."""

    def __init__(self, color_spectra: dict[int, darsia.ColorSpectrum] | None = None):
        """Initialize the LabelColorSpectrumMap.

        Args:
            color_spectra (dict[int, darsia.ColorSpectrum], optional): Initial mapping of
                labels to color spectra.

        """
        super().__init__(color_spectra or {})

    def __repr__(self) -> str:
        str_repr = ""
        for label, color_spectrum in self.items():
            str_repr += f"Label {label}: {color_spectrum}\n"
        return str_repr

    def __str__(self) -> str:
        str_str = ""
        for label, color_spectrum in self.items():
            str_str += f"Label {label}: {str(color_spectrum)}\n"
        return str_str

    def save(self, directory: darsia.Path) -> None:
        """Save color spectra to a directory.

        Stores each color spectrum in a separate file named `color_spectrum_{label}.json`.

        Args:
            directory (darsia.Path): The directory to save the color spectrum files.

        """
        directory.mkdir(parents=True, exist_ok=True)
        for label, color_spectrum in self.items():
            color_spectrum.save(directory / f"color_spectrum_{label}.json")

    @classmethod
    def load(cls, directory: darsia.Path) -> "LabelColorSpectrumMap":
        """Load color spectra from a directory.

        Identifies files named `color_spectrum_{label}.json` and loads them.

        Args:
            directory (darsia.Path): The directory containing color spectrum files.

        """
        color_spectra = {}
        for file in directory.glob("color_spectrum_*.json"):
            label = int(file.stem.split("_")[-1])
            color_spectrum = darsia.ColorSpectrum.load(file)
            color_spectra[label] = color_spectrum
        return cls(color_spectra)
