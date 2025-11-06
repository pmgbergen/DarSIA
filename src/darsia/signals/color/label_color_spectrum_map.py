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
