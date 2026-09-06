###############
Colour analysis
###############

.. currentmodule:: darsia

Tools for describing colour in a way that supports concentration detection:
paths through colour space traced by a dye as its concentration changes, ranges
and spectra around reference colours, and maps associating labels with colours.

Colour paths, ranges and spectra
================================

.. autosummary::
   :toctree: generated/

   ColorPath
   define_color_path
   ColorRange
   DiscreteColorRange
   ColorSpectrum
   ColorMode
   get_mean_color

Label-to-colour maps
====================

.. autosummary::
   :toctree: generated/

   LabelColorMap
   LabelColorPathMap
   LabelColorSpectrumMap
   LabelColorPathMapRegression
