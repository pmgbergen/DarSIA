Shape and color corrections (:mod:`darsia.corrections`)
=======================================================

.. automodule:: darsia.corrections
   :members:
   :undoc-members:
   :show-inheritance:

The base correction
-------------------

All corrections follow the design of a base correction object.

.. currentmodule:: darsia.corrections.basecorrection

.. autosummary::
   
   BaseCorrection
  
Color corrections with known location of a color checker
--------------------------------------------------------
General color correction requires the presence of a color checker allowing to match test colors with reference colors. The location of color checkers in the images can be provided explicitly or only approximately. Color corrections aim at calibrating colors identified in color checkers with reference colors, either defined by standards or customly by the user. Provided the exact location of a color checker, the colors in the entire image can be corrected using the following objects in the :mod:`darsia.corrections.color.colorcorrection`:

.. currentmodule:: darsia.corrections.color.colorcorrection

.. rubric:: Classes

.. autosummary::

   ColorCorrection
   ColorCheckerAfter2014
   CustomColorChecker

Color corrections with approximate/unknown location of a color checker
----------------------------------------------------------------------
Knowing the position of a color checker only approximately (or not at all), machine learning based routines may identify the precise location. In addition, reference colors for calibrated illumination conditions can be used to fully automatically correct for color impurities. This features are not stable for high-resolution images and should be used with care. The central objects in the :mod:`darsia.corrections.color.experimentalcolorcorrection` are:

.. currentmodule:: darsia.corrections.color.experimentalcolorcorrection

.. rubric:: Classes

.. autosummary::
   ExperimentalColorCorrection
   ClassicColorChecker
   EOTF


Shape corrections
-----------------

Shape correction alter the shape of the data container of the image. Several corrections are provided in the :mod:`darsia.corrections.shape` subpackage.

.. currentmodule:: darsia.corrections.shape

.. rubric:: Classes

.. autosummary::

   affine.AffineCorrection
   curvature.CurvatureCorrection
   deformation.DeformationCorrection
   drift.DriftCorrection
   rotation.RotationCorrection
   translation.TranslationCorrection

Some of these employ tailored transformations, which may also be used as stand-alone.

.. currentmodule:: darsia.corrections.shape

.. rubric:: Classes

.. autosummary::

   affine.AffineTransformation
   piecewiseperspective.PiecewisePerspectiveTransform

To detect suitable translations (used in several contexts including image registration, when restricted to patches), feature-based translation estimators are essential:

.. currentmodule:: darsia.corrections.shape

.. rubric:: Classes

.. autosummary::

   translation.TranslationEstimator
