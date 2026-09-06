#############
API reference
#############

.. note::

   A curated, topic-grouped API reference is being assembled. Until it lands,
   every public object is still importable directly from the top-level
   ``darsia`` namespace, e.g. ``darsia.Image``, ``darsia.imread``,
   ``darsia.ColorCorrection``, ``darsia.wasserstein_distance``.

Planned groups
==============

* **Image containers & coordinates** -- ``Image``, ``ScalarImage``,
  ``OpticalImage``, ``CoordinateSystem``, ``CoordinateTransformation``,
  ``Patches``.
* **Input / output** -- ``imread`` and the ``imread_from_*`` readers.
* **Corrections** -- colour, curvature, affine / generalized perspective, drift,
  deformation, translation and illumination corrections.
* **Restoration** -- ``TVD``, ``H1Regularization``, median filtering.
* **Segmentation** -- ``segment``.
* **Registration & multi-image analysis** -- ``ConcentrationAnalysis``,
  ``TranslationAnalysis``, ``ImageRegistration``.
* **Single-image analysis** -- contour, skeleton and path-evolution analysis.
* **Distances** -- ``wasserstein_distance``, ``EMD``, the Beckmann solver family.
* **Signal models** -- ``Model``, ``LinearModel``, ``StaticThresholdModel``,
  ``KernelInterpolation``, ``SignalReduction``, ``MonochromaticReduction``.
* **Managers** -- ``AnalysisBase``, ``TracerAnalysis``, ``CO2Analysis``.
* **Interactive assistants** -- ROI, crop and rotation helpers.
* **Presets** -- ``FluidFlowerCO2Analysis``, ``SimpleFluidFlower``.
* **Utilities** -- grids, finite volumes, derivatives, kernels, linear solvers,
  plotting helpers.
