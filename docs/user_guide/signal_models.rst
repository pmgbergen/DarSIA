#############
Signal models
#############

.. currentmodule:: darsia

A concentration analysis (:doc:`concentration_analysis`) has two swappable
mathematical pieces: a **signal reduction** that turns an image difference into a
scalar, and a **model** that turns that scalar into a physical quantity.

Signal reduction
================

* :class:`MonochromaticReduction` -- take one colour channel (``"red"``,
  ``"hue"``, ...).
* :class:`SignalReduction` -- general projection / dimension reduction.
* :class:`AxisReduction`, :func:`reduce_axis`, :func:`extrude_along_axis` --
  reduce or extend a spatial axis.

Models
======

All models share the :class:`Model` interface (``__call__`` maps signal to
quantity) and compose with :class:`CombinedModel`.

============================================  ==================================================
:class:`LinearModel`, :class:`ScalingModel`   linear / affine maps
:class:`ClipModel`                            clamp to a range
:class:`StaticThresholdModel`,                fixed / data-driven thresholds
:class:`DynamicThresholdModel`
:class:`StandardOtsu`,                        histogram-based thresholding
:class:`TwoPeakHistogrammAnalysis` (and kin)
:class:`KernelInterpolation`                  non-linear map learned from labelled
                                              colour samples; handles curved colour paths
:class:`ColorPathInterpolation`               interpolation along a colour path
:class:`PWTransformation`                     piecewise transformation
:class:`HeterogeneousModel`,                  apply a different model per region / facies
:class:`HeterogeneousLinearModel`
============================================  ==================================================

:class:`BinaryDataSelector` plus the criterion classes
(:class:`ValueCriterion`, :class:`GradientModulusCriterion`,
:class:`CombinedCriterion`, ...) build the masks that decide *where* a model is
applied.

See also
========

* :doc:`../auto_examples/analysis/plot_kernel_interpolation`
* :doc:`../api/signal_models`, :doc:`../api/color`
