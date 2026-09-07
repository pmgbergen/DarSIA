######################
Concentration analysis
######################

.. currentmodule:: darsia

The core scientific task: given a baseline image and a sequence of later images,
turn the colour changes into a physical field -- a tracer concentration, a CO2
saturation, a mass map.

The pipeline
============

:class:`ConcentrationAnalysis` chains four stages, each pluggable:

#. **baseline difference** -- subtract the pre-injection image;
#. **signal reduction** -- collapse the (possibly multichannel) difference to a
   scalar (:class:`MonochromaticReduction`, :class:`SignalReduction`; see
   :doc:`signal_models`);
#. **restoration** -- denoise the scalar signal (:class:`TVD`; see
   :doc:`restoration`);
#. **model** -- map the signal to a physical quantity
   (:class:`LinearModel`, :class:`KernelInterpolation`,
   :class:`CombinedModel`, ...).

.. code-block:: python

   import darsia

   analysis = darsia.ConcentrationAnalysis(
       base=baseline,
       signal_reduction=darsia.MonochromaticReduction(color="red"),
       restoration=darsia.TVD(),
       model=darsia.CombinedModel([
           darsia.LinearModel(scaling=4.0),
           darsia.ClipModel(min_value=0.0, max_value=1.0),
       ]),
   )
   concentration = analysis(test_image)

Calibration
===========

The model's parameters are rarely known a priori. DarSIA fits them against
images with known concentrations:

* :class:`KernelInterpolation` learns a non-linear colour-to-concentration map
  from labelled samples (:doc:`../auto_examples/analysis/plot_kernel_interpolation`);
* the calibration mixins (:class:`AbstractModelObjective`,
  :class:`InjectionRateModelObjectiveMixin`, ...) fit a model so that integrated
  mass matches a known injection rate;
* :class:`PriorPosteriorConcentrationAnalysis` adds a prior/posterior correction
  step.

Managers
========

:class:`TracerAnalysis` and :class:`CO2Analysis` (both :class:`AnalysisBase`)
wire a full baseline-plus-config pipeline together for the common experiment
types; subclass and override ``define_*_analysis`` to customise.

See also
========

* :doc:`../auto_examples/analysis/plot_co2_analysis`
* :doc:`signal_models`, :doc:`distances`
* :doc:`../api/analysis`
