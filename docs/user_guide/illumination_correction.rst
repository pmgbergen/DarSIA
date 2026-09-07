#######################
Illumination correction
#######################

.. currentmodule:: darsia

Lab lighting is rarely uniform across a metre-wide rig, and it drifts over a
multi-hour experiment. Illumination corrections flatten this so that a colour
change means a concentration change, not a lamp getting warmer.

============================================  ==================================================
:class:`IlluminationCorrection`               global illumination model from reference images
:class:`PatchwiseIlluminationCorrection`      per-patch correction on an ``nw`` x ``nw`` grid;
                                              handles spatially varying lighting
:class:`DynamicIlluminationCorrection`        re-estimates illumination per frame for drift
============================================  ==================================================

Like the other corrections, an illumination correction is constructed from one or
more baseline images plus a small config dict, then passed to :func:`imread` via
``transformations=``. See the class reference for the exact arguments, and
:func:`illumination_interpolation` for the underlying interpolation.

Choosing the patch resolution and which facies/labels drive the fit has a large
effect on quality -- see :doc:`gui/06-troubleshooting-faq`.

See also
========

* :doc:`color_correction`
* :doc:`../api/corrections`
