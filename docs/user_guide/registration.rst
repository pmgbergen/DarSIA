##################
Image registration
##################

.. currentmodule:: darsia

Registration recovers an *unknown* displacement between two images of the same
scene -- sand settling between a well test and a later baseline, compaction over
an injection, a rig nudged by the lab bench. Contrast this with
:doc:`shape_correction`, where the transform is known in advance.

* :class:`ImageRegistration` -- patch-wise registration; returns a displacement
  field that can be evaluated anywhere or applied to another image.
* :class:`DiffeomorphicImageRegistration` -- smooth, invertible deformation;
  :class:`MultiscaleDiffeomorphicImageRegistration` adds a coarse-to-fine sweep
  for large motion.
* :class:`TranslationAnalysis` -- per-patch pure translation, a lightweight
  special case.
* :class:`SegmentationComparison` -- compare two label fields region by region.

Typical use: align a source image to a destination, inspect the deformation on a
grid of :class:`Patches`, then apply the same field to a derived quantity.

See also
========

* :doc:`../auto_examples/registration/image_registration`
* :doc:`distances` -- a transport-based alternative for comparing distributions
* :doc:`../api/analysis`
