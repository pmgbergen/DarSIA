###########
Restoration
###########

.. currentmodule:: darsia

Restoration algorithms clean up a field -- removing noise, resampling,
smoothing -- without changing what it physically represents.

Denoising
=========

* :class:`TVD` / :func:`tvd` -- total-variation denoising; preserves sharp
  interfaces (plume edges) while removing speckle. :func:`split_bregman_tvd`
  and the ``"heterogeneous bregman"`` method allow a spatially varying weight.
* :func:`H1_regularization` -- smoother, cheaper H1 (Tikhonov) regularization.
* :class:`Median` -- median filtering.

.. code-block:: python

   import darsia

   clean = darsia.tvd(img=noisy, method="isotropic Bregman",
                      weight=0.03, max_num_iter=100)

Resampling and averaging
========================

* :class:`Resize` / :func:`resize`, :func:`equalize_voxel_size`,
  :func:`uniform_refinement` -- change resolution.
* :class:`VolumeAveraging`, :class:`REV`, :func:`volume_average`,
  :func:`porosity_based_averaging` -- upscale a fine field to representative
  elementary volumes.

Binary clean-up
===============

:class:`BinaryFillHoles`, :class:`BinaryRemoveSmallObjects` and
:class:`BinaryLocalConvexCover` tidy the masks produced by segmentation and
thresholding.

See also
========

* :doc:`../auto_examples/restoration/regularization`
* :doc:`../api/restoration`
