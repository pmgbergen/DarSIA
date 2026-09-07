############
Segmentation
############

.. currentmodule:: darsia

:func:`segment` partitions an image into labelled regions -- typically the sand
layers ("facies") of a multi-layered rig. It runs a watershed on a gradient or
monochromatic signal; you supply markers and edge parameters through a config
dict.

.. code-block:: python

   import darsia

   image = darsia.imread("baseline.jpg", dim=2)

   config = {
       "median disk radius": 20,
       "rescaling factor": 0.3,
       "markers disk radius": 10,
       "threshold": 20,
       "gradient disk radius": 2,
   }
   labels = darsia.segment(
       image,
       markers_method="gradient_based",   # or "supervised" with marker_points
       edges_method="gradient_based",     # or "scharr"
       **config,
   )
   labels.show()

Working with the label field
============================

* :func:`label_image`, :func:`make_consecutive`, :func:`reassign_labels`,
  :func:`group_labels` -- manipulate the integer label array.
* :class:`Masks`, :func:`roi_to_mask` -- build boolean masks from labels or ROIs.
* :class:`LabelsAssistant` and friends (:doc:`assistants`) -- edit labels
  interactively (merge, split, pick).

The workflow presets turn a hand-segmented reference image into named facies via
``[labeling]`` and ``[facies]`` -- see :doc:`gui/04-configuration-reference`.

See also
========

* :doc:`../auto_examples/segmentation/plot_segmentation`
* :doc:`../api/segmentation`
