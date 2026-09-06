############
Segmentation
############

.. currentmodule:: darsia

:func:`segment` partitions an image into labelled regions (for example the sand
layers of a FluidFlower rig) using a watershed on a gradient or monochromatic
signal. The remaining helpers manage label arrays produced by segmentation.

.. autosummary::
   :toctree: generated/

   segment
   label_image
   group_labels
   reassign_labels
   make_consecutive

Masks
=====

.. autosummary::
   :toctree: generated/

   Masks
   roi_to_mask
