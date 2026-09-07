#################
Colour correction
#################

.. currentmodule:: darsia

Cameras and lighting distort recorded colour. Because DarSIA turns colour
*changes* into physical quantities, calibrating colour against a known reference
is usually the first correction in a pipeline.

Colour-checker correction
=========================

:class:`ColorCorrection` matches the swatches of a Classic X-Rite colour checker
visible in the image to their reference values and applies the resulting
transform to the whole frame.

.. code-block:: python

   import darsia

   base = darsia.imread("baseline.jpg", width=2.8, height=1.5)

   # give the pixel coordinates of the four white L-marks on the chart
   cfg = {"roi": darsia.make_voxel([[154, 176], [222, 176], [222, 68], [154, 68]])}
   cc = darsia.ColorCorrection(base=base, config=cfg)
   corrected = darsia.imread("test.jpg", transformations=[cc], width=2.8, height=1.5)

* :class:`ColorChecker`, :class:`ColorCheckerAfter2014`,
  :class:`ClassicColorChecker`, :class:`CustomColorChecker` -- reference charts.
* :class:`ExperimentalColorCorrection` -- locates the chart automatically with a
  machine-learning detector; convenient, less robust on high-resolution images.
* :class:`RelativeColorCorrection` -- calibrate one image *relative* to another
  rather than to an absolute standard.

Colour balance
==============

When there is no chart, :class:`WhiteBalance`, :class:`ColorBalance`,
:class:`AffineBalance` and :class:`AdaptiveBalance` (and the function forms
:func:`white_balance`, :func:`color_balance`, :func:`affine_balance`) correct a
global cast from a neutral reference region.

See also
========

* :doc:`illumination_correction` -- spatially varying lighting
* :doc:`../auto_examples/corrections/plot_color_correction`
* :doc:`../api/corrections`
