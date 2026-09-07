################
Shape correction
################

.. currentmodule:: darsia

Shape corrections change the *geometry* of the pixel container -- dewarping lens
curvature, straightening perspective, aligning frames that have shifted. They
derive from :class:`BaseCorrection` and are usually passed to :func:`imread` via
``transformations=``.

============================================  ==================================================
:class:`CurvatureCorrection`                  crop and dewarp to the four rig corners; the
                                              workhorse for wide-angle FluidFlower photos
:class:`AffineCorrection`                     affine map from matched point pairs
:class:`GeneralizedPerspectiveCorrection`     perspective / homography, incl. piecewise
:class:`DriftCorrection`                      compensate small frame-to-frame camera drift
                                              using a fixed reference patch
:class:`DeformationCorrection`                apply a known displacement field
:class:`TranslationCorrection`                pure translation; :class:`TranslationEstimator`
                                              finds it from image features
:class:`RotationCorrection`                   rigid rotation
:class:`ResizeCorrection`                     rescale (also available as
                                              :func:`resize` / :class:`Resize`)
============================================  ==================================================

.. code-block:: python

   import json, darsia

   cfg = json.load(open("config.json"))
   curvature = darsia.CurvatureCorrection(config=cfg["curvature"])

   photo = darsia.imread("photo.jpg", transformations=[curvature],
                         width=2.8, height=1.5)

The stand-alone transforms (:class:`AffineTransformation`,
:class:`GeneralizedPerspectiveTransformation`,
:class:`PiecewisePerspectiveTransform`) can also be used directly on arrays of
points.

.. note::

   Shape corrections change pixels *and* keep the physical metadata consistent.
   To align two images by editing only their metadata, use
   :class:`CoordinateTransformation` (see :doc:`coordinate_systems`).

See also
========

* :doc:`../auto_examples/corrections/plot_optical_images`
* :doc:`registration` -- recovering an *unknown* deformation between images
* :doc:`../api/corrections`
