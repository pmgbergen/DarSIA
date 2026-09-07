#####################
Writing documentation
#####################

The documentation source lives in ``docs/`` and is built with Sphinx
(``pydata-sphinx-theme``). Install the toolchain with
``uv sync --extra docs`` and build with
``uv run sphinx-build -b html docs docs/_build/html``.

Layout
======

``docs/getting_started/``
   install and first steps
``docs/user_guide/``
   narrative concept pages (reStructuredText)
``docs/user_guide/gui/``
   the GUI tutorial (Markdown, via ``myst-nb``)
``docs/auto_examples/``
   **generated** by Sphinx-Gallery from ``examples/`` -- git-ignored
``docs/api/``
   curated ``autosummary`` pages; ``api/generated/`` is generated and git-ignored
``docs/development/``
   this guide
``docs/_templates/autosummary/``
   class / module page templates

Never edit or commit ``docs/auto_examples/`` or ``docs/api/generated/``.

Adding an API object
====================

Every public object must appear in exactly one curated ``.. autosummary::`` list
under ``docs/api/``. Add the name to the appropriate topic page (e.g.
``docs/api/corrections.rst``); the per-object stub page is generated on build.

Adding a narrative page
=======================

Create ``docs/user_guide/<name>.rst``, add it to the toctree in
``docs/user_guide/index.rst``, and cross-link the relevant ``api/`` and
``auto_examples/`` pages with ``:doc:``.

Adding an example
=================

Examples are ``.py`` files in ``examples/<topic>/``. The
`contributor cheat-sheet
<https://github.com/pmgbergen/darsia/blob/dev/examples/CHEATSHEET.md>`_ in that
directory is the reference: name executed examples ``plot_*.py``, start each with
a reStructuredText docstring, split code with ``# %%``, and reference bundled
data as ``../images/<file>``.

Docstrings
==========

Docstrings feed the API reference. They are being migrated to **NumPy style**;
new and edited docstrings should follow it. Run ``numpydoc lint`` on a file to
check.

Maintaining the workflow docs
=============================

The ``src/darsia/presets/workflows/doc/`` Markdown tree (surfaced in the GUI
guide) has its own conventions in ``DEVELOPER_NOTES.md``:

* ``presets/workflows/doc/README.md`` is the single navigation entry point;
* add new workflow docs as ``workflow-<name>.md`` and link them from
  ``README.md`` / ``overview.md``;
* keep schema detail in ``config-reference.md``; other pages link to it rather
  than duplicate option lists;
* prefer links to code modules over duplicating volatile internals;
* record limitations in ``known-issues.md``.
