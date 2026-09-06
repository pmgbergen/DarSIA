# DarSIA examples

Usage examples, organised by topic. Every `.py` here is also a page in the
online documentation ("Examples"), rendered by
[Sphinx-Gallery](https://sphinx-gallery.github.io/) from `docs/conf.py`.

This file is a contributor cheat-sheet; it is **not** part of the published
docs (the gallery reads `GALLERY_HEADER.rst`, not this file).

## Layout

| Directory | Topic |
|---|---|
| `io/` | reading optical, NumPy, DICOM and simulation (vtu) data |
| `corrections/` | colour and curvature / perspective correction |
| `restoration/` | total-variation and H1 regularization |
| `segmentation/` | watershed segmentation of layered media |
| `registration/` | deformation / compaction analysis |
| `analysis/` | tracer and CO2 concentration analysis, model calibration |
| `distances/` | Earth Mover's / Wasserstein distances |
| `paper/` | scripts reproducing figures from publications |
| `notebooks/` | introductory Jupyter notebooks (not part of the gallery) |
| `images/` | data files shared by the examples |

`docs/auto_examples/` is the generated gallery output. It is git-ignored and
rebuilt from these sources; never edit or commit it.

## The `plot_` convention

* **`plot_<name>.py`** &ndash; executed during the docs build; every figure it
  creates (including `image.show()`) is captured and shown on the page, and the
  script becomes a downloadable notebook. A `plot_*` script that raises fails the
  build.
* **`<name>.py`** (no prefix) &ndash; shown as source only, not executed. Use
  this when the example needs data that is not in the repository, or is too slow
  for CI.

## Anatomy of an example

```python
"""
Short page title
================

One or two sentences of introduction (reStructuredText).
"""

import darsia

# %%
# A comment block right after a "# %%" separator renders as prose between code
# cells, notebook-style.

image = darsia.imread("../images/baseline.jpg", width=2.8, height=1.5)
image.show("baseline")
```

* The module docstring is **required**; its first line is the page title.
* `# %%` on its own line starts a new cell. `#`-comment lines immediately after
  it become narrative text.
* Sphinx-Gallery runs each script with the working directory set to the script's
  own folder, so reference shared data as `../images/<file>` (not `__file__`).

## Adding an example

1. Put the file in the right `examples/<topic>/` folder, named `plot_*.py` if it
   should run.
2. Give it a docstring (see above). Nothing else &ndash; it is picked up
   automatically, ordered by filename within the topic.
3. New data files go in `examples/images/`. The repo `.gitignore` excludes
   `*.jpg`, `*.png`, `*.npy` etc., so add them explicitly:
   `git add -f examples/images/<file>`.

## Adding a topic (new subdirectory)

1. `mkdir examples/<topic>/`
2. Add `examples/<topic>/GALLERY_HEADER.rst` (a title + a sentence; copy an
   existing one).
3. Add `"../examples/<topic>"` to `subsection_order` in the `sphinx_gallery_conf`
   in `docs/conf.py`, at the position you want in the navigation.

## Rebuilding

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Only changed scripts re-execute (tracked via `.py.md5` hashes). To force a full
re-run, delete `docs/auto_examples/`.

## Relationship to the test suite

The gallery build already executes every `plot_*.py`. `tests/integration/
test_examples.py` additionally runs a selected set under `pytest` (each from its
own directory); add a `test_*` entry there if you want a new example &ndash;
especially a source-only one &ndash; covered by CI test runs as well.

## Recommended reading order

See `first_steps.md`.
