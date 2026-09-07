# 1. Getting Started

## What is DarSIA?

[DarSIA](https://github.com/pmgbergen/DarSIA) ("Darcy scale image analysis") is
a Python toolbox for turning photographs of a lab experiment into quantitative
scientific data. It is widely used on **FluidFlower** experiments:
time-lapse photos of a transparent rig filled with sand, where CO₂ is injected
and its migration is tracked from color changes in the images.

Concretely, DarSIA takes a folder of JPEGs plus a small amount of metadata
(the rig's physical dimensions, where the sand layers are, a color calibration)
and turns each image into physical quantities: CO₂ concentration, mass,
saturation. The GUI is a Qt application that wraps this pipeline so you never
have to write a driver script — you fill in a config file and click buttons.

## Installing

DarSIA is not on PyPI yet; install it from a clone (Python 3.12+).

```bash
git clone https://github.com/pmgbergen/darsia.git
cd darsia
uv sync --extra dev
```

The GUI needs the Qt bindings (`PySide6`), which are a normal dependency and
are pulled in by the command above. If GUI imports fail with
`ModuleNotFoundError: PySide6`, re-run `uv sync --extra dev` from the repo
root. `pip install -e ".[dev]"` works too.

## Launching the GUI

```bash
uv run python -m darsia.gui
```

You should see an empty window like this:

![Blank DarSIA window on first launch](images/00-blank-window.png)

## A tour of the main window

- **Sidebar (left)** — six categories: **Setup**, **Calibration**, **Analysis**,
  **Helper**, **Comparison**, **Utils**. Each expands into concrete steps
  (e.g. Setup → Rig, Protocols, Depth, Facies). Clicking a step opens the
  matching settings form on the right and marks it as the "selected workflow."
- **Settings panel (right)** — a form built automatically from your config
  file. Fields are grouped into tabs (`Rig`, `Corrections`, `Color Registry`,
  …) that mirror the TOML sections in [Core Concepts](03-core-concepts.md).
- **Toolbar**, left to right: **New** config, **Open** config, **Save**
  config, **Open Full Config** (shows every section, not just the ones
  relevant to the selected step), **Run** ▶ / **Stop** ■ the selected
  workflow, **Streaming Preview** toggle, **Logging** toggle.
- **File menu** also has `Ctrl+O` (Open Config), `Ctrl+S` (Save Config),
  `Ctrl+N` (New Config), and recently-opened files under *Open Recent*.
- **Run menu**: `Ctrl+Enter` runs the currently selected sidebar step;
  `Ctrl+Esc` stops it.
- **Logging dock** (bottom) shows what the app is doing; keep it open while
  you learn the tool. **Streaming Preview** (right dock, `Ctrl+P`) shows a
  live image of the current analysis step while a workflow runs.

Nothing is loaded yet because no config file has been opened. That is the
subject of the next chapter.

## The five-phase mental model

Every DarSIA workflow project moves through the same five
phases, and the sidebar categories map onto them one-to-one:

1. **Setup** — one-time, per-experiment preparation: rig geometry, depth map,
   sand-layer segmentation ("facies"), imaging/injection protocol files, and
   the geometric (curvature) correction.
2. **Calibration** — turn raw pixel colors into physical quantities: a color
   embedding ("color path") calibrated against known reference images, then a
   color-to-mass calibration for CO₂.
3. **Analysis** — the actual batch processing: cropping, mass/volume/
   concentration maps, segmentation contours, finger detection, thresholding.
4. **Helper** — interactive inspection tools (ROI picker, ROI viewer, color
   histograms, result viewer) that don't produce pipeline output themselves,
   but help you choose good settings for the other phases.
5. **Comparison** — after Analysis has run (possibly for several experiments),
   compare event timings or compute Wasserstein distances between runs.

You do these once, top to bottom, per experiment; Setup and Calibration are
rarely revisited, Analysis is where you iterate.

Continue to the **[Quickstart walkthrough](02-quickstart-walkthrough.md)**,
which runs through all five phases using a real config file.
