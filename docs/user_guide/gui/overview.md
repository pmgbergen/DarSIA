# DarSIA GUI Guide

A from-scratch, beginner-friendly tutorial for the DarSIA GUI, for people
picking up DarSIA for the first time. Every screenshot here is a real capture of
the app (not a mockup), and every TOML snippet is taken from `sample_config.toml`
(next to this guide), the running example throughout.

:::{note}
`sample_config.toml` is a complete, realistic FluidFlower configuration; only the
file-system paths (`/path/to/experiment/...`) are placeholders. The screenshots
were taken with a real run of it. To use it on your own experiment, edit the
paths in `[data]`, `[protocols]`, `[labeling]`, `[facies]` and `[depth]`, and
re-run Setup and Calibration from scratch (the crop corners and calibration
values are specific to one rig, camera and lighting).
:::

## Who this is for

You know Python and are comfortable editing a text file, but you have never
opened DarSIA before. You want to go from "installed DarSIA" to "produced a mass
map from a FluidFlower image" without reading the source code first.

## How the guide is organized

1. **[Getting started](01-getting-started.md)** — install, launch the GUI, tour
   the main window, and the five-phase mental model (Setup → Calibration →
   Analysis → Helper → Comparison).
2. **[Quickstart walkthrough](02-quickstart-walkthrough.md)** — the core
   tutorial. Load `sample_config.toml` and walk through every tab with
   screenshots and the matching config snippet.
3. **[Core concepts](03-core-concepts.md)** — the ideas that make DarSIA
   configs click: config layering, data selectors, the ROI registry, color
   embeddings, and the format registry.
4. **[Configuration reference](04-configuration-reference.md)** — an annotated,
   field-by-field reference, grounded in `sample_config.toml`.
5. **[Advanced analysis](05-advanced-analysis.md)** — segmentation styling,
   thresholding, finger detection, expert-knowledge ROI constraints, and the
   auto/manual color calibration modes.
6. **[Troubleshooting & FAQ](06-troubleshooting-faq.md)** — known issues and
   the questions beginners hit first.

## Prerequisites

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- DarSIA installed (see [Getting started](01-getting-started.md))
- A FluidFlower image set — the walkthrough uses `sample_config.toml`, but the
  same steps apply to your own data once you swap the paths in `[data]`

Start with [Getting started](01-getting-started.md).
