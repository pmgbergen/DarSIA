# 2. Quickstart Walkthrough

This chapter walks the whole pipeline once, top to bottom, using
`sample_config.toml` — a real,
complete config for a FluidFlower experiment. The screenshots below are
placeholders for now (see the warning on the guide's overview page).

## Step 0 — Open the config

`File > Open Config…` (`Ctrl+O`) and pick `sample_config.toml`. Once
loaded, the sidebar's completion dots update to reflect what has already been
computed on disk for this run, and every field in the panels below is filled
in from the file — nothing here is placeholder text.

> A config file is just TOML. You are always free to hand-edit it in a text
> editor instead of using the GUI forms — the GUI is a convenience layer over
> the same file. See [Configuration reference](04-configuration-reference.md).

## Step 1 — Setup

Sidebar → **Setup**. This phase prepares experiment metadata that the rest of
the pipeline depends on.

**Rig** (`[rig]`) — the physical geometry of the FluidFlower cell:

```toml
[rig]
width = 0.92
height = 0.55
dim = "2"
```

![Setup → Rig tab, width/height/dim fields](images/placeholder.png)

**Protocols** (`[protocols]`) — because a protocol file is a per-folder
mapping, selecting *Protocols* also surfaces the underlying **Data** section
(where the images actually live):

```toml
[data]
folders = ["/path/to/experiment/images"]
format = "JPG"
baseline = "/path/to/experiment/images/baseline.JPG"
results = "/path/to/experiment/results"
use_cache = false

[protocols]
imaging_mode = "exif"
injection = "/path/to/experiment/protocols/injection_protocol.csv"
pressure_temperature = "/path/to/experiment/protocols/pressure_temperature_protocol.csv"

[protocols.imaging]
"/path/to/experiment/images" = "/path/to/experiment/protocols/imaging_protocol.csv"
```

![Setup → Protocols surfaces the Data section: folders, format, baseline image, results folder](images/placeholder.png)

Other Setup steps not screenshotted here but present in `sample_config.toml`:

- **Facies** (`[facies]`, `[facies.0..2]`) — maps segmented sand-layer labels
  to named geological layers, used later by color-path calibration.
- **Depth** (`[depth]`) — depth measurements CSV and target grid resolution
  for building the depth map.
- **Corrections** (`[corrections]`) — the image-correction pipeline (type
  cast, drift, curvature/crop, patchwise illumination, resize); see
  [Configuration reference](04-configuration-reference.md#corrections).

To actually run a step, select it in the sidebar and press **▶ Run** (or
`Ctrl+Enter`). Progress and any errors stream into the **Logging** dock.

Equivalent CLI, if you prefer scripting the same step:

```bash
uv run python -m darsia.presets.workflows.user_interface_setup \
  --all --config sample_config.toml
```

## Step 2 — Calibration

Sidebar → **Calibration**. Two things get calibrated: the **color embedding**
(how pixel color maps to a scalar signal) and **mass** (how that scalar maps
to CO₂ mass).

**Color Path** (`[color_path]`, referenced from `[calibration.color]`):

```toml
[[color_path]]
name = "color_path"
basis = "facies"
reference_label = 2

[calibration.color]
embedding = "color_path"
data_selection = ["calibration"]
baseline = ["baseline"]
rois = ["calibration"]

[calibration.color.color_path]
num_segments = 3
resolution = 21
calibration_mode = "auto"
threshold_baseline = 0.0
threshold_calibration = 1e-6
ignore_baseline_spectrum = "none"
histogram_weighting = "wls"
ignore_labels = [0]
```

![Calibration → Color Path settings](images/placeholder.png)

**Mass** (`[calibration.mass]`) — selecting *Mass* pulls in the same color
embedding plus the mass-specific fields further down the form:

```toml
[calibration.mass]
mode = "manual"
fluid = "co2"
threshold = 0.2
data_selection = ["all"]
rois = ["calibration"]
embedding = "color_path"
```

![Calibration → Mass settings, with the referenced color embedding above it](images/placeholder.png)

`calibration_mode = "auto"` fits the color-path automatically from the
calibration images; `mode = "manual"` for mass calibration means the
threshold above was set by hand rather than fitted. See
[Advanced analysis](05-advanced-analysis.md#calibration-modes) for when to
choose which.

CLI equivalent:

```bash
uv run python -m darsia.presets.workflows.user_interface_calibration \
  --color-embedding --config sample_config.toml
uv run python -m darsia.presets.workflows.user_interface_calibration \
  --mass --config sample_config.toml
```

## Step 3 — Analysis

Sidebar → **Analysis**. This is where images actually get processed in bulk.
`sample_config.toml` enables mass, segmentation, and fingers.

**Mass** (`[analysis.mass]`):

```toml
[analysis.mass]
data_selection = ["calibration"]
formats = ["jpg"]
export = [
  "concentration_aq", "mass", "rescaled_concentration_aq",
  "rescaled_saturation_g", "saturation_g",
]
roi = ["full"]
embedding = "color_path"
color = "color_path"
active = ["contour_smoother_selection"]
```

![Analysis → Mass settings: export fields, color embedding, data selection, formats, ROI](images/placeholder.png)

Each entry in `export` becomes its own output folder under
`<results>/<mode>/<format>/`, e.g. `results/mass/jpg/`.

**Segmentation** (`[analysis.segmentation]`) — draws contour layers on top of
a scalar field. `sample_config.toml` defines two: `aqueous` (CO₂ dissolved in
brine) and `gas` (free-phase CO₂):

```toml
[[analysis.segmentation.config]]
name = "aqueous"
label = "CO2(aq)"
mode = "concentration_aq"
thresholds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
color = [120, 210, 255]
alpha = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
linewidth = 16
```

![Analysis → Segmentation, aqueous layer config](images/placeholder.png)

**Fingers** (`[analysis.fingers]`) — detects viscous-fingering features:

```toml
[[analysis.fingers.config]]
name = "primary"
mode = "mass"
roi = ["calibration"]
threshold = "0.5"
reduce_to_main_contour = true
fill_holes = true
gradient_mode = "saturation_g"
```

![Analysis → Fingers configuration](images/placeholder.png)

Run everything configured for this run:

```bash
uv run python -m darsia.presets.workflows.user_interface_analysis \
  --mass --segmentation --fingers \
  --config sample_config.toml
```

In the GUI, tick each step you want in the sidebar and press **▶ Run**, or
use **Open Full Config** + the **Run** menu to batch multiple selections.
Open the **Streaming Preview** dock (`Ctrl+P`) beforehand to watch each image
as it's processed instead of waiting for the whole batch to finish.

## Step 4 — Helper (optional, but very useful while tuning)

Sidebar → **Helper**. These tools don't write pipeline output; they help you
pick good settings for Setup/Calibration/Analysis interactively.

**ROI** (`[helper.roi]`) — draw a rectangle on an image and get a ready-to-paste
`[roi.*]` TOML block back:

```toml
[helper.roi]
mode = "none"
data_selection = ["baseline"]
```

![Helper → ROI, mode + data selection](images/placeholder.png)

**ROI Viewer** (`[helper.roi_viewer]`) — page through images with a chosen
ROI mask overlaid, to sanity-check ROI definitions before using them in
Analysis:

```toml
[helper.roi_viewer]
data_selection = ["baseline"]
```

![Helper → ROI Viewer, data selection](images/placeholder.png)

`sample_config.toml` also configures `[helper.results]` (browse stored `npz`/`csv`
analysis output with min/max/sum/integral stats) and `[helper.color]`
(RGB/HSV histograms, absolute vs. relative to baseline) — open them the same
way, via the sidebar.

## Step 5 — Comparison (multi-run, optional)

Sidebar → **Comparison** compares *across* runs (event timing, Wasserstein
distances) once Analysis has produced output for two or more experiments.
`sample_config.toml` is a single-run config, so this phase isn't exercised here —
see the [DarSIA comparison workflow
doc](https://github.com/pmgbergen/darsia/blob/dev/src/darsia/presets/workflows/doc/workflow-comparison.md)
for the underlying CLI if you need it.

## What's next

- [Core concepts](03-core-concepts.md) explains *why* the config is organized
  the way it is (registries, layering) — read this before writing your own
  config from scratch.
- [Configuration reference](04-configuration-reference.md) is the field-by-field
  lookup table for every section you just saw.
- [Advanced analysis](05-advanced-analysis.md) covers thresholding, expert
  knowledge ROI constraints, and calibration-mode tradeoffs.
