# 3. Core Concepts

This chapter explains the ideas behind the config format, so that writing
your own config (rather than editing a copy of `sample_config.toml`) makes sense.

## Config layering

Workflows accept **multiple** TOML files, loaded in order; later files
override earlier ones. `sample_config.toml` is a single self-contained file, but
the recommended pattern for a new experiment family is to split it:

1. `common.toml` — defaults shared across every run of an experiment family
   (e.g. rig geometry, correction pipeline).
2. `run.toml` — one per physical run: data paths, protocol files.
3. `analysis.toml` — analysis-only settings you iterate on without touching
   setup/calibration.

```bash
uv run python -m darsia.presets.workflows.user_interface_analysis \
  --mass --config common.toml run.toml analysis.toml
```

The GUI's `Open Config…` opens **one** file at a time; if your project layers
several files, load the most specific one (it should be self-contained enough
for interactive use), and keep the CLI for reproducible multi-file batch runs.

## Registries: define once, reference by key

Four kinds of reusable objects live in **top-level** sections, independent of
any single workflow phase. Every other section references them by string key
instead of repeating the definition. This is the single most important
pattern in the config format — get it right and everything else falls into
place.

### 1. Data selectors (`[data.*]`, plus top-level `[[data_path]]` etc.)

Pick a subset of images. Three styles, combinable:

```toml
[data.interval.calibration]   # uniformly-sampled time window
start = "01:00:00"
end = "05:00:00"
num = 5
tol = "00:05:00"

[data.time.qa]                # explicit timestamps
times = ["01:00:00", "02:30:00"]
tol = "00:05:00"

[[data_path]]                 # explicit filenames / globs — sample_config.toml's style
name = "baseline"
paths = ["baseline.JPG"]

[[data_path]]
name = "all"
paths = ["*"]
```

Then reference the name from any `data_selection` field:

```toml
[analysis.mass]
data_selection = ["calibration"]
```

### 2. ROIs (`[roi.*]` / `[[roi]]`)

A rectangle in **physical** (meter) coordinates, defined by two opposite
corners:

```toml
[[roi]]
name = "calibration"
corner_1 = [0.01, 0.45]
corner_2 = [0.9, 0.0]
```

Referenced from `analysis.mass.roi`, `analysis.fingers.roi`,
`color_path.rois`, `analysis.expert_knowledge.*`, and more. The **ROI
Helper** (Helper tab) draws a rectangle interactively and hands you a
copy-ready `[roi.*]` block — see
[Quickstart, Step 4](02-quickstart-walkthrough.md#step-4--helper-optional-but-very-useful-while-tuning).

### 3. Color embeddings (`[color_path.*]` / `[[color_path]]`)

A recipe for turning an image's color into a single scalar signal — the
basis of both segmentation/thresholding modes (`color.<id>`) and mass
calibration. `sample_config.toml` defines one, `color_path`, based on labeled
facies:

```toml
[[color_path]]
name = "color_path"
basis = "facies"
reference_label = 2
```

There are two sibling forms not used in `sample_config.toml` but available for
simpler cases: `[color.channel.*]` (a single RGB/HSV/LAB channel) and
`[color.range.*]` (a bounding box in color space). See
[Configuration reference](04-configuration-reference.md#color-embedding-registry).

### 4. Export formats (`[[format]]` / `[format.<type>.<id>]`)

A named preset for how to write an output image/array:

```toml
[[format]]
name = "jpg"
type = "jpg"
resolution = [1080, 4]
cmap = "color_path.from_facies.7"
keep_ratio = true
quality = 100
```

Referenced by identifier from any `formats` list:

```toml
[analysis.mass]
formats = ["jpg"]
```

Output lands in `<results>/<mode>/<type>_<identifier>/` (or `<results>/<mode>/<type>/`
when the identifier matches the type name, as in `sample_config.toml`).

## Why registries matter

Without them, you'd repeat the same ROI corners or color-path parameters in
every section that needs them — and they'd silently drift apart as you tweak
one copy and forget the others. With them, tightening the `calibration` ROI
by 2 pixels is a one-line edit that every consumer picks up.

## The correction pipeline

Every raw image passes through `[corrections]` before anything else touches
it. `sample_config.toml`'s pipeline, in the order it actually runs:

```toml
[corrections]
active = ["curvature", "drift", "patchwise_illumination", "resize", "type"]
```

- **curvature** — crops/dewarps to the four rig corners (from a one-time
  interactive point-picker during Setup).
- **drift** — corrects for camera/lighting drift using a colorchecker patch.
- **patchwise_illumination** — flattens uneven lighting across the rig using
  one or more baseline images.
- **resize** — downscales for speed (`scale = 0.25` here).
- **type** — casts to a numeric dtype (`float32`) for downstream math.

Order matters: this is why `active` is a list, not a set. See
[Configuration reference](04-configuration-reference.md#corrections) for
every correction's own sub-table.

## Next

[Configuration reference](04-configuration-reference.md) puts all of this
into one lookup table.
