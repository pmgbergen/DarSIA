# 6. Troubleshooting & FAQ

## Known issues

Carried over from the DarSIA workflow docs — these are current limitations,
not bugs specific to any one project:

- **Labeling quality** depends strongly on the quality of the input segmented
  reference image (`[labeling].colored_image`). A noisy or imprecise
  segmentation propagates into every facies-based color embedding downstream.
- **Curvature calibration** quality is sensitive to how carefully the four
  rig corners were picked during Setup. Redo the crop step if analysis output
  looks geometrically skewed.
- **Illumination correction** quality depends on which labels/facies were
  used and the interpolation options chosen for
  `[corrections.patchwise_illumination]`.
- **Relative color correction** requires careful, workflow-specific
  verification — it is not a drop-in toggle in setup-heavy pipelines.

## "Nothing happens when I click Run"

Check three things, in order:

1. Is a sidebar item actually **selected** (highlighted)? Run acts on the
   currently selected step, not on "whatever tab is open."
2. Is a config file loaded (title bar / config path label at the top of the
   window is not "No config loaded.")?
3. Open the **Logging** dock (`Ctrl+L`) — errors during a run print there,
   even when nothing visibly changes in the UI.

## "My config loaded but a form section is empty"

The settings panel only shows sections **relevant to the currently selected
sidebar step** — this is intentional, not a bug. Use **Settings > Open Full
Config** (`Ctrl+E`) to see every section in the file at once, regardless of
selection.

## "Where did my output go?"

Everything is written under `[data].results` from the config you ran, in a
subfolder named after the analysis mode and format, e.g.:

```
<results>/mass/jpg/
<results>/rescaled_saturation_g/npz/
<results>/fingers/tips/
```

Use **Helper > Results** (`[helper.results]`) to browse `npz`/`csv` output
directly in the GUI without leaving it, including per-image min/max/sum and
a rig-geometry integral.

## "The color/mass calibration result looks wrong"

- Verify the calibration ROI (`calibration.color.rois` /
  `calibration.mass.rois`) is actually the region you think it is — use
  **Helper > ROI Viewer** to overlay it on a real image first.
- Check `histogram_weighting` and `ignore_baseline_spectrum` — see
  [Advanced analysis](05-advanced-analysis.md#calibration-modes).
- If `calibration_mode = "auto"` isn't converging well, consider switching to
  `manual` with values from a previous known-good run (or an imported
  calibration bundle, `[utils.calibration].import_bundle`).

## "Can I use my own experiment?"

Yes — this guide's config, `sample_config.toml`, is a real run config, not a
special tutorial fixture. To adapt it:

1. Copy it to `config/run/<your-experiment>.toml`.
2. Update `[data]` (folders, baseline, results), `[rig]` (your physical
   dimensions), and `[protocols]` (your CSV paths).
3. Re-run **Setup > Crop correction** interactively for your rig's corners —
   don't reuse `sample_config.toml`'s pixel coordinates.
4. Re-do **Calibration** from scratch — color/mass calibration is specific
   to your camera, lighting, and sand.

See [Core concepts](03-core-concepts.md) for how to structure a
multi-file (`common.toml` + `run.toml`) setup once you have more than one run
of the same experiment family.

## Where to ask for help

- Re-read [Core concepts](03-core-concepts.md) if a config field's *purpose*
  is unclear — most confusion comes from not knowing which registry a section
  references.
- The underlying CLI docs in
  `src/darsia/presets/workflows/doc/` cover the same
  sections from the command-line-workflow angle, useful for scripting batch
  runs outside the GUI.
- `the DarSIA README` has DarSIA-level installation and citation
  information.
