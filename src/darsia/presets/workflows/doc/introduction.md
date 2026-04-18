# Introduction

This guide gives a practical overview of how to work with DarSIA presets/workflows config files.

## How config files are composed
Workflows accept multiple TOML files via repeated `--config` arguments. They are loaded in order, and later files are used to override or refine earlier defaults.

Typical layering pattern:
1. `common.toml` – shared defaults for an experiment family.
2. `run.toml` – run-specific paths, data source, and protocol specifics.
3. `analysis.toml` – analysis-only settings and optional ROIs.

Example:
```bash
python -m darsia.presets.workflows.user_interface_analysis \
  --mass --segmentation --fingers \
  --config /abs/path/common.toml /abs/path/run.toml /abs/path/analysis.toml
```

## Recommended config organization
- Keep reusable data selectors in top-level `[data.*]` registries.
- Keep reusable ROIs in top-level `[roi.*]` registries.
- Reference registry keys from workflow sections (`color_paths`, `analysis.mass`, `analysis.fingers`, ...).
- Keep calibration and analysis sections in dedicated files when possible.

## Typical end-to-end sequence
1. Setup (`user_interface_setup`)
2. Calibration (`user_interface_calibration`)
3. Analysis (`user_interface_analysis`)
4. Helper (`user_interface_helper`, optional)
5. Comparison (`user_interface_comparison`, optional)

## Where to continue
- [Overview](./overview.md)
- [Quickstart](./quickstart.md)
- [Config reference](./config-reference.md)
- [Image selection](./image-selection.md)
- [Analysis deep-dive](./analysis.md)
