# Helper Workflow

Module: `darsia.presets.workflows.user_interface_helper`

## Main flags
- `--roi`
- `--roi-viewer`
- `--results`
- `--config`
- `--show`

## Typical command
```bash
python -m darsia.presets.workflows.user_interface_helper --roi --config /abs/path/common.toml /abs/path/run.toml
```

## Notes
- Configure helper behavior in `[helper.roi]`.
- Configure ROI viewer with either:
  - `[helper.roi_viewer] data = ["data_key"]`, or
  - shorthand `[helper] data = ["data_key"]`.
- Configure ResultReader in `[helper.results]`:
  - `data`: selector key(s) from top-level `[data.*]` registry
  - `mode`: result mode folder (for example `rescaled_mass`)
  - `format`: `npz`/`csv` or a format registry key resolving to `npz`/`csv`
  - `cmap`: optional colormap (`matplotlib.*` or `color_path.*`)
  - `roi`: optional ROI key(s) from top-level `[roi.*]` (first key is used)
- ROI helper supports:
  - direct rectangle selection
  - zoom-window ROI extraction (via current zoom extents)
- Clicking `ROI` opens a copy-ready ROI template for `[roi.*]`.
- ROI Viewer supports:
  - image traversal with `Previous` / `Next`
  - ROI selector dropdown with `all`, `none`, and every ROI registry key
  - immediate redraw of the active-region overlay when selection changes
  - coarse-image and coarse-ROI-mask preloading for responsive switching
- ResultReader supports:
  - loading scalar results from `npz` and `csv`
  - previous/next navigation (buttons and image click)
  - per-image stats overlay: min, max, sum, and Rig-geometry integral
