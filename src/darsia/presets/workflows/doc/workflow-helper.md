# Helper Workflow

Module: `darsia.presets.workflows.user_interface_helper`

## Main flags
- `--roi`
- `--roi-viewer`
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
- ROI helper supports:
  - direct rectangle selection
  - zoom-window ROI extraction (via current zoom extents)
- Clicking `ROI` opens a copy-ready ROI template for `[roi.*]`.
- ROI Viewer supports:
  - image traversal with `Previous` / `Next`
  - ROI selector dropdown with `all`, `none`, and every ROI registry key
  - immediate redraw of the active-region overlay when selection changes
  - coarse-image and coarse-ROI-mask preloading for responsive switching
