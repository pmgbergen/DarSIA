# Helper Workflow

Module: `darsia.presets.workflows.user_interface_helper`

## Main flags
- `--roi`
- `--config`
- `--show`

## Typical command
```bash
python -m darsia.presets.workflows.user_interface_helper --roi --config /abs/path/common.toml /abs/path/run.toml
```

## Notes
- Configure helper behavior in `[helper.roi]`.
- ROI helper supports:
  - direct rectangle selection
  - zoom-window ROI extraction (via current zoom extents)
- Clicking `ROI` opens a copy-ready ROI template for `[roi.*]`.
