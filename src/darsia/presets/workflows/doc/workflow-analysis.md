# Analysis Workflow

Module: `darsia.presets.workflows.user_interface_analysis`

## Main flags
- `--cropping`
- `--mass`
- `--volume`
- `--segmentation`
- `--fingers`
- `--all`
- `--save-jpg`
- `--save-npz`
- `--show`

## Typical command
```bash
python -m darsia.presets.workflows.user_interface_analysis \
  --mass --segmentation --fingers \
  --config /abs/path/common.toml /abs/path/run.toml /abs/path/analysis.toml
```

## Common outputs
- processed crops
- mass/volume products
- segmentation visualizations
- fingers analysis outputs

See [ROI reference](./roi-reference.md), [image selection](./image-selection.md), and [config reference](./config-reference.md).
