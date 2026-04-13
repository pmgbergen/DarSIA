# Analysis Workflow

Module: `darsia.presets.workflows.user_interface_analysis`

## Main flags
- `--cropping`
- `--mass`
- `--volume`
- `--segmentation`
- `--fingers`
- `--all`
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

For cropping outputs, configure formats in TOML under:
```toml
[analysis.cropping]
formats = ["npz", "jpg"]
```

See [ROI reference](./roi-reference.md), [image selection](./image-selection.md), and [config reference](./config-reference.md).
