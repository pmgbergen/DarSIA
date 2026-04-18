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

For analysis exports, define global format identifiers under `[analysis].formats`:
```toml
[analysis]
formats = ["my_npy", "4k"]
```

These keys reference top-level `[format.<type>.<identifier>]` presets and produce
output folders named `<type>_<identifier>` (for example `jpg_4k`).

See [ROI reference](./roi-reference.md), [image selection](./image-selection.md), and [config reference](./config-reference.md).
