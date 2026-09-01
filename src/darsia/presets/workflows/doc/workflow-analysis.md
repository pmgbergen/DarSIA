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

Color-based scalar modes are now configured via centralized color embeddings and used as
`color.<id>` in segmentation/fingers/thresholding modes.

For analysis exports, each analysis mode defines its own format identifiers:
```toml
[analysis.mass]
formats = ["my_npy", "4k"]

[analysis.thresholding]
formats = ["jpg", "csv"]
```

These keys reference top-level `[format.<type>.<identifier>]` presets and produce
output folders named `<type>_<identifier>` (for example `jpg_4k`).

Each mode also specifies its own `data_selection` for independent image selection:
```toml
[analysis.mass]
data_selection = ["injection", "analysis"]
formats = ["my_npy", "4k"]
```

See [ROI reference](./roi-reference.md), [image selection](./image-selection.md), and [config reference](./config-reference.md).
