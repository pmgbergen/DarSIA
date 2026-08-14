# Calibration Workflow

Module: `darsia.presets.workflows.user_interface_calibration`

## Main flags
- `--color-embedding`
- `--mass`
- `--default-mass`
- `--volume`
- `--reset`
- `--delete`
- `--ref-config`
- `--show`

## Typical sequence
```bash
python -m darsia.presets.workflows.user_interface_calibration --color-embedding --config /abs/path/common.toml /abs/path/run.toml
python -m darsia.presets.workflows.user_interface_calibration --mass --config /abs/path/common.toml /abs/path/run.toml
```

## Notes
- Configure all color embeddings under `[color.path.*]`, `[color.range.*]`, `[color.channel.*]`.
- Use `[calibration.color]` and `[calibration.mass]` as calibration entrypoints.
- Mass calibration currently supports only color-path embeddings.

See [image selection](./image-selection.md) and [config reference](./config-reference.md).
