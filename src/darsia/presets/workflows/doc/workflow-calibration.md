# Calibration Workflow

Module: `darsia.presets.workflows.user_interface_calibration`

## Main flags
- `--color-paths`
- `--mass`
- `--default-mass`
- `--volume`
- `--reset`
- `--delete`
- `--ref-config`
- `--show`

## Typical sequence
```bash
python -m darsia.presets.workflows.user_interface_calibration --color-paths --config /abs/path/common.toml /abs/path/run.toml
python -m darsia.presets.workflows.user_interface_calibration --mass --config /abs/path/common.toml /abs/path/run.toml
```

## Notes
- Keep calibration basis settings aligned across `[color_paths]` and `[color_to_mass]`.
- Prefer reusable data registry keys for calibration datasets.

See [image selection](./image-selection.md) and [config reference](./config-reference.md).
