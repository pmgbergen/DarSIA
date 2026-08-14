# Quickstart

Use one or more config files with `--config`.

## 1) Setup
```bash
python -m darsia.presets.workflows.user_interface_setup --all --config /abs/path/common.toml /abs/path/run.toml
```

## 2) Calibration
```bash
python -m darsia.presets.workflows.user_interface_calibration --color-paths --config /abs/path/common.toml /abs/path/run.toml
python -m darsia.presets.workflows.user_interface_calibration --mass --config /abs/path/common.toml /abs/path/run.toml
```

## 3) Analysis
```bash
python -m darsia.presets.workflows.user_interface_analysis --mass --segmentation --fingers --config /abs/path/common.toml /abs/path/run.toml /abs/path/analysis.toml
```

## 4) Comparison (optional)
```bash
python -m darsia.presets.workflows.user_interface_comparison --events --config /abs/path/multi.toml
```

## 5) Helper (optional)
```bash
python -m darsia.presets.workflows.user_interface_helper --roi --config /abs/path/common.toml /abs/path/run.toml
```

See [config reference](./config-reference.md) for required sections.
