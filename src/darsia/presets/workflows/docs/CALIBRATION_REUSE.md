# Reusing calibrated data across users and machines

This guide describes a user-friendly workflow where:

1. an expert calibrates once,
2. exports a zip bundle,
3. another user imports it,
4. and links it in their config for analysis reuse.

## 1) Expert: export a calibration bundle

Use the preset workflow utils interface with your calibration config:

```bash
python <your_utils_entrypoint>.py \
  --config /absolute/path/to/config.toml \
  --export-calibration \
  --calibration-bundle /absolute/path/to/calibration_bundle.zip
```

If `--calibration-bundle` is omitted, a default bundle is written under:

`<results>/calibration/bundles/`

The bundle includes (when available):

- `color_paths/`
- `color_to_mass/`
- `baseline_color_spectrum/`
- `color_range/color_range.json`
- `manifest.json`

## 2) User: import a calibration bundle

```bash
python <your_utils_entrypoint>.py \
  --config /absolute/path/to/config.toml \
  --import-calibration \
  --calibration-bundle /absolute/path/to/calibration_bundle.zip \
  --calibration-target /absolute/path/to/shared/calibration
```

Optional:

- `--overwrite` to overwrite all conflicting files in the destination.

Without `--overwrite`, import aborts if conflicting files already exist.

After import, DarSIA writes:

- `CONFIG_SNIPPET.toml`

inside the import target folder.

## 3) Link imported calibration in your config

Copy the generated snippet entries into your config file. Typical keys are:

- `[color_paths].calibration_file`
- `[color_paths].baseline_color_spectrum_folder`
- `[color_paths].color_range_file`
- `[color_to_mass].calibration_folder`

Use absolute paths to make sharing and deployment robust.

## 4) Run analysis with reused calibration

Run your normal analysis workflow. It will load calibration from the linked paths.

## Notes on compatibility and safety

- Calibration artifacts contain metadata (`basis`, `label_ids`).
- Loading performs compatibility checks and raises errors on mismatches.
- This protects users from accidentally applying incompatible calibrations.
- GUI import performs a preflight conflict scan and prompts for
  overwrite-all / skip-all / abort before writing files.
