# FluidFlower work on DarSIA: review guide

This branch adds the FluidFlower analysis workflow on top of DarSIA `dev`. It
is split into four commits so that each part can be reviewed on its own:

1. changes to existing DarSIA core modules under `src/darsia` (53 modules) and
   one new core module;
2. the new packages `src/darsia/fluidflower`, `src/darsia_cache` and
   `src/darsia_service`;
3. the tests;
4. root files, tools, scripts, documentation and this guide.

The core changes are re-applied onto the current upstream versions of the same
modules. Where upstream had changed the same lines, the upstream change is kept
and the FluidFlower change is applied on top of it; see
[Merge notes](#merge-notes).

## What the FluidFlower work adds

| Path | Files | Purpose |
| --- | ---: | --- |
| `src/darsia/fluidflower/` (top level) | 492 | Application and workflow modules: series projects and registries, run configurations, rig geometry and frame models, painting tools for sand maps and their placement in every run, image preparation and light normalisation, calibration and review windows, an Events page that finds when something happens in an area of the rig in every run, the native DarSIA FluidFlower application and the phase model studio. Pages read their files off the window's thread, optionally in a helper process (a setting). Series decision settings and a per-computer path map for shared storage (for example on a Mac) are read from configuration. |
| `src/darsia/fluidflower/native/` | 25 | Camera RAW reading on a shared, calibrated image grid; explicit method parameters and source sidecars. |
| `src/darsia/fluidflower/geometry/` | 6 | Measured rig geometry, physical areas and static component transport. |
| `src/darsia/fluidflower/phase/` | 35 | Phase inference with explicit models, geometry and rule parameters. |
| `src/darsia/fluidflower/phase_training/` | 23 | Training core for phase models from reviewed regions. |
| `src/darsia/fluidflower/mass/` | 30 | Concentration, saturation and mass methods: titration-based flash, statistics, overflow-governed free gas, the pixel pipeline and an optional GPU provider. |
| `src/darsia/fluidflower/pipeline/` | 77 | Whole-run job queue, workers, job inputs, capacity admission and publication of products. |
| `src/darsia/fluidflower/published_dataset/` | 37 | Layout, builder and checks of a published dataset of a series, and the `darsia-reproduce` launcher. |
| `src/darsia/fluidflower/pages/`, `ui/` | 81 | Qt pages and shared shell components of the FluidFlower application. |
| `src/darsia/fluidflower/tools/` | 43 | Packaged workflow tools, e.g. the calibration review desktop and pairwise review. |
| `src/darsia_cache/` | 26 | Operational cache admission for workers. Its import closure holds no DarSIA, NumPy, Numba, Pillow or Qt. |
| `src/darsia_service/` | 2 | Standard-library helpers that run before DarSIA, e.g. a supervised worker start. |
| `tests/` | 1020 new, 4 changed | Unit tests (`tests/unit`, `tests/unit/gui`), workflow tests (`tests/fluidflower`) and their fixtures. Three changed upstream tests follow the core changes; `tests/unit/gui/conftest.py` lets the window's background reads answer at once in tests. |
| Root and `docs/` | 20 | Launchers for Windows and macOS, the pinned worker environment `requirements-worker.lock`, `README_METHOD.md` (method core), `.gitattributes`, job scripts under `scripts/`, a qualification tool under `tools/`, six documents under `docs/` (one is the Mac setup guide), this guide, and `INVENTORY.md` and `REMOVED.md`, the accounting records of the earlier code cleanup (see the note at the top of each). |

`pyproject.toml` gains one console script, `darsia-reproduce`.

## Changes to existing core modules

Many core readers now open files through `darsia.fluidflower.preset_wasserstein_inputs`
("storage aliases" below). A project folder can then be cloned to another
machine and its saved absolute paths read at the mapped location, without
rewriting files that are bound by SHA.

| Module | Why |
| --- | --- |
| `src/darsia/__init__.py` | Runs the worker cache bootstrap before the package loads, and applies the scientific OpenCV policy in the importing thread. |
| `src/darsia/_opencv_runtime.py` (new) | The scientific OpenCV policy (IPP off) for each thread, with checks at scientific entry and exit points. |
| `corrections/color/colorcorrection.py` | Loads saved colour corrections through storage aliases. |
| `corrections/color/illuminationcorrection.py` | Adds the closed-form `reference_sample` method next to `group_average`; loads through storage aliases. |
| `corrections/color/patchwiseilluminationcorrection.py` | Loads through storage aliases. |
| `corrections/color/relativecolorcorrection.py` | Loads through storage aliases. |
| `corrections/readcorrection.py` | Reads through storage aliases and maps the older saved class name `Resize` to `ResizeCorrection`. |
| `corrections/shape/curvature.py` | Optional `origin` of the curvature crop, for a crop area that is separate from the reference frame; loads through storage aliases. |
| `corrections/shape/drift.py` | Records `linear_input` for baselines built from linear RAW values; loads through storage aliases. |
| `corrections/shape/translation.py` | Exactly equal views get a zero translation without feature detection, also for float images outside the 8-bit conversion range. |
| `corrections/typecorrection.py` | Loads through storage aliases. |
| `experiment/experiment.py` | Optional overflow protocol on the experiment. |
| `experiment/protocols.py` | `OverflowProtocol`, a measured overflow table from a CSV file or an Excel sheet; protocol files read through storage aliases. |
| `gui/__main__.py` | `darsia dataset ...` runs the published-dataset command line instead of the GUI. |
| `gui/gui_main.py` | Installs the interaction-safety handlers of the FluidFlower shell, and reuses `SettingsFactory` for nested-key access instead of a second copy. |
| `gui/ui/file_dialog.py` | Table previews read `[workbook, worksheet]` references and open inside the main window when it has a page stack. |
| `gui/ui/main_window.py` | Installs the interaction-safety handlers. |
| `gui/ui/settings.py` | Keeps protocol `[workbook, worksheet]` pairs when path-map rows are saved. |
| `gui/ui/sidebar.py` | Keyboard focus, accessibility names and left-click-only activation for sidebar rows. |
| `gui/ui/table_viewer.py` | `load_table_reference` reads a path or a literal `[workbook, worksheet]` without evaluating code. |
| `gui/ui/toolbar.py` | The toolbar builder is a `QObject` with slots, so its signal connections are owned by the window. |
| `image/imread.py` | Folder listings and existence checks through storage aliases. |
| `multiphase/flash.py` | `TitrationFlash`, a flash model from dissolved inorganic carbon to the indicator colour, chosen when saved data names the titration model. |
| `presets/workflows/analysis/expert_knowledge.py` | Dynamic, per-image expert knowledge (e.g. the allowed region of a phase in that image's phase map), combined with the static ROIs of the same mode. |
| `presets/workflows/calibration/calibration_color_to_mass_analysis.py` | Initial flash bounds from one shared helper instead of fixed literals. |
| `presets/workflows/comparison/comparison_wasserstein.py` | Optional injected mass reader and assembled result records, so the comparison can run on admitted inputs. |
| `presets/workflows/config/color_embedding_registry.py` | Reads registry files through storage aliases. |
| `presets/workflows/config/color_to_mass.py` | A relative `calibration_folder` is read against the project folder. |
| `presets/workflows/config/corrections.py` | Curvature crop `origin` and a switch for the optional curvature stages; explicit illumination sample windows, `method`, `reference_sample` and a porosity-mask switch. |
| `presets/workflows/config/data.py` | RAW formats (`ARW`, `NEF`, `DNG`) and an optional calibrated `raw_camera` binding; relative paths read against the project folder. |
| `presets/workflows/config/data_registry.py` | Merges consistent entries inherited from several configuration files; reads through storage aliases. |
| `presets/workflows/config/depth.py` | Relative paths read against the project folder. |
| `presets/workflows/config/facies.py` | Relative paths read against the project folder. |
| `presets/workflows/config/format_registry.py` | Reads registry files through storage aliases. |
| `presets/workflows/config/labeling.py` | Relative paths read against the project folder. |
| `presets/workflows/config/multi_fluidflower_config.py` | Creates result folders through the storage layer. |
| `presets/workflows/config/protocols.py` | Optional overflow protocol; relative protocol paths read against the project folder. |
| `presets/workflows/config/rig.py` | Optional native label product and analysis-exclusion records for the rig, decoded strictly; relative rig path read against the project folder. |
| `presets/workflows/config/roi.py` | `MultiRoiConfig.load` reads `[[roi]]` entries through the ROI registry; it was not implemented before. |
| `presets/workflows/config/roi_registry.py` | Merges consistent ROI entries inherited from several files; reads through storage aliases. |
| `presets/workflows/config/time_data.py` | Globbing and existence checks through storage aliases. |
| `presets/workflows/config/utils.py` | Shared helpers: project-relative paths, storage aliases and the merging of inherited registry entries. Configuration paths are read as this computer reads stored paths, so a Windows spelling of a shared path is understood on another system. |
| `presets/workflows/config/video.py` | Uses the shared RGB parser of `config/analysis.py` instead of a second copy. |
| `presets/workflows/config/wasserstein.py` | Result folders through the storage layer; the image times are read per section; ROIs without a plain box or polygon geometry are refused. |
| `presets/workflows/facies_props.py` | Records which labels have explicitly assigned porosity and permeability; property tables read through storage aliases. |
| `presets/workflows/heterogeneous_color_analysis.py` | Uses the shared `get_mean_color` instead of a second copy. |
| `presets/workflows/rig.py` | RAW reading through the calibrated camera binding, owned outputs for RAW rig builds, analysis-domain support, optional explicit illumination samples and method, a colour correction that fails when the colour checker is not found, and storage aliases. |
| `presets/workflows/setup/setup_protocols.py` | Validates an existing workbook-backed protocol set instead of generating CSV templates for it. |
| `presets/workflows/setup/setup_rig.py` | RAW rig builds from a build request (no data cache for RAW), with analysis exclusions and native products. |
| `presets/workflows/utils/images.py` | Scientific OpenCV policy checks and a cache folder named after that policy. |
| `presets/workflows/utils/mass.py` | `select_data_path`, the nearest-match selection without decoding the image, and storage scopes. |
| `restoration/resize.py` | Loads through storage aliases. |
| `signals/color/label_color_spectrum_map.py` | Reuses the `LabelColorPathMap` representations instead of a second copy. |
| `utils/features.py` | Float images are clipped to [0, 1] before the 8-bit conversion for feature detection, as linear RAW values can exceed 1. |

## Install and run

The application needs Python 3.12 or later. The FluidFlower launchers expect a
virtual environment named `.venv-fluidflower` in the repository root:

```bash
python -m venv .venv-fluidflower
.venv-fluidflower/Scripts/python -m pip install -e ".[dev]"   # Windows
# .venv-fluidflower/bin/python -m pip install -e ".[dev]"     # Linux/macOS
```

`requirements-worker.lock` pins the exact environment used by job workers.
Camera RAW reading needs `rawpy`, which the lock file pins but
`pyproject.toml` does not declare.

GUI:

- Windows: `start_fluidflower_darsia.cmd` opens the native DarSIA FluidFlower
  application; `start_phase_model_studio.cmd` opens the phase model studio.
  Both call `launch_fluidflower.ps1`, which starts `run_fluidflower.pyw` with
  the environment above.
- macOS: `start_darsia.command` (double-click in Finder, or run it with
  `studio` or `experiments`) creates `.venv-fluidflower` on the first start
  and installs DarSIA and `rawpy` into it. A Mac sets up series; jobs run on
  Windows only. See `docs/mac_setup_station.md`.
- Any platform: `python run_fluidflower.pyw native` (or `studio`,
  `experiments`); the upstream GUI still starts with `darsia`.

Tests (one file per call is the most robust on Windows):

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest -q tests/unit/test_rig_workflow.py
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest -q tests/fluidflower
```

`tests/unit/test_current_plan_strict39_prospective.py` compares the
qualification tool under `tools/qualification/` with its historical original
at a fixed local path, so it fails where that file is missing.

## Where configuration lives

- **Series project folder**: `series.toml` with `config/` (native run
  configurations in the DarSIA TOML format), `protocols/`, `data/`,
  `geometry_facies/` and `image_selection/`. Relative paths in these files are
  read against the project folder. See `docs/project_repository.md`. The
  optional `[decisions]` table of `series.toml` names who records approvals
  and saves, and which recorded choices apply
  (`darsia.fluidflower.series_decisions`); the code checks only that a
  recorded decision exists and is unchanged, and a feature whose setting is
  missing stays unavailable.
- **Per-user files**, never in a repository: `series_registry.toml` (projects
  and `[storage.path_aliases]`; `%APPDATA%\DarSIA` on Windows, otherwise
  `$XDG_CONFIG_HOME/DarSIA`) and `stores.toml` (machine and store roots for
  shared jobs; `%LOCALAPPDATA%\DarSIA` on Windows, otherwise
  `$XDG_CONFIG_HOME/darsia`).
- **Per-user setting**: where this computer opens each shared storage root
  (Settings > Shared storage on this computer, kept in the user's Qt
  settings). A Mac that opens the share at `/Volumes/<share>` then reads and
  writes stored paths in the share's own spelling. Workers never read it, and
  without rows nothing changes.
- **Environment**: `DARSIA_FLUIDFLOWER_CACHE` (local cache root),
  `DARSIA_FLUIDFLOWER_DATA_ROOT` (a parent folder whose children are series
  folders), `DARSIA_SERIES_REGISTRY` (registry location),
  `DARSIA_OPERATOR_NAME` (name recorded for job actions) and, for the macOS
  launcher, `DARSIA_PYTHON` (the Python that creates its environment).
- **Shared jobs**: job folders on an SMB/NFS share, described in
  `docs/shared_nas_worker.md` and `README_METHOD.md`.

## Merge notes

The FluidFlower changes were made on DarSIA `dev` at `aece8c49`. Upstream
`dev` (here at `f75198d3`) has 59 further commits, which changed 68 files.
Twelve of these files were also changed here; eight merged without conflict,
and four needed a manual resolution.

Resolved conflicts:

- `presets/workflows/analysis/expert_knowledge.py`: both sides extended
  `ExpertKnowledgeAdapter`. Upstream's negative ROIs, ROI shapes, label and
  facies images and registry fields are kept, and its positive/negative mask
  logic is unchanged. It is now the static part `_static_mask`. The FluidFlower
  `mask_for(image, mode, *, state=None)` and `apply(..., state=None)` combine
  it with the dynamic, per-image sources in the new `dynamic` field.
- `presets/workflows/rig.py`, drift correction: upstream's choice between a
  named ROI and an auto-detected colour checker is kept. The FluidFlower
  `linear_input` flag for RAW baselines is passed to every `DriftCorrection`,
  including the new named-ROI branch.
- `presets/workflows/rig.py`, colour correction: upstream's choice between a
  named ROI and an auto-detected colour checker is kept. In the colour-checker
  branch, a colour checker that cannot be found raises an error, as in the
  FluidFlower version, instead of continuing with an uncalibrated correction.
- `presets/workflows/rig.py`, imports and helpers: `TYPE_CHECKING` and `Any`
  are both imported; upstream's `_resolve_named_roi_voxels` and the FluidFlower
  `_preserve_native_setup` are both kept.
- `presets/workflows/config/roi.py`: upstream's key-list import is kept. The
  `utils` import keeps only `_get_key`, because `MultiRoiConfig.load` now reads
  through `RoiRegistry`.
- `presets/workflows/heterogeneous_color_analysis.py`: both new imports are
  kept.

Merged without conflict (each FluidFlower change applies unchanged on
upstream's version): `__init__.py`, `corrections/color/colorcorrection.py`,
`corrections/shape/drift.py`, `gui/ui/settings.py`,
`presets/workflows/calibration/calibration_color_to_mass_analysis.py`,
`presets/workflows/config/corrections.py`,
`presets/workflows/config/roi_registry.py` and
`presets/workflows/setup/setup_rig.py`.

Interactions with upstream changes that still need a follow-up. They are left
as they are in this merge, so that the tree is exactly the two sides combined:

- Upstream removed the `label` field of `RoiConfig`; label ROIs now use
  `shape = "label"`. `config/wasserstein.py` still checks `entry.label`, so
  loading a Wasserstein section with ROIs fails, and the preset Wasserstein
  tests that build `label = ...` ROIs fail. The check should become
  `entry.shape not in ("box", "polygon")`, and those tests should use the new
  shapes.
- `Rig.setup` now passes `roi_registry` to `setup_reading`,
  `setup_shape_corrections` and `setup_color_corrections`. The phase-map rig
  subclass in the FluidFlower RAW setup chain overrides
  `setup_color_corrections` without that parameter. It should accept and
  forward `roi_registry`.
- `setup_rig` now reads `config.roi_registry`. Several FluidFlower test
  doubles of the configuration do not have that attribute yet.
- The settings form now holds the new `roi` fields of the drift and colour
  corrections. The FluidFlower curvature setup save therefore writes empty
  `[corrections.drift]` and `[corrections.color]` tables where none existed
  before.
- Upstream moved the contour smoother from the analysis to the calibration
  configuration. The FluidFlower mass calibration still looks for it in the
  analysis configuration and quietly applies no smoothing when it is not
  found there.
- Upstream's drift and colour configurations now require `colorchecker` in
  mode `"colorchecker"`, and `RoiRegistry.resolve_rois` no longer drops
  label-restricted entries. `painting_analysis_box_sources.load_legacy_box_source`
  relied on that drop to refuse a label-restricted `box1` or `box2`; such a
  box is now accepted, and two cases in
  `tests/unit/test_painting_analysis_box_sources.py` fail. FluidFlower
  configurations are otherwise not affected as far as the tests show, but
  these are behaviour changes for existing files.
