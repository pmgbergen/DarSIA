# Developer Notes

## Workflows documentation maintenance (presets/workflows)

### Minimal conventions for future updates
1. Keep `src/darsia/presets/workflows/doc/README.md` as the single navigation entry point.
2. Add new workflow docs as `workflow-<name>.md` and link them from `README.md` and `overview.md`.
3. Keep schema details centralized in `config-reference.md`; workflow pages should link to it instead of duplicating option lists.
4. When adding config keys, update `config-reference.md`, then update only impacted workflow guides.
5. For deprecations, add a short "Deprecated" note and preferred replacement in the affected page.
6. Prefer links to code modules over duplicating implementation details likely to drift.

### Risks
- **Fast drift in config schema**: workflow config objects evolve quickly; docs can become stale.
- **Output-path drift**: default folder conventions may change with config defaults.
- **Workflow surface drift**: flags and optional paths (CLI/GUI) may change across releases.
- **Correction pipeline drift**: behavior and persistence details may change as correction internals evolve.
- **Advanced features drift**: ROI/data registries and calibration options are actively evolving.

### Acceptance criteria for future documentation PRs
1. `src/darsia/presets/workflows/doc/README.md` exists and links all workflow docs.
2. Docs cover setup, calibration, analysis, comparison, and GUI entry points.
3. `config-reference.md` matches currently loaded workflow config sections.
4. Image selection and ROI usage are documented with registry-based examples.
5. New/changed workflow flags are reflected in relevant workflow pages.
6. Known limitations are captured in `known-issues.md`.
7. Links between pages resolve and avoid dead references.
8. No unnecessary duplication of volatile implementation internals.

## Lazy imports of the package (`src/darsia/__init__.py`)

`import darsia` imports none of its modules; `darsia.<name>` imports the module providing
the name on first use ([lazy_loader](https://github.com/scientific-python/lazy-loader),
Scientific Python SPEC 1, as in scikit-image). The names are listed once, in the stub
`__init__.pyi`, which is read by `lazy_loader.attach_stub` at runtime and by type checkers
and IDEs. This keeps the GUI (which only needs the configuration classes) small and fast.

- A new public class or function needs a line `from .<module> import <name> as <name>`
  in `__init__.pyi` (a new subpackage: `from . import <name> as <name>`). Only relative
  imports of darsia's own modules are possible; names of other packages (numpy, pathlib,
  ...) are imported from these packages directly.
- Check that all listed names resolve:
  `python -c "import darsia; print([n for n in darsia.__all__ if not hasattr(darsia, n)])"`.
  (`EAGER_IMPORT=1` of lazy_loader does not work for darsia: its modules use `darsia.<name>`
  while being imported.)
- Modules imported by the GUI (`darsia.gui`) must not import analysis, rig or calibration
  modules (or pandas, scipy, numba) at module level; import them inside functions or
  under `TYPE_CHECKING`. Check with `python -X importtime -c "import darsia.gui.ui.main_window"`.

## Synthetic workflow test (`tests/integration/workflows`)

An end-to-end run of the workflows (setup -> colour calibration -> analysis) on a small
synthetic experiment, part of the normal test run in CI (about 10 s). Run it alone with
`uv run pytest -m workflow`.

- `synthetic.py`, `make_synthetic_experiment(folder, shape, num_images, seed, ...)`,
  writes everything into `folder`: PNG images (64 x 128 by default; a blue sand, a
  dark blue seal, the same sand below; a growing white gas cap and a yellow aqueous halo
  under the seal; static dark grains; slight noise), the coloured segmentation, the
  protocols, the depth and facies CSVs, and `config.toml`. It returns the ground truth
  (layers, grains, gas and aqueous masks per image). No binary files are in the
  repository.
- `test_synthetic_workflow.py` calls the same entry points as the CLI and the GUI
  (`setup_*`, `calibration_*_from_context` without interaction, `run_analysis`) in
  module-scoped fixtures and compares the results with the ground truth.
- `DARSIA_SYNTHETIC_KEEP=<folder> uv run pytest -m workflow` writes the experiment and its
  results to `<folder>` and keeps them. `<folder>/config.toml` opens in the GUI as a small
  demo.
- Variants (other corrections, illumination gradients, noise for the restoration, ...):
  add options to `make_synthetic_experiment` and a test module per variant. Keep the
  images small; most of the time and memory is imports (about 250 MB).
