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
