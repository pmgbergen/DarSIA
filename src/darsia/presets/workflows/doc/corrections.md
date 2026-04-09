# Corrections

Workflow corrections are configured under `[corrections]`.

## Supported correction blocks
- `[corrections.type]`
- `[corrections.resize]` (restricted support)
- `[corrections.drift]`
- `[corrections.curvature.*]`
- `[corrections.color]`
- `relative_color` (boolean flag)
- `[corrections.illumination]`

## Practical notes
- Shape and color corrections are persisted separately (`shape_correction_*` and `color_correction_*`) with legacy fallback loading.
- `active_corrections` is not implemented.
- Relative color correction is currently limited in setup behavior; verify current rig warnings before enabling in production workflows.

Use this together with [workflow setup](./workflow-setup.md).
