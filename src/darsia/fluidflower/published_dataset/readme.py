"""The record's README.md, written from its manifest (English, no names)."""

from __future__ import annotations

from .layout import CHECKSUMS, EXPECTED_ARCHIVE, MANIFEST, README, SERIES_ARCHIVE


def _gb(value) -> str:
    """Bytes as decimal GB, or "not recorded"."""
    return "not recorded" if value is None else f"{value / 1e9:.2f} GB"


def _link(value) -> str:
    """A link, or the note that it comes at upload."""
    return value if value else "to be added at upload"


def _seconds(value) -> str:
    """Seconds with one decimal, or "not recorded"."""
    return "not recorded" if value is None else f"{value:.1f}"


def render_readme(manifest: dict) -> str:
    """The README text of a record, from its manifest only."""
    identity = manifest["identity"]
    record = identity["record"]
    runs = manifest["runs"]
    frames = sum(len(run["frames"]) for run in runs)
    code = manifest["code"]
    machine = manifest.get("machine") or {}
    rule = manifest["reproducibility"]["other_machines"]
    lines = [
        f"# {identity['title']}",
        "",
        f"Series `{identity['series_id']}`, record **{record}** "
        f"({'reproduction: raw images as acquired' if record == 'R' else 'simulator products'}), "
        f"mass maps **{export_resolution(manifest)}**: {len(runs)} runs, {frames} frames.",
        "",
        "## Links",
        "",
        f"- Other record of this series: {_link(identity['links'].get('other_record'))}",
        f"- DarSIA code: {_link(identity['links'].get('code'))}",
        f"- Article: {_link(identity['links'].get('article'))}",
        "- Licence and citation: see the record's metadata.",
        "",
        "## Files",
        "",
        f"- `{MANIFEST}`: what this record holds, with every SHA-256 (format `{manifest['format']}`).",
        f"- `{README}`: this file.",
        f"- `{CHECKSUMS}`: SHA-256 of every file (check with `sha256sum --check {CHECKSUMS}`).",
    ]
    if record == "R":
        lines += [
            f"- `{SERIES_ARCHIVE}`: the series folder (configuration, protocols, geometry, calibration, "
            "phase model, execution plan, path bindings and neutral provenance records).",
            f"- `{EXPECTED_ARCHIVE}`: the expected results (masses per frame and the stored coarse layer).",
            "- `raw_<run>.zip`: the run's raw images exactly as acquired (stored, not recompressed).",
        ]
    else:
        lines += ["- `simulator_<run>.zip`: the run's coarse products, static fields and time series."]
        if (manifest.get("simulator") or {}).get("geometry"):
            lines += ["  It also holds `products/<run>/geometry/simulator_geometry.npz` and `.json`: the complete "
                      "porous domain inside the inner frame, with the fixtures removed and the layers continued "
                      "across them and to the walls, floor and sand surface. Every filled cell is flagged "
                      "(fixture_filled, edge_filled); the analysis itself keeps the hidden sand hidden."]
    lines += [
        "",
        f"Size: {_gb(manifest['size'].get('bytes'))} in {manifest['size']['files']} files.",
        "",
        "## Unpacking",
        "",
        ("Download every file into one empty folder, then unpack "
         f"`{SERIES_ARCHIVE}` and `{EXPECTED_ARCHIVE}` there. "
         + ("The original staged inputs are preserved below `staged/`; use the reproduction launcher. "
            if manifest.get("staged_reproduction") else "The folder is then a DarSIA series folder. ") +
         ("The RAW archives may stay packed; the launcher copies the selected runs' exact inputs "
          "into its results folder before execution." if manifest.get("staged_reproduction") else
          "The raw archives may stay packed; the reproduction launcher unpacks one run at a time "
          "into its results folder.") if record == "R" else
         "Download every file into one empty folder, then unpack each `simulator_<run>.zip` there. "
         "Each run's geometry, mass maps and time series are under `products/<run>/`."),
        "",
    ]
    if record == "R":
        staged = manifest.get("staged_reproduction")
        if staged:
            lines += ["## Original staged RAW reproduction", "",
                "This package retains immutable original producer receipts, including their local paths "
                "and producer identifiers. Transport aliases bind those original identities to the "
                "packaged bytes; sealed configuration and calibration files are unchanged.", "",
                "The route runs the complete original RAW-to-native/phase/statistics stage, verifies "
                "the regenerated scientific payloads byte for byte, and runs original N1 with the saved "
                "calibration. It does not refit calibration. The statistics comparison uses exact "
                "decompressed bytes because the original gzip writer records a new container timestamp.", "",
                "For each stage listed below, prepare a clean checkout at its original commit and an "
                "environment with its exact Python patch version and locked libraries. Included bundles "
                "and wheels are verified export assets; the launcher does not install software.", ""]
            for key, stage in staged["stages"].items():
                software = stage["software"]
                lines.append(f"- `{key}`: commit `{software['code']['commit']}`, Python {software['python']}; "
                             f"inventory `{software['inventory']['path']}`.")
            lines += ["", "Select a local JSON runtime mapping with format `darsia-staged-runtimes-v1` "
                "and a `stages` object containing every listed stage ID. Each entry has `python` and "
                "`repository` absolute paths to the selected interpreter and checkout. Run "
                "`darsia-reproduce --dataset <folder> --results <empty-folder> --stage-runtimes <mapping.json>`, "
                "or choose that mapping in the launcher. Each stage verifies the actual runtime and original "
                "input catalogue before scientific execution. The complete selected frame order is required; "
                "quick prefixes and commit overrides are unavailable.", ""]
        software = manifest.get("software")
        if software is not None:
            lines += ["## Software", "",
                      f"Python {software['python']}. Exact code and library references are in "
                      "`software/versions.json`; the producing job's unchanged lock is in "
                      "`software/requirements-worker.lock`."]
            if software["mode"] == "include":
                lines += ["The package includes `software/source.bundle` and every locked library wheel "
                          "under `software/wheels/`. Clone the bundle into a new folder and check out the "
                          "published commit. Install the compatible wheels for the recorded Python using "
                          "the included lock. No software is installed by the export."]
            lines += [""]
        lines += [
            "## Reproducing the results",
            "",
            ("1. Prepare the original per-stage runtimes as described above. The launcher itself may run "
             "from a newer checkout; scientific stages use only their verified original code." if staged else
            f"1. Install DarSIA at commit `{code['published_commit']}` from a Git clone checked out at that "
            "commit (not a source archive: the launcher checks the code it runs against the commit) "
            f"with Python {(software or {}).get('python') or manifest['environment'].get('python') or '3'} and the pinned packages of "
            f"`{((software or {}).get('environment_lock') or manifest['environment'].get('lock') or {}).get('path', 'requirements-worker.lock')}`."),
            ("2. Run `darsia-reproduce` and choose the dataset, empty results folder and original stage runtimes, "
             "or run `darsia-reproduce --dataset <folder> --results <folder> --stage-runtimes <mapping.json>`. "
             "Options include `--runs`, `--workers`, `--threads` and `--backend cpu|cuda`." if staged else
            "2. Run `darsia-reproduce` and choose the unpacked folder and an empty results folder, or run "
            "`darsia-reproduce --dataset <folder> --results <folder>` "
            "(options: `--runs`, `--quick`, `--workers`, `--threads`, `--backend cpu|cuda`)."),
            "3. The launcher writes `report.html`, `report.json` and figures to the results folder.",
            "",
            ("The staged dataset remains read-only. Original inputs and fresh stage attempts are stored "
             "inside the selected results folder." if staged else
             "Alternatively, add the unpacked folder in DarSIA under Start > Add existing series folder. "
             "It opens read-only; results go to a location outside the dataset."),
            "",
            ("Measured P2 stage only (complete P1 plus P2 time is not recorded): " if staged else
             "Measured on the reference machine: ") + f"{_seconds(machine.get('seconds_per_frame'))} s per frame "
            f"with {machine.get('threads')} thread(s) per worker, backend {machine.get('backend')}, "
            f"precision {machine.get('precision')}; peak memory per worker {_gb(machine.get('peak_memory_bytes'))}.",
            "",
            "## Reproducibility",
            "",
            ("Staged RAW reproduction requires bit-identical final results. The following legacy limits "
             "are retained as diagnostics; meeting a tolerance does not satisfy this route." if staged else
             "A reproduction must be bit-identical on the reference machine. On other machines the following "
             "comparison limits apply per frame:"),
            "",
            f"- labels: the changed CO2 area is at most max({rule['labels']['co2_relative'] * 100:g} % of the "
            f"reference CO2 area, {rule['labels']['absolute_cm2']:g} cm²); the changed gas area at most "
            f"max({rule['labels']['gas_relative'] * 100:g} % of the reference gas area, "
            f"{rule['labels']['absolute_cm2']:g} cm²);",
            f"- masses and areas: |a − b| ≤ {rule['masses']['relative']:g}·|a| + "
            f"{rule['masses']['absolute']:g} (g or cm²);",
            f"- maps: per cell {rule['maps']['relative_to_frame_maximum']:g} × the frame's largest value, "
            f"with at most {rule['maps']['allowed_fraction_above'] * 100:g} % of the cells above; cells whose "
            "labels differ count under the label limit instead.",
            "",
            f"Quick check (the first frames of one run): {'valid' if manifest['quick_check_valid'] else 'not valid'} "
            "for this record.",
            "",
        ]
        variation = manifest.get("known_variation") or []
        if variation:
            lines += ["Known variation between machines:", ""]
            lines += [f"- {entry.get('summary', '')}" for entry in variation]
            lines += [""]
    layer = (manifest.get("expected") or manifest.get("simulator") or {}).get("layer") or {}
    maps = manifest.get("export_mass_maps")
    if maps is not None:
        lines += ["## Export mass maps", "",
                  f"The chosen export resolution is **{export_resolution(manifest)}**. "
                  f"The original stored results remain at {layer.get('resolution_mm', 'not recorded')} mm."]
        if maps.get("source") == "original stored layer":
            lines += ["The chosen resolution uses the original stored layer files listed in the manifest."]
        else:
            lines += ["Each frame also carries `export_<resolution>mm.npz` and its JSON metadata. "
                      "These attachments use the original extensive-cell reduction, with field units, "
                      "channel availability and conservation results recorded in the JSON."]
        if record == "S":
            lines += [f"The simulator geometry has its own grid spacing of "
                      f"{manifest['simulator']['resolution_mm']:g} mm; it is independent of the attached mass maps."]
        lines += [""]
    units = manifest.get("units") or {}
    lines += [
        "## Units and conventions",
        "",
        f"- Masses in {units.get('masses', 'g')}, areas in {units.get('areas', 'cm2')}, times in hours "
        "since the injection start.",
        f"- Stored grid: {layer.get('resolution_mm', 'n/a')} mm cells, {layer.get('shape', 'n/a')} "
        "(rows, columns); rows run upward in y, columns along x; the origin is the rig's metric origin.",
        "- Labels: " + ", ".join(f"{key} = {value}" for key, value in (units.get("label_legend") or {}).items())
        + ".",
        "",
    ]
    return "\n".join(lines)


def export_resolution(manifest):
    """The physical mass-map resolution, never an image subset or size policy."""
    maps = manifest.get("export_mass_maps") or {}
    layer = ((manifest.get("expected") or manifest.get("simulator") or {}).get("layer") or {})
    value = maps.get("resolution_mm", layer.get("resolution_mm"))
    return f"{value:g} mm" if isinstance(value, (int, float)) and not isinstance(value, bool) else "not recorded"
