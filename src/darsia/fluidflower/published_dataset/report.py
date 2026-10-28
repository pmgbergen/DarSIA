"""The reproduction report: report.json, report.html and English figures.

The report shows the verdicts (frame, run, dataset), the largest differences
and the share of each limit used next to the dataset's known variation, the
first divergence, the settings and the environment differences. It names no
person, machine or user and holds no local path: the dataset is identified
by its manifest, the machine by its facts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path

from .comparison import IDENTICAL, OUTSIDE, WITHIN
from .readme import export_resolution

REPORT_FORMAT = "darsia-reproduction-report-v1"
FIGURES = "figures"
EXPECTED_COLOUR, REPRODUCED_COLOUR = "#2a78d6", "#eb6834"  # Categorical slots 1 and 2.
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
MASS_LABELS = (("total", "Total CO2"), ("aqueous", "Dissolved CO2"), ("gas", "Free gas"))


def report_document(*, manifest: dict, check: dict, settings: dict, summary: dict, frames: list,
                    machine: dict, timing: dict | None = None, differences: list | None = None) -> dict:
    """The report as one JSON document.

    ``check`` holds the code comparison and environment differences,
    ``settings`` the scope and machine settings, ``summary`` the verdicts,
    ``frames`` the per-frame comparisons (as dictionaries) and ``differences``
    the settings that differ from the published plan.
    """
    identity = manifest["identity"]
    reproduction = bool(check["code"].get("same"))
    return {
        "format": REPORT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"series_id": identity["series_id"], "title": identity["title"], "record": identity["record"],
                    "tier": identity.get("tier"), "mass_map_resolution": export_resolution(manifest),
                    "published_commit": manifest["code"]["published_commit"]},
        "reproduction": reproduction,
        "note": "" if reproduction else "Not a reproduction: another DarSIA commit or local changes ran.",
        "code": check["code"],
        "environment_differences": list(check.get("environment") or ()),
        "machine": machine,
        "settings": settings,
        "differences_from_published_plan": list(differences or ()),
        "rule": manifest["reproducibility"],
        "known_variation": manifest.get("known_variation") or [],
        "verdict": summary,
        "frames": frames,
        "timing": timing or {},
    }


def mass_rows(manifest: dict, comparisons: dict, reproduced: dict) -> dict:
    """Per run, the frames' time and expected and reproduced masses, in time order."""
    rows = {}
    for row in manifest["runs"]:
        run = row["run"]
        expected = manifest["expected"]["runs"].get(run, {}).get("frames", {})
        for frame in row["frames"]:
            case = frame["case_id"]
            if (run, case) not in comparisons:
                continue
            rows.setdefault(run, []).append({
                "case_id": case, "elapsed_h": frame.get("elapsed_h"),
                "expected": expected[case]["masses_g"], "reproduced": reproduced.get((run, case), {}),
                "verdict": comparisons[(run, case)]})
    return rows


def render_mass_figure(path: str | Path, run: str, rows: list[dict], rule: dict) -> Path:
    """One run's mass curves: reproduced against expected, with the limit band.

    Three panels (total, dissolved, free gas) share the time axis; the
    expected curve is a line with its limit band, the reproduction markers.
    """
    from matplotlib.figure import Figure  # The object API needs no global backend.

    times = [row["elapsed_h"] for row in rows]
    figure = Figure(figsize=(7.5, 6.8), facecolor=SURFACE)
    axes = figure.subplots(3, 1, sharex=True)
    relative, absolute = rule["masses"]["relative"], rule["masses"]["absolute"]
    for axis, (key, label) in zip(axes, MASS_LABELS):
        expected = [row["expected"].get(key) for row in rows]
        reproduced = [row["reproduced"].get(key) for row in rows]
        lower = [value - (relative * abs(value) + absolute) if value is not None else math.nan for value in expected]
        upper = [value + (relative * abs(value) + absolute) if value is not None else math.nan for value in expected]
        axis.set_facecolor(SURFACE)
        axis.fill_between(times, lower, upper, color=EXPECTED_COLOUR, alpha=0.18, linewidth=0,
                          label="Limit band")
        axis.plot(times, [math.nan if v is None else v for v in expected], color=EXPECTED_COLOUR, linewidth=2,
                  label="Expected")
        axis.plot(times, [math.nan if v is None else v for v in reproduced], linestyle="none", marker="o",
                  markersize=5, markerfacecolor=REPRODUCED_COLOUR, markeredgecolor=SURFACE, markeredgewidth=1,
                  label="Reproduced")
        axis.set_ylabel(f"{label} (g)", color=INK)
        axis.grid(True, color="#e2e1dc", linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color("#c3c2b7")
        axis.tick_params(colors=MUTED)
    axes[0].legend(loc="upper left", frameon=False, labelcolor=INK)
    axes[-1].set_xlabel("Time since injection start (h)", color=INK)
    figure.suptitle(f"Run {run}: CO2 mass, reproduced against expected", color=INK)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=110, facecolor=SURFACE)
    return target


def _escape(value) -> str:
    """Text for HTML."""
    return html.escape(str(value), quote=True)


def _share(value) -> str:
    """A share of a limit as a percentage."""
    return "n/a" if value is None else f"{100 * value:.3g} % of the limit"


def render_html(report: dict, figures: dict[str, str]) -> str:
    """A self-contained HTML page for the report; figures are files beside it."""
    verdict = report["verdict"]
    colours = {IDENTICAL: "#008300", WITHIN: "#2a78d6", OUTSIDE: "#e34948"}
    badge = lambda value: (f'<span class="badge" style="border-color:{colours.get(value, MUTED)}">'
                           f"{_escape(value)}</span>")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Reproduction report</title><style>",
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--rule:#e2e1dc}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3936}}",
        "body{background:var(--surface);color:var(--ink);font-family:system-ui,sans-serif;max-width:60rem;",
        "margin:0 auto;padding:1rem}table{border-collapse:collapse;width:100%;margin:.5rem 0 1rem}",
        "td,th{border-bottom:1px solid var(--rule);padding:.3rem .5rem;text-align:left;vertical-align:top}",
        ".muted{color:var(--muted)}.badge{border:2px solid;border-radius:4px;padding:0 .4rem;font-weight:600}",
        "img{max-width:100%;height:auto;background:#fcfcfb}</style></head><body>",
        f"<h1>Reproduction report: {_escape(report['dataset']['title'])}</h1>",
        f"<p>Series {_escape(report['dataset']['series_id'])}, record {_escape(report['dataset']['record'])}, "
        f"mass maps {_escape(report['dataset'].get('mass_map_resolution', 'not recorded'))}. "
        f"Dataset verdict: {badge(verdict['dataset'])}</p>",
    ]
    if report["note"]:
        parts.append(f"<p><strong>{_escape(report['note'])}</strong></p>")
    if report["settings"].get("route"):
        parts.append(f"<p>Route: {_escape(report['settings']['route'])} on this machine, without the job queue. "
                     "Reproduction runs are not qualified and not production.</p>")
    parts.append("<h2>Runs</h2><table><tr><th>Run</th><th>Verdict</th></tr>")
    parts += [f"<tr><td>{_escape(run)}</td><td>{badge(value)}</td></tr>" for run, value in verdict["runs"].items()]
    parts.append("</table>")
    staged_runs = (report.get("staged_reproduction") or {}).get("runs")
    if staged_runs:
        parts.append("<h2>Successful replay settings</h2><table><tr><th>Run</th><th>Backend</th>"
                     "<th>Threads per stage</th><th>Concurrent runs</th><th>Precision</th></tr>")
        for run, receipt in staged_runs.items():
            settings = receipt["settings"]
            parts.append("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in
                (run, settings["backend"], settings["threads"], settings["workers"], settings["precision"])) + "</tr>")
        parts.append("</table>")
    counts = verdict["frames"]
    parts.append(f"<p>Frames: {counts.get(IDENTICAL, 0)} identical, {counts.get(WITHIN, 0)} within expected "
                 f"variation, {counts.get(OUTSIDE, 0)} outside.</p>")
    parts.append("<h2>Margins</h2><table><tr><th>Quantity</th><th>Largest share of its limit</th></tr>")
    names = {"label": "Labels (changed CO2 and gas area)", "mass": "Masses", "map": "Maps (stored layer)"}
    parts += [f"<tr><td>{_escape(names.get(kind, kind))}</td><td>{_share(value)}</td></tr>"
              for kind, value in verdict["largest_share_of_limit"].items()]
    parts.append("</table>")
    if report["known_variation"]:
        parts.append("<h3>Known variation between machines</h3><ul>")
        parts += [f"<li>{_escape(entry.get('summary', json.dumps(entry)))}</li>" for entry in report["known_variation"]]
        parts.append("</ul>")
    first = verdict.get("first_divergence")
    if first:
        parts.append(f"<h2>First divergence</h2><p>Run {_escape(first['run'])}, frame {_escape(first['case_id'])}: "
                     f"{_escape(first['quantity'])} ({_escape(first.get('detail', ''))}).</p>")
    if report.get("differences_from_published_plan"):
        parts.append("<h2>Settings that differ from the published plan</h2><ul>")
        parts += [f"<li>{_escape(line)}</li>" for line in report["differences_from_published_plan"]]
        parts.append("</ul>")
    if report["environment_differences"]:
        parts.append("<h2>Environment differences</h2><ul>")
        parts += [f"<li>{_escape(line)}</li>" for line in report["environment_differences"]]
        parts.append("</ul>")
    parts.append("<h2>Settings</h2><table>")
    parts += [f"<tr><td>{_escape(key)}</td><td>{_escape(value)}</td></tr>" for key, value in report["settings"].items()]
    parts.append("</table><h2>Machine</h2><table>")
    parts += [f"<tr><td>{_escape(key)}</td><td>{_escape(value)}</td></tr>" for key, value in report["machine"].items()]
    parts.append("</table>")
    if figures:
        parts.append("<h2>Mass curves</h2>")
        parts += [f"<figure><img src='{_escape(name)}' alt='Mass curves of run {_escape(run)}, reproduced against "
                  f"expected, with the limit band'><figcaption class='muted'>Run {_escape(run)}</figcaption></figure>"
                  for run, name in figures.items()]
    requirement = ("Staged RAW reproduction requires bit-identical final results. Cross-machine limits "
                   "are diagnostic only; meeting a tolerance does not satisfy this route."
                   if verdict.get("bit_identical_required") else
                   "Same machine: bit-identical results are required. Other machines: the "
                   "cross-machine limits of the dataset's manifest apply.")
    parts.append("<p class='muted'>" + requirement + "</p></body></html>")
    return "\n".join(parts)


def write_report(folder: str | Path, report: dict, rows: dict) -> dict[str, Path]:
    """Write report.json, report.html and one mass figure per run into ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    figures = {}
    for run, run_rows in rows.items():
        name = f"{FIGURES}/mass_{run}.png"
        render_mass_figure(folder / name, run, run_rows, report["rule"]["other_machines"])
        figures[run] = name
    json_path = folder / "report.json"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    html_path = folder / "report.html"
    html_path.write_text(render_html(report, figures), encoding="utf-8")
    return {"json": json_path, "html": html_path, **{f"figure {run}": folder / name for run, name in figures.items()}}
