"""Passive counts from an already read manifest, never a job/scientific GO."""

from collections.abc import Mapping

from darsia.fluidflower.series_image_selection import FORMAT, _validate_saved_table


def saved_selection_summary(document):
    """Summarize only stored choices from an already read series manifest.

    This is an artefact status, never proof that protocol files are current or
    that review/analysis has run. The explicit resolver performs those checks.
    """
    allowed = (document.get("runs") or {}).get("ids")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(run, str) or not run for run in allowed)
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError("Choose a version-1 registered series with distinct runs.")
    table = document.get("image_selection")
    if table is None:
        table = {"format": FORMAT, "schema_version": 1, "runs": {}}
    _validate_saved_table(table, allowed)
    ready = {}
    for run, row in table["runs"].items():
        sources = row.get("source_metadata")
        if sources is not None:
            if (
                not isinstance(sources, list)
                or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("path"), str)
                    or not item["path"]
                    or not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in item["sha256"])
                    or type(item.get("size_bytes")) is not int
                    or item["size_bytes"] < 0
                    for item in sources
                )
                or len({item["path"].replace("\\", "/").casefold() for item in sources})
                != len(sources)
            ):
                raise ValueError(
                    "Saved image choices have malformed source metadata binding."
                )
        if row["selected"] and sources:
            ready[run] = len(row["selected"])
    missing = [run for run in allowed if run not in ready]
    count = sum(ready.values())
    return {
        "complete": not missing,
        "run_count": len(allowed),
        "saved_run_count": len(ready),
        "selected_frame_count": count,
        "missing_runs": missing,
        "summary": f"Stored explicit choices: {count} images in {len(ready)} of {len(allowed)} runs.",
        "note": "Load the run protocol to validate or edit choices; job creation must validate the saved source binding. This is not a computed result.",
        "saved_utc": max(
            (
                row.get("saved_utc", "")
                for row in table["runs"].values()
                if isinstance(row.get("saved_utc"), str)
            ),
            default="",
        ),
    }


def setup_selection_summary(document):
    """Count saved choices for the active Setup set, not analysis admission.

    Exclusions outside runs.ids describe runs already removed from this set.
    They cannot turn the remaining saved choices into missing choices. Only
    explicit, valid reasons may remove a still-registered run from this count.
    """
    stored = saved_selection_summary(document)
    ids = document["runs"]["ids"]
    names = {run.casefold(): run for run in ids}
    exclusions = document["runs"].get("excluded", {})
    active, outside = {}, {}
    if isinstance(exclusions, Mapping):
        for run, reason in exclusions.items():
            if isinstance(run, str) and run and isinstance(reason, str) and reason.strip():
                canonical = names.get(run.casefold())
                (active if canonical is not None else outside)[canonical or run] = reason
    eligible = [run for run in ids if run not in active]
    saved = [run for run in eligible if run not in stored["missing_runs"]]
    missing = [run for run in eligible if run not in saved]
    rows = document.get("image_selection", {}).get("runs", {})
    frames = sum(len(rows[run]["selected"]) for run in saved)
    summary = f"Stored explicit choices: {frames} images in {len(saved)} of {len(eligible)} active runs."
    if active:
        summary += f" {len(active)} registered run(s) explicitly excluded."
    if outside:
        summary += f" {len(outside)} recorded exclusion(s) are outside the active run list."
    return dict(
        stored,
        complete=bool(eligible) and not missing,
        saved_run_count=len(saved),
        selected_frame_count=frames,
        missing_runs=missing,
        eligible_count=len(eligible),
        excluded_count=len(active),
        excluded_runs=active,
        inactive_excluded_count=len(outside),
        inactive_excluded_runs=outside,
        summary=summary,
    )


def production_selection_note(document):
    """Describe a separately recorded production reference without opening it."""
    choice = document.get("production_image_selection")
    if not choice:
        return "Production grid: no separate saved grid is recorded."
    if isinstance(choice, Mapping) and isinstance(choice.get("selection_manifest"), Mapping):
        return (
            "Production grid: a separate saved-manifest reference is recorded. "
            "Load that grid to inspect and validate its sources; it is not counted above."
        )
    return "Production grid: the recorded reference is incomplete; it is not counted above."
