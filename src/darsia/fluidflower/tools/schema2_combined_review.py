#!/usr/bin/env python3
"""Serve one live review UI for the matched FF_AC and FF_UM campaigns."""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import calibration_pairwise_review as review


CAMPAIGN_ID = "schema2_causal_phase_smoothed_ab_20260829"
DEFAULT_FFAC_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
DEFAULT_FFUM_ROOT = configured_tool_path('ffum', 'analysis_v2') / f'{CAMPAIGN_ID}'
DEFAULT_COMBINED_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}_combined_review'


def _read_manifest(path: Path) -> dict[str, Any]:
    from darsia.fluidflower.execution import _read_json_object
    return _read_json_object(path)


def _write_json_if_changed(path: Path, payload: Mapping[str, Any]) -> bool:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if path.is_file() and path.read_bytes() == encoded:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return True


def sync_combined_manifest(
    combined_root: Path,
    sources: Sequence[tuple[str, Path]],
    run_filters: Mapping[str, set[str]] | None = None,
    *,
    campaign_id: str = CAMPAIGN_ID,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    run_datasets: dict[str, str] = {}
    for dataset, source_root in sources:
        source_root = source_root.resolve()
        manifest_path = source_root / "postflight_manifest.json"
        row: dict[str, Any] = {
            "dataset": dataset,
            "root": str(source_root),
            "manifest": str(manifest_path),
            "available": manifest_path.is_file(),
            "job_count": 0,
        }
        if not manifest_path.is_file():
            source_rows.append(row)
            continue
        source_manifest = _read_manifest(manifest_path)
        source_jobs = source_manifest.get("jobs") or []
        if not isinstance(source_jobs, list):
            raise ValueError(f"Invalid jobs list in {manifest_path}")
        for raw_job in source_jobs:
            if not isinstance(raw_job, Mapping):
                continue
            job = dict(raw_job)
            run = str(job.get("run") or "").casefold()
            variant = str(job.get("variant") or "")
            if not run or not variant:
                continue
            selected_runs = (run_filters or {}).get(dataset)
            if selected_runs is not None and run not in selected_runs:
                continue
            previous_dataset = run_datasets.setdefault(run, dataset)
            if previous_dataset != dataset:
                raise RuntimeError(
                    f"Run identifier {run!r} occurs in both {previous_dataset} and {dataset}"
                )
            quality_flags = (
                dict(job.get("quality_flags") or {})
                if isinstance(job.get("quality_flags"), Mapping)
                else {}
            )
            quality_flags["dataset"] = dataset
            job.update(
                {
                    "dataset": dataset,
                    "quality_flags": quality_flags,
                    "candidate_folder": str(
                        source_root / "postflight" / variant / run
                    ),
                    "review_output_dir": str(
                        source_root / "manual_pairwise_review"
                    ),
                    "source_selection_dir": str(source_root),
                }
            )
            jobs.append(job)
        row["job_count"] = len(
            [job for job in jobs if job.get("dataset") == dataset]
        )
        row["selected_runs"] = sorted((run_filters or {}).get(dataset) or [])
        row["source_schema"] = source_manifest.get("schema")
        source_rows.append(row)

    jobs.sort(
        key=lambda item: (
            str(item.get("dataset") or ""),
            str(item.get("run") or ""),
            int(item.get("grid_order") or 999),
        )
    )
    payload = {
        "schema": 2,
        "campaign_id": str(campaign_id),
        "kind": "combined-cross-dataset-review-manifest",
        "sources": source_rows,
        "jobs": jobs,
    }
    manifest_path = combined_root / "postflight_manifest.json"
    _write_json_if_changed(manifest_path, payload)
    return payload


class ManifestWatcher:
    def __init__(
        self,
        combined_root: Path,
        sources: Sequence[tuple[str, Path]],
        interval: float,
        run_filters: Mapping[str, set[str]] | None = None,
        campaign_id: str = CAMPAIGN_ID,
    ) -> None:
        self.combined_root = combined_root
        self.sources = tuple(sources)
        self.interval = max(2.0, float(interval))
        self.run_filters = dict(run_filters or {})
        self.campaign_id = str(campaign_id)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._loop,
            name="ffac-ffum-review-manifest-watcher",
            daemon=True,
        )
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                sync_combined_manifest(
                    self.combined_root,
                    self.sources,
                    self.run_filters,
                    campaign_id=self.campaign_id,
                )
            except Exception as exc:  # pragma: no cover - live NAS safety net
                print(
                    f"Combined review manifest refresh failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval + 1.0)
            self.thread = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID)
    parser.add_argument("--ffac-root", type=Path, default=DEFAULT_FFAC_ROOT)
    parser.add_argument("--ffum-root", type=Path, default=DEFAULT_FFUM_ROOT)
    parser.add_argument("--combined-root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--blind-seed", default=review.DEFAULT_BLIND_SEED)
    parser.add_argument("--watch-interval", type=float, default=5.0)
    parser.add_argument("--ffac-runs", nargs="+")
    parser.add_argument("--ffum-runs", nargs="+")
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    combined_root = args.combined_root.resolve()
    sources = (
        ("ffac", args.ffac_root.resolve()),
        ("ffum", args.ffum_root.resolve()),
    )
    run_filters = {
        dataset: {str(run).casefold() for run in runs}
        for dataset, runs in (("ffac", args.ffac_runs), ("ffum", args.ffum_runs))
        if runs is not None
    }
    manifest = sync_combined_manifest(
        combined_root,
        sources,
        run_filters,
        campaign_id=args.campaign_id,
    )
    inventory = review.discover_review_inventory(
        combined_root,
        blind_seed=args.blind_seed,
        allow_incomplete=True,
    )
    output_dir = combined_root / "manual_pairwise_review"
    store = review.ReviewStore(output_dir, inventory.pairs)
    app = review.ReviewApp(
        inventory,
        store,
        default_reviewer=args.reviewer,
        selection_dir=combined_root,
        blind_seed=args.blind_seed,
        watch=True,
        prepare_new=False,
        watch_interval=args.watch_interval,
    )
    server = ThreadingHTTPServer((args.host, args.port), review.make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(
        json.dumps(
            {
                "url": url,
                "combined_root": str(combined_root),
                "source_count": len(manifest["sources"]),
                "job_count": len(manifest["jobs"]),
                "ready_runs": [pair.run for pair in inventory.pairs],
                "pending_runs": list(inventory.pending_runs),
                "decision_json": str(store.json_path),
            },
            indent=2,
        ),
        flush=True,
    )
    watcher = ManifestWatcher(
        combined_root,
        sources,
        args.watch_interval,
        run_filters,
        args.campaign_id,
    )
    watcher.start()
    app.start_live_updates()
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        app.stop_live_updates()
        server.server_close()


if __name__ == "__main__":
    main()
