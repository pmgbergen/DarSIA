"""Non-destructive discovery of run folders and per-run source roles."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from darsia.fluidflower.image_formats import (
    ALL_IMAGE_SUFFIXES,
    DEFAULT_IMAGE_FILE_POLICY,
    PROCESSED_IMAGE_SUFFIXES,
    image_file_counts,
    missing_image_family,
    normalize_image_file_policy,
    select_preferred_images,
)

# Backward-compatible name for callers that explicitly expect display-ready files.
IMAGE_SUFFIXES = set(PROCESSED_IMAGE_SUFFIXES)
TABLE_SUFFIXES = {".csv", ".xls", ".xlsx"}
SOURCE_LAYOUT_FORMAT = "darsia-fluidflower-run-layout-v2"

ROLE_IGNORE = "ignore"
ROLE_INJECTION = "co2_injection_images"
ROLE_EXPERIMENT = "experiment_images"
ROLE_CLEARWATER = "clearwater_images"
ROLE_INDICATOR_REFERENCE = "indicator_reference_images"
ROLE_ALTERNATE_COLOR_REFERENCE = "alternate_color_reference_images"
ROLE_PROTOCOL_TABLE = "protocol_table"
ROLE_OVERFLOW_IMAGES = "overflow_images"
ROLE_OVERFLOW_TABLE = "overflow_table"
ROLE_OVERFLOW_COMBINED = "overflow_images_and_table"

_OVERFLOW_TOKENS = {"overflow", "scale", "weight", "vekt"}
_PROTOCOL_TOKENS = {
    "event",
    "events",
    "experimentalprotocol",
    "protocol",
    "protokoll",
}
_CLEARWATER_TOKENS = {
    "btb",
    "clear",
    "clearwater",
    "flush",
    "wash",
    "washing",
}
_INDICATOR_REFERENCE_TOKENS = {
    "buffer",
    "buffers",
    "indicatorcalibration",
    "indicatorreference",
    "ph",
    "phbuffer",
    "phseries",
    "titrationreference",
}
_ALTERNATE_COLOR_REFERENCE_TOKENS = {
    "alternate",
    "colorstate",
    "colorreference",
    "colourstate",
    "colourreference",
    "dye",
    "knowncolor",
    "knowncolour",
    "opticalreference",
    "rigcolor",
    "rigcolour",
}
_INJECTION_TOKENS = {
    "co2",
    "injection",
}
_EXPERIMENT_TOKENS = {
    "rest",
}
_GENERIC_SIGNATURE_TOKENS = {
    "ac",
    "data",
    "experiment",
    "experiments",
    "image",
    "images",
    "photo",
    "photos",
    "run",
}
_SIGNATURE_TOKEN_ALIASES = {
    "inject": "injection",
    "resting": "rest",
    "washing": "wash",
    "vekt": "weight",
}


def _natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def _shallow_files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    with os.scandir(root) as entries:
        paths = [
            Path(entry.path)
            for entry in entries
            if entry.is_file() and Path(entry.name).suffix.casefold() in suffixes
        ]
    return sorted(paths, key=lambda path: _natural_key(path.name))


def _shallow_images(
    root: Path, image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY
) -> list[Path]:
    return select_preferred_images(
        _shallow_files(root, set(ALL_IMAGE_SUFFIXES)), image_file_policy
    )


def _shallow_tables(root: Path) -> list[Path]:
    return [
        path
        for path in _shallow_files(root, TABLE_SUFFIXES)
        if not path.name.startswith("~$")
    ]


def _images(
    root: Path, image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY
) -> list[Path]:
    """Recursively list images for explicit legacy reference roots."""

    if not root.is_dir():
        return []
    candidates = sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in ALL_IMAGE_SUFFIXES
        ),
        key=lambda path: _natural_key(path.relative_to(root)),
    )
    by_folder: dict[Path, list[Path]] = {}
    for path in candidates:
        by_folder.setdefault(path.parent, []).append(path)
    return [
        path
        for folder in sorted(by_folder, key=lambda value: _natural_key(value))
        for path in select_preferred_images(by_folder[folder], image_file_policy)
    ]


def _candidate(
    reference_root: Path | None,
    run_id: str,
    image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> str:
    if reference_root is None or not reference_root.is_dir():
        return ""
    matching = reference_root / run_id
    candidates = (
        _images(matching, image_file_policy)
        if matching.is_dir()
        else _images(reference_root, image_file_policy)
    )
    return str(candidates[0]) if candidates else ""


def discover_run_candidates(experiment_root: Path) -> list[dict[str, Any]]:
    """Propose immediate child folders as runs without scanning their content."""

    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return [
        {
            "run_id": folder.name.strip().casefold(),
            "folder": str(folder.resolve()),
            "included": True,
        }
        for folder in sorted(root.iterdir(), key=lambda path: _natural_key(path.name))
        if folder.is_dir()
    ]


def _folder_tokens(relative_path: str) -> tuple[str, ...]:
    # Keep domain tokens such as ``co2`` intact while still removing run IDs below.
    raw_tokens = re.findall(r"[a-z]+(?:\d+)?|\d+", relative_path.casefold())
    tokens = [
        token
        for token in raw_tokens
        if not token.isdigit()
        and token not in _GENERIC_SIGNATURE_TOKENS
        and not re.fullmatch(r"(?:ac|run)\d+", token)
    ]
    return tuple(
        dict.fromkeys(_SIGNATURE_TOKEN_ALIASES.get(token, token) for token in tokens)
    )


def folder_signature(relative_path: str) -> str:
    """Return a run-independent signature for one relative source folder."""

    tokens = _folder_tokens(relative_path)
    return "_".join(tokens) or "."


def _suggest_role(
    relative_path: str,
    *,
    image_count: int,
    table_count: int,
    table_names: Sequence[str] = (),
) -> str:
    tokens = set(_folder_tokens(relative_path))
    if image_count and tokens & _OVERFLOW_TOKENS:
        return ROLE_OVERFLOW_IMAGES
    if image_count and tokens & _ALTERNATE_COLOR_REFERENCE_TOKENS:
        return ROLE_ALTERNATE_COLOR_REFERENCE
    if image_count and tokens & _INDICATOR_REFERENCE_TOKENS:
        return ROLE_INDICATOR_REFERENCE
    if image_count and tokens & _CLEARWATER_TOKENS:
        return ROLE_CLEARWATER
    if image_count and (tokens & _INJECTION_TOKENS or relative_path == "."):
        return ROLE_INJECTION
    if image_count and tokens & _EXPERIMENT_TOKENS:
        return ROLE_EXPERIMENT
    return ROLE_IGNORE


def suggest_table_role(path: str | Path) -> str:
    """Suggest one semantic role for an individual CSV/Excel file."""

    table = Path(path)
    tokens = set(_folder_tokens(table.stem)) | set(_folder_tokens(table.parent.name))
    has_overflow = bool(tokens & _OVERFLOW_TOKENS)
    has_protocol = bool(tokens & _PROTOCOL_TOKENS)
    if has_overflow and not has_protocol:
        return ROLE_OVERFLOW_TABLE
    if has_protocol and not has_overflow:
        return ROLE_PROTOCOL_TABLE
    return ROLE_IGNORE


def inspect_run_structure(
    run_root: Path,
    *,
    image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> list[dict[str, Any]]:
    """Describe source-bearing folders without reading or changing image content."""

    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    # A single os.walk pass avoids per-file network stats and repeated directory
    # scans, which is important for experiments stored on SMB/NAS shares.
    discovered: list[tuple[Path, list[Path], list[Path], list[Path]]] = []
    for current_root, directory_names, file_names in os.walk(root):
        directory_names.sort(key=_natural_key)
        folder = Path(current_root)
        image_candidates = sorted(
            (
                folder / name
                for name in file_names
                if Path(name).suffix.casefold() in ALL_IMAGE_SUFFIXES
            ),
            key=lambda path: _natural_key(path.name),
        )
        images = select_preferred_images(image_candidates, image_file_policy)
        tables = sorted(
            (
                folder / name
                for name in file_names
                if Path(name).suffix.casefold() in TABLE_SUFFIXES
                and not name.startswith("~$")
            ),
            key=lambda path: _natural_key(path.name),
        )
        # A folder whose images a strict policy rejects stays visible, with
        # its reason, instead of silently dropping out of the series.
        if images or tables or missing_image_family(image_candidates, image_file_policy):
            discovered.append((folder, images, tables, image_candidates))
    records: list[dict[str, Any]] = []
    for folder, images, tables, image_candidates in discovered:
        relative = "." if folder == root else folder.relative_to(root).as_posix()
        records.append(
            {
                "relative_path": relative,
                "absolute_path": str(folder),
                "image_count": len(images),
                "available_image_count": len(image_candidates),
                "image_file_counts": image_file_counts(image_candidates),
                "image_file_policy": normalize_image_file_policy(image_file_policy),
                "missing_image_family": missing_image_family(
                    image_candidates, image_file_policy
                ),
                "selected_image_suffixes": sorted(
                    {path.suffix.casefold() for path in images}
                ),
                "table_count": len(tables),
                "first_image": str(images[0]) if images else "",
                "last_image": str(images[-1]) if images else "",
                "tables": [str(path) for path in tables],
                "table_candidates": [
                    {
                        "path": str(path),
                        "relative_path": path.relative_to(root).as_posix(),
                        "folder_relative_path": relative,
                        "name": path.name,
                        "suffix": path.suffix.casefold(),
                        "signature": folder_signature(path.stem),
                        "suggested_role": suggest_table_role(path),
                    }
                    for path in tables
                ],
                "signature": folder_signature(relative),
                "suggested_role": _suggest_role(
                    relative,
                    image_count=len(images),
                    table_count=len(tables),
                    table_names=[path.name for path in tables],
                ),
            }
        )
    return sorted(
        records,
        key=lambda row: (
            str(row["relative_path"]) != ".",
            _natural_key(str(row["relative_path"])),
        ),
    )


def build_run_layout_template(
    example_run_root: Path,
    *,
    assignments: Mapping[str, str] | None = None,
    sequence_orders: Mapping[str, int] | None = None,
    selected_table_files: Mapping[str, str] | None = None,
    table_assignments: Mapping[str, str] | None = None,
    baseline_rule: str = "first_injection_image",
    clearwater_selector: str = "first",
    image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> dict[str, Any]:
    """Build a serializable folder-role template from one representative run."""

    if baseline_rule not in {"first_injection_image", "first_experiment_image"}:
        raise ValueError("Unsupported baseline-selection rule")
    if clearwater_selector not in {"first", "last"}:
        raise ValueError("Clear-water selector must be first or last")
    root = Path(example_run_root).expanduser().resolve()
    rows = inspect_run_structure(root, image_file_policy=image_file_policy)
    selected: list[dict[str, Any]] = []
    table_specs: list[dict[str, Any]] = []
    explicit = {str(key): str(value) for key, value in (assignments or {}).items()}
    explicit_orders = {
        str(key): int(value) for key, value in (sequence_orders or {}).items()
    }
    explicit_tables = {
        str(key): Path(str(value)).name
        for key, value in (selected_table_files or {}).items()
        if str(value).strip()
    }
    explicit_table_roles = {
        str(key).replace("\\", "/").casefold(): str(value)
        for key, value in (table_assignments or {}).items()
    }
    next_sequence_order = 1
    for row in rows:
        relative = str(row["relative_path"])
        role = explicit.get(relative, str(row["suggested_role"]))
        legacy_table_role = (
            role
            if role
            in {
                ROLE_PROTOCOL_TABLE,
                ROLE_OVERFLOW_TABLE,
                ROLE_OVERFLOW_COMBINED,
            }
            else ""
        )
        if legacy_table_role:
            role = ROLE_IGNORE
        # Ignore is an explicit, transferable folder role. Omitting it from the
        # template allowed automatic discovery to add the same folder back later.
        specification: dict[str, Any] = {
            "relative_path": relative,
            "signature": str(row["signature"]),
            "role": role,
        }
        if role in {ROLE_INJECTION, ROLE_EXPERIMENT}:
            specification["sequence_order"] = int(
                explicit_orders.get(relative, next_sequence_order)
            )
            next_sequence_order = max(
                next_sequence_order + 1,
                int(specification["sequence_order"]) + 1,
            )
        selected.append(specification)

        requested_legacy_table = explicit_tables.get(relative, "")
        for candidate in row.get("table_candidates") or []:
            table_relative = str(candidate.get("relative_path") or "").replace(
                "\\", "/"
            )
            absolute = str(candidate.get("path") or "")
            role_key = next(
                (
                    key
                    for key in (
                        table_relative.casefold(),
                        absolute.replace("\\", "/").casefold(),
                        Path(absolute).name.casefold(),
                    )
                    if key in explicit_table_roles
                ),
                "",
            )
            table_role = (
                explicit_table_roles[role_key]
                if role_key
                else str(candidate.get("suggested_role") or ROLE_IGNORE)
            )
            if legacy_table_role:
                requested_name = Path(requested_legacy_table).name.casefold()
                if (
                    not requested_name
                    or Path(absolute).name.casefold() == requested_name
                ):
                    table_role = (
                        ROLE_OVERFLOW_TABLE
                        if legacy_table_role == ROLE_OVERFLOW_COMBINED
                        else legacy_table_role
                    )
                elif requested_name:
                    table_role = ROLE_IGNORE
            if table_role not in {
                ROLE_IGNORE,
                ROLE_PROTOCOL_TABLE,
                ROLE_OVERFLOW_TABLE,
            }:
                table_role = ROLE_IGNORE
            table_specs.append(
                {
                    "relative_path": table_relative,
                    "folder_relative_path": relative,
                    "name": str(candidate.get("name") or Path(absolute).name),
                    "signature": str(candidate.get("signature") or "."),
                    "suffix": str(candidate.get("suffix") or ""),
                    "role": table_role,
                }
            )
    return {
        "schema": 2,
        "format": SOURCE_LAYOUT_FORMAT,
        "example_run_root": str(root),
        "example_run_name": root.name,
        "baseline_rule": baseline_rule,
        "clearwater_selector": clearwater_selector,
        "image_file_policy": normalize_image_file_policy(image_file_policy),
        "unmatched_folder_policy": "ignore",
        "unmatched_table_policy": "ignore",
        "folders": selected,
        "table_files": table_specs,
    }


def _signature_score(left: str, right: str) -> float:
    left_tokens = set(left.split("_")) - {"."}
    right_tokens = set(right.split("_")) - {"."}
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _sequence_path_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    relative = str(record.get("relative_path") or "")
    timestamp = re.search(r"(?<!\d)(\d{6,8})[_-](\d{6})(?!\d)", relative)
    if timestamp:
        return (0, int(timestamp.group(1)), int(timestamp.group(2)))
    return (1, _natural_key(relative))


def _normalise_sequence_orders(records: list[dict[str, Any]]) -> None:
    co2_records = [
        record
        for record in records
        if str(record.get("role") or "") in {ROLE_INJECTION, ROLE_EXPERIMENT}
    ]
    orders = [record.get("sequence_order") for record in co2_records]
    explicit = (
        bool(co2_records)
        and all(isinstance(order, int) and order > 0 for order in orders)
        and len(set(orders)) == len(orders)
    )
    if explicit:
        return
    for order, record in enumerate(
        sorted(co2_records, key=_sequence_path_key), start=1
    ):
        record["sequence_order"] = order


def _assigned_structure(
    run_root: Path,
    layout_template: Mapping[str, Any] | None,
    *,
    image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = inspect_run_structure(run_root, image_file_policy=image_file_policy)
    if not layout_template:
        assigned = [
            {**row, "role": str(row["suggested_role"]), "match": "automatic"}
            for row in records
        ]
        _assign_table_files(run_root, assigned, None)
        _normalise_sequence_orders(assigned)
        return assigned, []

    specifications = [
        dict(row)
        for row in layout_template.get("folders") or []
        if isinstance(row, Mapping)
    ]
    exact = {
        str(row.get("relative_path") or "").casefold(): (index, row)
        for index, row in enumerate(specifications)
    }
    assigned: list[dict[str, Any]] = []
    selection_warnings: list[str] = []
    ignore_unmatched = (
        str(layout_template.get("unmatched_folder_policy") or "automatic") == "ignore"
    )
    active_template_roles = {
        str(row.get("role") or ROLE_IGNORE)
        for row in specifications
        if str(row.get("role") or ROLE_IGNORE) != ROLE_IGNORE
    }
    matched_specs: set[int] = set()
    for record in records:
        relative = str(record["relative_path"])
        exact_match = exact.get(relative.casefold())
        specification = exact_match[1] if exact_match is not None else None
        match = "exact"
        if exact_match is not None:
            matched_specs.add(exact_match[0])
        if specification is None:
            available = [
                (index, row)
                for index, row in enumerate(specifications)
                if index not in matched_specs
            ] or list(enumerate(specifications))
            scored = [
                (
                    _signature_score(
                        str(record["signature"]), str(row.get("signature") or ".")
                    ),
                    index,
                    row,
                )
                for index, row in available
            ]
            score, index, best = max(scored, default=(0.0, -1, {}))
            if score >= 0.5:
                specification = best
                matched_specs.add(index)
                match = f"signature:{score:.2f}"
        selected_table = _match_selected_table(record, specification)
        if (
            specification is not None
            and specification.get("selected_table_name")
            and not selected_table
        ):
            selection_warnings.append(
                "Selected table file was not matched in "
                f"{relative}: {specification.get('selected_table_name')}"
            )
        suggested_role = str(record.get("suggested_role") or ROLE_IGNORE)
        ignore_extra = (
            specification is None
            and ignore_unmatched
            and suggested_role not in active_template_roles
        )
        if ignore_extra and suggested_role != ROLE_IGNORE:
            selection_warnings.append(f"Unassigned folder was ignored: {relative}")
        assigned.append(
            {
                **record,
                "role": (
                    str(specification.get("role") or ROLE_IGNORE)
                    if specification is not None
                    else ROLE_IGNORE if ignore_extra else suggested_role
                ),
                "match": (
                    match
                    if specification is not None
                    else (
                        "ignored-extra"
                        if ignore_extra and suggested_role != ROLE_IGNORE
                        else (
                            "automatic-compatible-extra"
                            if ignore_unmatched
                            and suggested_role in active_template_roles
                            else (
                                "automatic-extra"
                                if suggested_role != ROLE_IGNORE
                                else "unmatched"
                            )
                        )
                    )
                ),
                "sequence_order": (
                    int(specification["sequence_order"])
                    if specification is not None
                    and str(specification.get("sequence_order") or "").isdigit()
                    else None
                ),
                "selected_table": selected_table,
            }
        )
    warnings = [
        "Expected folder was not matched: "
        f"{specification.get('relative_path') or specification.get('signature')}"
        for index, specification in enumerate(specifications)
        if index not in matched_specs
        and str(specification.get("role") or ROLE_IGNORE) != ROLE_IGNORE
    ] + selection_warnings
    warnings.extend(_assign_table_files(run_root, assigned, layout_template))
    _normalise_sequence_orders(assigned)
    return assigned, warnings


def _match_selected_table(
    record: Mapping[str, Any], specification: Mapping[str, Any] | None
) -> str:
    """Resolve one explicitly selected table in the matched run folder."""

    tables = [Path(str(value)) for value in record.get("tables") or []]
    if not tables:
        return ""
    if specification is None:
        return str(tables[0]) if len(tables) == 1 else ""
    selected_name = str(specification.get("selected_table_name") or "")
    selected_signature = str(specification.get("selected_table_signature") or "")
    if not selected_name and not selected_signature:
        return str(tables[0]) if len(tables) == 1 else ""
    for path in tables:
        if path.name.casefold() == Path(selected_name).name.casefold():
            return str(path)
    if not selected_signature:
        selected_signature = folder_signature(Path(selected_name).stem)
    requested_suffix = Path(selected_name).suffix.casefold()
    scored = [
        (
            _signature_score(folder_signature(path.stem), selected_signature)
            + (
                0.25
                if requested_suffix and path.suffix.casefold() == requested_suffix
                else 0.0
            ),
            path,
        )
        for path in tables
    ]
    score, best = max(scored, default=(0.0, Path()), key=lambda value: value[0])
    return str(best) if score >= 0.5 else ""


def _assign_table_files(
    run_root: Path,
    records: list[dict[str, Any]],
    layout_template: Mapping[str, Any] | None,
) -> list[str]:
    """Attach an independently matched role to every discovered table file."""

    root = Path(run_root).expanduser().resolve()
    specifications = [
        dict(row)
        for row in (layout_template or {}).get("table_files") or []
        if isinstance(row, Mapping)
        and str(row.get("role") or "")
        in {ROLE_IGNORE, ROLE_PROTOCOL_TABLE, ROLE_OVERFLOW_TABLE}
    ]
    schema_two = bool(
        layout_template
        and (
            int((layout_template or {}).get("schema") or 1) >= 2
            or "table_files" in (layout_template or {})
        )
    )
    ignore_unmatched = (
        str((layout_template or {}).get("unmatched_table_policy") or "automatic")
        == "ignore"
    )

    candidates: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        for path_text in record.get("tables") or []:
            path = Path(str(path_text))
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                relative = path.name
            candidates.append(
                {
                    "record_index": record_index,
                    "path": path,
                    "relative_path": relative,
                    "signature": folder_signature(path.stem),
                    "folder_relative_path": str(record.get("relative_path") or "."),
                }
            )

    # Resolve all files together. A per-file greedy pass can consume the desired
    # overflow role with an older similarly named workbook before it reaches the
    # exact *_overflow_darsia file.
    ranked_pairs: list[tuple[float, int, int, str]] = []
    for candidate_index, candidate in enumerate(candidates):
        path = candidate["path"]
        relative = str(candidate["relative_path"])
        candidate_signature = str(candidate["signature"])
        for specification_index, specification in enumerate(specifications):
            specification_relative = str(
                specification.get("relative_path") or ""
            ).replace("\\", "/")
            exact_relative = relative.casefold() == specification_relative.casefold()
            score = (
                0.75
                * _signature_score(
                    candidate_signature,
                    str(specification.get("signature") or "."),
                )
                + 0.25
                * _signature_score(
                    folder_signature(str(candidate["folder_relative_path"])),
                    folder_signature(
                        str(specification.get("folder_relative_path") or ".")
                    ),
                )
                + (
                    0.10
                    if path.suffix.casefold()
                    == str(specification.get("suffix") or "").casefold()
                    else 0.0
                )
            )
            if exact_relative:
                score += 10.0
            elif candidate_signature == str(specification.get("signature") or "."):
                score += 2.0
            if score >= 0.5:
                ranked_pairs.append(
                    (
                        score,
                        candidate_index,
                        specification_index,
                        "exact" if exact_relative else f"signature:{score:.2f}",
                    )
                )

    candidate_matches: dict[int, tuple[Mapping[str, Any], str]] = {}
    matched_specs: set[int] = set()
    matched_candidates: set[int] = set()
    for _score, candidate_index, specification_index, match in sorted(
        ranked_pairs, reverse=True
    ):
        if (
            candidate_index in matched_candidates
            or specification_index in matched_specs
        ):
            continue
        matched_candidates.add(candidate_index)
        matched_specs.add(specification_index)
        candidate_matches[candidate_index] = (
            specifications[specification_index],
            match,
        )

    assigned_by_record: list[list[dict[str, Any]]] = [[] for _record in records]
    unexpected_warnings: list[str] = []
    for candidate_index, candidate in enumerate(candidates):
        path = candidate["path"]
        relative = str(candidate["relative_path"])
        candidate_signature = str(candidate["signature"])
        matched = candidate_matches.get(candidate_index)
        specification = matched[0] if matched is not None else None
        match = matched[1] if matched is not None else "unmatched"

        if specification is not None:
            role = str(specification.get("role") or ROLE_IGNORE)
        elif layout_template and not schema_two:
            record = records[int(candidate["record_index"])]
            folder_role = str(record.get("role") or ROLE_IGNORE)
            selected = str(record.get("selected_table") or "")
            selected_matches = not selected or Path(selected) == path
            if selected_matches and folder_role in {
                ROLE_PROTOCOL_TABLE,
                ROLE_OVERFLOW_TABLE,
                ROLE_OVERFLOW_COMBINED,
            }:
                role = (
                    ROLE_OVERFLOW_TABLE
                    if folder_role == ROLE_OVERFLOW_COMBINED
                    else folder_role
                )
                match = "legacy-folder-role"
            else:
                role = ROLE_IGNORE
        elif layout_template:
            suggested_role = suggest_table_role(path)
            role = ROLE_IGNORE if ignore_unmatched else suggested_role
            match = (
                "ignored-extra"
                if ignore_unmatched and suggested_role != ROLE_IGNORE
                else "automatic-extra" if suggested_role != ROLE_IGNORE else "unmatched"
            )
            if ignore_unmatched and suggested_role != ROLE_IGNORE:
                unexpected_warnings.append(
                    f"Unassigned table file was ignored: {relative}"
                )
        else:
            role = suggest_table_role(path)
            match = "automatic" if role != ROLE_IGNORE else "unmatched"
        assigned_by_record[int(candidate["record_index"])].append(
            {
                "path": str(path),
                "relative_path": relative,
                "folder_relative_path": str(candidate["folder_relative_path"]),
                "name": path.name,
                "suffix": path.suffix.casefold(),
                "signature": candidate_signature,
                "role": role,
                "match": match,
            }
        )
    for record, assigned_tables in zip(records, assigned_by_record, strict=True):
        record["table_files"] = assigned_tables

    return [
        "Expected table file was not matched: "
        f"{specification.get('relative_path') or specification.get('name')}"
        for index, specification in enumerate(specifications)
        if index not in matched_specs
        and str(specification.get("role") or ROLE_IGNORE) != ROLE_IGNORE
    ] + unexpected_warnings


def _role_in(role: str, accepted: Sequence[str]) -> bool:
    return role in set(accepted)


def _image_summary(
    records: Sequence[Mapping[str, Any]], roles: Sequence[str]
) -> tuple[int, str, str]:
    selected = sorted(
        [
            record
            for record in records
            if _role_in(str(record.get("role") or ""), roles)
            and int(record.get("image_count") or 0) > 0
        ],
        key=lambda record: (
            int(record.get("sequence_order") or 10**9),
            _sequence_path_key(record),
        ),
    )
    if not selected:
        return 0, "", ""
    return (
        sum(int(record.get("image_count") or 0) for record in selected),
        str(selected[0].get("first_image") or ""),
        str(selected[-1].get("last_image") or ""),
    )


def _select_folders(
    records: Sequence[Mapping[str, Any]], roles: Sequence[str]
) -> list[str]:
    selected = [
        record for record in records if _role_in(str(record.get("role") or ""), roles)
    ]
    if set(roles) & {ROLE_INJECTION, ROLE_EXPERIMENT}:
        selected.sort(
            key=lambda record: (
                int(record.get("sequence_order") or 10**9),
                _sequence_path_key(record),
            )
        )
    return [str(record["absolute_path"]) for record in selected]


def _overflow_tables(records: Sequence[Mapping[str, Any]]) -> list[Path]:
    table_rows = [
        dict(table)
        for record in records
        for table in record.get("table_files") or []
        if isinstance(table, Mapping)
    ]
    if table_rows:
        return sorted(
            {
                Path(str(table.get("path") or ""))
                for table in table_rows
                if str(table.get("role") or "") == ROLE_OVERFLOW_TABLE
                and str(table.get("path") or "")
            },
            key=lambda path: _natural_key(path),
        )
    selected: list[Path] = []
    for record in records:
        if _role_in(
            str(record.get("role") or ""),
            (ROLE_OVERFLOW_TABLE, ROLE_OVERFLOW_COMBINED),
        ):
            explicit = str(record.get("selected_table") or "")
            selected.extend(
                [Path(explicit)]
                if explicit
                else [Path(path) for path in record.get("tables") or []]
            )
    if not selected:
        selected = [
            Path(path)
            for record in records
            for path in record.get("tables") or []
            if set(_folder_tokens(Path(path).as_posix())) & _OVERFLOW_TOKENS
        ]
    return sorted(set(selected), key=lambda path: _natural_key(path))


def _protocol_tables(records: Sequence[Mapping[str, Any]]) -> list[Path]:
    table_rows = [
        dict(table)
        for record in records
        for table in record.get("table_files") or []
        if isinstance(table, Mapping)
    ]
    if table_rows:
        return sorted(
            {
                Path(str(table.get("path") or ""))
                for table in table_rows
                if str(table.get("role") or "") == ROLE_PROTOCOL_TABLE
                and str(table.get("path") or "")
            },
            key=lambda path: _natural_key(path),
        )
    selected: list[Path] = []
    for record in records:
        if str(record.get("role") or "") != ROLE_PROTOCOL_TABLE:
            continue
        explicit = str(record.get("selected_table") or "")
        selected.extend(
            [Path(explicit)]
            if explicit
            else [Path(path) for path in record.get("tables") or []]
        )
    if not selected:
        selected = [
            Path(path)
            for record in records
            for path in record.get("tables") or []
            if set(_folder_tokens(Path(path).stem)) & _PROTOCOL_TOKENS
        ]
    return sorted(set(selected), key=lambda path: _natural_key(path))


def discover_image_source_mapping(
    experiment_root: Path,
    *,
    structure: str = "run_subfolders",
    baseline_root: Path | None = None,
    clearwater_root: Path | None = None,
    flat_run_id: str = "run",
    layout_template: Mapping[str, Any] | None = None,
    run_definitions: Sequence[Mapping[str, Any]] | None = None,
    protocol_layout: str = "",
    image_file_policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> list[dict[str, Any]]:
    """Return editable per-run proposals; no source file is moved or modified."""

    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    image_file_policy = normalize_image_file_policy(image_file_policy)
    baseline = Path(baseline_root).expanduser().resolve() if baseline_root else None
    clearwater = (
        Path(clearwater_root).expanduser().resolve() if clearwater_root else None
    )
    if structure == "single_run":
        groups = [(str(flat_run_id).strip().casefold() or "run", root)]
    elif structure == "run_subfolders":
        if run_definitions:
            groups = [
                (
                    str(row.get("run_id") or Path(str(row.get("folder") or "")).name)
                    .strip()
                    .casefold(),
                    Path(str(row.get("folder") or "")).expanduser().resolve(),
                )
                for row in run_definitions
                if bool(row.get("included", True))
                and bool(str(row.get("folder") or "").strip())
                and Path(str(row.get("folder") or "")).expanduser().is_dir()
            ]
        else:
            groups = [
                (str(row["run_id"]), Path(str(row["folder"])))
                for row in discover_run_candidates(root)
            ]
        if not groups and run_definitions:
            return []
        if not groups:
            groups = [(str(flat_run_id).strip().casefold() or "run", root)]
    else:
        raise ValueError("Image structure must be single_run or run_subfolders")

    rows: list[dict[str, Any]] = []
    clearwater_selector = str(
        (layout_template or {}).get("clearwater_selector") or "first"
    )
    baseline_rule = str(
        (layout_template or {}).get("baseline_rule") or "first_injection_image"
    )
    for run_id, run_root in groups:
        assigned, warnings = _assigned_structure(
            run_root,
            layout_template,
            image_file_policy=image_file_policy,
        )
        # A strict image-file policy never falls back: name each used folder
        # that lacks the required kind of file.
        for record in assigned:
            reason = str(record.get("missing_image_family") or "")
            if reason and str(record.get("role") or ROLE_IGNORE) != ROLE_IGNORE:
                warnings.append(
                    f"{record.get('relative_path') or record.get('absolute_path')}: {reason}."
                )
        image_count, first_image, last_image = _image_summary(
            assigned, (ROLE_INJECTION, ROLE_EXPERIMENT)
        )
        _, first_injection_image, _ = _image_summary(assigned, (ROLE_INJECTION,))
        experiment_folders = _select_folders(
            assigned, (ROLE_INJECTION, ROLE_EXPERIMENT)
        )
        co2_records = sorted(
            (
                record
                for record in assigned
                if str(record.get("role") or "") in {ROLE_INJECTION, ROLE_EXPERIMENT}
            ),
            key=lambda record: int(record.get("sequence_order") or 10**9),
        )
        co2_sequences = [
            {
                "order": int(record.get("sequence_order") or index),
                "kind": (
                    "injection"
                    if str(record.get("role") or "") == ROLE_INJECTION
                    else "continuation"
                ),
                "folder": str(record.get("absolute_path") or ""),
                "image_count": int(record.get("image_count") or 0),
                "first_image": str(record.get("first_image") or ""),
                "last_image": str(record.get("last_image") or ""),
                "match": str(record.get("match") or ""),
            }
            for index, record in enumerate(co2_records, start=1)
        ]
        _, first_clearwater, last_clearwater = _image_summary(
            assigned, (ROLE_CLEARWATER,)
        )
        indicator_reference_count, _, _ = _image_summary(
            assigned, (ROLE_INDICATOR_REFERENCE,)
        )
        alternate_color_reference_count, _, _ = _image_summary(
            assigned, (ROLE_ALTERNATE_COLOR_REFERENCE,)
        )
        overflow_image_count, _, _ = _image_summary(
            assigned, (ROLE_OVERFLOW_IMAGES, ROLE_OVERFLOW_COMBINED)
        )
        overflow_folders = _select_folders(
            assigned, (ROLE_OVERFLOW_IMAGES, ROLE_OVERFLOW_COMBINED)
        )
        tables = _overflow_tables(assigned)
        protocol_tables = _protocol_tables(assigned)
        baseline_candidate = _candidate(baseline, run_id, image_file_policy) or (
            (
                first_injection_image
                if baseline_rule == "first_injection_image" and first_injection_image
                else first_image
            )
            if image_count
            else ""
        )
        clearwater_candidate = _candidate(clearwater, run_id, image_file_policy)
        if not clearwater_candidate and first_clearwater:
            clearwater_candidate = (
                last_clearwater if clearwater_selector == "last" else first_clearwater
            )
        if not image_count:
            warnings.append("No CO2 experiment-image sequence was matched.")
        elif baseline_rule == "first_injection_image" and not first_injection_image:
            warnings.append(
                "No folder was marked as a CO2 injection sequence; baseline fell "
                "back to the first CO2 continuation image."
            )
        if not clearwater_candidate:
            warnings.append("No clear-water candidate was found.")
        if protocol_layout == "per_run_files" and not protocol_tables:
            warnings.append("No per-experiment protocol workbook was found.")
        rows.append(
            {
                "run_id": run_id,
                "image_file_policy": image_file_policy,
                "image_file_counts": {
                    family: sum(
                        int((record.get("image_file_counts") or {}).get(family) or 0)
                        for record in assigned
                        if str(record.get("role") or "")
                        in {ROLE_INJECTION, ROLE_EXPERIMENT}
                    )
                    for family in ("processed", "raw")
                },
                "experiment_folder": str(run_root),
                "experiment_folders": experiment_folders,
                "co2_image_folders": experiment_folders,
                "co2_sequences": co2_sequences,
                "image_count": image_count,
                "first_image": first_image,
                "last_image": last_image,
                "baseline_candidate": baseline_candidate,
                "baseline_rule": baseline_rule,
                "clearwater_candidate": clearwater_candidate,
                "clearwater_folders": _select_folders(assigned, (ROLE_CLEARWATER,)),
                "clearwater_selector": clearwater_selector,
                "indicator_reference_folders": _select_folders(
                    assigned, (ROLE_INDICATOR_REFERENCE,)
                ),
                "indicator_reference_image_count": indicator_reference_count,
                "alternate_color_reference_folders": _select_folders(
                    assigned, (ROLE_ALTERNATE_COLOR_REFERENCE,)
                ),
                "alternate_color_reference_image_count": (
                    alternate_color_reference_count
                ),
                "protocol_tables": [str(path) for path in protocol_tables],
                "protocol_table": (str(protocol_tables[0]) if protocol_tables else ""),
                "overflow_image_folders": overflow_folders,
                "overflow_image_count": overflow_image_count,
                "overflow_tables": [str(path) for path in tables],
                "overflow_table": str(tables[0]) if tables else "",
                "layout_match": "template" if layout_template else "automatic",
                "source_folder_matches": [
                    {
                        "relative_path": str(record.get("relative_path") or ""),
                        "role": str(record.get("role") or ROLE_IGNORE),
                        "match": str(record.get("match") or ""),
                        "image_count": int(record.get("image_count") or 0),
                        "available_image_count": int(
                            record.get("available_image_count") or 0
                        ),
                        "image_file_counts": dict(
                            record.get("image_file_counts") or {}
                        ),
                        "selected_image_suffixes": list(
                            record.get("selected_image_suffixes") or []
                        ),
                        "table_count": int(record.get("table_count") or 0),
                        "selected_table": str(record.get("selected_table") or ""),
                        "sequence_order": record.get("sequence_order"),
                    }
                    for record in assigned
                ],
                "source_table_matches": [
                    {
                        "relative_path": str(table.get("relative_path") or ""),
                        "folder_relative_path": str(
                            table.get("folder_relative_path") or "."
                        ),
                        "path": str(table.get("path") or ""),
                        "role": str(table.get("role") or ROLE_IGNORE),
                        "match": str(table.get("match") or ""),
                    }
                    for record in assigned
                    for table in record.get("table_files") or []
                    if isinstance(table, Mapping)
                ],
                "discovery_warnings": warnings,
                "candidate_status": "requires_user_confirmation",
                "mapping_status": (
                    "ready_for_review" if image_count else "incomplete_source_mapping"
                ),
            }
        )
    return rows


__all__ = [
    "IMAGE_SUFFIXES",
    "ROLE_ALTERNATE_COLOR_REFERENCE",
    "ROLE_CLEARWATER",
    "ROLE_EXPERIMENT",
    "ROLE_IGNORE",
    "ROLE_INDICATOR_REFERENCE",
    "ROLE_INJECTION",
    "ROLE_OVERFLOW_COMBINED",
    "ROLE_OVERFLOW_IMAGES",
    "ROLE_OVERFLOW_TABLE",
    "ROLE_PROTOCOL_TABLE",
    "SOURCE_LAYOUT_FORMAT",
    "TABLE_SUFFIXES",
    "build_run_layout_template",
    "discover_image_source_mapping",
    "discover_run_candidates",
    "folder_signature",
    "inspect_run_structure",
    "suggest_table_role",
]
