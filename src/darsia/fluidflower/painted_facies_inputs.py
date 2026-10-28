"""Check declared run-configuration ids against an explicit painted table.

Only the additive painted-adoption contract selects this check. It reports
missing ids before Apply or Build; it never rewrites inputs or interprets
retirement metadata as a separate reason to refuse a current table id.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

KEY = "painted_adoption_table"
_MAP_FIELDS = {"porosity_by_facies", "swi_by_facies"}
_LIST_FIELDS = {"unquantified_facies", "ignore_labels", "ignored_label_ids"}


def _integer(value, label):
    """Read an existing literal category without bool or float coercion."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    raise ValueError(label + " needs a nonnegative integer facies id")


def _current_ids(table):
    """Read the current row identities already recorded by the adopted table."""
    rows = table.get("provenance") if isinstance(table, Mapping) else None
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("The painted table lacks its adopted row provenance")
    return {_integer(row["facies_id"], "Adopted table row") for row in rows}


def _references(document, path=""):
    """Locate declared categories, keeping native segment-label values separate."""
    if not isinstance(document, Mapping):
        raise ValueError("Facies-id inspection needs a decoded run configuration")
    for key, value in document.items():
        own = path + "." + str(key) if path else str(key)
        if key == "facies" and isinstance(value, Mapping):
            for number in value:
                if number not in {"path", "props"}:
                    yield own + "." + str(number), _integer(number, own)
        elif key in _MAP_FIELDS:
            if not isinstance(value, Mapping):
                raise ValueError(own + " must be an explicit id-keyed mapping")
            for number in value:
                yield own + "." + str(number), _integer(number, own)
        elif key in _LIST_FIELDS:
            if not isinstance(value, (list, tuple)):
                raise ValueError(own + " must be an explicit facies-id list")
            for index, number in enumerate(value):
                yield own + f"[{index}]", _integer(number, own)
        if isinstance(value, Mapping):
            yield from _references(value, own)
        elif isinstance(value, (list, tuple)):
            for index, row in enumerate(value):
                if isinstance(row, Mapping):
                    yield from _references(row, own + f"[{index}]")


def id_keyed_input_problems(table, document, *, source="configuration"):
    """List exact source fields whose declared ids are absent from the table."""
    current = _current_ids(table)
    return tuple(
        f"{source}: {path} uses facies id {number} absent from the adopted table"
        for path, number in _references(document) if number not in current
    )


def require_current_facies_inputs(configuration, documents):
    """Refuse only missing ids for a new painted table; keep old calls idle."""
    if KEY not in configuration:
        return
    contract = configuration[KEY]
    table = contract.get("table") if isinstance(contract, Mapping) else None
    _current_ids(table)
    if not isinstance(documents, Mapping):
        raise ValueError("Named decoded run configurations are required")
    problems = []
    for source, document in documents.items():
        problems.extend(id_keyed_input_problems(table, document, source=str(source)))
    if problems:
        raise ValueError(
            "Rebuild the run configuration by facies name before Apply or Build: "
            + "; ".join(problems)
        )
