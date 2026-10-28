"""Pure name-bound rebuilding of new run DATA from an adopted painting table.

The caller owns the exact current revision, source bytes, fresh destinations
and publication. Nothing here opens a path, changes a Store, starts a workflow
or edits a fitted model. Existing R5G6 pointer/lineage APIs remain separate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import tomllib

_MAPS = {"porosity_by_facies", "swi_by_facies"}
_SETS = {"unquantified_facies", "ignore_labels", "ignored_label_ids", "nonporous_label_ids"}
_BASIS_SETS = {"calibration_label_ids"}


@dataclass(frozen=True)
class PaintedConfigurationRebuild:
    """New DATA and explicit evidence; no publication or model validity claim."""
    document: dict
    changes: tuple[dict, ...]
    evidence: dict
    toml_bytes: bytes | None = None


def _integer(value):
    """Read a literal nonnegative category without numeric coercion."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    raise ValueError("A facies category must be a nonnegative integer")


def _require_established_frame_row(row, table, parent):
    """Recognize only the original automatic nonphysical frame-row record.

    Outside-pixel proof remains owned by the immutable adopted table. This
    pure recipe reader neither classifies an interior gap nor supplies a new
    physical value; an independently approved gap retains its separate mask.
    """
    from .layer_painting import OUTSIDE_FACIES, OUTSIDE_POROSITY

    try:
        number = _integer(row["facies_id"])
        matching = [item for item in table["provenance"]
                    if _integer(item["facies_id"]) == number]
        columns = table["columns"]
        if (len(matching) != 1 or not isinstance(columns, list)
                or any(type(column) is not str for column in columns)):
            raise ValueError("Malformed automatic frame-row provenance")
        provenance = matching[0]
        names = {column.strip().casefold(): column for column in columns}
        identity = names.get("id") or names["facies_id"]
        established = {identity, *(names[name] for name in
            ("name", "porosity", "permeability", "permeability_unit", "analysis_role", "enabled"))}
        fields = [[column, ("established:nonphysical-frame" if column in established
                            else "not-applicable:proven-nonphysical-frame")]
                  for column in columns]
        previous_ids = [identity for identity, _name, _value in parent]
        previous_ids.extend(_integer(item["facies_id"]) for item in table["provenance"]
                            if _integer(item["facies_id"]) != number)
        if (len(names) != len(columns) or not previous_ids
                or number != max(previous_ids) + 1
                or row["name"] != OUTSIDE_FACIES or provenance["name"] != OUTSIDE_FACIES
                or _integer(row["label_id"]) != 0 or provenance["layer_ids"] != [0]
                or table["layer_to_facies"].get("0") != number
                or provenance["parent_ids"] != [identity for identity, name, _value in parent
                                                 if name == OUTSIDE_FACIES]
                or provenance["fields"] != fields
                or row.get("enabled") is not False
                or type(row.get("porosity")) not in (int, float)
                or row["porosity"] != OUTSIDE_POROSITY or row["permeability"] is not None):
            raise ValueError("The established nonphysical frame row differs")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The adopted frame role lacks its original automatic nonphysical row") from error


def _recipe(adopted, parent_csv_bytes):
    """Join the existing strict parent reader to the adopted row/layer recipe."""
    from .adoption_facies_table import _parent_table
    from .layer_painting import ROLE_POROSITY
    contract = adopted["painted_adoption_table"]
    table = contract["table"]
    source = contract["source"]["csv"]
    columns, _, _, _, parent, digest = _parent_table(parent_csv_bytes, source["sha256"])
    if len(parent_csv_bytes) != source["size_bytes"]:
        raise ValueError("The supplied parent CSV size differs from the adopted binding")
    names, parents, groups = {}, {}, {}
    for row in table["provenance"]:
        name, number = row["name"], _integer(row["facies_id"])
        if name in names or number in names.values():
            raise ValueError("Adopted facies names and ids must be unique")
        names[name] = number
        parents[number] = tuple(_integer(value) for value in row["parent_ids"])
        groups[str(number)] = {"labels": []}
    aliases = {}
    for alias in table["aliases"]:
        target = names[alias["target_name"]]
        if _integer(alias["target_id"]) != target:
            raise ValueError("An explicit name alias differs from its adopted target")
        key = (_integer(alias["source_id"]), alias["source_name"])
        if key in aliases and aliases[key] != target:
            raise ValueError("An explicit source name alias is ambiguous")
        aliases[key] = target
    old_to_new = {}
    for old_id, name, _row in parent:
        target = aliases.get((old_id, name), names.get(name))
        if target is not None:
            if old_id not in parents[target] and (old_id, name) not in aliases:
                raise ValueError("The adopted row does not bind its parent name/id")
            old_to_new[old_id] = target
    declared = {}
    for row in table["provenance"]:
        for label in row["layer_ids"]:
            label = _integer(label)
            if label in declared:
                raise ValueError("An adopted layer occurs in more than one facies")
            declared[label] = _integer(row["facies_id"])
    layer_map = {_integer(key): _integer(value) for key, value in table["layer_to_facies"].items()}
    if layer_map != declared:
        raise ValueError("Adopted layer mapping differs from its row provenance")
    for label, number in sorted(layer_map.items()):
        groups[str(number)]["labels"].append(label)
    roles = {}
    for row in adopted["regions"]:
        number = _integer(row["facies_id"])
        if names.get(row["name"]) != number or layer_map.get(_integer(row["label_id"])) != number:
            raise ValueError("Adopted region name/id differs from its actual layer recipe")
        role = row["analysis_role"]
        if role == "frame":
            _require_established_frame_row(row, table, parent)
        elif role not in {"porous_medium", *ROLE_POROSITY}:
            raise ValueError("The adopted row has no existing analysis role")
        if number in roles and roles[number] != role:
            raise ValueError("Adopted layers disagree on their facies role")
        roles[number] = role
    if set(roles) != set(names.values()):
        raise ValueError("Adopted regions omit a declared facies row")
    return dict(names=names, parent_names={number: name for number, name, _ in parent},
        old_to_new=old_to_new, groups=groups, roles=roles,
        parent_csv=dict(source), parent_sha256=digest, columns=columns,
        table=deepcopy(table))


def rebuild_painted_facies_configuration(document, adopted_configuration, *, parent_csv_bytes):
    """Return new run DATA, preserving supplied numeric recipes by exact name.

    Native groups come from adopted LAYER ids, never the former segment PNG.
    Existing id-keyed physical maps retain their literal numeric types/values;
    duplicate parent names collapse only when those values agree. New nonporous
    rows enter existing ignored/unquantified declarations without invented phi
    or Swi. Fitted model descriptors and metadata are never rewritten.
    """
    if not isinstance(document, Mapping):
        raise ValueError("Rebuilding requires a decoded run configuration")
    recipe = _recipe(adopted_configuration, parent_csv_bytes)
    changes, dropped, missing_recipes = [], [], []
    nonporous = {number for number, role in recipe["roles"].items() if role != "porous_medium"}
    porous = set(recipe["roles"]) - nonporous
    reverse = {number: name for name, number in recipe["names"].items()}

    def numeric(values, path):
        """Carry a physical recipe only through exact parent names and typed equality."""
        if not isinstance(values, Mapping):
            raise ValueError(path + " must be an id-keyed mapping")
        result = {}
        for key, value in values.items():
            old = _integer(key)
            if old not in recipe["parent_names"] and old not in recipe["roles"]:
                raise ValueError(path + " has no pinned parent name for id " + str(old))
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(path + " requires explicit finite numerical values")
            number = (recipe["old_to_new"].get(old) if old in recipe["parent_names"] else old)
            if number is None:
                dropped.append(dict(path=path, source_id=old, source_name=recipe["parent_names"][old]))
                continue
            key = str(number)
            if key in result and (type(result[key]) is not type(value) or result[key] != value):
                raise ValueError(path + ": conflicting numerical recipes for facies " + reverse[number])
            result[key] = deepcopy(value)
        missing = porous - {int(key) for key in result}
        missing_recipes.extend(dict(path=path, facies_id=number, name=reverse[number])
                               for number in sorted(missing))
        return result

    def categories(values, path):
        """Keep selected named categories and apply the existing nonporous policy."""
        if not isinstance(values, (list, tuple)):
            raise ValueError(path + " must be an id list")
        result = []
        for value in values:
            old = _integer(value)
            if old not in recipe["parent_names"] and old not in recipe["roles"]:
                raise ValueError(path + " has no pinned parent name for id " + str(old))
            number = (recipe["old_to_new"].get(old) if old in recipe["parent_names"] else old)
            if number is not None and number not in result:
                result.append(number)
        result.extend(number for number in sorted(nonporous) if number not in result)
        return tuple(result) if isinstance(values, tuple) else result

    def walk(value, path=()):
        """Rebuild only declared run inputs, leaving fitted-model records opaque."""
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                own = (*path, key)
                # Fitted model payloads are historical source records, not new
                # run recipes. Their original identities remain caller-owned.
                if key in {"models", "metadata", "provenance"}:
                    result[key] = deepcopy(item)
                elif not path and key == "facies" and isinstance(item, Mapping):
                    result[key] = {k: deepcopy(v) for k, v in item.items() if k in {"path", "props"}}
                    result[key].update(deepcopy(recipe["groups"]))
                elif key in _MAPS:
                    result[key] = numeric(item, ".".join(map(str, own)))
                elif key in _SETS:
                    result[key] = categories(item, ".".join(map(str, own)))
                elif key in _BASIS_SETS:
                    # This is new configuration's requested basis, never an
                    # edit or validation of saved fitted model metadata.
                    if not isinstance(item, (list, tuple)):
                        raise ValueError("A declared calibration basis must be an id list")
                    ids = sorted(recipe["roles"])
                    result[key] = tuple(ids) if isinstance(item, tuple) else ids
                else:
                    result[key] = walk(item, own)
                if result[key] != item or type(result[key]) is not type(item):
                    if key == "facies" or key in _MAPS | _SETS | _BASIS_SETS:
                        changes.append(dict(path=list(own), before=deepcopy(item), after=deepcopy(result[key])))
            return result
        if isinstance(value, list):
            return [walk(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(walk(item, (*path, index)) for index, item in enumerate(value))
        return deepcopy(value)

    result = walk(document)
    return PaintedConfigurationRebuild(result, tuple(changes), dict(
        parent_csv=recipe["parent_csv"], adopted_table=recipe["table"],
        adopted_contract=deepcopy(adopted_configuration["painted_adoption_table"]),
        source_id_to_current={str(key): value for key, value in recipe["old_to_new"].items()},
        dropped_unused_recipes=dropped, missing_physical_recipes=missing_recipes,
        missing_recipe_admission="existing consumer owns required physical coverage; no values invented",
        native_grouping="adopted layer_to_facies; original segment groups replaced",
        numerical_values_reestimated=False, fitted_models_rewritten=False,
        fitted_model_eligibility_claimed=False,
        model_input_recipe=dict(label_ids=sorted(recipe["roles"]),
            calibration_label_ids=sorted(recipe["roles"]),
            ignore_labels=sorted(nonporous),
            facies_names={str(key): reverse[key] for key in sorted(reverse)}),
        model_input_recipe_is_validated_model=False))


def rebuild_painted_facies_toml(before, adopted_configuration, *, parent_csv_bytes):
    """Patch only affected TOML spans, retaining every other original byte.

    Normal table headers and multiline literal values are supported. A syntax
    that cannot be represented by this bounded patcher is reported before any
    write; callers can display that technical limitation without modifying the
    source. Parsed final DATA must equal the pure result including TOML types.
    """
    from .corrected_setup import _typed_equal
    if type(before) is not bytes:
        raise ValueError("Original TOML must be explicit bytes")
    document = tomllib.loads(before.decode("utf-8-sig"))
    rebuilt = rebuild_painted_facies_configuration(document, adopted_configuration,
                                                  parent_csv_bytes=parent_csv_bytes)
    if not rebuilt.changes:
        return PaintedConfigurationRebuild(rebuilt.document, (), rebuilt.evidence, before)
    bom = b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
    text = before[len(bom):].decode("utf-8")
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"

    def literal(value):
        """Encode only the rebuilt numeric maps and category-list literals."""
        if isinstance(value, Mapping):
            return "{ " + ", ".join(json.dumps(str(k)) + " = " + literal(v)
                                      for k, v in value.items()) + " }"
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def section(path, values):
        """Render one changed numeric table without serializing its neighbours."""
        header = "[" + ".".join(json.dumps(key) for key in path) + "]\n"
        return header + "".join(json.dumps(str(key)) + " = " + literal(value) + "\n"
                                for key, value in values.items()) + "\n"

    def header_path(line):
        """Read a literal table header through the same TOML parser."""
        if not re.match(r"^\s*\[[^\[]", line):
            return None
        try:
            value = tomllib.loads(line.rstrip("\r\n") + "\n__painted_probe__=true\n")
        except tomllib.TOMLDecodeError:
            return None
        path = []
        while isinstance(value, dict) and "__painted_probe__" not in value and len(value) == 1:
            key, value = next(iter(value.items())); path.append(key)
        return tuple(path) if isinstance(value, dict) and value.get("__painted_probe__") is True else None

    headers = [(index, header_path(line)) for index, line in enumerate(lines)]
    headers = [(index, path) for index, path in headers if path is not None]
    spans = [(index, next((j for j, _ in headers if j > index), len(lines)), path)
             for index, path in headers]
    edits, append = [], []
    for change in rebuilt.changes:
        path = tuple(change["path"])
        if any(not isinstance(key, str) for key in path):
            raise ValueError("A changed array-table recipe needs the caller's structured serializer")
        value = change["after"]
        if path == ("facies",):
            blocks = [span for span in spans if len(span[2]) == 2 and span[2][0] == "facies"]
            generated = "".join(section(("facies", key), row) for key, row in value.items()
                                if key not in {"path", "props"})
            if blocks:
                for index, end, _ in blocks:
                    edits.append((index, end, generated if index == blocks[0][0] else ""))
            else:
                append.append(generated)
            continue
        own = next((span for span in spans if span[2] == path), None)
        if own is not None and isinstance(value, Mapping):
            edits.append((own[0], own[1], section(path, value)))
            continue
        parent = path[:-1]
        parent_span = next((span for span in spans if span[2] == parent), None)
        start, end = (parent_span[0] + 1, parent_span[1]) if parent_span else (0, headers[0][0] if headers else len(lines))
        key = path[-1]
        candidates = [i for i in range(start, end) if re.match(r"^\s*(?:" + re.escape(key)
                     + r"|\"" + re.escape(key) + r"\"|'" + re.escape(key) + r"')\s*=", lines[i])]
        if len(candidates) != 1:
            raise ValueError("Cannot locate the exact TOML recipe span: " + ".".join(path))
        index = candidates[0]
        last = index + 1
        while last <= end:
            try:
                parsed = tomllib.loads("".join(lines[index:last]))
                if key in parsed:
                    break
            except tomllib.TOMLDecodeError:
                pass
            last += 1
        if last > end:
            raise ValueError("Cannot locate the complete TOML recipe value")
        edits.append((index, last, json.dumps(key) + " = " + literal(value) + "\n"))
    ordered = sorted(edits)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Changed TOML recipe spans overlap")
    for start, end, replacement in reversed(ordered):
        lines[start:end] = [replacement.replace("\n", newline)]
    if append:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines.append(newline)
        lines.extend(newline + value.replace("\n", newline) for value in append)
    after = bom + "".join(lines).encode("utf-8")
    if not _typed_equal(tomllib.loads(after.decode("utf-8-sig")), rebuilt.document):
        raise ValueError("TOML patch changed another value or type; original bytes retained")
    evidence = dict(rebuilt.evidence, source_sha256=hashlib.sha256(before).hexdigest(),
                    result_sha256=hashlib.sha256(after).hexdigest(), unrelated_byte_spans_retained=True)
    return PaintedConfigurationRebuild(rebuilt.document, rebuilt.changes, evidence, after)
