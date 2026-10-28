"""Pure, explicit facies-table derivation for a future painted-map adoption.

The caller must supply the session of the SHA-verified approved version. This
module does not establish that approval, read files, save a revision, or relax
the Store's existing imported-table contract.

Names match after stripping outer whitespace, with case and internal whitespace
kept exact. Porous facies carry extra cells verbatim by name; nonporous roles
record blank extra cells as not applicable. Numeric-looking text is never
converted, and column aliases such as ``swi CO2`` and ``swi`` remain distinct. Original header spelling/order is retained; absent painting-owned
columns are appended in the documented order below. Output is UTF-8 CSV with LF
record separators. The supplied parent bytes are never rewritten.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import math
from collections.abc import Mapping
from types import MappingProxyType

from .layer_painting import ROLE_POROSITY, PERMEABILITY_UNITS


MAX_PARENT_BYTES = 16 * 1024 * 1024
MAX_PARENT_ROWS = 100_000
NAME_POLICY = "strip-outer-whitespace; exact-case-and-internal-whitespace"
_OWNED = ("porosity", "permeability", "permeability_unit", "analysis_role", "enabled")


@dataclass(frozen=True)
class FaciesConflict:
    """The literal conflicting extra cells for one used porous facies."""
    name: str
    column: str
    values: tuple[tuple[int, str], ...]


class FaciesTableError(ValueError):
    """An actionable refusal; no partial table is returned."""

    def __init__(self, message, *, conflicts=(), unmatched_names=()):
        """Keep structured conflicts and missing names for a plain refusal."""
        super().__init__(message)
        self.conflicts = tuple(conflicts)
        self.unmatched_names = tuple(unmatched_names)


@dataclass(frozen=True)
class FaciesRowProvenance:
    """Join a derived row to drawn layers, original IDs and field origins."""
    name: str
    facies_id: int
    layer_ids: tuple[int, ...]
    parent_ids: tuple[int, ...]
    # (original/output column name, explicit source of its value).
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FaciesAlias:
    """An explicit role-authorized name alias preserving an established ID."""
    source_name: str
    target_name: str
    source_id: int
    target_id: int
    reason: str


@dataclass(frozen=True)
class AdoptionFaciesTable:
    """Immutable derived bytes and the complete stable-ID transition record."""
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]
    layer_to_facies: Mapping[int, int]
    facies_ids: Mapping[str, int]
    csv_bytes: bytes
    parent_sha256: str
    sha256: str
    provenance: tuple[FaciesRowProvenance, ...]
    name_policy: str = NAME_POLICY
    source_max_id: int = -1
    retired_source_ids: tuple[int, ...] = ()
    changed_source_ids: tuple[int, ...] = ()
    aliases: tuple[FaciesAlias, ...] = ()


def _name(value, *, where):
    """Normalize only outer whitespace while preserving literal name identity."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FaciesTableError(f"{where} needs a nonempty facies name")
    return value.strip()


def _parent_table(raw, expected_sha256):
    """Validate pinned parent bytes through the existing full-CSV reader."""
    if type(raw) is not bytes or len(raw) > MAX_PARENT_BYTES:
        raise FaciesTableError("Parent CSV must be bytes within the metadata limit")
    actual = hashlib.sha256(raw).hexdigest()
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or actual != expected_sha256):
        raise FaciesTableError("Parent CSV SHA-256 differs from its expected identity")
    from .geometry_raster_sources import _full_csv
    try:
        columns, values, ids = _full_csv(None, csv_bytes=raw, strict=True,
                                        max_rows=MAX_PARENT_ROWS)
        names = {column.strip().casefold(): column for column in columns}
        identity = names.get("id") or names["facies_id"]
        name_column = names["name"]
        rows = [(value, _name(row[name_column], where="Parent row"), row)
                for value, row in zip(ids, values)]
    except (UnicodeError, csv.Error, KeyError, ValueError) as error:
        raise FaciesTableError(f"Parent CSV is invalid: {error}") from error
    return columns, names, identity, name_column, rows, actual


def _number(value, *, where, positive=False):
    """Serialize a supplied painting property without falling back to old cells."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FaciesTableError(f"{where} needs an explicit finite{' positive' if positive else ''} value")
    try:
        number = float(value)
    except OverflowError:
        raise FaciesTableError(f"{where} exceeds the finite numeric range") from None
    if not math.isfinite(number) or (positive and number <= 0):
        raise FaciesTableError(f"{where} needs an explicit finite{' positive' if positive else ''} value")
    return repr(number)


def _painting_values(session, name, entry):
    """Reuse established role semantics and require explicit porous properties."""
    role = getattr(entry, "role", None)
    if not isinstance(role, str) or role not in {"porous_medium", *ROLE_POROSITY}:
        raise FaciesTableError(f"Facies {name!r} has an unsupported painting role {role!r}")
    if role == "porous_medium":
        porosity = getattr(entry, "porosity", None)
        porosity_text = _number(porosity, where=f"Facies {name!r} porosity", positive=True)
        if porosity > 1:
            raise FaciesTableError(f"Facies {name!r} porosity must be in (0, 1]")
        permeability = _number(getattr(entry, "permeability", None),
                               where=f"Facies {name!r} permeability", positive=True)
    else:
        # These are the existing painting roles' volume and exclusion semantics,
        # not a replacement for missing porous-medium properties.
        porosity_text, permeability = repr(ROLE_POROSITY[role]), ""
    unit = getattr(session, "permeability_unit", None)
    if not isinstance(unit, str) or unit not in PERMEABILITY_UNITS:
        raise FaciesTableError("The approved painting session needs its explicit supported permeability unit (darcy, mD, m2)")
    return dict(porosity=porosity_text, permeability=permeability,
                permeability_unit=unit, analysis_role=role,
                enabled="False" if role in {"excluded", "colorchecker"} else "True")


def _stable_ids(ordered, facies_names, values_by_name, by_name, old_rows):
    """Keep current name IDs and record the unambiguous renamed-water alias."""
    source_max = max((identity for identity, _, _ in old_rows), default=-1)
    ids, aliases, origins = {}, [], {}
    for original in ordered:
        name = facies_names[original]
        parents = by_name.get(name, ())
        if parents:
            ids[name] = min(identity for identity, _ in parents)
            origins[name] = "parent:lowest-id-by-exact-name"
    water = [original for original in ordered
             if values_by_name[original]["analysis_role"] == "free_water"
             and facies_names[original] not in ids]
    zero = [(identity, source_name) for identity, source_name, _ in old_rows if identity == 0]
    if (len(water) == 1 and zero and zero[0][1].casefold() == "water"
            and 0 not in ids.values()):
        original, = water
        name, source_name = facies_names[original], zero[0][1]
        ids[name], origins[name] = 0, "alias:explicit-free-water-role"
        aliases.append(FaciesAlias(source_name, name, 0, 0,
                                   "explicit-free-water-role-alias"))
    next_id = max(0, source_max, *ids.values()) + 1
    for original in ordered:
        name = facies_names[original]
        if name not in ids:
            ids[name], origins[name] = next_id, "new:above-source-maximum"
            next_id += 1
    return ids, origins, tuple(aliases), source_max


def build_adoption_facies_table(session, parent_csv_bytes, *, expected_sha256,
                                used_layer_ids):
    """Derive used painting rows with stable names/IDs and explicit blank cells.

    Porous facies require exact parent names for carried extra cells. Names
    unused by this painting do not block. Nonporous extra cells are blank and
    explicitly not applicable. Existing IDs use the lowest agreeing parent ID;
    new names allocate above the original maximum in saved painting order.
    An unambiguous renamed free-water facies retains the established water ID
    zero through a recorded name alias. Outside label zero remains a separate
    nonphysical row contract, allocated by the caller above every source ID.
    """
    columns, normalized, id_column, name_column, old_rows, parent_sha = _parent_table(
        parent_csv_bytes, expected_sha256)
    if (not isinstance(used_layer_ids, (list, tuple, set, frozenset))
            or not used_layer_ids or len(used_layer_ids) > 255
            or any(type(value) is not int or not 1 <= value <= 255 for value in used_layer_ids)):
        raise FaciesTableError("Used painting layer IDs must be explicit integers from 1 to 255; outside label 0 is separate")
    used = set(used_layer_ids)
    facies = getattr(session, "facies", None)
    layers = getattr(session, "layers", None)
    if (not isinstance(facies, Mapping) or not 0 < len(facies) <= 255
            or not isinstance(layers, (list, tuple)) or len(layers) > 255):
        raise FaciesTableError("The saved painting session needs explicit facies and layers")
    facies_names, folded = {}, set()
    for original in facies:
        name = _name(original, where="Painting facies")
        if name.casefold() in folded:
            raise FaciesTableError(f"Painting facies names are ambiguous after whitespace/case normalization: {name!r}")
        facies_names[original] = name
        folded.add(name.casefold())
    layer_names = {}
    for layer in layers:
        value, name = getattr(layer, "layer_id", None), getattr(layer, "facies", None)
        if type(value) is not int or not 1 <= value <= 255 or value in layer_names:
            raise FaciesTableError("Saved painting layer IDs must be distinct integers from 1 to 255")
        if not isinstance(name, str) or name not in facies_names:
            raise FaciesTableError(f"Painting layer {value} refers to missing facies {name!r}")
        layer_names[value] = name
    if used - layer_names.keys():
        raise FaciesTableError(f"Used layer IDs are absent from the saved painting: {sorted(used - layer_names.keys())}")
    selected = {layer_names[value] for value in used}
    ordered = [original for original in facies if original in selected]
    extra = [column for column in columns
             if column not in {id_column, name_column}
             and column.strip().casefold() not in _OWNED]
    by_name = {}
    for identity, name, row in old_rows:
        by_name.setdefault(name, []).append((identity, row))
    values_by_name = {original: _painting_values(session, facies_names[original], facies[original])
                      for original in ordered}
    porous = [original for original in ordered
              if values_by_name[original]["analysis_role"] == "porous_medium"]
    unmatched = [facies_names[original] for original in porous
                 if facies_names[original] not in by_name]
    if unmatched and extra:
        raise FaciesTableError(
            "No parent facies NAME match for " + ", ".join(repr(name) for name in unmatched)
            + "; required extra columns: " + ", ".join(repr(column) for column in extra)
            + ". Supply an explicit matching source; values will not be guessed.",
            unmatched_names=unmatched)
    conflicts = []
    for original in porous:
        name = facies_names[original]
        duplicates = by_name.get(name, ())
        for column in extra:
            values = tuple((identity, row[column]) for identity, row in duplicates)
            if len({value for _, value in values}) > 1:
                conflicts.append(FaciesConflict(name, column, values))
    if conflicts:
        detail = "; ".join(f"{item.name!r}, column {item.column!r}: "
                           + ", ".join(f"id {identity}={value!r}" for identity, value in item.values)
                           for item in conflicts)
        raise FaciesTableError("Conflicting parent rows for one facies name: " + detail,
                               conflicts=conflicts)
    facies_ids, id_origins, aliases, source_max = _stable_ids(
        ordered, facies_names, values_by_name, by_name, old_rows)
    output_columns = tuple([*columns, *(key for key in _OWNED if key not in normalized)])
    column_keys = {column: column.strip().casefold() for column in output_columns}
    rows, provenance = [], []
    for original in ordered:
        name = facies_names[original]
        facies_id, values = facies_ids[name], values_by_name[original]
        parents = by_name.get(name, [])
        parent_ids = tuple(identity for identity, _ in parents)
        alias_ids = [item.source_id for item in aliases if item.target_name == name]
        parent_ids = tuple(sorted(set(parent_ids) | set(alias_ids)))
        own_layers = tuple(sorted(value for value in used if layer_names[value] == original))
        row, fields = {}, []
        for column in output_columns:
            key = column_keys[column]
            if column == id_column:
                row[column], origin = str(facies_id), id_origins[name]
            elif column == name_column:
                row[column], origin = name, "painting:facies-name"
            elif key in values:
                row[column], origin = values[key], "painting:" + key
            elif values["analysis_role"] != "porous_medium":
                row[column], origin = "", "not-applicable:analysis-role-" + values["analysis_role"]
            else:
                row[column], origin = parents[0][1][column], "parent:exact-cell"
            fields.append((column, origin))
        rows.append(MappingProxyType(row))
        provenance.append(FaciesRowProvenance(name, facies_id, own_layers,
                                              parent_ids, tuple(fields)))
    layer_to_facies = {value: facies_ids[facies_names[layer_names[value]]]
                       for value in sorted(used)}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=output_columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = buffer.getvalue().encode("utf8")
    retired = tuple(sorted({identity for identity, _, _ in old_rows} - set(facies_ids.values())))
    return AdoptionFaciesTable(output_columns, tuple(rows),
        MappingProxyType(layer_to_facies), MappingProxyType(facies_ids), output,
        parent_sha, hashlib.sha256(output).hexdigest(), tuple(provenance),
        source_max_id=source_max, retired_source_ids=retired,
        changed_source_ids=(), aliases=aliases)
