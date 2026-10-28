"""Resolve saved storage through an immutable index and explicit collect receipts.

The original producer paths remain identities. A different machine never probes
them: it uses a published metadata copy, an explicitly supplied accessible alias,
or a SHA-bound publication/collection. No implicit collect, search or producer
execution takes place. The saved reader verifies payload bytes after resolution.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import storage_saved_products as saved
from .pipeline import products_publication as publication
from .pipeline.job_queue import name


def indexed_asset_resolver(index, *, current_machine_id, collections=(), source_paths=None):
    """Return a read-only resolver for exactly the supplied product-index closure.

    ``collections`` contains immutable collection.json or publication.json
    descriptors, never directories. ``source_paths`` explicitly maps product IDs
    to accessible aliases. Every resolved file is still checked by the caller's
    original descriptor; an alias does not change producer identity.
    """
    name(current_machine_id)
    root, document, rows = publication._read_index(index)
    aliases = dict(source_paths or {})
    if not aliases.keys() <= rows.keys():
        raise ValueError("Accessible aliases must name indexed product IDs")
    copies = {}
    for bound in collections:
        receipt = publication._bound(bound)
        bundle = publication._json(publication._verified_bytes(
            receipt["path"], receipt, limit=publication.MAX_INDEX_BYTES))
        if (bundle.get("format") not in {publication.BUNDLE_FORMAT, publication.COLLECTION_FORMAT}
                or publication._identity(publication._bound(bundle["products_index"]))
                != publication._identity(publication._bound(index))
                or bundle.get("run") != document["run"]
                or bundle.get("original_attempt_id") != document["attempt_id"]
                or bundle.get("file_writers_closed") is not True):
            raise ValueError("Collected storage differs from its original products index")
        for copy in bundle["copies"]:
            key = publication._relative(copy["product_id"])
            relative = publication._relative(copy["relative_path"])
            if key not in rows or key in copies or relative != "products/" + key:
                raise ValueError("Unknown, conflicting or escaping collected product")
            row = rows[key]
            if (publication._identity(copy) != publication._identity(row)
                    or copy.get("machine_id") != row["machine_id"]
                    or copy.get("local_path") != row["local_path"]):
                raise ValueError("Collected copy changes its producer identity")
            copies[key] = Path(receipt["path"]).parent.joinpath(*relative.split("/"))

    identities = {}
    for row in rows.values():
        key = row["local_path"], row["sha256"], row["bytes"]
        matches = identities.setdefault(key, [])
        if matches and any(other["machine_id"] != row["machine_id"] for other in matches):
            raise ValueError("One original asset has conflicting producer machines")
        matches.append(row)

    def resolve(bound):
        saved._descriptor(bound)
        key = bound["path"], bound["sha256"], bound["size_bytes"]
        matches = identities.get(key)
        if not matches:
            raise ValueError("Saved asset is outside the exact products-index closure")
        # Duplicate logical IDs can name the same original file (for example a
        # frame result and its producer inventory). Their byte identity is exact.
        for row in matches:
            if row["product_id"] in copies:
                return copies[row["product_id"]]
        for row in matches:
            copy = row.get("metadata_copy")
            if copy is not None:
                return root.joinpath(*copy["relative_path"].split("/"))
        for row in matches:
            if row["product_id"] in aliases:
                return Path(aliases[row["product_id"]])
        row = matches[0]
        if row["machine_id"] != current_machine_id:
            raise saved.StoredProductCollectionRequired(deepcopy(bound))
        return Path(row["local_path"])

    return resolve


def read_indexed_stored_run(index, *, current_machine_id, collections=(),
                            source_paths=None, load_before=False):
    """Read an original complete saved run using only admitted accessible files."""
    resolve = indexed_asset_resolver(index, current_machine_id=current_machine_id,
                                     collections=collections, source_paths=source_paths)
    completion = saved._json(publication.read_published_metadata(index))
    return saved.read_stored_run(completion, resolve_asset=resolve, load_before=load_before)


def read_indexed_phase_probabilities(index, case_id, *, current_machine_id,
                                     collections=(), source_paths=None):
    """Read one optional phase-one score pair from its exact published commit.

    This never runs inference or probes an old producer path on another host.
    The normal resolver still requires collection when a payload is remote.
    """
    import json
    from .mass.n1_pipeline import DataReader
    from .phase.probability_storage import read_product
    from .pipeline.staged_data import FRAME_FORMAT

    name(case_id)
    resolve = indexed_asset_resolver(index, current_machine_id=current_machine_id,
                                     collections=collections, source_paths=source_paths)
    _, document, rows = publication._read_index(index)
    prefix = "phase1/" + case_id + "/"

    def descriptor(leaf):
        row = rows[prefix + leaf]
        return dict(path=row["local_path"], sha256=row["sha256"], size_bytes=row["bytes"])

    commit = descriptor("phase1_commit.json")
    raw = publication._verified_bytes(resolve(commit), publication._bound(commit),
                                       limit=publication.MAX_METADATA_BYTES)
    record = json.loads(raw)
    roles = {key: descriptor(leaf) for key, leaf in (
        ("archive", "phase_probabilities.npz"), ("receipt", "phase_probabilities_receipt.json"))}
    if (record.get("format") != FRAME_FORMAT or record.get("complete") is not True
            or record.get("run") != document["run"] or record.get("case_id") != case_id
            or record.get("source", {}).get("run") != document["run"]
            or record.get("phase", {}).get("complete") is not True
            or record["phase"].get("source_sha256") != record["source"].get("source_sha256")
            or record.get("probabilities") != roles
            or any(record.get("files", {}).get(Path(bound["path"]).name) != bound
                   for bound in roles.values())):
        raise ValueError("Indexed probabilities differ from their actual phase-one commit")
    label = record["phase"]["labels"]
    if (label != descriptor("phase_labels.png")
            or record["files"].get(Path(label["path"]).name) != label):
        raise ValueError("Probability final labels differ from the exact commit inventory")
    aliases = [dict(logical=bound, physical=dict(bound, path=str(resolve(bound))))
               for bound in (*roles.values(), label)]
    return read_product(roles, reader=DataReader(aliases), source=record["source"], case_id=case_id)
