"""Read an explicitly approved chart mask already reflected in a saved Build.

The original chart-transfer Core must be integrated into this source checkout.
This reader neither loads another checkout nor creates a proposal or approval.
Chart confirmation is input provenance, not Build or scientific qualification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path


_ARRAYS = (
    "material_labels", "chart_mask", "effective_labels", "original_chart_mask",
    "vacated_mask", "unassigned_mask",
)


def read_chart(chart, *, reader, context):
    """Return the original approved native mask and its exact source bindings.

    ``reader`` is the existing bound DataReader (JSON and named NPZ archive
    readers); ``context`` is the already decoded BuildPhysicalContext. The
    three descriptors bind the saved proposal JSON, human approval JSON and
    original six result arrays. JSON formatting may differ from Core's payload
    formatting: only Core's own canonical encoder reconstructs that payload.
    """
    import numpy as np

    from ..mass.native_optical_source import _bound

    if not isinstance(chart, dict) or set(chart) not in ({"proposal", "approval", "arrays"}, {"preparation"}):
        raise ValueError("Chart inputs require the exact proposal, approval and arrays bindings")
    bindings = {key: deepcopy(_bound(value)) for key, value in chart.items()}

    # Do not source-load the experimental Core tree or replace its validation.
    # Absence is an explicit integration dependency, before any chart file read.
    try:
        from darsia.fluidflower.painting_chart_transfer import (
            ChartTransferProposal, _json, apply_chart_transfer,
        )
    except ImportError as error:
        raise ImportError(
            "Local chart inputs require the original painting_chart_transfer Core "
            "integrated into this source checkout"
        ) from error

    if "preparation" in bindings:
        preparation = reader.json(bindings["preparation"])
        row = preparation["chart_transfers"][context.run]
        metadata, approval = row["proposal"], row["approval"]
        artifact = row["artifact"]
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Chart artifact must remain inside its original preparation")
        bindings["arrays"] = dict(artifact, path=str(Path(bindings["preparation"]["path"]).parent / relative))
    else:
        metadata = reader.json(bindings["proposal"])
        approval = reader.json(bindings["approval"])
    if (not isinstance(metadata, dict) or metadata.get("run_id") != context.run
            or not isinstance(metadata.get("result"), dict)
            or set(metadata["result"]) != set(_ARRAYS)):
        raise ValueError("Chart proposal must bind this Build run and the original result arrays")
    arrays = reader.archive(bindings["arrays"], _ARRAYS)
    proposal = ChartTransferProposal(_json(metadata), arrays)
    applied = apply_chart_transfer(proposal, approval)

    labels = np.asarray(context.labels)
    effective = applied["effective_labels"]
    if (effective.ndim != 2 or not np.issubdtype(effective.dtype, np.integer)
            or effective.dtype != labels.dtype or effective.shape != labels.shape
            or not np.array_equal(effective, labels)):
        raise ValueError("Approved chart effective labels differ from the actual saved Build")
    # Equal categorical integer values with the same dtype are bit-exact; no
    # conversion, redraw, registration or physical-field reconstruction occurs.
    for key in ("chart_mask", "original_chart_mask", "vacated_mask", "unassigned_mask"):
        if applied[key].dtype != np.dtype(bool) or applied[key].shape != labels.shape:
            raise ValueError("Approved chart masks must be Boolean arrays on the saved Build grid")
    mask = applied["chart_mask"]
    chart_id = metadata.get("chart_id")
    if (type(chart_id) is not int or not 0 < chart_id < 256
            or not np.array_equal(mask, effective == chart_id)):
        raise ValueError("Approved chart occlusion differs from its original effective chart role")

    # Preserve the original reader's postread identity guards across the whole
    # join, including JSON files read before the archive.
    for bound in bindings.values():
        reader.path(bound)
    provenance = {
        "format": "darsia-local-build-chart-input-v1",
        "run": context.run,
        "inputs": bindings,
        "proposal_sha256": hashlib.sha256(proposal.payload).hexdigest(),
        "approval": deepcopy(approval),
        "source_binding": deepcopy(metadata["source_binding"]),
        "result": deepcopy(metadata["result"]),
        "physical_context": deepcopy(context.provenance),
    }
    return mask.copy(), provenance
