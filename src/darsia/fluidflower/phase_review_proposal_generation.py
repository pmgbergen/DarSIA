"""Requested blind-review boxes from exact, already saved phase results.

No predictor, RAW decoder, fitting or map writer is called here. The original
phase uncertainty selector remains available for its exact three saved fields.
Final categorical maps use their own boundary/rule selector and saved adapters.
The mass route is visibly unavailable until an original provenance bridge proves
its exact native source. Missing inputs are never reconstructed from masks or
decision-used gas scores.

One frame is read at a time through the existing saved readers. Generated boxes
live beside the review decisions; source packages and answers are untouched.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

SOURCE_FORMAT = "saved-mass-fields-blind-proposal-source-v1"
PHASE_SOURCE_FORMAT = "saved-phase-probabilities-blind-proposal-source-v1"
CATEGORICAL_SOURCE_FORMAT = "saved-final-phase-categories-blind-proposal-source-v1"
POOL_FORMAT = "saved-fields-incremental-blind-proposals-v1"
PHASE_FIELDS = ("untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ProposalSourceUnavailable(ValueError):
    """The retained producer cannot use this exact saved source."""


class PhaseProbabilityFieldsUnavailable(ProposalSourceUnavailable):
    """No complete saved phase probability fields exist for this exact source."""


class ProposalRequestCancelled(RuntimeError):
    """The review window closed or cancelled its finite request."""


class ProposalPublicationGuard:
    """Linearize a request's revocation against its final atomic publication.

    Expensive saved-source checks run before this short critical section. The
    GUI can invalidate a request while those checks run, without waiting for
    image hashing or accessing widgets from the saved-reader worker.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._revoked = threading.Event()

    def invalidate(self):
        with self._lock:
            self._revoked.set()

    def is_cancelled(self):
        return self._revoked.is_set()

    @contextmanager
    def critical_section(self):
        with self._lock:
            _cancelled(self.is_cancelled)
            yield


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file(path):
    from .phase_saved_assets import asset
    return asset(Path(path).resolve())


def _descriptor(value, *, base=None):
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or not value["path"] or not isinstance(value.get("sha256"), str)
            or not _SHA.fullmatch(value["sha256"])):
        raise ProposalSourceUnavailable("Saved proposal inputs require an exact path and SHA256.")
    path = Path(value["path"])
    if not path.is_absolute():
        if base is None:
            raise ProposalSourceUnavailable("A relative saved input needs its bound package directory.")
        path = Path(base) / path
    from .review_storage import resolve_review_path
    return dict(path=str(resolve_review_path(path).resolve()), sha256=value["sha256"])


def _reader():
    from .phase_saved_assets import VerifiedAssetReader
    return VerifiedAssetReader()


def _producer():
    from .tools import calibration_pairwise_review as original
    return original


def _phase_producer():
    from . import phase_validation, phase_learning
    return phase_validation, phase_learning


def _cancelled(callback):
    if callback is not None and callback():
        raise ProposalRequestCancelled("Blind-box generation was cancelled; existing boxes and answers were retained.")


def _geometry(native):
    canvas, box = native.get("canvas_wh"), native.get("content_box_xywh")
    if (not isinstance(canvas, (tuple, list)) or len(canvas) != 2
            or not isinstance(box, (tuple, list)) or len(box) != 4
            or any(type(v) is not int for v in [*canvas, *box])):
        raise ProposalSourceUnavailable("The native review has no exact integer canvas and content box.")
    width, height = canvas
    x, y, w, h = box
    if min(width, height, w, h) <= 0 or min(x, y) < 0 or x + w > width or y + h > height:
        raise ProposalSourceUnavailable("The native content box falls outside its recorded canvas.")
    return list(canvas), list(box)


def _publication_roots(root, job):
    """Keep the selected package and original candidate map-root policy bound."""
    from .review_storage import resolve_review_path
    raw = Path(job.get("phase_map_output_dir") or job.get("review_output_dir") or root)
    selected = resolve_review_path(raw if raw.is_absolute() else Path(root) / raw).resolve()
    return list(dict.fromkeys((str(Path(root).resolve()), str(selected))))


def _require_publication_visible(roots, run):
    from .phase_blind_review_policy import require_predictions_visible
    for root in roots:
        require_predictions_visible(root, run)


def prepare_saved_mass_source(campaign_root, run_id, *, candidate_id, params_sha256,
                             mass_result, reader=None, cancelled=None):
    """Refuse a mass result that cannot prove this exact native phase source.

    The original mass-result contract records a provenance JSON, input-manifest
    digest and revision lineage. No retained bridge currently proves those
    revisions are this review's exact pipeline fingerprint and source frames.
    Candidate IDs, identical times/shapes and an explicit result descriptor are
    insufficient. Never label old mass fields with the current frame hashes.
    """
    _cancelled(cancelled)
    raise ProposalSourceUnavailable(
        "Saved mass fields cannot generate boxes for this review: the original "
        "mass provenance/input-manifest lineage has no verified bridge to this "
        "exact phase fingerprint and native source-image hashes. Matching run, "
        "candidate, times or grid is insufficient; no mass fields were relabelled.")


def prepare_saved_categorical_source(*args, **kwargs):
    """Bind the selected saved FINAL categories through their original admission."""
    from .phase_review_saved_categories import prepare_saved_categorical_source as prepare
    return prepare(*args, **kwargs)


def prepare_saved_phase_source(campaign_root, run_id, *, candidate_id, params_sha256,
                              model_id, reader=None, cancelled=None):
    """Bind the three original saved phase probabilities, without reconstructing them.

    An explicit ``phase_probability_fields`` receipt on each native frame must
    name this model/candidate/source and exact per-field descriptors. Receipts
    containing only decision-used gas scores or categorical masks cannot satisfy
    this contract. This adapter does not write such receipts or manufacture arrays.
    """
    from .phase_saved_runtime_views import native_source
    reader = reader or _reader()
    _cancelled(cancelled)
    root = Path(campaign_root).resolve()
    run = str(run_id).strip().casefold()
    if not run or not candidate_id or not model_id or not _SHA.fullmatch(str(params_sha256)):
        raise ProposalSourceUnavailable("An exact selected model, candidate, run and fingerprint are required.")
    campaign_ref = _file(root / "postflight_manifest.json")
    campaign = reader.json(campaign_ref)
    jobs = [row for row in campaign.get("jobs", ()) if isinstance(row, Mapping)
            and str(row.get("run", "")).casefold() == run and row.get("candidate_id") == candidate_id
            and row.get("params_sha256") == params_sha256]
    if len(jobs) != 1:
        raise ProposalSourceUnavailable("The selected phase candidate is not uniquely bound in this campaign.")
    job = deepcopy(dict(jobs[0]))
    publication_roots = _publication_roots(root, job)
    _require_publication_visible(publication_roots, run)
    for key in ("analysis_source_review_root", "review_output_dir"):
        if job.get(key) and not Path(job[key]).is_absolute():
            job[key] = str(root / job[key])
    _review_root, native_path, _unused = native_source(job)
    native_ref = _file(native_path)
    native = reader.json(native_ref)
    if str(native.get("run", "")).casefold() != run:
        raise ProposalSourceUnavailable("The selected native timeline belongs to another experiment.")
    canvas, box = _geometry(native)
    records = native.get("frame_sources")
    if not isinstance(records, list) or not records:
        raise ProposalSourceUnavailable("The selected review has no source-bound frame timeline.")
    frames = []
    for index, row in enumerate(records):
        _cancelled(cancelled)
        receipt = row.get("phase_probability_fields") if isinstance(row, Mapping) else None
        if not isinstance(receipt, Mapping):
            raise PhaseProbabilityFieldsUnavailable(
                f"Frame {index} lacks the saved untouched-water, aqueous-CO2 and free-gas probability fields "
                "required by the original phase-box producer. Categorical maps and two gas-decision scores "
                "cannot replace them. This saved package cannot generate phase uncertainty boxes.")
        if (receipt.get("candidate_id") != candidate_id or receipt.get("params_sha256") != params_sha256
                or receipt.get("model_id") != model_id or receipt.get("source_sha256") != row.get("source_sha256")
                or not _SHA.fullmatch(str(row.get("source_sha256", "")))
                or row.get("frame_index", index) != index or type(row.get("elapsed_h")) not in (int, float)
                or not math.isfinite(row["elapsed_h"]) or receipt.get("shape_hw") != [box[3], box[2]]
                or receipt.get("coordinate_space") != "native_review_content"):
            raise ProposalSourceUnavailable(f"Frame {index} probability receipt differs from the selected model, source or native grid.")
        declared = receipt.get("fields")
        if not isinstance(declared, Mapping) or set(declared) != set(PHASE_FIELDS):
            raise ProposalSourceUnavailable(f"Frame {index} does not bind all three original phase probabilities.")
        fields = {}
        for name in PHASE_FIELDS:
            _cancelled(cancelled)
            fields[name] = _descriptor(declared[name], base=Path(native_ref["path"]).parent)
            if declared[name].get("array_key") is not None:
                if not isinstance(declared[name]["array_key"], str) or not declared[name]["array_key"]:
                    raise ProposalSourceUnavailable(f"Frame {index} has an invalid saved probability array key.")
                fields[name]["array_key"] = declared[name]["array_key"]
            reader.path(fields[name])
        classes = (row.get("rig_domain_revision") or {}).get("classes")
        if not classes:
            raise ProposalSourceUnavailable(f"Frame {index} lacks its recorded porous-domain classes.")
        classes = _descriptor(classes, base=Path(native_ref["path"]).parent)
        reader.path(classes)
        frames.append(dict(frame_index=index, elapsed_h=row["elapsed_h"], source_sha256=row["source_sha256"],
                           source_name=row.get("source_name"), classes=classes, fields=fields))
    validation, learning = _phase_producer()
    source = dict(format=PHASE_SOURCE_FORMAT, run=run, candidate_id=candidate_id, params_sha256=params_sha256,
                  model_id=model_id, campaign=campaign_ref, native_manifest=native_ref, frames=frames,
                  canvas_wh=canvas, content_box_xywh=box, publication_roots=publication_roots,
                  producer=dict(module="darsia.fluidflower.phase_validation", function="_canvas_proposals",
                                source=_file(validation.__file__), algorithm_source=_file(learning.__file__),
                                proposal_schema=_producer().PHASE_QUIZ_PROPOSAL_SCHEMA),
                  scientific_maps_recomputed=False, source_arrays_substituted=False)
    source["source_sha256"] = _digest(source)
    return source


def pool_path(output_dir, run, candidate_id, params_sha256):
    """Use local review storage, never the candidate's immutable map package."""
    identity = _digest(dict(run=run, candidate_id=candidate_id, params_sha256=params_sha256))
    return Path(output_dir) / "generated_phase_quiz" / identity / "proposals.json"


def categorical_pool_path(output_dir, run, candidate_id, params_sha256):
    """Keep saved-map requests separate from historical probability pools."""
    identity = _digest(dict(run=run, candidate_id=candidate_id, params_sha256=params_sha256))
    return Path(output_dir) / "generated_phase_map_quiz" / identity / "proposals.json"


def _validate_source(source, reader=None, *, require_current_assets=True):
    if source.get("format") == CATEGORICAL_SOURCE_FORMAT:
        from .phase_review_saved_categories import validate_source
        return validate_source(source, reader=reader, require_current_assets=require_current_assets)
    content = dict(source)
    digest = content.pop("source_sha256", None)
    if (content.get("format") not in {SOURCE_FORMAT, PHASE_SOURCE_FORMAT} or not _SHA.fullmatch(str(digest))
            or _digest(content) != digest or content.get("scientific_maps_recomputed") is not False
            or content.get("source_arrays_substituted") is not False):
        raise ProposalSourceUnavailable("The saved proposal source binding changed.")
    phase_only = content["format"] == PHASE_SOURCE_FORMAT
    if require_current_assets and not phase_only:
        raise ProposalSourceUnavailable(
            "Mass-field box generation has no verified original provenance bridge to the exact phase fingerprint and native source-image hashes.")
    producer = source["producer"]
    module = "darsia.fluidflower.phase_validation" if phase_only else "darsia.fluidflower.tools.calibration_pairwise_review"
    expected = "_canvas_proposals" if phase_only else "_phase_quiz_proposals_from_fields"
    if (producer.get("module") != module or producer.get("function") != expected
            or producer.get("proposal_schema") != 2):
        raise ProposalSourceUnavailable("The recorded blind-box producer or proposal schema is unsupported.")
    # Historical answers must survive application updates and source migration.
    # Validate the complete recorded receipt without reading old paths or code.
    keys = ("campaign", "native_manifest", "mass_result") if not phase_only else ("campaign", "native_manifest")
    references = [source[key] for key in keys] + [producer["source"]]
    if phase_only:
        references.append(producer["algorithm_source"])
    references.extend(source.get("products", {}).values())
    for frame in source["frames"]:
        references.append(frame["classes"])
        references.extend(frame["fields"].values())
    for reference in references:
        if (not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str)
                or not reference["path"] or not _SHA.fullmatch(str(reference.get("sha256")))):
            raise ProposalSourceUnavailable("A recorded blind-box source asset lacks its exact path and SHA256.")
    if not require_current_assets:
        return
    reader = reader or _reader()
    for key in keys:
        reader.path(_descriptor(source[key]))
    manifest = reader.json(source["campaign"])
    jobs = [row for row in manifest.get("jobs", ()) if isinstance(row, Mapping)
            and str(row.get("run", "")).casefold() == source["run"]
            and row.get("candidate_id") == source["candidate_id"]
            and row.get("params_sha256") == source["params_sha256"]]
    if len(jobs) != 1 or source.get("publication_roots") != _publication_roots(Path(source["campaign"]["path"]).parent, jobs[0]):
        raise ProposalSourceUnavailable("The proposal source lost its exact candidate-owned publication policy binding.")
    _require_publication_visible(source["publication_roots"], source["run"])
    original = _phase_producer()[0] if phase_only else _producer()
    if _file(original.__file__)["sha256"] != producer["source"]["sha256"]:
        raise ProposalSourceUnavailable("The original box producer changed; retain this pool and prepare a new source binding.")
    if phase_only:
        if _file(_phase_producer()[1].__file__)["sha256"] != producer["algorithm_source"]["sha256"]:
            raise ProposalSourceUnavailable("The original phase-region selection algorithm changed.")


def _proposal_rows(rows, *, source):
    clean, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Saved proposals must be records.")
        identity = row.get("proposal_id")
        if (not isinstance(identity, str) or not identity or identity in seen
                or row.get("candidate_id") != source["candidate_id"]
                or row.get("params_sha256") != source["params_sha256"]
                or (row.get("run") or row.get("run_id")) != source["run"]
                or any(key in row for key in ("human_choice", "phase_class", "answered_utc"))):
            raise ValueError("A proposal is duplicated, pre-answered or belongs to another candidate.")
        seen.add(identity)
        clean.append(deepcopy(dict(row)))
    return clean


def _read_pool(path, source):
    raw = path.read_bytes() if path.exists() else None
    if raw is None:
        return None, None
    payload = json.loads(raw.decode("utf-8-sig"))
    if (payload.get("format") != POOL_FORMAT or payload.get("source_sha256") != source["source_sha256"]
            or payload.get("source") != source
            or payload.get("candidate_id") != source["candidate_id"]
            or payload.get("params_sha256") != source["params_sha256"] or payload.get("run") != source["run"]):
        raise ValueError("The existing generated pool has another source binding; no boxes were replaced.")
    _proposal_rows(payload["proposals"], source=source)
    content = dict(payload)
    digest = content.pop("content_sha256", None)
    if _digest(content) != digest:
        raise ValueError("The generated pool or its generation history changed; no boxes were replaced.")
    initial = payload.get("initial_proposal_count")
    if type(initial) is not int or not 0 <= initial <= len(payload["proposals"]):
        raise ValueError("The generated pool has an invalid initial offered-box count.")
    if source.get("format") == CATEGORICAL_SOURCE_FORMAT:
        count = payload.get("retained_prefix_count")
        if (type(count) is not int or not 0 <= count <= len(payload["proposals"])
                or _digest(payload["proposals"][:count]) != payload.get("retained_prefix_sha256")):
            raise ValueError("The categorical pool lost its unchanged original proposal prefix.")
        prior = payload.get("prior_probability_pool")
        if prior is not None and (not isinstance(prior, Mapping)
                or not _SHA.fullmatch(str(prior.get("sha256")))
                or not _SHA.fullmatch(str(prior.get("source_sha256")))):
            raise ValueError("The categorical pool lost its original probability-pool identity.")
    return raw, payload


def _recorded_pool(path, identity):
    """Read local historical boxes without opening their old scientific assets."""
    if not path.exists():
        return None, None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Generated blind boxes lack their exact saved source binding.")
    _validate_source(source, require_current_assets=False)
    if tuple(source.get(key) for key in ("run", "candidate_id", "params_sha256")) != identity:
        raise ValueError("Generated blind boxes belong to another selected candidate.")
    return _read_pool(path, source)


def _category_rows(source, frame, reader, rows, count):
    from .phase_review_saved_categories import load_frame
    from .phase_review_categorical_proposals import propose_saved_category_regions
    saved = load_frame(source, frame, reader=reader)
    canvas, box = source["canvas_wh"], source["content_box_xywh"]
    return propose_saved_category_regions(saved["labels"], domain=saved["domain"],
        rule_masks=saved["rule_masks"], run_id=source["run"], frame_index=frame["frame_index"],
        elapsed_h=frame["elapsed_h"], model_id=source["model_id"], candidate_id=source["candidate_id"],
        fingerprint=source["params_sha256"], canvas_wh=canvas, content_box_xywh=box, count=count,
        facies_labels=saved.get("facies"),
        existing_regions=_local_existing_regions(rows, frame["frame_index"], canvas, box)), saved["rule_coverage"]


@contextmanager
def _pool_write_lock(path):
    """Hold a nonblocking OS lock; process exit releases it, the file stays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name("." + path.name + ".write.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another review is saving blind boxes; retry after that writer finishes.") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_pool(path, payload, previous, *, publication_guard=None, validate=None,
                 dependency_paths=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with pending.open("xb") as stream:
            stream.write((json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        with ExitStack() as locks:
            # A categorical pool retains its original probability prefix. Hold
            # both namespace locks so a compliant old writer cannot change that
            # prefix between the last comparison and the new publication.
            for locked_path in sorted({Path(path), *(Path(item) for item in dependency_paths)}, key=str):
                locks.enter_context(_pool_write_lock(locked_path))
            current = path.read_bytes() if path.exists() else None
            if current != previous:
                raise RuntimeError("Another review changed the proposal pool while this request ran; no boxes were replaced.")
            if validate is not None:
                validate()
            # Only the final metadata replacement holds the revocation lock;
            # closing/reloading can cancel while saved-source checks run.
            guard = publication_guard.critical_section() if publication_guard is not None else nullcontext()
            with guard:
                current = path.read_bytes() if path.exists() else None
                if current != previous:
                    raise RuntimeError("Another review changed the proposal pool while this request ran; no boxes were replaced.")
                os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)  # Only this writer's unpublished temporary file.


def _probability_array(descriptor, reader):
    import numpy as np
    loaded = np.load(reader.path(descriptor), mmap_mode="r", allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            key = descriptor.get("array_key")
            if not key or key not in loaded.files:
                raise ProposalSourceUnavailable("A saved probability archive lacks its explicitly bound array key.")
            array = loaded[key]
    else:
        if descriptor.get("array_key") is not None:
            raise ProposalSourceUnavailable("A named probability array requires an NPZ archive.")
        array = loaded
    if (array.ndim != 2 or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 1)):
        raise ProposalSourceUnavailable("Saved phase probabilities must be finite floating-point fields in [0,1].")
    return array


def _local_existing_regions(rows, frame_index, canvas, box):
    """Undo only the original canvas placement for the retained exclusion API.

    Original phase rows retain integer pixel boxes; use those exact coordinates
    after checking their canvas placement. Older original proposal rows expose
    canvas bounds only, whose inverse is used without rounding or resampling.
    """
    width, height = canvas
    x, y, w, h = box
    regions = []
    for row in rows:
        if row.get("frame_index") != frame_index:
            continue
        if row.get("coordinate_space") != "normalized_native_review_canvas_16x9":
            raise ProposalSourceUnavailable("An existing box has no exact native canvas coordinates for exclusion.")
        try:
            values = [float(row[name]) for name in ("x", "y", "width", "height")]
        except (KeyError, TypeError, ValueError) as error:
            raise ProposalSourceUnavailable("An existing box lacks finite native canvas bounds.") from error
        if not all(math.isfinite(value) for value in values) or min(values[2:]) <= 0:
            raise ProposalSourceUnavailable("An existing box has invalid native canvas bounds.")
        local = [(values[0] * width - x) / w, (values[1] * height - y) / h,
                 values[2] * width / w, values[3] * height / h]
        pixels = row.get("bbox_xywh_px")
        if pixels is not None:
            if (not isinstance(pixels, (list, tuple)) or len(pixels) != 4
                    or any(type(value) is not int for value in pixels)
                    or min(pixels[:2]) < 0 or min(pixels[2:]) <= 0
                    or pixels[0] + pixels[2] > w or pixels[1] + pixels[3] > h):
                raise ProposalSourceUnavailable("An existing phase box has invalid original pixel coordinates.")
            exact = [pixels[0] / w, pixels[1] / h, pixels[2] / w, pixels[3] / h]
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(local, exact)):
                raise ProposalSourceUnavailable("An existing phase box differs from its original content-grid coordinates.")
            local = exact
        regions.append(dict(zip(("x", "y", "width", "height"), local)))
    return regions


def generate_more(source, output_dir, *, requested_count, actor, existing_proposals=(),
                  progress: Callable[[str], None] | None = None,
                  cancelled: Callable[[], bool] | None = None,
                  publication_guard=None, validate_request=None, initial_proposal_count=None):
    """Append saved-result boxes until the request is met or selection ends.

    Existing proposals are never edited, reranked or removed. The phase selector
    excludes existing native boxes and can revisit frames on later requests. The
    original UI batch selector decides which boxes to offer. The optional request
    validator reads session guards; this function never writes answers or the
    session. Cancellation/failure leaves the previous pool intact.
    """
    if type(requested_count) is not int or not 1 <= requested_count <= 1_000_000:
        raise ValueError("Request between 1 and 1,000,000 additional boxes.")
    actor = str(actor or "").strip()
    if not actor:
        raise ValueError("A reviewer name is required before requesting more blind boxes.")
    _cancelled(cancelled)
    if publication_guard is not None:
        _cancelled(publication_guard.is_cancelled)
    reader = _reader()
    categorical = source["format"] == CATEGORICAL_SOURCE_FORMAT
    # Preparation already admitted the categorical source. Check the recorded
    # identity here; each selected frame authenticates its bytes when read.
    # Full current admission and fresh asset checks still run under the writer
    # locks before publication, so another early whole-timeline pass adds no
    # protection for the boxes that can actually be saved.
    _validate_source(source, reader, require_current_assets=not categorical)
    identity = (source["run"], source["candidate_id"], source["params_sha256"])
    path = (categorical_pool_path if categorical else pool_path)(output_dir, *identity)
    previous, stored = _read_pool(path, source)
    original_rows = _proposal_rows(existing_proposals, source=source)
    retained_count = len(original_rows)
    if initial_proposal_count is not None and (type(initial_proposal_count) is not int
            or not 0 <= initial_proposal_count <= len(original_rows)):
        raise ValueError("The initial offered-box count differs from the original pool.")
    prior_path = pool_path(output_dir, *identity)
    prior_bytes, prior_pool = (None, None)
    if categorical:
        prior_bytes, prior_pool = _recorded_pool(prior_path, identity)
        if prior_pool is not None and prior_pool["source"]["format"] == CATEGORICAL_SOURCE_FORMAT:
            raise ValueError("The historical probability namespace contains a categorical pool.")
        prior_binding = (dict(sha256=hashlib.sha256(prior_bytes).hexdigest(),
                              source_sha256=prior_pool["source_sha256"]) if prior_bytes is not None else None)
        if stored is not None:
            if stored.get("prior_probability_pool") != prior_binding:
                raise ValueError("The original probability pool changed; no categorical boxes were replaced.")
        elif prior_pool is not None:
            prefix = prior_pool["proposals"]
            if original_rows[:len(prefix)] != prefix:
                raise ValueError("The saved probability pool is not the exact live proposal prefix.")
            prior_initial = prior_pool["initial_proposal_count"]
            if initial_proposal_count is not None and initial_proposal_count != prior_initial:
                raise ValueError("The initial offered-box count differs from the saved probability pool.")
            initial_proposal_count = prior_initial
    payload = deepcopy(stored) if stored is not None else dict(
        format=POOL_FORMAT, run=source["run"], candidate_id=source["candidate_id"],
        params_sha256=source["params_sha256"], native_manifest_sha256=source["native_manifest"]["sha256"],
        source_sha256=source["source_sha256"], source=deepcopy(source), proposals=original_rows,
        processed_frames=[], exhausted_frames=[], history=[], initial_proposal_count=len(original_rows),
        scientific_maps_recomputed=False, source_arrays_substituted=False)
    if stored is None and categorical:
        payload.update(retained_prefix_count=len(original_rows), retained_prefix_sha256=_digest(original_rows),
                       prior_probability_pool=prior_binding, rule_coverage={})
    if stored is None and initial_proposal_count is not None:
        payload["initial_proposal_count"] = initial_proposal_count
    rows = payload["proposals"]
    if categorical and original_rows != rows[:len(original_rows)]:
        raise ValueError("The live categorical proposal prefix changed; existing boxes were preserved.")
    by_id = {row["proposal_id"]: row for row in rows}
    for row in original_rows:
        if row["proposal_id"] not in by_id or by_id[row["proposal_id"]] != row:
            raise ValueError("The live proposal list changed independently; existing boxes were preserved.")
    processed = set(payload["processed_frames"])
    exhausted = set(payload["exhausted_frames"])
    indices = [row["frame_index"] for row in source["frames"]]
    if (len(processed) != len(payload["processed_frames"]) or not processed.issubset(indices)
            or len(exhausted) != len(payload["exhausted_frames"]) or not exhausted.issubset(processed)):
        raise ValueError("Generated-frame progress does not match the pinned timeline.")
    added, visited = 0, []
    producer = None if categorical else _phase_producer()[0]._canvas_proposals
    for frame in source["frames"]:
        index = frame["frame_index"]
        if index in exhausted:
            continue
        _cancelled(cancelled)
        if publication_guard is not None:
            _cancelled(publication_guard.is_cancelled)
        if progress:
            progress(f"Preparing blind boxes from saved frame {index + 1}; {added} new boxes")
        count = requested_count - added
        if categorical:
            new, coverage = _category_rows(source, frame, reader, rows, count)
            payload["rule_coverage"][str(index)] = coverage
        else:
            new = _probability_rows(source, frame, reader, rows, count, producer)
        if len(new) < count:
            exhausted.add(index)
        for row in _proposal_rows(new, source=source):
            row["source_name"] = frame["source_name"]
            row["source_sha256"] = frame["source_sha256"]
            proposal_id = row["proposal_id"]
            if proposal_id in by_id:
                # An existing box retains its original metadata and stable ID.
                continue
            rows.append(row)
            by_id[proposal_id] = row
            added += 1
        processed.add(index)
        visited.append(index)
        if added >= requested_count:
            break
    _cancelled(cancelled)

    def validate_publication():
        # Recheck inside the OS publication lock, before taking the short
        # revocation lock. No GUI object is consulted by this worker callback.
        _cancelled(cancelled)
        if publication_guard is not None:
            _cancelled(publication_guard.is_cancelled)
        final_reader = _reader()
        _validate_source(source, final_reader)
        if not categorical:
            for frame in source["frames"]:
                if frame["frame_index"] in processed:
                    final_reader.path(frame["classes"])
                    for descriptor in frame["fields"].values():
                        final_reader.path(descriptor)
        if validate_request is not None:
            validate_request()
        if categorical:
            current_prior = prior_path.read_bytes() if prior_path.exists() else None
            if current_prior != prior_bytes:
                raise RuntimeError("The original probability pool changed during generation; existing boxes were preserved.")
        _cancelled(cancelled)

    if not visited and stored is not None:
        validate_publication()
        if publication_guard is not None:
            _cancelled(publication_guard.is_cancelled)
        return dict(path=str(path), payload=stored, added_count=0, exhausted=True,
                    message="No additional eligible boxes remain on these exact saved frames.")
    payload["processed_frames"] = [index for index in indices if index in processed]
    payload["exhausted_frames"] = [index for index in indices if index in exhausted]
    payload["exhausted"] = len(exhausted) == len(indices)
    if stored is None and retained_count == 0 and initial_proposal_count is None:
        payload["initial_proposal_count"] = min(len(rows), _producer().PHASE_QUIZ_INITIAL_PROPOSALS_PER_CANDIDATE)
    payload["history"].append(dict(action="requested more blind validation boxes", person=actor,
                                   recorded_utc=datetime.now(timezone.utc).isoformat(),
                                   requested_count=requested_count, added_count=added,
                                   processed_frames=visited, prior_pool_sha256=hashlib.sha256(previous).hexdigest() if previous else None))
    payload.pop("content_sha256", None)
    payload["content_sha256"] = _digest(payload)
    _cancelled(cancelled)
    if publication_guard is not None:
        _cancelled(publication_guard.is_cancelled)
    _atomic_pool(path, payload, previous, publication_guard=publication_guard, validate=validate_publication,
                 dependency_paths=(prior_path,) if categorical else ())
    coverage_message = ""
    if categorical:
        coverage = payload["rule_coverage"]
        if any(item.get("missing") or not item.get("available", False) for item in coverage.values()):
            coverage_message = " Saved rule-change evidence is incomplete for some frames; final boundaries and the available recorded changes remain usable."
    return dict(path=str(path), payload=payload, added_count=added, exhausted=payload["exhausted"],
                message=f"Added {added} boxes from {len(visited)} exact saved frames."
                        + (" No additional eligible boxes remain." if payload["exhausted"] else "") + coverage_message)


def _probability_rows(source, frame, reader, rows, count, producer):
    """Preserve the original saved-probability selector and exact placement."""
    import numpy as np
    canvas, box = source["canvas_wh"], source["content_box_xywh"]
    index = frame["frame_index"]
    arrays = {name: _probability_array(frame["fields"][name], reader) for name in PHASE_FIELDS}
    if any(list(array.shape) != [box[3], box[2]] for array in arrays.values()):
        raise ProposalSourceUnavailable(f"Frame {index} probability arrays differ from the original native content shape.")
    with np.load(reader.path(frame["classes"]), allow_pickle=False) as archive:
        if "rig_domain" not in archive.files:
            raise ProposalSourceUnavailable(f"Frame {index} has no saved rig_domain array.")
        mask = np.asarray(archive["rig_domain"])
        facies = np.asarray(archive["facies"]) if "facies" in archive.files else None
    if mask.dtype != np.bool_:
        raise ProposalSourceUnavailable(f"Frame {index} porous-domain mask is not explicitly boolean.")
    if list(mask.shape) != [canvas[1], canvas[0]]:
        raise ProposalSourceUnavailable(f"Frame {index} porous-domain mask differs from the native canvas; no warp is substituted.")
    x, y, width, height = box
    mask = mask[y:y + height, x:x + width]
    if not np.any(mask):
        raise ProposalSourceUnavailable(f"Frame {index} has no porous-domain pixels inside the native content box.")
    if facies is None or list(facies.shape) != [canvas[1], canvas[0]]:
        raise ProposalSourceUnavailable(f"Frame {index} lacks original native facies labels for balanced phase-box selection.")
    return producer(arrays, run_id=source["run"], frame_index=index, elapsed_h=frame["elapsed_h"],
                    model_id=source["model_id"], candidate_id=source["candidate_id"],
                    fingerprint=source["params_sha256"], canvas_shape=(canvas[1], canvas[0]),
                    content_box=tuple(box), eligible_mask=mask,
                    facies_labels=facies[y:y + height, x:x + width], count=count,
                    existing_regions=_local_existing_regions(rows, index, canvas, box))


def load_generated_pool(output_dir, candidate):
    """Original review-loader hook: read the local pool with its saved bindings.

    Return None when no overlay exists so the caller retains its original reader.
    A corrupt/stale overlay raises visibly; it never falls back to different boxes.
    """
    identity = (candidate.run, candidate.candidate_id, candidate.params_sha256)
    path = categorical_pool_path(output_dir, *identity)
    if path.exists():
        _raw, payload = _recorded_pool(path, identity)
        if payload["source"]["format"] != CATEGORICAL_SOURCE_FORMAT:
            raise ValueError("The categorical namespace contains another proposal source format.")
        prior_path = pool_path(output_dir, *identity)
        if prior_path.exists():
            prior_bytes, prior = _recorded_pool(prior_path, identity)
            expected = dict(sha256=hashlib.sha256(prior_bytes).hexdigest(),
                            source_sha256=prior["source_sha256"])
            if (payload.get("prior_probability_pool") != expected
                    or payload["proposals"][:len(prior["proposals"])] != prior["proposals"]):
                raise ValueError("The original probability pool changed; categorical history was preserved.")
        return payload
    _raw, payload = _recorded_pool(pool_path(output_dir, *identity), identity)
    return payload
