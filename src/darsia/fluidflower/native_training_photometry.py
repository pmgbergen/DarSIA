"""Native UI binding of saved photometry to the unchanged training producer.

Only JSON metadata is read. Scientific arrays are checked for declared SHA and
file presence here; their bytes are verified by the original trainer before use.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from PySide6.QtWidgets import QLabel

from darsia.fluidflower.phase_training.chemistry_fit_scope import EXCLUDED_RUNS
from darsia.fluidflower.phase_training.log_linear_input import (
    EXACT_VALID_MASK_POLICY,
    MODEL_FORMAT,
    SOURCE_FORMAT,
    V5_UNIT_DENOMINATOR,
)
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.phase_workflow import parse_training_bundles
from darsia.fluidflower.review_storage import resolve_review_path


# The same selected inputs are checked again in the background at most this often.
INPUT_RECHECK_S = 60.0
CHECKING_TEXT = ("Checking the selected training inputs: their saved review files are read on the shared drive. "
                 "Fresh training can start when the check is done.")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _read(path, expected=None):
    path = resolve_review_path(path)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(
            "Training metadata is too large for this UI binding: " + str(path)
        )
    raw = path.read_bytes()
    if expected is not None and (not _digest(expected) or _sha(raw) != expected):
        raise ValueError("Saved training metadata SHA changed: " + str(path))

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate training metadata key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite training metadata: " + value)

    return raw, json.loads(
        raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=nonfinite
    )


def _fragment(raw, key):
    text = raw.decode("utf-8-sig")
    matches = list(re.finditer(r'"' + key + r'"\s*:\s*', text))
    if len(matches) != 1:
        raise ValueError("One unambiguous saved " + key + " block is required")
    start = matches[0].end()
    _value, length = json.JSONDecoder().raw_decode(text[start:])
    return text[start : start + length].encode("utf-8")


def _series(value):
    return str(value or "").casefold().replace("_", "").replace("-", "")


@dataclass(frozen=True)
class SavedTrainingPhotometry:
    model_id: str
    request_path: Path
    request_sha256: str
    request_model_id: str
    settings: dict
    fragments: dict
    reference_paths: tuple[Path, ...]

    @property
    def label(self):
        return f"Photometry: v5, from request {self.request_sha256} of model {self.model_id}"

    def require_request(self, request):
        """The real producer's persisted blocks must retain their exact bytes."""
        _read(self.request_path, self.request_sha256)
        raw, saved = _read(request["request_path"])
        for key, expected in self.fragments.items():
            if (
                _fragment(raw, key) != expected
                or saved["settings"].get(key) != self.settings[key]
            ):
                raise ValueError(
                    "The original request producer did not preserve "
                    + key
                    + " byte-identically"
                )
        if saved["settings"].get("radiometry_profile") not in (None, "", "legacy"):
            raise ValueError("The V5 request unexpectedly carries legacy radiometry")


@dataclass(frozen=True)
class ReviewedRawTrainingPhotometry:
    """The actual prepared photographs, without inherited parent image settings."""
    basis: dict

    @property
    def label(self):
        return "Photometry and crop: the selected prepared RAW photographs"

    @property
    def settings(self):
        from .phase_training.raw_review_training import FAMILY
        return dict(training_family=FAMILY, raw_image_basis=deepcopy(self.basis),
                    v5_raw_photometry=deepcopy(self.basis["input_photometry"]), adapters=["generic"])

    def require_request(self, request):
        from .phase_training.raw_review_training import FAMILY, reviewed_input_basis
        _raw, saved = _read(request["request_path"])
        pairs = [(row["decisions"]["snapshot_path"], row["combined_manifest"]["snapshot_path"])
                 for row in saved["bundles"]]
        basis = reviewed_input_basis(pairs)
        if (saved["settings"].get("training_family") != FAMILY
                or basis != saved["settings"].get("raw_image_basis")
                or basis["input_photometry"] != saved["settings"].get("v5_raw_photometry")):
            raise ValueError("The fresh request differs from its prepared RAW image basis")


def read_saved_training_photometry(entry, series_id):
    """Resolve the selected indexed model's explicit request; no path discovery."""
    if not isinstance(entry, dict) or not entry.get("root"):
        raise ValueError("Choose an indexed saved model as the photometry source")
    identity = entry.get("library_source_identity") or {}
    expected = (identity.get("metadata_sha256") or {}).get("training_request.json")
    if not _digest(expected) or identity.get("model_id") != entry.get("model_id"):
        raise ValueError("The selected model lacks its indexed training-request SHA")
    selected = _series(series_id)
    covered = entry.get("datasets") or [entry.get("series_scope")]
    if isinstance(covered, str):
        covered = [covered]
    if not selected or selected not in {_series(item) for item in covered}:
        raise ValueError("The saved photometry model belongs to another series")
    path = resolve_review_path(entry["root"]) / "training_request.json"
    raw, document = _read(path, expected)
    settings = document.get("settings") or {}
    photometry, chemistry = settings.get("v5_raw_photometry"), settings.get(
        "v5_chemistry_fit_scope"
    )
    if (
        not isinstance(photometry, dict)
        or photometry.get("format") != MODEL_FORMAT
        or photometry.get("full_native_resolution") is not True
        or photometry.get("unit_denominator") != V5_UNIT_DENOMINATOR
        or photometry.get("display_jpeg_is_training_input") is not False
    ):
        raise ValueError(
            "The indexed request has no complete recorded V5 input contract"
        )
    if not isinstance(chemistry, dict) or chemistry.get("excluded_runs") != list(
        EXCLUDED_RUNS
    ):
        raise ValueError(
            "The indexed V5 request lacks its original chemistry-fit scope"
        )
    if settings.get("radiometry_profile") not in (None, "", "legacy"):
        raise ValueError(
            "The saved V5 source request also carries incompatible legacy radiometry"
        )
    references = []
    for key, value in photometry.items():
        if key.endswith("_path"):
            expected_sha = photometry.get(key[:-5] + "_sha256")
            if not isinstance(value, str) or not value:
                raise ValueError("Missing saved V5 reference path: " + key)
            if not _digest(expected_sha):
                raise ValueError("Missing saved V5 reference SHA: " + key)
            _read(value, expected_sha)
            references.append(Path(value))
    if not references:
        raise ValueError("The saved V5 request has no SHA-bound source metadata")
    copied = {
        key: deepcopy(settings[key])
        for key in ("v5_raw_photometry", "v5_chemistry_fit_scope")
    }
    if "v5_review_fit_partition" in settings:
        copied["v5_review_fit_partition"] = deepcopy(
            settings["v5_review_fit_partition"]
        )
    return SavedTrainingPhotometry(
        str(entry["model_id"]),
        path,
        expected,
        str(document.get("model_id") or ""),
        copied,
        {key: _fragment(raw, key) for key in copied},
        tuple(references),
    )


def _raw_review_inputs(manifest_path):
    """Declared training inputs of a RAW photographic review: SHA form and presence, never pixels.

    Baseline B and every frame name their native log-linear RGB and exact valid
    mask through the SHA-bound records the review binds; the trainer verifies
    their bytes before use.
    """
    from darsia.fluidflower.phase_training.raw_frame_context import RawFrameGeometryCohort

    cohort = RawFrameGeometryCohort.from_manifest(manifest_path)
    paths = set()
    for index in (None, *range(len(cohort.rows))):
        for descriptor in cohort.native_centre(index).values():
            if not _digest(descriptor.get("sha256")):
                raise ValueError("Missing RAW training input SHA in " + str(manifest_path))
            if not resolve_review_path(descriptor["path"]).is_file():
                raise FileNotFoundError("RAW training input is missing: " + descriptor["path"])
            paths.add(descriptor["path"])
    return paths


def require_saved_sidecars(bundles, binding):
    """Check exactly the selected bundles' declared V5 files or RAW review inputs, never their pixels."""
    if not bundles:
        raise ValueError("Choose saved reviewed evidence before training")
    checked, payloads = set(), set()
    for decisions_path, combined_path in bundles:
        _raw, decisions = _read(decisions_path)
        _raw, combined = _read(combined_path)
        jobs = combined.get("jobs") or []
        for run, decision in (decisions.get("decisions") or {}).items():
            if not any(
                decision.get(key)
                for key in ("annotations", "phase_quiz_answers", "frame_verdicts")
            ):
                continue
            rows = [
                job
                for job in jobs
                if str(job.get("run", "")).casefold() == str(run).casefold()
            ]
            if not rows:
                raise ValueError(
                    "Saved reviewed evidence has no original source job for " + str(run)
                )
            job = rows[0]
            root = resolve_review_path(
                job.get("analysis_source_review_root") or job["review_output_dir"]
            )
            root = analysis_review_root(root, str(run).casefold())
            path = root / "native_frames" / str(run).casefold() / "manifest.json"
            if path in checked:
                continue
            expected = job.get("native_manifest_sha256") or (
                job.get("v5_raw_photometry") or {}
            ).get("materialized_manifest_sha256")
            if not _digest(expected):
                raise ValueError("The reviewed V5 source lacks its native-manifest SHA")
            _raw, native = _read(path, expected)
            from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT
            if native.get("format") == RAW_REVIEW_FORMAT:
                # A RAW photographic review names SHA-bound native records, not V5 sidecars.
                payloads.update(_raw_review_inputs(path))
                checked.add(path)
                continue
            if (native.get("v5_photometry") or {}).get("format") != SOURCE_FORMAT:
                raise ValueError(
                    "Selected evidence uses historical JPEG input instead of saved V5 sidecars"
                )
            if (
                native.get("v5_track1_photo_geometry")
                and "v5_review_fit_partition" not in binding.settings
            ):
                raise ValueError(
                    "Track1 evidence needs its original saved V5 review-fit partition; this request does not contain it"
                )
            records = [
                native.get("reference") or {},
                *(native.get("frame_sources") or []),
            ]
            if len(records) < 2:
                raise ValueError("Selected V5 source has no recorded frame timeline")
            for record in records:
                keys = ["v5_log_linear_rgb_path"]
                if native.get("v5_float_support_policy") == EXACT_VALID_MASK_POLICY:
                    keys.append("v5_log_linear_valid_mask_path")
                if record.get("v5_extended_ring_path") is not None:
                    keys.append("v5_extended_ring_path")
                for key in keys:
                    value, digest = record.get(key), record.get(key[:-5] + "_sha256")
                    if not isinstance(value, str) or not value or not _digest(digest):
                        raise ValueError(
                            "Missing saved sidecar path/SHA: "
                            + key
                            + " in "
                            + str(path)
                        )
                    if not resolve_review_path(value).is_file():
                        raise FileNotFoundError("Saved V5 sidecar is missing: " + value)
                    payloads.add(value)
            checked.add(path)
    if not checked:
        raise ValueError(
            "No reviewed V5 run source is available in the selected evidence"
        )
    return {
        "native_manifests": len(checked),
        "declared_existing_sidecars": len(payloads),
        "payload_sha_verified_here": False,
        "payload_sha_validation": "Original trainer before use",
    }


def check_training_inputs(entry, bundle_text, series_id, reader):
    """Bind the photometry of the selected review bundles, or raise why it is unavailable.

    It reads every selected bundle and its saved sources, which takes minutes
    for many bundles on a shared drive; the page runs it in the background.
    """
    bundles = parse_training_bundles(bundle_text)
    from .phase_training.raw_review_training import reviewed_input_basis
    basis = reviewed_input_basis(bundles, reader=reader)
    binding = (ReviewedRawTrainingPhotometry(basis) if basis is not None else
               read_saved_training_photometry(entry, series_id))
    require_saved_sidecars(bundles, binding)
    return binding


class NativeTrainingPhotometry:
    """One native instance's settings hook, source label and fresh-route gate."""

    def __init__(self, steps):
        self.steps, self.studio = steps, steps.studio
        self.binding, self.reason = None, "Choose saved V5 photometry"
        from .phase_saved_assets import VerifiedAssetReader
        from .ui_metadata_preview import MetadataPreview
        self._raw_basis_reader = VerifiedAssetReader()
        self.label = QLabel(self.reason)
        self.label.setWordWrap(True)
        steps.layouts[4].addWidget(self.label)
        # (inputs, binding, problem, checked_at) of the last finished check.
        self._answer = None
        self._checking = None
        self._check_reader = MetadataPreview(self.label)
        self._check_reader.completed.connect(self._checked)
        self.studio._native_training_photometry = self
        for name, signal_name in (
            ("validation_model", "currentIndexChanged"),
            ("bundles", "textChanged"),
            ("reviewer", "textChanged"),
            ("training_series", "textChanged"),
            ("training_scope", "currentIndexChanged"),
            ("radiometry_mode", "currentIndexChanged"),
        ):
            control = getattr(self.studio, name, None)
            signal = getattr(control, signal_name, None)
            if signal is not None:
                signal.connect(self.changed)
        profile = getattr(self.studio, "radiometry_profile_path", None)
        profile_changed = getattr(profile, "changed", None)
        if profile_changed is not None:
            profile_changed.connect(self.changed)
        self._attach_effect_summary()

    def _attach_effect_summary(self):
        original = getattr(self.studio, "_update_training_effect_summary", None)
        if not callable(original):
            return

        def summary(*args, **kwargs):
            result = original(*args, **kwargs)
            self._apply_effect_summary()
            return result

        self.studio._update_training_effect_summary = summary
        # Original signal slots stay connected. Add a final presentation-only
        # label update; fresh training retains its backend and concurrency wording.
        for name in (
            "radiometry_mode",
            "radiometry_profile_path",
            "backend",
            "stage_workers",
            "extraction_workers",
            "pixels",
            "mil_iterations",
            "change_ablation",
            "training_evaluation",
        ):
            control = getattr(self.studio, name, None)
            for signal_name in (
                "changed",
                "currentIndexChanged",
                "stateChanged",
                "valueChanged",
            ):
                signal = getattr(control, signal_name, None)
                if signal is not None:
                    signal.connect(self._apply_effect_summary)
                    break
        for name in ("existing_basis", "scratch_basis", "retrain_basis", "continue_basis"):
            control = getattr(self.steps, name, None)
            if control is not None:
                control.toggled.connect(summary)
        self._apply_effect_summary()

    def _apply_effect_summary(self, *_args):
        label = getattr(self.studio, "training_effect_summary", None)
        if label is None:
            return
        existing = getattr(self.steps, "existing_basis", None)
        prepared = getattr(self.steps, "prepared_retraining", None)
        chosen = getattr(self.steps, "existing_choice", None)
        choice = chosen() if callable(chosen) else ""
        if (existing is not None and existing.isChecked() and choice != "continue"
                and prepared is not None and prepared.available()):
            label.setText("Existing saved labels are fitted anew on current Setup crop, lighting and colour images. "
                "No new boxes are required. Architecture and training settings come from the selected model; "
                "image-preparation workers follow the Jobs capacity choice.")
            return
        if (existing is not None and existing.isChecked() and choice != "retrain"
                and getattr(self.steps, "continuation_training", None) is not None
                and (choice == "continue" or not self.uses_prepared_raw_basis)):
            label.setText(
                "Existing-model continuation uses CPU with saved parent weights and scaling; "
                "training settings come from the parent request. No new cross-validation is run. "
                "Backend, worker, sampling, MIL and CV controls apply only to fresh-model training."
            )
            return
        _old, marker, remainder = label.text().partition(" Current effect: ")
        prefix = (
            ("Manual new-model training uses the selected prepared RAW image basis."
             if isinstance(self.binding, ReviewedRawTrainingPhotometry) else
             "Manual new-model training uses saved V5 photometry.")
            if self.binding is not None
            else "Manual new-model training: V5 photometry unavailable."
        )
        label.setText(prefix + marker + remainder)

    def fresh_ui_ready(self):
        """Original UI eligibility, excluding only the unused legacy profile.

        The source/evidence admission remains separate and force-revalidated by
        Train. Standalone and non-fresh callbacks retain their original gate.
        """
        studio = self.studio
        required = ("reviewer", "bundles", "training_series", "training_scope")
        if self.binding is None or any(not hasattr(studio, name) for name in required):
            return False
        ready = bool(studio.reviewer.text().strip()) and bool(
            studio.bundles.toPlainText().strip()
        )
        if studio.training_scope.currentData() == "single-series":
            ready = (
                ready
                and len(
                    [
                        value
                        for value in studio.training_series.text().split(",")
                        if value.strip()
                    ]
                )
                == 1
            )
        return (
            ready
            and getattr(studio, "training_process", None) is None
            and not getattr(studio, "_series_batch_active", False)
        )

    @property
    def uses_prepared_raw_basis(self):
        return isinstance(self.binding, ReviewedRawTrainingPhotometry)

    @property
    def checking(self):
        """True while the selected inputs are being checked in the background."""
        return self._checking is not None

    def changed(self, *_args):
        # The selected model, series and bundles are the inputs of the check.
        self.steps.refresh()

    def _inputs(self):
        """What the check reads: series, selected model and review bundles (no file access)."""
        getter = getattr(self.studio, "_selected_validation_model", None)
        entry = getter() if callable(getter) else None
        bundles = getattr(self.studio, "bundles", None)
        inputs = (str(self.steps.series_id or ""), str((entry or {}).get("root") or ""),
                  json.dumps(entry.get("library_source_identity") if entry else None, sort_keys=True),
                  bundles.toPlainText() if bundles is not None else "")
        return inputs, entry

    def refresh(self, *, force=False):
        """Show the last answer for the selected inputs; check them again in the background.

        The check reads every selected review bundle on the shared drive. It
        runs in the background when the selected series, model or bundles
        change, or when the last answer is older than INPUT_RECHECK_S; until
        new inputs are answered, fresh training waits and the label says that
        they are being checked. A failed check is kept like a passed one, so
        it is not repeated on every refresh. ``force`` (Train and settings
        preparation) checks at once, as before.
        """
        inputs, entry = self._inputs()
        if force:
            try:
                answer = (inputs, check_training_inputs(entry, inputs[3], inputs[0], self._raw_basis_reader), "")
            except Exception as error:
                answer = (inputs, None, str(error))
            self._answer = (*answer, time.monotonic())
        elif self._checking != inputs and (
                self._answer is None or self._answer[0] != inputs
                or time.monotonic() - self._answer[3] >= INPUT_RECHECK_S):
            self._start_check(inputs, entry)
        self._show(inputs)
        return self.binding is not None

    def _start_check(self, inputs, entry):
        reader, entry = self._raw_basis_reader, deepcopy(entry)

        def check(_cancelled):
            try:
                return inputs, check_training_inputs(entry, inputs[3], inputs[0], reader), ""
            except Exception as error:
                return inputs, None, str(error)

        self._checking = inputs
        self._check_reader.start(check)

    def _checked(self, value, error):
        inputs = value[0] if value is not None else self._checking
        if inputs != self._checking:
            return
        self._checking = None
        binding, problem = (value[1], value[2]) if value is not None else (None, error or "no answer was returned")
        self._answer = (inputs, binding, problem, time.monotonic())
        self.steps.refresh()

    def _show(self, inputs):
        """Apply the answer for ``inputs``; without one, say that they are being checked."""
        answer = self._answer if self._answer is not None and self._answer[0] == inputs else None
        self.binding = answer[1] if answer is not None else None
        if self.binding is not None:
            self.reason = (
                self.binding.label
                + "\nSaved sidecar path/SHA metadata and presence checked; the original trainer verifies payload SHA before use."
            )
            self.steps.training_routes["scratch"] = self.train
        else:
            self.reason = ("Fresh training is unavailable: " + answer[2]) if answer is not None else CHECKING_TEXT
            self.steps.training_routes.pop("scratch", None)
        self.label.setText(self.reason)
        self._apply_effect_summary()

    def prepare_settings(self, settings, bundles):
        if not self.refresh(force=True) or self.binding is None:
            raise ValueError(self.reason)
        try:
            require_saved_sidecars(bundles, self.binding)
        except Exception as error:
            self._disable(error)
            raise
        result = deepcopy(settings)
        result.pop("radiometry_profile", None)
        result.update(deepcopy(self.binding.settings))
        if isinstance(self.binding, ReviewedRawTrainingPhotometry):
            from .phase_training.raw_review_training import apply_settings
            result = apply_settings(result, bundles, reader=self._raw_basis_reader)
            existing = getattr(self.steps, "existing_basis", None)
            if existing is not None and existing.isChecked():
                from .phase_saved_assets import VerifiedAssetReader
                from .phase_training.model import DualPhaseSignalModel
                from .phase_features import OpticalFeatureContract
                entry = self.studio._selected_validation_model()
                if not entry:
                    raise ValueError("Choose the model architecture to retrain on the prepared RAW photographs")
                path = resolve_review_path(entry["root"]) / "models/generic/co2_presence_signal.json"
                expected = ((entry.get("library_source_identity") or {}).get("metadata_sha256") or {}).get(
                    "models/generic/co2_presence_signal.json")
                if not expected:
                    raise ValueError("The selected model has no indexed generic optical head to retrain")
                head = dict(path=str(path), sha256=expected)
                reader = VerifiedAssetReader()
                model = DualPhaseSignalModel.from_dict(reader.json(head))
                contract = OpticalFeatureContract.from_dict(model.feature_contract)
                if contract.chemistry_adapter != "none" or contract.reference_adapter != "none":
                    raise ValueError("Select a generic optical model for fresh RAW retraining")
                gas_name = "models/generic/gas_within_co2_signal.json"
                gas_sha = ((entry.get("library_source_identity") or {}).get("metadata_sha256") or {}).get(gas_name)
                if not gas_sha:
                    raise ValueError("The selected model has no indexed generic gas head to retrain")
                gas = dict(path=str(resolve_review_path(entry["root"]) / gas_name), sha256=gas_sha)
                gas_model = DualPhaseSignalModel.from_dict(reader.json(gas))
                if gas_model.feature_contract != model.feature_contract or gas_model.texture_sigma != model.texture_sigma:
                    raise ValueError("The selected model's CO2 and gas heads use different input architectures")
                result.update(feature_version=contract.generic_version, texture_sigma=model.texture_sigma,
                    template_model=dict(model_id=entry["model_id"], co2_head=head, gas_head=gas,
                                        source_model_root=str(resolve_review_path(entry["root"])),
                                        library_source_identity=deepcopy(entry.get("library_source_identity")),
                                        feature_contract=contract.to_dict(), saved_weights_reused=False))
        return result

    def require_request(self, request):
        if self.binding is None:
            raise ValueError(self.reason)
        try:
            self.binding.require_request(request)
        except Exception as error:
            self._disable(error)
            raise

    def _disable(self, error):
        inputs, _entry = self._inputs()
        self._answer = (inputs, None, str(error), time.monotonic())
        self._show(inputs)

    def train(self):
        if not self.refresh(force=True):
            self.steps.refresh()
            raise ValueError(self.reason)
        try:
            return self.steps._original_start()
        finally:
            self.steps.refresh()
