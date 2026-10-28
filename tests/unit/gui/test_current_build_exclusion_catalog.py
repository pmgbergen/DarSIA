"""UNRUN original dependency-reader tests, never physical/approval acceptance.

All files are in-memory TESTONLY bytes. The JSON is sufficient only for the
original metadata dependency collectors; binary leaves are invalid archives.
No Build, source pixel replay, startup, capacity or eligibility is mocked.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from darsia.fluidflower.pipeline.job_inputs import _key
from .test_alias_job_inputs import MemoryFiles


class ExclusionMetadataFiles(MemoryFiles):
    def __init__(self, root):
        super().__init__(root)
        self.display_metadata = set()

    def read(self, path):
        path = str(path).replace("\\", "/")
        if path.endswith(".json"):
            return super().read(path)
        assert path in self.display_metadata, "Preparation opened a binary or unlisted TOML"
        self.opened.append(path)
        return self.files[path]


def exclusion_metadata_TESTONLY(tmp_path, *, display=False):
    files = ExclusionMetadataFiles(tmp_path)
    source_map = files.file("source_map.npz", b"TESTONLY invalid source map")
    labels = files.file("ac52/labels.npz", b"TESTONLY invalid labels")
    mask = files.file("ac52/mask.npz", b"TESTONLY invalid exclusion mask")
    target = files.file("ac52/target.npz", b"TESTONLY invalid target pixels")
    model = files.doc("model.json", {"TESTONLY": "unsealed model"})
    index = files.doc("geometry_index.json", {"TESTONLY": "unsealed index"})
    row = dict(run_id="ac52", product=labels, contract={}, gap={"pixels": 0}, components=[], findings=[])
    row["review_row_sha256"] = hashlib.sha256(json.dumps(
        row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf8")).hexdigest()
    review_data = dict(format="darsia-painted-gap-review-v1",
        meaning="outside-analysis-unquantified", source={"painting_facies": {}, "geometry_index": index},
        source_map=source_map, runs=[row], issues=[])
    manifest = None
    if display:
        manifest = files.file("series.toml", b'schema_version = 1\n[runs]\nids = ["ac52"]\n')
        files.display_metadata.add(manifest["path"])
        review_data["display_context"] = dict(format="darsia-painted-review-display-context-v1",
            series_manifest=manifest, declared_run_order=["ac52"], declared_runs={"ids": ["ac52"]},
            image_roles={}, limitations={}, anchor_metric_mm=[0., 0.])
    review = files.doc("review.json", review_data)
    approval = files.doc("approval.json", dict(TESTONLY="not a human approval", review=review))
    version = files.doc("target_version.json", dict(TESTONLY="unsealed target edit", product=target,
                                                   basis={"review": review}))
    target_approval = files.doc("target_approval.json", {"TESTONLY": "not an approval"})
    target_binding = dict(version=version, approval=target_approval)
    continuation = files.doc("continuation.json", dict(format="darsia-photo-edge-continuation-v1",
        policy="nearest-photo-edge-to-target-frame-v1", adopted=False, runs={"ac52": {}}))
    def relative(bound):
        return dict(bound, path=bound["path"][len(files.root) + 1:])
    preparation = dict(TESTONLY="metadata dependency specimen, no accepted science",
        painting_facies={}, geometry_index=index, source_map=relative(source_map),
        continuation=dict(file="continuation.json", sha256=continuation["sha256"], size_bytes=continuation["size_bytes"]),
        runs=[dict(run_id="ac52", labels=relative(labels), reference_model=model, target_model=model)],
        analysis_exclusions=dict(approval=approval, runs={"ac52": {"artifact": mask}}),
        target_corrections=dict(bindings=[target_binding], runs={"ac52": {
            "artifact": relative(target), "bindings": [target_binding]}}))
    pin = files.doc("preparation.json", preparation)
    configuration = dict(painted_adoption_table={"preparation": pin},
                         analysis_exclusions=deepcopy(preparation["analysis_exclusions"]))
    contract = dict(preparation=pin, run_id="ac52", native_grid={})
    return files, configuration, contract, dict(source_map=source_map, labels=labels, mask=mask,
        target=target, review=review, approval=approval, version=version, target_approval=target_approval,
        manifest=manifest)


def catalogue_exclusions_TESTONLY(files, config, contract, *, roots=None, aliases=()):
    catalog = files.catalog(roots=roots, aliases=aliases)
    def source(bound, use, role="configuration", *, metadata=True, original=False):
        if original and _key(catalog.physical(bound)["path"]) != _key(bound["path"]):
            raise ValueError("Original physical locator changed")
        return (catalog.doc if metadata else catalog.add)(bound, role, "current_build." + use)
    catalog.current_build_exclusions(config, contract, source)
    return catalog


def test_original_exclusion_and_target_collectors_admit_all_leaves_without_binary_open(tmp_path):
    files, config, contract, pins = exclusion_metadata_TESTONLY(tmp_path)
    before = deepcopy((config, contract))
    first = catalogue_exclusions_TESTONLY(files, config, contract)
    second = catalogue_exclusions_TESTONLY(files, deepcopy(config), deepcopy(contract))
    assert first.entries == second.entries and (config, contract) == before
    admitted = {row["reference"]["relative_path"] for row in first.entries.values()}
    assert {"source_map.npz", "ac52/labels.npz", "ac52/mask.npz", "ac52/target.npz",
            "review.json", "approval.json", "target_version.json", "target_approval.json"} <= admitted
    assert all(path.endswith(".json") for path in files.opened)
    assert all(pins[name]["path"] in files.statted for name in ("source_map", "labels", "mask", "target"))
    uses = {use for row in first.entries.values() for use in row["uses"]}
    assert any(use.startswith("current_build.exclusions.approval.") for use in uses)
    assert any(use.startswith("current_build.exclusions.target.") for use in uses)


@pytest.mark.parametrize("fault", [None, "changed_manifest", "relocated_manifest"])
def test_original_display_context_reads_only_its_exact_pinned_series_toml(tmp_path, fault):
    """Original read_gap_review/validate_display_context/TOML parser stay real."""
    files, config, contract, pins = exclusion_metadata_TESTONLY(tmp_path, display=True)
    aliases = []
    if fault == "changed_manifest":
        # Same-size replacement must fail the byte hash, not just file length.
        files.files[pins["manifest"]["path"]] = files.files[pins["manifest"]["path"]].replace(b"ac52", b"ac53")
    elif fault == "relocated_manifest":
        copied = files.file("copy.toml", files.files[pins["manifest"]["path"]])
        aliases = [dict(logical=pins["manifest"], physical=copied)]
    if fault is None:
        catalog = catalogue_exclusions_TESTONLY(files, config, contract)
        uses = {use: row["reference"] for row in catalog.entries.values() for use in row["uses"]}
        assert uses["current_build.exclusions.display_series.0"]["sha256"] == pins["manifest"]["sha256"]
        assert uses["current_build.exclusions.display_series.0"]["relative_path"] == "series.toml"
        assert pins["manifest"]["path"] in files.opened
    else:
        with pytest.raises(ValueError):
            catalogue_exclusions_TESTONLY(files, config, contract, aliases=aliases)
    assert all(path.endswith(".json") or path == pins["manifest"]["path"] for path in files.opened)


@pytest.mark.parametrize("fault", ["missing_source_map", "changed_approval", "relocated_preparation"])
def test_optional_closure_refuses_missing_changed_or_relocated_originals(tmp_path, fault):
    files, config, contract, pins = exclusion_metadata_TESTONLY(tmp_path)
    aliases = []
    if fault == "missing_source_map":
        del files.files[pins["source_map"]["path"]]
    elif fault == "changed_approval":
        files.files[pins["approval"]["path"]] = b"{}"
    else:
        logical = contract["preparation"]
        physical = files.file("copy.json", files.files[logical["path"]])
        aliases = [dict(logical=logical, physical=physical)]
    with pytest.raises((ValueError, KeyError)):
        catalogue_exclusions_TESTONLY(files, config, contract, aliases=aliases)
    assert all(path.endswith(".json") for path in files.opened)


@pytest.mark.parametrize("operation", ["binary", "unlisted_toml", "exists", "inventory", "table"])
def test_preparation_scope_refuses_unimplemented_reader_operations(tmp_path, monkeypatch, operation):
    """A faulting dependency consumer cannot add a hidden runtime operation."""
    from darsia.fluidflower import painted_analysis_exclusions as exclusions
    from darsia.fluidflower import preset_wasserstein_inputs as io
    files, config, contract, pins = exclusion_metadata_TESTONLY(tmp_path)
    def forbidden(_preparation):
        scope = io.current_scope()
        if operation == "binary":
            scope.read(pins["source_map"]["path"])
        elif operation == "unlisted_toml":
            other = files.file("unlisted.toml", b'schema_version = 1\n')
            scope.descriptor(other["path"], other)
            scope.read(other["path"])
        elif operation == "exists":
            scope.exists(files.root, "is_dir")
        elif operation == "inventory":
            scope.inventory(files.root, "*")
        else:
            scope.role("protocol")
        pytest.fail("Unsupported preparation operation returned")
    monkeypatch.setattr(exclusions, "exclusion_sources", forbidden)
    with pytest.raises(ValueError, match="Current Build exclusion preparation"):
        catalogue_exclusions_TESTONLY(files, config, contract)
    assert io.current_scope() is None
    assert all(path.endswith(".json") for path in files.opened)
