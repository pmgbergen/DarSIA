from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from darsia.fluidflower.phase_training import annotation_basis_transfer as transfer
from darsia.fluidflower.phase_training.review_regions import annotation_mask
from darsia.fluidflower.phase_saved_assets import asset
from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return asset(path)


def config(root, name="config", *, coefficients=None, lighting=1):
    registration = dict(common=dict(sensor=dict(camera_wh=[200, 120], raw_wh=[202, 122]),
        profiles=dict(camera=dict(coefficients=coefficients or [1, 2], distortion=[3, 4])),
        lens=dict(ca_red=[5, 6], ca_blue=[7, 8])))
    original = write(root, name + "-original.json", dict(registration=registration,
        source_modules=[dict(path="original_renderer.py", git_blob_sha256="f" * 64)]))
    return write(root, name + ".json", dict(registration=registration,
        native_parameter_source=original, photometry=dict(light=lighting)))


def certificate(h, *, canvas=(120, 90), box=(10, 10, 100, 60), raw="a" * 64):
    return dict(profile="camera", sources=dict(arw_sha256=raw), review_to_camera_h=np.asarray(h).tolist(),
        camera_jpeg_wh=[200, 120], raw_visible_wh=[202, 122], sony_distortion_table=[3, 4],
        sony_ca_red_table=[5, 6], sony_ca_blue_table=[7, 8], review_canvas_wh=list(canvas),
        content_box_xywh=list(box))


def old_review(root, h, *, count=3):
    rows = []
    for i in range(count):
        raw = chr(ord("a") + i) * 64
        cert = certificate(h, raw=raw)
        receipt = write(root, f"receipt-{i}.json", dict(run="run1", frame_index=i, elapsed_h=float(i),
            source=dict(camera_arw=f"unopened-{i}.ARW", camera_arw_sha256=raw),
            processor=dict(renderer_module_sha256="f" * 64),
            certificate=dict(renderer_certificate=cert,
                transform_sha256=sha256(json.dumps(cert, sort_keys=True, default=str).encode()).hexdigest())))
        rows.append(dict(frame_index=i, elapsed_h=float(i), source_sha256="e" * 64,
                         v5_raw_sha256=raw, v5_frame_receipt_path=receipt["path"],
                         v5_frame_receipt_sha256=receipt["sha256"]))
    baseline = write(root, "baseline.json", dict(source=dict(camera_arw="B.ARW", camera_arw_sha256="d" * 64)))
    return write(root, "old/native_frames/run1/manifest.json", dict(run="run1", dataset="series",
        frame_count=count, frame_sources=rows, canvas_wh=[120, 90], content_box_xywh=[10, 10, 100, 60],
        reference=dict(v5_baseline_receipt_path=baseline["path"],v5_baseline_receipt_sha256=baseline["sha256"])))


def raw_review(root, h, configuration, *, box=(5, 5, 80, 50), order=(0, 1, 2)):
    rows = []
    for i, original_index in enumerate(order):
        camera = dict(path=f"unopened-{original_index}.ARW", sha256=chr(ord("a") + original_index) * 64)
        renderer = certificate(h, raw=camera["sha256"])
        renderer.update(source_kind="raw", profile_id="camera", native_to_camera_u_h=np.asarray(h).tolist(),
                        native_shape_hw=[box[3], box[2]], camera_u_wh=[200, 120])
        audit = write(root, f"target/audit-{i}.json", dict(contract="darsia-raw-rig-source-v1", jpeg_inputs=False,
            sources=dict(camera_arw=camera), source_identity=dict(kind="raw-acquisition", raw_sha256=camera["sha256"]),
            processor=dict(source_binding=dict(path="binding.json",sha256="1"*64)),
            renderer=renderer))
        dummy = dict(path="never-loaded.npy", sha256="0" * 64)
        full = dict(complete=True, source_sha256=camera["sha256"], phase_shape_hw=[70, 100],
            native_origin_xy=[-box[0], -box[1]], native_centre=dict(native_shape_hw=[box[3], box[2]],
            log_linear_rgb=dummy, valid_mask=dummy))
        record = write(root, f"target/record-{i}.json", dict(full=full, audit=audit))
        rows.append(dict(frame_index=i, elapsed_h=float(original_index), acquisition=camera,
                         native_record=record, audit=audit, case_id=f"case{i}", profile_id="camera"))
    return write(root, "target/native_frames/run1/manifest.json", dict(format=RAW_REVIEW_FORMAT, source_kind="raw",
        run="run1", frame_count=len(rows), frames=rows, native_configuration=configuration,
        raw_binding=dict(path="binding.json", sha256="1" * 64), canvas_wh=[100, 70],
        content_box_xywh=list(box), phase_shape_hw=[70, 100], native_shape_hw=[box[3], box[2]],
        reference=dict(acquisition=dict(path="B.ARW",sha256="d"*64))))


@pytest.fixture
def inputs(tmp_path):
    before = config(tmp_path)
    after = config(tmp_path, "after", lighting=9)
    old = old_review(tmp_path, np.eye(3))
    new = raw_review(tmp_path, [[1, 0, 20], [0, 1, 5], [0, 0, 1]], after)
    annotation = dict(id="label", reviewer_id="reviewer", category="confirmed_free_gas", shape="rectangle",
        frame_index=2, elapsed_h=2., x=40/120, y=30/90, width=20/120, height=10/90,
        photo_identity=dict(dataset="series", run="run1", source_sha256="e"*64))
    return old, new, before, annotation


def test_metadata_only_projection_keeps_exact_label_and_updates_photo_identity(inputs):
    old, new, configuration, annotation = inputs
    original = deepcopy(annotation)
    result = transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration)
    projected = result["annotation"]
    # Old native [30:50,20:30] -> target native [10:30,15:25], then target padding.
    assert np.allclose(projected["points"], [[.15,20/70],[.35,20/70],[.35,30/70],[.15,30/70]])
    assert projected["category"] == annotation["category"] and projected["reviewer_id"] == "reviewer"
    assert projected["photo_identity"]["source_sha256"] == "c" * 64
    assert projected["basis_transfer"]["source_annotation"] == original
    assert annotation == original
    mask = annotation_mask(projected, canvas_wh=[100,70], content_box=[5,5,80,50])
    assert mask[15:26,10:31].all() and mask.sum() == 11 * 21


def test_target_index_is_found_by_raw_sha_not_current_position(tmp_path, inputs):
    old, _, configuration, annotation = inputs
    new = raw_review(tmp_path, np.eye(3), configuration, order=(2,))
    result = transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration)
    assert result["annotation"]["frame_index"] == 0 and result["annotation"]["elapsed_h"] == 2.


def test_annotation_on_a_left_out_frame_is_excluded_with_its_reason(tmp_path, inputs):
    old, _, configuration, annotation = inputs
    new = raw_review(tmp_path, np.eye(3), configuration, order=(0, 1))
    with pytest.raises(ValueError, match="one exact current target frame"):
        transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration)
    reason = "ROIs cannot be aligned by translation."
    result = transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration,
                                              left_out={"c" * 64: reason})
    assert result["annotation"] is None and result["exclusion_reason"] == "frame-left-out: " + reason
    assert result["provenance"]["acquisition"]["sha256"] == "c" * 64


def test_perspective_rotation_projects_polygon_with_same_homography(tmp_path, inputs):
    old, _, configuration, annotation = inputs
    h = np.array([[.9, -.1, 3], [.1, 1.1, -2], [.0001,.0002,1]])
    new = raw_review(tmp_path, h, configuration)
    result = transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration)
    native = np.array([[30,20,1],[50,20,1],[50,30,1],[30,30,1]]) @ np.linalg.inv(h).T
    expected = (native[:,:2]/native[:,2,None] + [5,5]) / [100,70]
    assert np.allclose(result["annotation"]["points"], expected)


def test_outside_crop_is_explicit_exclusion_without_moving_label_to_edge(tmp_path, inputs):
    old, _, configuration, annotation = inputs
    new = raw_review(tmp_path, [[1,0,150],[0,1,0],[0,0,1]], configuration)
    result = transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration)
    assert result["annotation"] is None and result["exclusion_reason"] == "outside-current-native-crop"
    assert result["provenance"]["source_annotation"] == annotation


def test_partial_crop_clips_polygon_not_each_vertex(tmp_path, inputs):
    old, _, configuration, annotation = inputs
    new = raw_review(tmp_path, [[1,0,40],[0,1,0],[0,0,1]], configuration)
    result = transfer.transfer_raw_annotation(annotation, old, new, source_native_configuration=configuration)
    assert result["annotation"] is not None and result["provenance"]["target_clipped"]
    assert min(p[0] for p in result["annotation"]["points"]) == .05


@pytest.mark.parametrize("kind", ["lens", "missing-configuration", "time", "photo", "singular", "missing-source", "renderer"])
def test_unbound_or_inconsistent_projection_is_refused(tmp_path, inputs, kind):
    old, new, configuration, annotation = inputs
    if kind == "lens":
        new = raw_review(tmp_path, np.eye(3), config(tmp_path,"changed",coefficients=[99,2]))
    elif kind == "missing-configuration":
        configuration = None
    elif kind == "time":
        annotation["elapsed_h"] = 9.
    elif kind == "photo":
        annotation["photo_identity"]["source_sha256"] = "9" * 64
    elif kind == "singular":
        new = raw_review(tmp_path, np.zeros((3,3)), configuration)
    elif kind == "missing-source":
        new = raw_review(tmp_path, np.eye(3), configuration,order=(0,1))
    elif kind == "renderer":
        doc = json.loads(open(configuration["path"]).read())
        doc["native_parameter_source"] = write(tmp_path,"wrong-original.json",dict(
            registration=doc["registration"],source_modules=[]))
        configuration = write(tmp_path,"wrong-config.json",doc)
    with pytest.raises(ValueError):
        transfer.transfer_raw_annotation(annotation,old,new,source_native_configuration=configuration)


def test_required_union_includes_true_immediate_predecessor_and_separate_baseline(tmp_path, inputs):
    old,new,configuration,annotation = inputs
    decisions = write(tmp_path,"decisions.json",dict(decisions=dict(run1=dict(annotations=[annotation]))))
    root = str((tmp_path/"old").resolve())
    combined = write(tmp_path,"combined.json",dict(jobs=[dict(run="run1",dataset="series",review_output_dir=root,
        v5_raw_photometry=dict(materialized_manifest_sha256=old["sha256"]))]))
    required = transfer.required_annotation_sources(decisions,combined)
    rows = required["runs"]["run1"]["acquisitions"]
    assert [(r["frame_index"],r["roles"]) for r in rows] == [(None,["baseline"]),(1,["history"]),(2,["annotation"])]
    assert [r["acquisition"]["sha256"] for r in rows] == ["d"*64,"b"*64,"c"*64]
    projected = transfer.transfer_bundle(decisions,combined,dict(run1=new),source_native_configuration=configuration)
    job = projected["combined_manifest"]["jobs"][0]
    assert "v5_raw_photometry" not in job and job["native_manifest_sha256"] == new["sha256"]
    assert json.loads(open(decisions["path"]).read())["decisions"]["run1"]["annotations"][0] == annotation


def test_raw_to_raw_uses_same_verified_camera_basis(tmp_path, inputs):
    _,old,configuration,annotation = inputs
    annotation.update(x=.3,y=.4,width=.1,height=.1)
    annotation.pop("photo_identity")
    new=raw_review(tmp_path/"other",np.eye(3),configuration)
    result=transfer.transfer_raw_annotation(annotation,old,new)
    assert result["annotation"] is not None


def test_unbound_legacy_box_may_be_explicitly_excluded_but_never_guessed(tmp_path, inputs):
    old,new,configuration,annotation=inputs
    invalid=dict(annotation,id="ambiguous",elapsed_h=99.)
    decisions=write(tmp_path,"decisions.json",dict(decisions=dict(run1=dict(annotations=[annotation,invalid]))))
    combined=write(tmp_path,"combined.json",dict(jobs=[dict(run="run1",dataset="series",
        review_output_dir=str(tmp_path/"old"),v5_raw_photometry=dict(materialized_manifest_sha256=old["sha256"]))]))
    with pytest.raises(transfer.UnboundAnnotation):
        transfer.required_annotation_sources(decisions,combined)
    result=transfer.transfer_bundle(decisions,combined,dict(run1=new),
        source_native_configuration=configuration,on_unbound="exclude")
    assert len(result["decisions"]["decisions"]["run1"]["annotations"])==1
    assert result["provenance"]["excluded_annotations"][0]["source_annotation"]==invalid


def test_equivalent_candidate_jobs_share_source_but_conflicting_sources_refuse(tmp_path,inputs):
    old,new,configuration,annotation=inputs
    decisions=write(tmp_path,"decisions.json",dict(decisions=dict(run1=dict(annotations=[annotation]))))
    job=dict(run="run1",dataset="series",review_output_dir=str(tmp_path/"old"),
        v5_raw_photometry=dict(materialized_manifest_sha256=old["sha256"]))
    combined=write(tmp_path,"combined.json",dict(jobs=[job,dict(job,candidate_id="B")]))
    result=transfer.required_annotation_sources(decisions,combined)
    assert list(result["runs"])==["run1"]


def test_source_specific_lens_wrapper_proves_only_its_unchanged_acquisitions(tmp_path,inputs):
    old,new,configuration,annotation=inputs
    base=dict(processor_module_sha256="b"*64,renderer_module_sha256="f"*64)
    wrapper=dict(processor_module_sha256="d"*64,renderer_module_sha256="e"*64,
        base_processor_module_sha256=base["processor_module_sha256"],
        source_exact_alt_lens_exception=dict(arw_sha256="c"*64,all_other_frames_use_original_renderer=True))
    document=json.loads(open(old["path"]).read())
    for frame in document["frame_sources"]:
        receipt=json.loads(open(frame["v5_frame_receipt_path"]).read())
        receipt["processor"]=wrapper
        pin=write(tmp_path,f"wrapper-{frame['frame_index']}.json",receipt)
        frame.update(v5_frame_receipt_path=pin["path"],v5_frame_receipt_sha256=pin["sha256"])
    old=write(tmp_path,"old/native_frames/run1/manifest.json",document)
    batch=write(tmp_path,"batch.json",dict(processor_identities=dict(base=base,variant=wrapper)))
    unchanged=dict(annotation,frame_index=1,elapsed_h=1.)
    result=transfer.transfer_raw_annotation(unchanged,old,new,source_native_configuration=configuration,
        source_processor_provenance=batch)
    assert result["annotation"] is not None
    assert result["provenance"]["source_renderer_declaration"]["materialization"]==batch
    with pytest.raises(transfer.UnboundAnnotation,match="alternative-lens"):
        transfer.transfer_raw_annotation(annotation,old,new,source_native_configuration=configuration,
            source_processor_provenance=batch)
    decisions=write(tmp_path,"decisions.json",dict(decisions=dict(run1=dict(annotations=[unchanged,annotation]))))
    combined=write(tmp_path,"combined.json",dict(jobs=[dict(run="run1",dataset="series",
        review_output_dir=str(tmp_path/"old"),v5_raw_photometry=dict(materialized_manifest_sha256=old["sha256"],
        batch_receipt_path=batch["path"],batch_receipt_sha256=batch["sha256"]))]))
    plan=transfer.required_annotation_sources(decisions,combined,source_native_configuration=configuration,
        on_unbound="exclude")
    assert plan["runs"]["run1"]["annotation_ids"]==["label"]
    assert len(plan["excluded_annotations"])==1
    wrapper["source_exact_alt_lens_exception"]["all_other_frames_use_original_renderer"]=False
    wrong=write(tmp_path,"wrong-batch.json",dict(processor_identities=dict(base=base,variant=wrapper)))
    with pytest.raises(transfer.UnboundAnnotation):
        transfer.transfer_raw_annotation(unchanged,old,new,source_native_configuration=configuration,
            source_processor_provenance=wrong)
