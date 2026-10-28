"""Read-only camera-source joins for the opt-in selected Studio timeline.

Presentation indices and rendered JPEG hashes are never evidence identities.
Historical payloads are copied for display; no session or label file is written.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math

SOURCE_KEY = "ffac_selected_policy_a_sources"
CONTRACT = "ffac-selected-camera-annotation-binding-v1"
# Saved with every newly saved annotation: the photograph it was drawn on.
PHOTO_IDENTITY_KEY = "photo_identity"


def is_selected(native):
    return SOURCE_KEY in (native or {})


def _rows(native):
    return (native or {}).get("frame_sources") or []


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdefABCDEF" for c in value)


def camera_sha(row):
    identity=row.get("source_label_identity") or {}
    digest=identity.get("camera_source_sha256") or row.get("source_sha256")
    if not _digest(digest):
        raise ValueError("Camera source has no exact acquisition SHA-256")
    if identity and (identity.get("camera_source_sha256") != row.get("source_sha256")):
        raise ValueError("Camera label identity differs from its acquisition")
    return digest.lower()


def _read_bound(descriptor,read_document):
    if not isinstance(descriptor,dict) or not _digest(descriptor.get("sha256")) or not descriptor.get("path"):
        raise ValueError("Original camera alias lacks a bound manifest")
    result=read_document(Path(descriptor["path"]),required=True,expected_sha256=descriptor["sha256"])
    if result is None:
        raise ValueError("Original camera alias changed or is unavailable")
    return result


def resolve_camera_lineage(native,read_document,*,_seen=()):
    """Follow existing SHA-bound native aliases, without opening camera pixels."""
    result=deepcopy(native or {})
    alias=result.get("v5_native_source_alias")
    if alias:
        original=_read_bound(alias["original_native_manifest"],read_document)
        descriptor=alias["original_native_manifest"]
        if descriptor["sha256"] in _seen:
            raise ValueError("Original camera source alias contains a cycle")
        parent=resolve_camera_lineage(original,read_document,_seen=(*_seen,descriptor["sha256"]))
        if parent.get("run") != result.get("run") or len(_rows(parent)) != len(_rows(result)):
            raise ValueError("Original camera alias changes its run or timeline")
        for child,old in zip(_rows(result),_rows(parent)):
            if child.get("source_sha256") and child["source_sha256"] != camera_sha(old):
                raise ValueError("Original camera alias changes an acquisition SHA")
            child["source_sha256"]=camera_sha(old)
        return result
    marker=result.get("ffac_rig_camera_sources")
    if marker:
        inputs=_read_bound(marker["input_manifest"],read_document)
        parent_descriptor=inputs["parent_native_manifest"]
        parent=resolve_camera_lineage(_read_bound(parent_descriptor,read_document),read_document,
            _seen=(*_seen,parent_descriptor["sha256"]))
        bindings=inputs.get("frames") or []
        if len(bindings)!=len(_rows(result)):
            raise ValueError("Rig camera alias differs from its bound timeline")
        for row,binding in zip(_rows(result),bindings):
            if binding.get("source_changed"):
                # New acquisition: the rig receipt owns this literal camera SHA.
                digest=camera_sha(row)
                if binding.get("source_sha256") not in (None,digest):
                    raise ValueError("Changed rig source differs from its camera binding")
            else:
                index=binding.get("parent_source_frame_index")
                if type(index) is not int or not 0<=index<len(_rows(parent)):
                    raise ValueError("Retained rig source has no exact parent acquisition")
                digest=camera_sha(_rows(parent)[index])
                if row.get("source_sha256") not in (None,digest):
                    raise ValueError("Retained rig source differs from its parent camera SHA")
            row["source_sha256"]=digest
    return result


def _original_row(payload,original):
    rows=_rows(original)
    explicit=(payload.get("source_label_identity") or {}).get("camera_source_sha256") or payload.get("source_sha256")
    if explicit:
        matches=[row for row in rows if camera_sha(row)==str(explicit).lower()]
    else:
        index=payload.get("original_frame_index",payload.get("frame_index"))
        matches=[row for row in rows if row.get("frame_index")==index]
        elapsed=payload.get("elapsed_h")
        if matches and elapsed is not None and not math.isclose(float(matches[0].get("elapsed_h")),float(elapsed),abs_tol=1e-6):
            matches=[row for row in rows if row.get("elapsed_h") is not None
                and math.isclose(float(row["elapsed_h"]),float(elapsed),abs_tol=1e-6)]
    return matches[0] if len(matches)==1 else None


def align_selected_annotation(payload,original,current,*,geometry=True):
    """Return a display copy, or an honest old-only probe with no selected index."""
    result=dict(payload=deepcopy(dict(payload)),displayable=False,mapped_by="camera_source_sha256",
        mapping_error=None,mapping_warnings=[],source_binding_contract=CONTRACT)
    try:
        identity=payload.get("source_label_identity") or {}
        if identity:
            if identity.get("run") and str(identity["run"]).casefold()!=str(current.get("run","")).casefold():
                raise ValueError("Annotation camera identity belongs to another run")
            if payload.get("source_sha256") and identity.get("camera_source_sha256")!=payload["source_sha256"]:
                raise ValueError("Annotation camera identity has conflicting SHA bindings")
        for field in ("run","dataset"):
            if original and str(original.get(field,"")).casefold()!=str(current.get(field,"")).casefold():
                raise ValueError("Original/current camera run identity differs")
        old=_original_row(payload,original)
        explicit=(payload.get("source_label_identity") or {}).get("camera_source_sha256")
        if old is None and not explicit:
            raise ValueError("Original camera acquisition is not uniquely bound; rendered JPEG SHA and current index are not used")
        digest=camera_sha(old) if old else str(explicit).lower()
        if not _digest(digest):
            raise ValueError("Annotation has no camera acquisition SHA")
        choices=[row for row in _rows(current) if camera_sha(row)==digest]
        result["camera_source_sha256"]=digest
        if len(choices)!=1:
            result["payload"]["frame_index"]=None
            result["old_only_studio_probe"]=True
            raise ValueError("Original Studio probe is absent from the selected timeline; annotation retained without an overlay")
        target=choices[0]
        result["payload"]["frame_index"]=target["frame_index"]
        result["payload"]["elapsed_h"]=target.get("elapsed_h")
        if geometry:
            if not original:
                raise ValueError("Original native geometry unavailable; overlay disabled")
            old_box,new_box=original.get("content_box_xywh"),current.get("content_box_xywh")
            if not old_box or not new_box or old_box[2:]!=new_box[2:] or original.get("analysis_image_shape_hw")!=current.get("analysis_image_shape_hw"):
                raise ValueError("Camera review content geometry differs; overlay disabled")
            ow,oh=original["canvas_wh"];nw,nh=current["canvas_wh"]
            dx,dy=new_box[0]-old_box[0],new_box[1]-old_box[1]
            mapped=result["payload"]
            for key,scale,offset in (("x",ow/nw,dx/nw),("y",oh/nh,dy/nh),("width",ow/nw,0),("height",oh/nh,0)):
                if key in mapped: mapped[key]=float(mapped[key])*scale+offset
            if mapped.get("points"):
                mapped["points"]=[[(float(p["x"] if isinstance(p,dict) else p[0])*ow+dx)/nw,
                    (float(p["y"] if isinstance(p,dict) else p[1])*oh+dy)/nh] for p in mapped["points"]]
            result["geometry_mapping"]="same native content pixels on the recorded review canvases"
        result["displayable"]=True
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        result["mapping_error"]=str(exc)
    return result


def load_annotation_binding(review_root,run):
    """Load the completed review's bound parent-camera lineage off the GUI thread."""
    from .phase_selected_review import load_native_source,_READER,_document
    state=load_native_source(review_root,run,verify_geometry=False)
    phase_plan=_document(state["inputs"]["phase_plan"],_READER)
    descriptor=phase_plan["runs"][str(run).casefold()]["parent_native_manifest"]
    def read(path,**kwargs):
        return _document(dict(path=str(path),sha256=kwargs["expected_sha256"]),_READER)
    parent=resolve_camera_lineage(_read_bound(descriptor,read),read)
    return dict(contract=CONTRACT,current=state["native"],parent=parent,parent_native_manifest=descriptor)


def presentation_frame(payload,binding):
    if binding is None:
        return int(payload.get("frame_index") or 0)
    original=binding["current"] if payload.get("source_binding_contract")==CONTRACT else binding["parent"]
    result=align_selected_annotation(payload,original,binding["current"],geometry=False)
    return result["payload"].get("frame_index") if result["displayable"] else None


def _series_id(value):
    """Canonical series ID of a dataset name, or "" when there is none."""
    from .series_paths import series_key
    try:
        return series_key(value) if value else ""
    except ValueError:
        return ""


def photo_identity(row,*,dataset,run):
    """The photograph one frame record shows: {dataset, run, source_name, source_sha256}.

    Returns None when the record names neither the photograph nor its SHA-256.
    """
    row=row or {}
    digest=(row.get("source_label_identity") or {}).get("camera_source_sha256") or row.get("source_sha256")
    digest=str(digest).lower() if _digest(digest) else None
    name=str(row.get("source_name") or str(row.get("source_path") or "").replace("\\","/").rsplit("/",1)[-1])
    if not name and digest is None:
        return None
    return dict(dataset=_series_id(dataset),run=str(run).casefold(),source_name=name or None,source_sha256=digest)


def stamp_camera_identity(payload,row,run,*,dataset,selected_camera=False):
    """Return newly saved evidence with the identity of the photograph it is on.

    ``row`` is the frame record of the evidence's frame_index. By default the
    copy gains ``photo_identity`` (see ``photo_identity``). The selected-camera
    review contract (``selected_camera=True``) instead writes that contract's
    original three fields, with unchanged values. Only newly saved evidence is
    stamped; legacy payloads stay intact and keep loading without it.
    """
    result=deepcopy(payload)
    if not selected_camera:
        identity=photo_identity(row,dataset=dataset,run=run)
        if identity is not None:
            result[PHOTO_IDENTITY_KEY]=identity
        return result
    result["source_label_identity"]=dict(dataset=_series_id(dataset),run=str(run).casefold(),camera_source_sha256=camera_sha(row))
    result["source_sha256"]=camera_sha(row)
    result["source_binding_contract"]=CONTRACT
    return result


def presentation_annotation(payload,binding):
    if binding is None:
        return payload
    original=binding["current"] if payload.get("source_binding_contract")==CONTRACT else binding["parent"]
    result=align_selected_annotation(payload,original,binding["current"],geometry=True)
    return result["payload"] if result["displayable"] else dict(payload,frame_index=None)
