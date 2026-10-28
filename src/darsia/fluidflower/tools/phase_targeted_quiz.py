"""Frozen, separately stored proposal pools for targeted photographic labeling.

Only locations are proposed. No choices are prefilled and no model prediction is
treated as a training label. The standard ReviewStore remains the answer writer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_proposals(candidate, side):
    path = Path(candidate.phase_quiz_proposals_path)
    if not candidate.phase_quiz_proposals_sha256 or sha(path) != candidate.phase_quiz_proposals_sha256:
        raise ValueError("Targeted quiz proposal file changed")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (payload.get("candidate_id") != candidate.candidate_id
            or payload.get("params_sha256") != candidate.params_sha256
            or payload.get("run") != candidate.run):
        raise ValueError("Targeted quiz belongs to another candidate or experiment")
    native_path = Path(candidate.review_output_dir) / "native_frames" / candidate.run / "manifest.json"
    if sha(native_path) != payload.get("native_manifest_sha256"):
        raise ValueError("Targeted quiz native geometry changed")
    native = json.loads(native_path.read_text(encoding="utf-8-sig"))
    sources = native["frame_sources"]
    seen = set()
    rows = []
    for row in payload["proposals"]:
        index = row["frame_index"]
        if not isinstance(index, int) or not 0 <= index < len(sources):
            raise ValueError("Targeted quiz has an invalid frame")
        if (row.get("source_name") != sources[index].get("source_name")
                or row.get("candidate_id") != candidate.candidate_id
                or row.get("params_sha256") != candidate.params_sha256
                or row.get("coordinate_space") != "normalized_native_review_canvas_16x9"
                or row.get("suggested_phase") != "unscored"
                or any(k in row for k in ("human_choice", "phase_class", "answered_utc"))):
            raise ValueError("Targeted quiz source binding or unlabeled status differs")
        if not row.get("proposal_id") or row["proposal_id"] in seen:
            raise ValueError("Targeted quiz proposal IDs must be unique")
        seen.add(row["proposal_id"])
        x, y, w, h = (float(row[k]) for k in ("x", "y", "width", "height"))
        if not (0 <= x < x+w <= 1 and 0 <= y < y+h <= 1):
            raise ValueError("Targeted quiz box outside native canvas")
        rows.append(dict(row, candidate=side))
    return dict(proposals=rows, initial_proposal_count=min(len(rows), int(payload.get("initial_proposal_count", 12))))



if __name__ == "__main__":
    raise SystemExit("Use ordinary Phase Model Studio: Check disagreements or Open blind phase labeling. Separate quiz windows have been retired.")
