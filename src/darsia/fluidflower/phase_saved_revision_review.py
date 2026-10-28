"""Read-only admissions for registered historical saved candidates.

Every guard comes from the original saved-reader/head-validator AST.
No trainer, candidate calculator, inference entrypoint or CLI is imported.
"""
from .phase_saved_review_metadata import saved_revision_bindings

def combined_saved_review(model_root):
    return saved_revision_bindings('combined_floor').saved_review(model_root)

def t3_saved_review(model_root):
    return saved_revision_bindings('trained_floor').saved_review(model_root)

def t4_saved_review(model_root):
    return saved_revision_bindings('epoch_floor').saved_review(model_root)

def t4b_saved_review(model_root):
    return saved_revision_bindings('joint_floor').saved_review(model_root)

def l1_neighbourhood_saved_review(model_root):
    return saved_revision_bindings('neighbourhood_floor').saved_review(model_root)
