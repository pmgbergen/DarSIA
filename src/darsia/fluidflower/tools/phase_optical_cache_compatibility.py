"""Exact declared producer cache identities; unknown code retains its own SHA."""
from darsia.fluidflower.phase_saved_review_metadata import configured_parameters

def canonical_implementation(name,digest):
    key=canonical_implementation_name(name)
    p=configured_parameters("training_code_identity")
    known=p["AUDITED_IMPLEMENTATIONS"].get(key) or p["AUDITED_STORAGE_IMPLEMENTATIONS"].get(key)
    return known[0] if known is not None and digest in set(known[1]) else digest

def canonical_implementation_name(name):
    prefix='darsia.fluidflower.tools.'
    return 'scripts.'+name[len(prefix):] if name.startswith(prefix) else name
