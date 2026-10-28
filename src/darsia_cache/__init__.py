"""Operational cache admission only; importing this package has no side effects.

No DarSIA, NumPy, Numba, Pillow or Qt import is allowed in this module closure.
The explicit bootstrap call owns admission; there is no default root or quota.
"""
