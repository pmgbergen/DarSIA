"""Small operational helpers that run before DarSIA itself, or without it.

Importing this package has no side effects. No DarSIA, NumPy, Numba, Pillow
or Qt import is allowed here, so these helpers still work when DarSIA cannot
load, for example when its cache bootstrap refuses a worker start.
"""
