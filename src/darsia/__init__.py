"""DarSIA provides tools for quantitative image processing with support for various and
common data types in computational and laboratory porous media research.

The modules are imported on first use of one of their names (`darsia.Image` imports
`darsia.image.image`); the public names are listed in `__init__.pyi`.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
