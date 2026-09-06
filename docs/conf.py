"""Configuration file for the Sphinx documentation builder.

Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from __future__ import annotations

import inspect
import os
import sys
from importlib.metadata import version as _get_version

# -- Project information ---------------------------------------------------------

project = "DarSIA"
copyright = "2023-2026, DarSIA developers"
author = "DarSIA developers"

# Single source of truth: the installed package version.
release = _get_version("darsia")
version = ".".join(release.split(".")[:2])

# -- General configuration -----------------------------------------------------

root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # transitional: NumPy-style is the migration target
    "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "myst_nb",
    # "numpydoc",                    # TODO(step 6): swap in for napoleon after migration
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # Sphinx-Gallery writes .ipynb download companions next to its .rst pages;
    # myst-nb would otherwise pick them up as duplicate source documents.
    "auto_examples/**/*.ipynb",
    # Sphinx-Gallery timing pages; they cross-reference labels that only exist
    # for executed examples, so with source-only examples they emit dead links.
    "**/sg_execution_times.rst",
    "sg_execution_times.rst",
]

suppress_warnings = ["toc.no_title"]

# Do not prefix every documented object with its (long) dotted module path.
add_module_names = False

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autoclass_content = "both"  # DarSIA documents constructor args in __init__

# Case-insensitive filesystems (Windows/macOS) collapse e.g. ``darsia.TVD`` and
# ``darsia.tvd`` onto one stub file. Give the lowercase callables a distinct name.
autosummary_filename_map = {
    "darsia.tvd": "darsia-tvd-function",
    "darsia.resize": "darsia-resize-function",
}
autodoc_class_signature = "mixed"
autodoc_member_order = "groupwise"
autodoc_typehints = "none"  # TODO(step 6): flip to "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

# -- Napoleon (renders both Google- and NumPy-style during the migration) ----

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # render Attributes as a field list, avoids autodoc dupes

# -- MyST / notebooks -------------------------------------------------------

myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
nb_execution_mode = "off"

# -- sphinx-gallery -----------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "ignore_pattern": r"/notebooks/",
    "within_subsection_order": "FileNameSortKey",
    "subsection_order": [
        "../examples/io",
        "../examples/corrections",
        "../examples/restoration",
        "../examples/segmentation",
        "../examples/registration",
        "../examples/analysis",
        "../examples/distances",
        "../examples/paper",
    ],
    "image_scrapers": ("matplotlib",),
    "doc_module": ("darsia",),
    "default_thumb_file": os.path.join(
        os.path.dirname(__file__), "_static", "darsia-logo.png"
    ),
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "download_all_examples": False,
}

# -- Intersphinx ----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
}

# -- HTML output ---------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/darsia-logo.png"
html_favicon = "_static/darsia-logo.png"
html_short_title = "DarSIA"
html_title = f"DarSIA {version}"
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_split_index = True

html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "github_url": "https://github.com/pmgbergen/darsia",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pmgbergen/darsia",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "content",
    "header_links_before_dropdown": 6,
}

html_context = {
    "github_user": "pmgbergen",
    "github_repo": "darsia",
    "github_version": "dev",
    "doc_path": "docs",
}

# -- linkcode: map documented objects back to GitHub source ------------------

_REVISION = os.environ.get("DARSIA_DOCS_REVISION", "dev")


def linkcode_resolve(domain, info):
    """Return the GitHub URL of the source for a documented Python object."""
    if domain != "py" or not info["module"]:
        return None

    module = sys.modules.get(info["module"])
    if module is None:
        return None

    obj = module
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    obj = inspect.unwrap(obj)
    try:
        source_file = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None
    if not source_file:
        return None

    import darsia

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(darsia.__file__)))
    try:
        rel_path = os.path.relpath(source_file, repo_root).replace(os.sep, "/")
    except ValueError:
        return None
    if rel_path.startswith(".."):
        return None

    end = start + len(lines) - 1
    return (
        f"https://github.com/pmgbergen/darsia/blob/{_REVISION}/"
        f"{rel_path}#L{start}-L{end}"
    )
