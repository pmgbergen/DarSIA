# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DarSIA'
copyright = '2023, Jakub Wiktor Both, Jan Martin Nordbotten, Erlend Storvik'
author = 'Jakub Wiktor Both, Jan Martin Nordbotten, Erlend Storvik'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Name of the root document or "homepage"
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

# Removes the module name space in front of classes and functions
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_short_title = "DarSIA"
html_split_index = True
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False

html_theme_options = {
  "show_toc_level": 4 # TODO
}

# -- Autodoc Settings -------------------------------------------------------------------------

# autoclass concatenates docs strings from init and class.
autoclass_content = "class"  # class-both-init

# Display the signature next to class name
autodoc_class_signature = "mixed"  # mixed-separated

# orders the members of an object group wise, e.g. private, special or public methods
autodoc_member_order = "groupwise"  # alphabetical-groupwise-bysource

# type hints will be shortened:
autodoc_typehints_format = "short"

# default configurations for all autodoc directives
autodoc_default_options = {
    "members": True,
    "special-members": False,
    "private-members": False,
    "show-inheritance": True,
    "inherited-members": True,
    "no-value": False
}

# uses type hints in signatures for e.g. linking (default)
autodoc_typehints = "none" #TODO "description"

# Avoid double appearance of documentation if child member has no docs
autodoc_inherit_docstrings = False

# Used to shorten the parsing of type hint aliases
autodoc_type_aliases = {}

# -- Intersphinx Settings ---------------------------------------------------------------------

intersphinx_mapping = {
    'python3': ("https://docs.python.org/3", None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'skimage': ('https://scikit-image.org/docs/stabe', None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

