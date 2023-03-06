# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

# Warning: do not change the path here. To use autodoc, you need to install the
# package first.
from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------

project = "Hist"
copyright = "2020, Henry Schreiner"
author = "Henry Schreiner and Nino Lau"
version = get_distribution("hist").version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinxcontrib.programoutput",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    ".env",
    "user-guide/notebooks/SVGHistogram.ipynb",
    "Thumbs.db",
    "_build",
    "banner_slides.md",
    "build",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Config for the Sphinx book

html_baseurl = "https://hist.readthedocs.io/en/latest/"

html_theme_options = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/scikit-hep/hist",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: list[str] = []


# -- Options for Notebook input ----------------------------------------------

html_logo = "_images/histlogo.png"
html_title = f"Hist {version}"

nbsphinx_execute = "auto"  # auto, never

highlight_language = "python3"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png2x'}",
    "--InlineBackend.rc=figure.dpi=96",
]

nbsphinx_kernel_name = "python3"
