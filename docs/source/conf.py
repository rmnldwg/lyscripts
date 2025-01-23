"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html.
"""

import lyscripts

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lyscripts"
copyright = "2022, Roman Ludwig"  # noqa: A001
author = "Roman Ludwig"
gh_username = "rmnldwg"
version = lyscripts.__version__
release = lyscripts.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.programoutput",
    "sphinxcontrib.autodoc_pydantic",
    "myst_parser",
]

# markdown to reST
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = []

# document classes and their constructors
autoclass_content = "class"

# sort members by source
autodoc_member_order = "bysource"

# show type hints
autodoc_typehints = "signature"

# create links to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "lymph": ("https://lymph-model.readthedocs.io/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "lydata": ("https://lydata.readthedocs.io/stable/", None),
    "emcee": ("https://emcee.readthedocs.io/en/stable/", None),
}

# autodoc_pydantic settings
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_member_order = "bysource"
autodoc_pydantic_field_show_constraints = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/{gh_username}/{project}",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_navbar_depth": 3,
    "home_page_in_toc": True,
}
html_favicon = "_static/favicon.png"

# import sphinx_modern_theme
# html_theme = "sphinx_modern_theme"
# html_theme_path = [sphinx_modern_theme.get_html_theme_path()]

# html_theme = "bootstrap-astropy"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["./_static"]
html_css_files = [
    "css/custom.css",
]
