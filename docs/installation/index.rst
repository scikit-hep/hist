.. _installation:

Installation
============

PyPI
----------

You can set up a development environment using PyPI.

::

   python3 -m venv .env            # Make a new environment in ./.env/
   source .env/bin/activate        # Use the new environment
   (.env)$ pip install -e .[dev]
   (.env)$ python -m ipykernel install --user --name hist

*You should have pip 10 or later*.

Conda
-----------

You can also set up a development environment using Conda. With Conda, you can search some channels for development.

::

   $ conda env create -f dev-environment.yml -n hist
   $ conda activate hist
   (hist)$ python -m ipykernel install --name hist
