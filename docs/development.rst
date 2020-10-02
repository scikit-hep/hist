Development
===========================

We welcome you to contribute to this project. If you want to develop this package, you can
use the following methods.

Pip
------------------------

You can set up a development environment using pip.

.. code-block:: bash

   python3 -m venv .env            # Make a new environment in ./.env/
   source .env/bin/activate        # Use the new environment
   (.env)$ pip install -e .[dev]
   (.env)$ python -m ipykernel install --user --name hist

*You should have pip 10 or later*.

Conda
-------------------------

You can also set up a development environment using Conda. With Conda, you can search some channels for development.

.. code-block:: bash

   $ conda env create -f dev-environment.yml -n hist
   $ conda activate hist
   (hist)$ python -m ipykernel install --name hist
