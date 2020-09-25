
See the [Scikit-HEP Developer introduction][skhep-dev-intro] for a
detailed description of best practices for developing Scikit-HEP packages.

[skhep-dev-intro]: https://scikit-hep.org/developer/intro

# Setting up a development environment

### PyPI

You can set up a development environment using PyPI.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv)$ pip install -e .[dev]
(venv)$ python -m ipykernel install --user --name hist
```

You should have pip 10 or later.

### Conda

You can also set up a development environment using Conda. With conda, you can search some channels for development.

```bash
$ conda env create -f dev-environment.yml -n hist
$ conda activate hist
(hist)$ python -m ipykernel install --name hist
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits
pass required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or `pre-commit
run --all-files` to check even without installing the hook.

# Testing

Use PyTest to run the unit checks:

```bash
pytest
```
