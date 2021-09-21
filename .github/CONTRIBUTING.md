
See the [Scikit-HEP Developer introduction][skhep-dev-intro] for a
detailed description of best practices for developing Scikit-HEP packages.

[skhep-dev-intro]: https://scikit-hep.org/developer/intro

# Contributing

## Setting up a development environment

### Nox

The fastest way to start with development is to use nox. If you don't have nox,
you can use `pipx run nox` to run it without installing, or `pipx install nox`.
If you don't have pipx (pip for applications), then you can install with with
`pip install pipx` (the only case were installing an application with regular
pip is reasonable). If you use macOS, then pipx and nox are both in brew, use
`brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests-3.9  # Python 3.9 tests only
$ nox -s docs -- serve  # Build and serve the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run. On Linux, it will run the `--mpl` tests. You can
run the linux tests from anywhere with Docker:

```bash
docker run --rm -v $PWD:/nox -w /nox -t quay.io/pypa/manylinux2014_x86_64:latest pipx run nox -s tests-3.9
# Regenerate the MPL comparison images:
docker run --rm -v $PWD:/nox -w /nox -t quay.io/pypa/manylinux2014_x86_64:latest pipx run nox -s tests-3.9 -- --mpl-generate-path=tests/baseline
```

### PyPI

For extended development, you can set up a development environment using PyPI.

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

## Post setup

You should prepare pre-commit, which will help you by checking that commits
pass required checks:

```bash
pipx install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or `pre-commit
run --all-files` to check even without installing the hook.

## Testing

Use PyTest to run the unit checks:

```bash
pytest
```
