
## Development install

### PyPI

You can set up a development environment using PyPI.

```bash
$ python -m venv .env
$ source .env/bin/activate
(.env)$ pip install -r requirements.txt
(.env)$ pip install -e .[dev]
(.env)$ python -m ipykernel install --user --name hist
```

- You should have pip 10 or later.

### Conda

You can also set up a development environment using Conda. With conda, you can search some channels for development.

```bash
$ conda env create -f environment.yml -n hist
$ conda activate hist
(hist)$ python -m ipykernel install --name hist
```
