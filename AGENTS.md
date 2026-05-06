# Agent Notes: hist

## Package

Single `src/hist/` package, built with `hatchling` + `hatch-vcs`. Version is auto-generated from git tags into `src/hist/version.py`.

## Architecture

- **Core hierarchy**: `BaseHist` subclasses `boost_histogram.Histogram` and provides all named-axis logic, UHI+ indexing, plotting, and fills. `Hist` and `NamedHist` are thin subclasses in the `hist` family.
  - `Hist` — allows positional and named access.
  - `NamedHist` — enforces name-only fill, index, and project (raises on positional args).
- **Axes**: `hist.axis.*` wraps `boost_histogram.axis` classes with `AxesMixin`, adding `name` and `label` metadata stored in `_raw_metadata`.
- **Quick construct**: `Hist.new.Reg(...).Var(...).Int64()` is powered by `MetaConstructor` (metaclass on `BaseHist`) + `QuickConstruct`/`ConstructProxy`. Exposed as `hist.new`.
- **Stack**: A standalone generic container (`Stack[S]`) over `BaseHist` instances, not a histogram subclass. Created via `h.stack(axis)` or `Stack(h1, h2, ...)`.
- **Plotting**: `BaseHist.plot*()` methods delegate to the `hist.plot` module (requires `mplhep` + `matplotlib`). `plot_pull`/`plot_ratio` additionally need `scipy` + `iminuit`.
- **Dask support**: `hist.dask` is an optional subpackage requiring `dask_histogram`. Provides `dask.Hist` and `dask.NamedHist`.
- **Interop / flattening**: `fill_flattened()` uses `interop.destructure()` and `interop.broadcast_and_flatten()` for Awkward Array support.
- **Serialization**: `hist.serialization` wraps `boost_histogram.serialization` and injects a `hist` writer_info block into UHI dicts.
- **Entry points**: `from hist import Hist, NamedHist, Stack, BaseHist`; CLI `hist` points to `hist.classichist:main`.

## Key Commands

| Goal | Command |
|------|---------|
| Install for dev | `uv sync` |
| Lint / format / type check | `prek -a` or `nox -s lint` |
| Type check only | `prek -a mypy` (strict on `src/`) |
| Run tests | `nox -s tests` or `uv run pytest` |
| Run full test suite (incl. mpl) | `uv run pytest --mpl` |
| Run single test file | `uv run pytest tests/test_plot.py --mpl` |
| Test minimum deps | `nox -s minimums` |
| Build docs | `nox -s docs` |
| Build package | `nox -s build` |

## Important Toolchain Details

- **pytest `--mpl`**: Matplotlib image comparison tests only run with `--mpl`. On non-Linux they are silently skipped because baselines are Linux-generated. Regenerating baselines (`nox -s regenerate`) requires Linux / Docker.
- **pytest treats warnings as errors** (`filterwarnings = ["error"]`). Any deprecation warning will fail the test suite.
- **pytest `xfail_strict = true`**: An unexpected pass on an `xfail` test is treated as a failure.
- **mypy is strict** (`strict = true`, `warn_unreachable = true`). Do not relax without justification.
- **Ruff requires `from __future__ import annotations`** in every file (enforced via `isort.required-imports`).
- **Pre-commit mypy** runs on `^src` only and has pinned stub dependencies in `.pre-commit-config.yaml`; update those if needed.

## Dependency Groups

| Group | Contents |
|-------|----------|
| `test-core` | pytest, pytest-mpl, packaging, plot deps |
| `test` | test-core + fit + awkward |
| `plot` | matplotlib + mplhep |
| `fit` | scipy + iminuit |
| `dask` | dask[dataframe], dask_awkward, dask_histogram |
| `dev` | test + plot + dask + ipykernel |
| `docs` | dev + sphinx stack |

CI installs `test` + `dask`, then separately `plot` for the `--mpl` test job.

## CI / Workflow

- `ci.yml` runs: pylint, Python 3.10–3.14 checks, minimum-dependency checks.
- `cd.yml` runs on push to `main` and releases; builds via `hynek/build-and-inspect-python-package` and publishes on release.
- Do not run `git commit` or `git push` unless explicitly asked.
