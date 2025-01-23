from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import nox

nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"

DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lint(session):
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run pylint.
    """

    session.install("pylint~=3.3.3")
    session.install("-e.")
    session.run("pylint", "hist", *session.posargs)


@nox.session(reuse_venv=True)
def tests(session):
    """
    Run the unit and regular tests.
    """
    session.install("-e", ".[test,plot]")
    args = ["--mpl"] if sys.platform.startswith("linux") else []
    session.run("pytest", *args, *session.posargs)


@nox.session
def regenerate(session):
    """
    Regenerate MPL images.
    """
    session.install("-e", ".[test,plot]")
    if not sys.platform.startswith("linux"):
        session.error(
            "Must be run from Linux, images will be slightly different on macOS"
        )
    session.run("pytest", "--mpl-generate-path=tests/baseline", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to serve.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    args = parser.parse_args(session.posargs)

    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if args.serve:
        print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
        session.run("python", "-m", "http.server", "8000", "-d", "_build/html")


@nox.session(reuse_venv=True)
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install("sphinx")
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "reference/",
        "--separate",
        "--force",
        "--module-first",
        "../src/hist",
    )


@nox.session(reuse_venv=True)
def build(session):
    """
    Build an SDist and wheel.
    """

    args = [] if shutil.which("uv") else ["uv"]
    session.install("build==1.2.0", *args)
    session.run("python", "-m", "build", "--installer=uv")


@nox.session()
def boost(session):
    """
    Build against latest boost-histogram.
    """

    tmpdir = session.create_tmp()
    session.chdir(tmpdir)
    # Support running with -r/-R
    if not Path("boost-histogram").is_dir():
        session.run(
            "git",
            "clone",
            "--recursive",
            "https://github.com/scikit-hep/boost-histogram",
            external=True,
        )
        session.chdir("boost-histogram")
        session.install(".")
    session.chdir(DIR)
    session.install("-e.[test,plot]", "pip")
    session.run("pip", "list")
    session.run("pytest", *session.posargs)
