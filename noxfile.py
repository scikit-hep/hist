from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nox

ALL_PYTHONS = ["3.7", "3.8", "3.9", "3.10"]

nox.options.sessions = ["lint", "tests"]


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

    session.install("pylint~=2.14.0")
    session.install("-e", ".")
    session.run("pylint", "src", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
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
def docs(session):
    """
    Build the docs. Pass "serve" to serve.
    """

    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8001/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8001", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")


@nox.session
def build(session):
    """
    Build an SDist and wheel.
    """

    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build")
