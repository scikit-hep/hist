import shutil
from pathlib import Path

import nox

ALL_PYTHONS = ["3.6", "3.7", "3.8", "3.9"]

nox.options.sessions = ["lint", "tests"]


DIR = Path(__file__).parent.resolve()


@nox.session
def lint(session):
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session):
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def docs(session):
    """
    Build the docs. Pass "serve" to serve.
    """

    session.install(".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
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
