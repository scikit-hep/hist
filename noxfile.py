#!/usr/bin/env -S uv run -q
# /// script
# dependencies = ["nox>=2025.2.9"]
# ///

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import nox

nox.needs_version = ">=2025.2.9"
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


@nox.session
def tests(session):
    """
    Run the unit and regular tests.
    """
    session.install("-e.[plot]", "--group=test")
    args = ["--mpl"] if sys.platform.startswith("linux") else []
    session.run("pytest", *args, *session.posargs)


@nox.session(venv_backend="uv", default=False)
def minimums(session):
    """
    Run with the minimum dependencies.
    """

    session.install("-e.", "--group=test", "--resolution=lowest-direct")
    session.run("pytest", *session.posargs)


@nox.session(default=False)
def regenerate(session):
    """
    Regenerate MPL images.
    """
    session.install("-e.[plot]", "--group=test")
    if not sys.platform.startswith("linux"):
        session.error(
            "Must be run from Linux, images will be slightly different on macOS"
        )
    session.run("pytest", "--mpl-generate-path=tests/baseline", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Use "--non-interactive" to avoid serving. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install("-e.", "--group=docs", *extra_installs)

    session.chdir("docs")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(reuse_venv=True, default=False)
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs. Requires Linux.

    For example:

        docker run -v $PWD:/histogram -w /histogram -it quay.io/pypa/manylinux_2_28_x86_64
        uvx nox -s regenerate
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


@nox.session(reuse_venv=True, default=False)
def build(session):
    """
    Build an SDist and wheel.
    """

    args = [] if shutil.which("uv") else ["uv"]
    session.install("build", *args)
    session.run("python", "-m", "build", "--installer=uv")


@nox.session(default=False)
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


if __name__ == "__main__":
    nox.main()
