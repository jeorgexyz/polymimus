"""Entry point for `python -m polymimus`.

The console script installed by pyproject.toml points at `polymimus.cli:app`
and works once the package is installed. Without this module, though,
`python -m polymimus` does nothing at all, and `python -m polymimus.cli`
exits 0 while printing nothing -- cli.py defines the Typer app but never
calls it, so importing the module as __main__ just builds the app and
returns.

Silence with a zero exit code is a bad failure: it looks like the command
ran and had nothing to say.
"""

from polymimus.cli import app

if __name__ == "__main__":
    app()
