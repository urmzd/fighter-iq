"""Shared terminal UI helpers — world-class CLI output via rich.

All output goes to stderr so stdout remains clean for piped data.
"""

import sys
from contextlib import contextmanager
from typing import Generator

from rich.console import Console

console = Console(stderr=True)


def header(cmd: str) -> None:
    """Print a cyan bold command header with a 40-char dim rule."""
    console.print(f"\n  [cyan bold]{cmd}[/]")
    console.print(f"  [dim]{'─' * 40}[/]\n")


def phase_ok(msg: str, detail: str = "") -> None:
    """Print a green checkmark with an optional dim detail."""
    suffix = f" [dim]· {detail}[/]" if detail else ""
    console.print(f"  [green bold]✓[/] {msg}{suffix}")


def warn(msg: str) -> None:
    """Print a yellow warning."""
    console.print(f"  [yellow bold]⚠[/] [yellow]{msg}[/]")


def info(msg: str) -> None:
    """Print a cyan info marker with dim text."""
    console.print(f"  [cyan]ℹ[/] [dim]{msg}[/]")


def error(msg: str) -> None:
    """Print a red error message."""
    console.print(f"  [red bold]✗[/] [red]{msg}[/]")


@contextmanager
def spinner(msg: str) -> Generator[None, None, None]:
    """Show a spinner while a long operation runs."""
    with console.status(f"  [cyan]{msg}[/]", spinner="dots"):
        yield
