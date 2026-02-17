"""Graceful shutdown via signal handling."""

import signal
import threading

_shutdown_event = threading.Event()


def request_shutdown(signum: int, frame: object) -> None:
    """Signal handler that sets the shutdown flag without raising."""
    _shutdown_event.set()


def is_shutdown_requested() -> bool:
    """Check whether a shutdown has been requested."""
    return _shutdown_event.is_set()


def install_signal_handlers() -> None:
    """Register SIGINT and SIGTERM to trigger graceful shutdown."""
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
