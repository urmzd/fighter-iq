"""Video frame ingestor — produces frames from video files."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from PIL import Image

from fighter_iq.extractor import extract_frames


class VideoIngestor:
    """Extracts frames from a video file at configurable intervals.

    Implements the Ingestor protocol.
    """

    def extract(
        self,
        source: Path,
        interval: float = 1.0,
        max_duration: float | None = None,
    ) -> Generator[tuple[float, Image.Image], None, None]:
        """Yield (timestamp_seconds, PIL.Image) tuples from a video file."""
        yield from extract_frames(source, interval, max_duration)
