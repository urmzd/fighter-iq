"""Frame extraction from video files using OpenCV."""

from collections.abc import Generator
from pathlib import Path

import cv2
from PIL import Image


def extract_frames(
    video_path: Path,
    interval: float = 1.0,
    max_duration: float | None = None,
) -> Generator[tuple[float, Image.Image], None, None]:
    """Extract frames from a video at the given interval (seconds).

    Yields (timestamp_seconds, PIL.Image) tuples. When max_duration is set,
    stops after the first frame whose timestamp exceeds that limit.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Invalid FPS in video: {video_path}")

    frame_interval = int(fps * interval)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                if max_duration is not None and timestamp > max_duration:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                yield (timestamp, pil_image)

            frame_idx += 1
    finally:
        cap.release()
