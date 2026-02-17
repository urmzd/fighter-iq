"""Pre-render annotated video from analysis JSON and source video."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from rich.progress import Progress

from fight_analyzer import AnalysisResult, BBox, FighterDetection, FrameAnalysis, Keypoint
from fight_analyzer.visualizer import draw_annotations


def _find_surrounding_frames(analysis: AnalysisResult, timestamp: float) -> tuple[int | None, int | None]:
    """Find indices of the nearest preceding and following analyzed frames."""
    prev_idx = None
    next_idx = None
    for i, frame in enumerate(analysis.frames):
        if frame.timestamp <= timestamp:
            prev_idx = i
        else:
            next_idx = i
            break
    return prev_idx, next_idx


def _interpolate_fighters(
    prev_frame: FrameAnalysis,
    next_frame: FrameAnalysis,
    alpha: float,
) -> list[FighterDetection]:
    """Linearly interpolate fighter keypoints and bboxes between two frames.

    alpha=0.0 → prev_frame positions, alpha=1.0 → next_frame positions.
    Matches fighters by identity. Unmatched fighters use whichever frame has them.
    """
    # Build identity → detection maps
    prev_map = {f.identity: f for f in prev_frame.fighters if f.identity is not None}
    next_map = {f.identity: f for f in next_frame.fighters if f.identity is not None}

    all_identities = list(dict.fromkeys(
        list(prev_map.keys()) + list(next_map.keys())
    ))

    result = []
    for ident in all_identities:
        pf = prev_map.get(ident)
        nf = next_map.get(ident)

        if pf is not None and nf is not None:
            # Interpolate bbox
            bbox = BBox(
                x1=pf.bbox.x1 + alpha * (nf.bbox.x1 - pf.bbox.x1),
                y1=pf.bbox.y1 + alpha * (nf.bbox.y1 - pf.bbox.y1),
                x2=pf.bbox.x2 + alpha * (nf.bbox.x2 - pf.bbox.x2),
                y2=pf.bbox.y2 + alpha * (nf.bbox.y2 - pf.bbox.y2),
            )
            # Interpolate keypoints
            keypoints = None
            if pf.keypoints and nf.keypoints and len(pf.keypoints) == len(nf.keypoints):
                keypoints = [
                    Keypoint(
                        x=pk.x + alpha * (nk.x - pk.x),
                        y=pk.y + alpha * (nk.y - pk.y),
                        confidence=min(pk.confidence, nk.confidence),
                    )
                    for pk, nk in zip(pf.keypoints, nf.keypoints)
                ]
            confidence = pf.confidence + alpha * (nf.confidence - pf.confidence)
            result.append(FighterDetection(bbox=bbox, confidence=confidence, keypoints=keypoints, identity=ident))
        elif pf is not None:
            result.append(pf)
        else:
            result.append(nf)

    return result


def render_annotated_video(
    video_path: Path,
    analysis: AnalysisResult,
    output_path: Path,
) -> Path:
    """Read original video frame-by-frame, draw analysis annotations, write annotated MP4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_duration = analysis.settings.get("max_duration")
    if max_duration is not None:
        total_frames = min(total_frames, int(max_duration * fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    prev_fighters: list[FighterDetection] | None = None

    try:
        with Progress() as progress:
            task = progress.add_task("Rendering annotated video...", total=total_frames)
            frame_idx = 0

            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps
                if max_duration is not None and timestamp > max_duration:
                    break
                prev_idx, next_idx = _find_surrounding_frames(analysis, timestamp)

                if prev_idx is not None:
                    fa = analysis.frames[prev_idx]

                    # Attempt interpolation to the next analyzed frame
                    fighters_to_draw = fa.fighters
                    if (
                        next_idx is not None
                        and not fa.incomplete
                        and not analysis.frames[next_idx].incomplete
                    ):
                        next_fa = analysis.frames[next_idx]
                        span = next_fa.timestamp - fa.timestamp
                        if span > 0:
                            alpha = (timestamp - fa.timestamp) / span
                            fighters_to_draw = _interpolate_fighters(fa, next_fa, alpha)

                    pil_image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                    annotated = draw_annotations(
                        pil_image,
                        fighters_to_draw,
                        fa.control_score,
                        fa.impact,
                        fa.impact_type,
                        fa.movement_vectors,
                        prev_fighters,
                        incomplete=fa.incomplete,
                    )
                    writer.write(annotated)

                    if not fa.incomplete:
                        prev_fighters = fa.fighters
                else:
                    writer.write(bgr_frame)

                frame_idx += 1
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()

    return output_path
