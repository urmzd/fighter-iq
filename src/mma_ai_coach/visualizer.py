"""Frame annotation and display using OpenCV."""

import cv2
import numpy as np
from PIL import Image

from mma_ai_coach import FighterDetection

# Fighter colors (BGR for OpenCV)
_COLOR_A = (0, 255, 0)   # green
_COLOR_B = (255, 100, 0)  # blue
_COLOR_IMPACT = (0, 0, 255)  # red
_COLOR_VECTOR = (0, 255, 255)  # yellow
_COLOR_REFEREE = (128, 128, 128)  # gray

_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),      # face
    (3, 5), (4, 6),                                  # ears → shoulders
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),                      # torso
    (11, 13), (12, 14), (13, 15), (14, 16),          # legs
]
_KEYPOINT_VISIBILITY_THRESHOLD = 0.5


def _draw_skeleton(
    frame: np.ndarray,
    keypoints: list,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw skeleton connections and keypoint dots for a fighter."""
    # Draw limb connections
    for i, j in _SKELETON_PAIRS:
        kp_a, kp_b = keypoints[i], keypoints[j]
        if kp_a.confidence >= _KEYPOINT_VISIBILITY_THRESHOLD and kp_b.confidence >= _KEYPOINT_VISIBILITY_THRESHOLD:
            pt1 = (int(kp_a.x), int(kp_a.y))
            pt2 = (int(kp_b.x), int(kp_b.y))
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoint dots
    radius = max(2, thickness)
    for kp in keypoints:
        if kp.confidence >= _KEYPOINT_VISIBILITY_THRESHOLD:
            cv2.circle(frame, (int(kp.x), int(kp.y)), radius, color, -1)


def draw_annotations(
    image: Image.Image,
    fighters: list[FighterDetection],
    control_score: float | None,
    impact: bool,
    impact_type: str | None,
    movement_vectors: list[tuple[float, float]],
    prev_fighters: list[FighterDetection] | None,
    incomplete: bool = False,
    referee: FighterDetection | None = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, movement vectors, and overlays on a frame."""
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    # Scale line thickness and font relative to frame size
    scale = max(w, h) / 1000.0
    thickness = max(1, int(2 * scale))
    font_scale = 0.5 * scale
    font = cv2.FONT_HERSHEY_SIMPLEX

    colors = [_COLOR_A, _COLOR_B]
    labels = ["Fighter A", "Fighter B"]

    for i, fighter in enumerate(fighters[:2]):
        color = colors[i]
        label = labels[i]
        b = fighter.bbox

        # Skeleton or bounding box
        if fighter.keypoints is not None:
            _draw_skeleton(frame, fighter.keypoints, color, thickness)
        else:
            pt1 = (int(b.x1), int(b.y1))
            pt2 = (int(b.x2), int(b.y2))
            cv2.rectangle(frame, pt1, pt2, color, thickness)

        # Label with confidence
        text = f"{label} {fighter.confidence:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        label_y = max(int(b.y1) - 6, text_size[1] + 4)
        cv2.putText(frame, text, (int(b.x1), label_y), font, font_scale, color, thickness)

        # Movement vector (arrow from previous center to current center)
        if prev_fighters and i < len(prev_fighters) and i < len(movement_vectors):
            dx, dy = movement_vectors[i]
            if abs(dx) > 1 or abs(dy) > 1:
                prev_center = prev_fighters[i].bbox.center
                curr_center = fighter.bbox.center
                cv2.arrowedLine(
                    frame,
                    (int(prev_center[0]), int(prev_center[1])),
                    (int(curr_center[0]), int(curr_center[1])),
                    _COLOR_VECTOR,
                    thickness,
                    tipLength=0.3,
                )

    # Referee bounding box (gray)
    if referee is not None:
        b = referee.bbox
        pt1 = (int(b.x1), int(b.y1))
        pt2 = (int(b.x2), int(b.y2))
        cv2.rectangle(frame, pt1, pt2, _COLOR_REFEREE, thickness)
        ref_text = "Referee"
        text_size = cv2.getTextSize(ref_text, font, font_scale, thickness)[0]
        label_y = max(int(b.y1) - 6, text_size[1] + 4)
        cv2.putText(frame, ref_text, (int(b.x1), label_y), font, font_scale, _COLOR_REFEREE, thickness)

    bar_h = max(12, int(20 * scale))
    bar_y = 4

    if incomplete:
        # Orange "INCOMPLETE" notice instead of control bar
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (0, 140, 255), -1)
        cv2.putText(
            frame, "INCOMPLETE — <2 fighters detected",
            (4, bar_y + bar_h - 3), font, font_scale * 0.8, (255, 255, 255), max(1, thickness - 1),
        )
    else:
        # Control score bar at top of frame
        mid_x = w // 2
        score = control_score if control_score is not None else 0.0

        # Background bar (dark gray)
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)

        # Fill: green from center-left for A, blue from center-right for B
        fill_width = int(abs(score) * mid_x)
        if score >= 0:
            cv2.rectangle(frame, (mid_x - fill_width, bar_y), (mid_x, bar_y + bar_h), _COLOR_A, -1)
        else:
            cv2.rectangle(frame, (mid_x, bar_y), (mid_x + fill_width, bar_y + bar_h), _COLOR_B, -1)

        # Center tick mark
        cv2.line(frame, (mid_x, bar_y), (mid_x, bar_y + bar_h), (255, 255, 255), 1)

        # Control label
        ctrl_dir = "A" if score >= 0 else "B"
        ctrl_text = f"Control: {abs(score):.2f} {ctrl_dir}"
        cv2.putText(frame, ctrl_text, (4, bar_y + bar_h + int(16 * scale)), font, font_scale * 0.8, (255, 255, 255), max(1, thickness - 1))

    # Impact border
    if impact:
        border = max(3, int(4 * scale))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _COLOR_IMPACT, border)
        impact_label = f"IMPACT: {impact_type or 'unknown'}"
        cv2.putText(frame, impact_label, (10, h - 10), font, font_scale, _COLOR_IMPACT, thickness)

    return frame


def show_frame(window_name: str, frame: np.ndarray, wait_ms: int) -> bool:
    """Display a frame in an OpenCV window. Returns False if user presses 'q'."""
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key != ord("q")
