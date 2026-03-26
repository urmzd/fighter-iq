"""Spatial analysis: control scoring, proximity, movement vectors, impact detection."""

import numpy as np

from fighter_iq import BBox, FighterDetection

# Keywords that indicate impact in VLM descriptions
_STRIKE_KEYWORDS = [
    "punch", "punches", "punching",
    "kick", "kicks", "kicking",
    "elbow", "elbows",
    "knee", "knees", "kneeing",
    "strike", "strikes", "striking",
    "hit", "hits", "hitting",
    "jab", "jabs", "jabbing",
    "cross", "hook", "uppercut",
    "roundhouse", "front kick", "side kick",
    "spinning", "landed", "connects",
]
_TAKEDOWN_KEYWORDS = [
    "takedown", "taken down", "takes down",
    "slam", "slams", "slamming",
    "sweep", "sweeps", "sweeping",
    "trip", "trips", "tripping",
    "throw", "throws", "throwing",
    "wrestle", "wrestling",
    "shoots", "shot", "double leg", "single leg",
    "suplex",
]
_KNOCKDOWN_KEYWORDS = [
    "knockdown", "knocked down", "drops", "dropped",
    "falls", "fell", "on the canvas", "wobbled",
]


def _compute_iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_control(
    fighters_prev: list[FighterDetection] | None,
    fighters_curr: list[FighterDetection],
    frame_width: int,
    frame_height: int,
) -> float:
    """Compute control score: -1.0 (Fighter B controls) to 1.0 (Fighter A controls).

    Factors:
    - Vertical position (lower y-center = higher on screen = top position in grappling)
    - Forward pressure (advancing toward opponent)
    - Cage/boundary proximity penalty
    """
    if len(fighters_curr) < 2:
        return 0.0

    a, b = fighters_curr[0], fighters_curr[1]
    score = 0.0

    # Vertical position: fighter higher on screen (lower y) may have top control
    a_cy = a.bbox.center[1] / frame_height
    b_cy = b.bbox.center[1] / frame_height
    vertical_diff = b_cy - a_cy  # positive if A is higher (has top position)
    score += vertical_diff * 0.4

    # Forward pressure: who is advancing toward the opponent
    if fighters_prev is not None and len(fighters_prev) >= 2:
        a_prev, b_prev = fighters_prev[0], fighters_prev[1]

        # Distance between fighters: current vs previous
        dist_curr = np.sqrt(
            (a.bbox.center[0] - b.bbox.center[0]) ** 2
            + (a.bbox.center[1] - b.bbox.center[1]) ** 2
        )
        dist_prev = np.sqrt(
            (a_prev.bbox.center[0] - b_prev.bbox.center[0]) ** 2
            + (a_prev.bbox.center[1] - b_prev.bbox.center[1]) ** 2
        )

        # A advancing = A moved toward B (distance decreased + A moved more)
        a_displacement = np.sqrt(
            (a.bbox.center[0] - a_prev.bbox.center[0]) ** 2
            + (a.bbox.center[1] - a_prev.bbox.center[1]) ** 2
        )
        b_displacement = np.sqrt(
            (b.bbox.center[0] - b_prev.bbox.center[0]) ** 2
            + (b.bbox.center[1] - b_prev.bbox.center[1]) ** 2
        )

        if dist_curr < dist_prev:  # Fighters getting closer
            if a_displacement > b_displacement:
                score += 0.3  # A is pressing
            elif b_displacement > a_displacement:
                score -= 0.3  # B is pressing

    # Cage proximity penalty: fighter closer to frame edge has less control
    a_edge_dist = min(
        a.bbox.center[0], frame_width - a.bbox.center[0],
        a.bbox.center[1], frame_height - a.bbox.center[1],
    ) / max(frame_width, frame_height)

    b_edge_dist = min(
        b.bbox.center[0], frame_width - b.bbox.center[0],
        b.bbox.center[1], frame_height - b.bbox.center[1],
    ) / max(frame_width, frame_height)

    edge_diff = a_edge_dist - b_edge_dist  # positive if A is further from edge
    score += edge_diff * 0.3

    return float(np.clip(score, -1.0, 1.0))


def compute_proximity(
    fighters: list[FighterDetection],
    frame_width: int,
    frame_height: int,
) -> list[float]:
    """Compute normalized distance of each fighter's center to the frame center (0-1)."""
    cx, cy = frame_width / 2, frame_height / 2
    diag = np.sqrt(cx**2 + cy**2)

    proximities = []
    for f in fighters:
        fx, fy = f.bbox.center
        dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
        proximities.append(float(dist / diag))

    return proximities


def compute_movement_vectors(
    fighters_prev: list[FighterDetection] | None,
    fighters_curr: list[FighterDetection],
) -> list[tuple[float, float]]:
    """Compute displacement vectors (dx, dy) between consecutive frames per fighter."""
    if fighters_prev is None or len(fighters_prev) == 0:
        return [(0.0, 0.0)] * len(fighters_curr)

    vectors = []
    for i, curr in enumerate(fighters_curr):
        if i < len(fighters_prev):
            prev = fighters_prev[i]
            dx = curr.bbox.center[0] - prev.bbox.center[0]
            dy = curr.bbox.center[1] - prev.bbox.center[1]
            vectors.append((float(dx), float(dy)))
        else:
            vectors.append((0.0, 0.0))

    return vectors


def detect_impact(
    fighters_prev: list[FighterDetection] | None,
    fighters_curr: list[FighterDetection],
    description: str,
) -> tuple[bool, str | None]:
    """Detect impact events (strikes, takedowns, sweeps).

    Uses bbox overlap, sudden displacement, and VLM description keywords.
    """
    desc_lower = description.lower()

    # Check for bbox overlap between fighters
    high_overlap = False
    if len(fighters_curr) >= 2:
        iou = _compute_iou(fighters_curr[0].bbox, fighters_curr[1].bbox)
        high_overlap = iou > 0.15

    # Check for sudden large displacement (possible takedown/knockdown)
    large_displacement = False
    if fighters_prev is not None and len(fighters_prev) >= 2 and len(fighters_curr) >= 2:
        for i in range(min(2, len(fighters_curr))):
            if i < len(fighters_prev):
                dx = abs(fighters_curr[i].bbox.center[0] - fighters_prev[i].bbox.center[0])
                dy = abs(fighters_curr[i].bbox.center[1] - fighters_prev[i].bbox.center[1])
                displacement = np.sqrt(dx**2 + dy**2)
                if displacement > 100:  # significant movement threshold
                    large_displacement = True
                    break

    # Check VLM description for impact keywords
    has_strike_keyword = any(kw in desc_lower for kw in _STRIKE_KEYWORDS)
    has_takedown_keyword = any(kw in desc_lower for kw in _TAKEDOWN_KEYWORDS)
    has_knockdown_keyword = any(kw in desc_lower for kw in _KNOCKDOWN_KEYWORDS)

    # Determine impact type
    # Takedown/knockdown: large displacement + keyword confirmation
    if large_displacement and (has_takedown_keyword or has_knockdown_keyword):
        if has_takedown_keyword:
            return True, "takedown"
        return True, "knockdown"

    # Strike: bbox overlap + strike keyword
    if high_overlap and has_strike_keyword:
        return True, "strike"

    # Sweep: sweep keyword
    if any(kw in desc_lower for kw in ["sweep", "sweeps", "sweeping"]):
        return True, "sweep"

    # Fallback: large displacement with takedown keyword
    if has_takedown_keyword and large_displacement:
        return True, "takedown"

    return False, None
