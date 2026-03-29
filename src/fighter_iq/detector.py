"""Fighter bounding box detection using YOLOv8 with referee/spectator filtering."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from fighter_iq import (
    BBox,
    ColorHistogram,
    DetectedPerson,
    FighterDetection,
    FighterIdentity,
    FighterProfile,
    Keypoint,
    PersonRole,
)


def load_detector() -> YOLO:
    """Load YOLOv8-nano pose model for person detection with keypoints."""
    return YOLO("yolov8n-pose.pt")


def detect_persons(model: YOLO, image: Image.Image) -> list[DetectedPerson]:
    """Stage 1: Raw YOLO pose detection — returns persons with keypoints, confidence >= 0.80."""
    results = model(image, device="cpu", verbose=False)

    persons: list[DetectedPerson] = []
    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            if cls != 0:
                continue
            conf = float(box.conf[0])
            if conf < 0.80:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Extract 17 COCO keypoints
            kp_xy = result.keypoints.xy[i]  # shape (17, 2)
            kp_conf = result.keypoints.conf[i]  # shape (17,)
            keypoints = [
                Keypoint(x=float(kp_xy[j][0]), y=float(kp_xy[j][1]), confidence=float(kp_conf[j])) for j in range(17)
            ]

            persons.append(
                DetectedPerson(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    keypoints=keypoints,
                )
            )

    persons.sort(key=lambda d: d.confidence, reverse=True)
    return persons


def filter_spectators(persons: list[DetectedPerson], frame_width: int, frame_height: int) -> list[DetectedPerson]:
    """Stage 2: Mark spectators based on bbox size and position.

    Only runs when >2 persons detected; with 2 or fewer, nothing to filter.
    """
    non_spectator_count = sum(1 for p in persons if p.role != PersonRole.SPECTATOR)
    if non_spectator_count <= 2:
        return persons

    frame_area = frame_width * frame_height
    margin_x = frame_width * 0.10
    margin_y = frame_height * 0.10

    for person in persons:
        if person.role != PersonRole.UNKNOWN:
            continue

        bbox = person.bbox
        cx, cy = bbox.center
        area_ratio = bbox.area / frame_area if frame_area > 0 else 0

        # Too small — far away or behind cage
        if area_ratio < 0.005:
            person.role = PersonRole.SPECTATOR
            continue

        # Center in peripheral margin
        in_margin = cx < margin_x or cx > frame_width - margin_x or cy < margin_y or cy > frame_height - margin_y
        if in_margin:
            person.role = PersonRole.SPECTATOR
            continue

        # Lower portion of frame AND small bbox — seated cageside
        if cy > frame_height * 0.75 and area_ratio < 0.02:
            person.role = PersonRole.SPECTATOR
            continue

    return persons


def filter_referee(persons: list[DetectedPerson], frame_width: int, frame_height: int) -> list[DetectedPerson]:
    """Stage 3: Mark the most referee-like candidate among non-spectators.

    Only runs when >2 non-spectator persons remain. Presumes the two largest
    bboxes (by area) are the fighters.
    """
    candidates = [p for p in persons if p.role == PersonRole.UNKNOWN]
    if len(candidates) <= 2:
        return persons

    # The two largest by area are presumed fighters
    candidates_by_area = sorted(candidates, key=lambda p: p.bbox.area, reverse=True)
    top_two = candidates_by_area[:2]
    remaining = candidates_by_area[2:]

    # Mark top two as fighters
    for p in top_two:
        p.role = PersonRole.FIGHTER

    # Midpoint between the two largest detections
    mid_x = (top_two[0].bbox.center[0] + top_two[1].bbox.center[0]) / 2
    mid_y = (top_two[0].bbox.center[1] + top_two[1].bbox.center[1]) / 2

    fighter_min_area = min(t.bbox.area for t in top_two)
    frame_diag = math.sqrt(frame_width**2 + frame_height**2)

    best_score = -1.0
    best_candidate: DetectedPerson | None = None

    for person in remaining:
        bbox = person.bbox
        score = 0.0

        # Upright aspect ratio (height/width > 1.8 → standing, not fighting) — 40%
        aspect = bbox.height / bbox.width if bbox.width > 0 else 0
        if aspect > 1.8:
            score += 0.4
        elif aspect > 1.2:
            score += 0.2 * ((aspect - 1.2) / 0.6)

        # Proximity to midpoint between the two largest — 30%
        dist = math.sqrt((bbox.center[0] - mid_x) ** 2 + (bbox.center[1] - mid_y) ** 2)
        proximity = 1.0 - min(dist / frame_diag, 1.0)
        score += proximity * 0.3

        # Smaller than both fighters — 30%
        if bbox.area < fighter_min_area:
            score += 0.3

        if score > best_score:
            best_score = score
            best_candidate = person

    if best_candidate is not None:
        best_candidate.role = PersonRole.REFEREE
        # Mark any other remaining as unknown (they stay UNKNOWN, handled as non-fighters)

    return persons


def _hsv_to_color_name(h: int, s: int, v: int) -> str:
    """Map HSV values to a human-readable color name."""
    if v < 40:
        return "black"
    if s < 30:
        return "white" if v > 200 else "gray"
    # Hue ranges (OpenCV uses 0-179)
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 170:
        return "purple"
    return "unknown"


def extract_color_histogram(image: Image.Image, bbox: BBox, lower_ratio: float = 0.5) -> ColorHistogram:
    """Extract an HSV color histogram from the lower portion (shorts region) of a bbox."""
    img_w, img_h = image.size
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(img_w, int(bbox.x2))
    y2 = min(img_h, int(bbox.y2))

    crop_h = y2 - y1
    lower_y1 = y1 + int(crop_h * (1 - lower_ratio))

    crop = image.crop((x1, lower_y1, x2, y2))

    if crop.width < 10 or crop.height < 10:
        empty = np.zeros(ColorHistogram.H_BINS * ColorHistogram.S_BINS, dtype=np.float32)
        return ColorHistogram(histogram=empty, dominant_color_name="unknown", dominant_hsv=(0, 0, 0))

    hsv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [ColorHistogram.H_BINS, ColorHistogram.S_BINS],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    flat = hist.flatten().astype(np.float32)

    # Dominant color from peak bin
    peak_idx = int(np.argmax(flat))
    peak_h_bin = peak_idx // ColorHistogram.S_BINS
    peak_s_bin = peak_idx % ColorHistogram.S_BINS
    dom_h = int(peak_h_bin * (180 / ColorHistogram.H_BINS))
    dom_s = int(peak_s_bin * (256 / ColorHistogram.S_BINS))
    dom_v = int(np.mean(hsv[:, :, 2]))
    color_name = _hsv_to_color_name(dom_h, dom_s, dom_v)

    return ColorHistogram(histogram=flat, dominant_color_name=color_name, dominant_hsv=(dom_h, dom_s, dom_v))


def initialize_profiles(fighters: list[DetectedPerson], image: Image.Image) -> list[FighterProfile]:
    """Create initial fighter profiles from the first confirmed fighter pair."""
    profiles: list[FighterProfile] = []
    identities = [FighterIdentity.FIGHTER_A, FighterIdentity.FIGHTER_B]
    for i, person in enumerate(fighters[:2]):
        hist = extract_color_histogram(image, person.bbox)
        profiles.append(
            FighterProfile(
                identity=identities[i],
                color_histogram=hist,
                appearance_description=f"{hist.dominant_color_name} shorts",
                last_bbox=person.bbox,
                last_confidence=person.confidence,
                frames_seen=1,
                frames_since_last_seen=0,
            )
        )
    return profiles


def update_profile(
    profile: FighterProfile, person: DetectedPerson, image: Image.Image, ema_alpha: float = 0.3
) -> FighterProfile:
    """Update a fighter profile with new observation via EMA blending."""
    new_hist = extract_color_histogram(image, person.bbox)
    blended = (1 - ema_alpha) * profile.color_histogram.histogram + ema_alpha * new_hist.histogram
    cv2.normalize(blended, blended, 0, 1, cv2.NORM_MINMAX)

    profile.color_histogram = ColorHistogram(
        histogram=blended,
        dominant_color_name=new_hist.dominant_color_name,
        dominant_hsv=new_hist.dominant_hsv,
    )
    profile.appearance_description = f"{new_hist.dominant_color_name} shorts"
    profile.last_bbox = person.bbox
    profile.last_confidence = person.confidence
    profile.frames_seen += 1
    profile.frames_since_last_seen = 0
    return profile


def match_profiles(
    persons: list[DetectedPerson],
    profiles: list[FighterProfile],
    image: Image.Image,
    similarity_threshold: float = 0.3,
    spatial_weight: float = 0.2,
) -> tuple[list[tuple[DetectedPerson, FighterProfile]], list[DetectedPerson]]:
    """Match detected persons to existing fighter profiles by appearance + proximity.

    Returns (matched_pairs, unmatched_persons).
    """
    if not profiles or not persons:
        return [], list(persons)

    # Compute score matrix
    n_persons = len(persons)
    n_profiles = len(profiles)
    scores = np.zeros((n_persons, n_profiles), dtype=np.float64)
    appearance_scores = np.zeros((n_persons, n_profiles), dtype=np.float64)

    person_hists: list[ColorHistogram] = []
    for person in persons:
        person_hists.append(extract_color_histogram(image, person.bbox))

    for i, (person, phist) in enumerate(zip(persons, person_hists)):
        for j, profile in enumerate(profiles):
            # Appearance score via histogram correlation [-1, 1]
            app = cv2.compareHist(
                phist.histogram.reshape(-1, 1),
                profile.color_histogram.histogram.reshape(-1, 1),
                cv2.HISTCMP_CORREL,
            )
            appearance_scores[i, j] = app

            # Spatial proximity — decay with staleness
            px, py = person.bbox.center
            lx, ly = profile.last_bbox.center
            dist = math.sqrt((px - lx) ** 2 + (py - ly) ** 2)
            max_dist = math.sqrt(image.width**2 + image.height**2)
            proximity = 1.0 - min(dist / max_dist, 1.0)
            staleness_factor = 1.0 / (1.0 + profile.frames_since_last_seen)
            spatial = proximity * staleness_factor

            scores[i, j] = (1 - spatial_weight) * app + spatial_weight * spatial

    # Greedy assignment (only 2 profiles)
    matched: list[tuple[DetectedPerson, FighterProfile]] = []
    used_persons: set[int] = set()
    used_profiles: set[int] = set()

    for _ in range(min(n_persons, n_profiles)):
        best_score = -float("inf")
        best_i, best_j = -1, -1
        for i in range(n_persons):
            if i in used_persons:
                continue
            for j in range(n_profiles):
                if j in used_profiles:
                    continue
                if scores[i, j] > best_score:
                    best_score = scores[i, j]
                    best_i, best_j = i, j
        if best_i < 0 or appearance_scores[best_i, best_j] < similarity_threshold:
            break
        matched.append((persons[best_i], profiles[best_j]))
        used_persons.add(best_i)
        used_profiles.add(best_j)

    unmatched = [p for i, p in enumerate(persons) if i not in used_persons]
    return matched, unmatched


def filter_referee_with_profiles(
    persons: list[DetectedPerson],
    matched_indices: set[int],
    frame_w: int,
    frame_h: int,
) -> list[DetectedPerson]:
    """Score unmatched persons as potential referee using appearance-informed filtering.

    Works even when <=2 persons are detected — the core bug fix.
    """
    # Mark matched persons as fighters
    fighters_for_ref: list[DetectedPerson] = []
    for i, p in enumerate(persons):
        if i in matched_indices:
            p.role = PersonRole.FIGHTER
            fighters_for_ref.append(p)

    remaining = [p for i, p in enumerate(persons) if i not in matched_indices and p.role == PersonRole.UNKNOWN]
    if not remaining or len(fighters_for_ref) < 1:
        return persons

    # Midpoint of matched fighters
    mid_x = sum(f.bbox.center[0] for f in fighters_for_ref) / len(fighters_for_ref)
    mid_y = sum(f.bbox.center[1] for f in fighters_for_ref) / len(fighters_for_ref)
    fighter_min_area = min(f.bbox.area for f in fighters_for_ref)
    frame_diag = math.sqrt(frame_w**2 + frame_h**2)

    best_score = -1.0
    best_candidate: DetectedPerson | None = None

    for person in remaining:
        bbox = person.bbox
        score = 0.0

        # Upright aspect ratio — 40%
        aspect = bbox.height / bbox.width if bbox.width > 0 else 0
        if aspect > 1.8:
            score += 0.4
        elif aspect > 1.2:
            score += 0.2 * ((aspect - 1.2) / 0.6)

        # Proximity to fighter midpoint — 30%
        dist = math.sqrt((bbox.center[0] - mid_x) ** 2 + (bbox.center[1] - mid_y) ** 2)
        proximity = 1.0 - min(dist / frame_diag, 1.0)
        score += proximity * 0.3

        # Smaller than fighters — 30%
        if bbox.area < fighter_min_area:
            score += 0.3

        if score > best_score:
            best_score = score
            best_candidate = person

    if best_candidate is not None and best_score > 0.4:
        best_candidate.role = PersonRole.REFEREE

    return persons


def detect_fighters(
    model: YOLO, image: Image.Image, frame_width: int, frame_height: int
) -> tuple[list[FighterDetection], FighterDetection | None, list[FighterDetection]]:
    """Orchestrate detection pipeline: detect, filter spectators, filter referee.

    Returns (fighters[:2], referee_or_None, spectators).
    """
    persons = detect_persons(model, image)
    persons = filter_spectators(persons, frame_width, frame_height)
    persons = filter_referee(persons, frame_width, frame_height)

    fighters: list[FighterDetection] = []
    referee: FighterDetection | None = None
    spectators: list[FighterDetection] = []

    for p in persons:
        if p.role == PersonRole.REFEREE:
            referee = FighterDetection(bbox=p.bbox, confidence=p.confidence, keypoints=None)
        elif p.role == PersonRole.SPECTATOR:
            spectators.append(FighterDetection(bbox=p.bbox, confidence=p.confidence, keypoints=None))
        else:
            # FIGHTER or UNKNOWN (when <=2 persons, they stay UNKNOWN = treated as fighters)
            fighters.append(FighterDetection(bbox=p.bbox, confidence=p.confidence, keypoints=p.keypoints))

    # Take top 2 fighters by confidence
    fighters.sort(key=lambda d: d.confidence, reverse=True)
    return fighters[:2], referee, spectators
