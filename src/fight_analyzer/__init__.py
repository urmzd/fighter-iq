"""Fight Analyzer — Video analysis pipeline for martial arts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import numpy as np


class PersonRole(Enum):
    FIGHTER = "fighter"
    REFEREE = "referee"
    SPECTATOR = "spectator"
    UNKNOWN = "unknown"


class FighterIdentity(Enum):
    FIGHTER_A = "fighter_a"
    FIGHTER_B = "fighter_b"


KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class ColorHistogram:
    histogram: np.ndarray  # shape (H_BINS * S_BINS,), normalized
    dominant_color_name: str  # e.g. "red", "black", "blue"
    dominant_hsv: tuple[int, int, int]
    H_BINS: ClassVar[int] = 30
    S_BINS: ClassVar[int] = 32


@dataclass
class FighterProfile:
    identity: FighterIdentity
    color_histogram: ColorHistogram
    appearance_description: str  # e.g. "red shorts"
    last_bbox: BBox
    last_confidence: float
    frames_seen: int = 0
    frames_since_last_seen: int = 0


@dataclass
class FighterDetection:
    bbox: BBox
    confidence: float
    keypoints: list[Keypoint] | None = None
    identity: FighterIdentity | None = None


@dataclass
class DetectedPerson:
    bbox: BBox
    confidence: float
    role: PersonRole = PersonRole.UNKNOWN
    keypoints: list[Keypoint] | None = None


@dataclass
class FrameAnalysis:
    timestamp: float
    description: str
    fighters: list[FighterDetection]
    control_score: float | None
    proximity_to_center: list[float]
    movement_vectors: list[tuple[float, float]]
    impact: bool
    impact_type: str | None
    incomplete: bool = False
    fighter_count: int = 2
    filtered_referee: FighterDetection | None = None
    filtered_spectators: list[FighterDetection] = field(default_factory=list)
    fighter_appearances: dict[str, str] = field(default_factory=dict)


@dataclass
class SegmentSummary:
    timestamps: list[float]
    narrative: str
    avg_control: float
    impacts: int
    incomplete_frames: int = 0


@dataclass
class AnalysisResult:
    video: str
    settings: dict
    frames: list[FrameAnalysis] = field(default_factory=list)
    segments: list[SegmentSummary] = field(default_factory=list)
    summary: str = ""
