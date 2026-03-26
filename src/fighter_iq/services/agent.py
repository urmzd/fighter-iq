"""Fight agent — detection, VLM description, spatial metrics, and summarization."""

from __future__ import annotations

import gc
from typing import Any

from PIL import Image

from fighter_iq import (
    FighterDetection,
    FighterIdentity,
    FighterProfile,
    FrameAnalysis,
    PersonRole,
    SegmentSummary,
)


class FightAgent:
    """Stateful agent that analyzes fight frames.

    Implements the Agent protocol. Owns model lifecycle and fighter profile
    tracking state. Delegates to detector, analyzer, spatial, and summarizer
    modules internally.
    """

    def __init__(self) -> None:
        # Models (lazy-loaded)
        self._yolo: Any = None
        self._vlm_model: Any = None
        self._vlm_processor: Any = None
        self._vlm_config: Any = None
        self._text_model: Any = None
        self._text_tokenizer: Any = None

        # Fighter identity tracking state
        self._profiles: list[FighterProfile] = []
        self._profiles_initialized: bool = False
        self._prev_detections: list[FighterDetection] | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_detection_models(self) -> None:
        """Load YOLO + VLM models for per-frame analysis (Phase 1)."""
        from fighter_iq.detector import load_detector
        from fighter_iq.analyzer import load_vision_model

        self._yolo = load_detector()
        self._vlm_model, self._vlm_processor, self._vlm_config = load_vision_model()

    def unload_detection_models(self) -> None:
        """Release detection models from memory."""
        del self._yolo, self._vlm_model, self._vlm_processor, self._vlm_config
        self._yolo = None
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_config = None
        gc.collect()

    def load_text_model(self) -> None:
        """Load text model for summarization (Phase 2)."""
        from fighter_iq.summarizer import load_text_model

        self._text_model, self._text_tokenizer = load_text_model()

    def unload_text_model(self) -> None:
        """Release text model from memory."""
        del self._text_model, self._text_tokenizer
        self._text_model = None
        self._text_tokenizer = None
        gc.collect()

    # ------------------------------------------------------------------
    # Per-frame analysis (Phase 1)
    # ------------------------------------------------------------------

    def analyze_frame(
        self,
        image: Image.Image,
        timestamp: float,
        context: dict | None = None,
    ) -> FrameAnalysis:
        """Detect fighters, describe action, compute spatial metrics for one frame."""
        from fighter_iq.detector import (
            detect_persons,
            detect_fighters,
            filter_spectators,
            initialize_profiles,
            update_profile,
            match_profiles,
            filter_referee_with_profiles,
        )
        from fighter_iq.analyzer import analyze_frame as vlm_analyze
        from fighter_iq.spatial import (
            compute_control,
            compute_proximity,
            compute_movement_vectors,
            detect_impact,
        )

        fighter_descriptions: list[tuple[str, str]] | None = None
        fighter_appearances: dict[str, str] = {}

        if self._profiles_initialized:
            fighters, referee, spectators, fighter_descriptions, fighter_appearances = (
                self._detect_with_profiles(image)
            )
        else:
            fighters, referee, spectators = detect_fighters(
                self._yolo, image, image.width, image.height
            )
            # Try to bootstrap profiles
            if len(fighters) == 2 and referee is not None:
                persons = detect_persons(self._yolo, image)
                persons = filter_spectators(persons, image.width, image.height)
                fighter_persons = [
                    p for p in persons
                    if p.role in (PersonRole.FIGHTER, PersonRole.UNKNOWN)
                ]
                fighter_persons.sort(key=lambda p: p.bbox.area, reverse=True)
                self._profiles = initialize_profiles(fighter_persons[:2], image)
                self._profiles_initialized = True
                for i, f in enumerate(fighters):
                    f.identity = (
                        self._profiles[i].identity if i < len(self._profiles) else None
                    )
                fighter_descriptions = [
                    (f"Fighter {p.identity.value[-1].upper()}", p.appearance_description)
                    for p in self._profiles
                ]
                fighter_appearances = {
                    p.identity.value: p.appearance_description
                    for p in self._profiles
                }

        incomplete = len(fighters) < 2

        # VLM description
        description = vlm_analyze(
            self._vlm_model,
            self._vlm_processor,
            self._vlm_config,
            image,
            timestamp,
            fighter_descriptions=fighter_descriptions,
            referee_detected=(referee is not None),
        )

        # Spatial metrics
        if incomplete:
            control = None
            impact, impact_type = False, None
        else:
            control = compute_control(
                self._prev_detections, fighters, image.width, image.height
            )
            impact, impact_type = detect_impact(
                self._prev_detections, fighters, description
            )

        proximity = compute_proximity(fighters, image.width, image.height)
        vectors = compute_movement_vectors(self._prev_detections, fighters)

        if not incomplete:
            self._prev_detections = fighters

        return FrameAnalysis(
            timestamp=round(timestamp, 2),
            description=description,
            fighters=fighters,
            control_score=round(control, 3) if control is not None else None,
            proximity_to_center=[round(p, 3) for p in proximity],
            movement_vectors=[(round(dx, 1), round(dy, 1)) for dx, dy in vectors],
            impact=impact,
            impact_type=impact_type,
            incomplete=incomplete,
            fighter_count=len(fighters),
            filtered_referee=referee,
            filtered_spectators=spectators,
            fighter_appearances=fighter_appearances,
        )

    def _detect_with_profiles(
        self, image: Image.Image
    ) -> tuple[
        list[FighterDetection],
        FighterDetection | None,
        list[FighterDetection],
        list[tuple[str, str]],
        dict[str, str],
    ]:
        """Profile-based detection path (warm start)."""
        from fighter_iq.detector import (
            detect_persons,
            filter_spectators,
            match_profiles,
            filter_referee_with_profiles,
            update_profile,
        )

        persons = detect_persons(self._yolo, image)
        persons = filter_spectators(persons, image.width, image.height)
        candidates = [p for p in persons if p.role != PersonRole.SPECTATOR]

        matched_pairs, _unmatched = match_profiles(candidates, self._profiles, image)

        matched_indices: set[int] = set()
        for person, _prof in matched_pairs:
            for i, p in enumerate(persons):
                if p is person:
                    matched_indices.add(i)
                    break

        persons = filter_referee_with_profiles(
            persons, matched_indices, image.width, image.height
        )

        for person, prof in matched_pairs:
            update_profile(prof, person, image)

        identity_map: dict[FighterIdentity, FighterDetection] = {}
        for person, prof in matched_pairs:
            identity_map[prof.identity] = FighterDetection(
                bbox=person.bbox,
                confidence=person.confidence,
                keypoints=person.keypoints,
                identity=prof.identity,
            )

        fighters = []
        for ident in [FighterIdentity.FIGHTER_A, FighterIdentity.FIGHTER_B]:
            if ident in identity_map:
                fighters.append(identity_map[ident])

        referee: FighterDetection | None = None
        spectators: list[FighterDetection] = []
        for p in persons:
            if p.role == PersonRole.REFEREE:
                referee = FighterDetection(bbox=p.bbox, confidence=p.confidence)
            elif p.role == PersonRole.SPECTATOR:
                spectators.append(
                    FighterDetection(bbox=p.bbox, confidence=p.confidence)
                )

        matched_ids = {prof.identity for _, prof in matched_pairs}
        for prof in self._profiles:
            if prof.identity not in matched_ids:
                prof.frames_since_last_seen += 1

        if all(p.frames_since_last_seen > 5 for p in self._profiles):
            self._profiles_initialized = False
            self._profiles = []

        fighter_descriptions = [
            (f"Fighter {p.identity.value[-1].upper()}", p.appearance_description)
            for p in self._profiles
        ]
        fighter_appearances = {
            p.identity.value: p.appearance_description for p in self._profiles
        }

        return fighters, referee, spectators, fighter_descriptions, fighter_appearances

    # ------------------------------------------------------------------
    # Summarization (Phase 2)
    # ------------------------------------------------------------------

    def summarize_segment(self, frames: list[FrameAnalysis]) -> str:
        """Produce a narrative summary from a batch of frame analyses."""
        from fighter_iq.summarizer import stitch_segment

        segment = stitch_segment(self._text_model, self._text_tokenizer, frames)
        return segment.narrative

    def stitch_segment(self, frames: list[FrameAnalysis]) -> SegmentSummary:
        """Produce a full SegmentSummary from a batch of frame analyses."""
        from fighter_iq.summarizer import stitch_segment

        return stitch_segment(self._text_model, self._text_tokenizer, frames)

    def summarize_fight(self, segment_narratives: list[str]) -> str:
        """Produce an overall fight summary from segment narratives."""
        from fighter_iq.summarizer import final_summary as _final_summary

        # Build minimal SegmentSummary objects for the existing API
        segments = [
            SegmentSummary(
                timestamps=[0.0],
                narrative=n,
                avg_control=0.0,
                impacts=0,
            )
            for n in segment_narratives
        ]
        return _final_summary(self._text_model, self._text_tokenizer, segments)
