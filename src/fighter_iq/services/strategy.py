"""Strategy service — identifies tactics and classifies strategies from fight data."""

from __future__ import annotations

import numpy as np

from fighter_iq import FrameAnalysis
from fighter_iq.models import (
    FrameEmbedding,
    Strategy,
    StrategyType,
    Tactic,
    TacticCategory,
)

# Keyword sets reused from spatial.py for tactic classification
_STRIKE_KEYWORDS = [
    "punch",
    "punches",
    "punching",
    "kick",
    "kicks",
    "kicking",
    "elbow",
    "elbows",
    "knee",
    "knees",
    "kneeing",
    "strike",
    "strikes",
    "striking",
    "hit",
    "hits",
    "hitting",
    "jab",
    "jabs",
    "jabbing",
    "cross",
    "hook",
    "uppercut",
    "roundhouse",
    "front kick",
    "side kick",
    "spinning",
    "landed",
    "connects",
]
_GRAPPLE_KEYWORDS = [
    "takedown",
    "taken down",
    "takes down",
    "clinch",
    "clinching",
    "wrestle",
    "wrestling",
    "grapple",
    "grappling",
    "mount",
    "mounted",
    "guard",
    "side control",
    "back control",
    "underhook",
    "overhook",
    "double leg",
    "single leg",
    "slam",
    "slams",
    "suplex",
    "ground",
]
_MOVEMENT_KEYWORDS = [
    "advancing",
    "retreating",
    "circling",
    "lateral",
    "angle",
    "cutting",
    "footwork",
    "stepping",
    "forward",
    "backward",
    "moving",
]
_DEFENSE_KEYWORDS = [
    "block",
    "blocking",
    "blocks",
    "parry",
    "parries",
    "slip",
    "slips",
    "slipping",
    "evade",
    "evading",
    "dodge",
    "dodging",
    "shell",
    "cover",
    "covering",
    "sprawl",
    "sprawls",
    "sprawling",
]

# Similarity threshold below which we consider a new action boundary
_ACTION_BOUNDARY_THRESHOLD = 0.85


class FightStrategyService:
    """Identifies tactics and classifies strategies from frame analyses and embeddings.

    Implements the StrategyService protocol.
    """

    def identify_tactics(
        self,
        frames: list[FrameAnalysis],
        embeddings: list[FrameEmbedding],
    ) -> list[Tactic]:
        """Identify individual tactics from analyzed frames.

        Algorithm:
        1. Classify each frame's description into a TacticCategory.
        2. Use CLIP embedding similarity to detect action boundaries.
        3. Merge consecutive same-category frames into Tactic spans.
        """
        if not frames:
            return []

        # Build per-frame category + name classifications
        frame_labels = [self._classify_frame(f) for f in frames]

        # Build embedding lookup by timestamp
        emb_by_ts: dict[float, np.ndarray] = {}
        for e in embeddings:
            emb_by_ts[round(e.timestamp, 2)] = e.embedding

        # Merge consecutive same-category frames into tactic spans,
        # splitting on embedding discontinuities
        tactics: list[Tactic] = []
        span_start = 0

        for i in range(1, len(frames)):
            cat_changed = frame_labels[i][0] != frame_labels[span_start][0]

            # Check embedding similarity for boundary detection
            emb_boundary = False
            ts_prev = round(frames[i - 1].timestamp, 2)
            ts_curr = round(frames[i].timestamp, 2)
            if ts_prev in emb_by_ts and ts_curr in emb_by_ts:
                sim = _cosine_similarity(emb_by_ts[ts_prev], emb_by_ts[ts_curr])
                emb_boundary = sim < _ACTION_BOUNDARY_THRESHOLD

            if cat_changed or emb_boundary:
                tactics.append(self._make_tactic(frames, frame_labels, span_start, i))
                span_start = i

        # Final span
        tactics.append(self._make_tactic(frames, frame_labels, span_start, len(frames)))
        return tactics

    def classify_strategies(
        self,
        tactics: list[Tactic],
        frames: list[FrameAnalysis],
    ) -> list[Strategy]:
        """Group tactics into higher-level strategies using sliding windows.

        Uses a 30-second window to classify dominant patterns.
        """
        if not tactics:
            return []

        window_duration = 30.0
        strategies: list[Strategy] = []
        start_time = tactics[0].start_time
        end_time = tactics[-1].end_time

        t = start_time
        while t < end_time:
            window_end = t + window_duration
            window_tactics = [tc for tc in tactics if tc.end_time > t and tc.start_time < window_end]
            if not window_tactics:
                t = window_end
                continue

            strategy_type = self._classify_window(window_tactics, frames, t, window_end)
            description = self._describe_strategy(strategy_type, window_tactics)

            strategies.append(
                Strategy(
                    strategy_type=strategy_type,
                    tactics=window_tactics,
                    start_time=t,
                    end_time=min(window_end, end_time),
                    confidence=self._compute_strategy_confidence(window_tactics, strategy_type),
                    description=description,
                )
            )
            t = window_end

        return strategies

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_frame(self, frame: FrameAnalysis) -> tuple[TacticCategory, str]:
        """Classify a single frame's description into a category and tactic name."""
        desc = frame.description.lower()

        # Check categories in priority order
        if frame.impact and frame.impact_type in ("strike", "knockdown"):
            name = self._extract_strike_name(desc)
            return TacticCategory.STRIKE, name

        if frame.impact and frame.impact_type in ("takedown", "sweep"):
            name = self._extract_grapple_name(desc)
            return TacticCategory.GRAPPLE, name

        strike_count = sum(1 for kw in _STRIKE_KEYWORDS if kw in desc)
        grapple_count = sum(1 for kw in _GRAPPLE_KEYWORDS if kw in desc)
        movement_count = sum(1 for kw in _MOVEMENT_KEYWORDS if kw in desc)
        defense_count = sum(1 for kw in _DEFENSE_KEYWORDS if kw in desc)

        counts = {
            TacticCategory.STRIKE: strike_count,
            TacticCategory.GRAPPLE: grapple_count,
            TacticCategory.MOVEMENT: movement_count,
            TacticCategory.DEFENSE: defense_count,
        }
        best_cat = max(counts, key=counts.get)  # type: ignore[arg-type]

        if counts[best_cat] == 0:
            best_cat = TacticCategory.MOVEMENT  # default: neutral movement

        if best_cat == TacticCategory.STRIKE:
            name = self._extract_strike_name(desc)
        elif best_cat == TacticCategory.GRAPPLE:
            name = self._extract_grapple_name(desc)
        elif best_cat == TacticCategory.DEFENSE:
            name = self._extract_defense_name(desc)
        else:
            name = self._extract_movement_name(desc)

        return best_cat, name

    def _extract_strike_name(self, desc: str) -> str:
        specific = [
            "jab",
            "cross",
            "hook",
            "uppercut",
            "roundhouse",
            "front kick",
            "side kick",
            "spinning",
            "elbow",
            "knee",
        ]
        for s in specific:
            if s in desc:
                return s
        return "strike"

    def _extract_grapple_name(self, desc: str) -> str:
        specific = [
            "double leg",
            "single leg",
            "clinch",
            "mount",
            "guard",
            "side control",
            "back control",
            "slam",
            "suplex",
            "takedown",
        ]
        for s in specific:
            if s in desc:
                return s
        return "grapple"

    def _extract_defense_name(self, desc: str) -> str:
        specific = ["slip", "block", "parry", "sprawl", "evade", "dodge"]
        for s in specific:
            if s in desc:
                return s
        return "defense"

    def _extract_movement_name(self, desc: str) -> str:
        specific = [
            "angle cut",
            "circling",
            "advancing",
            "retreating",
            "lateral movement",
            "footwork",
        ]
        for s in specific:
            if s in desc:
                return s
        return "movement"

    def _make_tactic(
        self,
        frames: list[FrameAnalysis],
        labels: list[tuple[TacticCategory, str]],
        start: int,
        end: int,
    ) -> Tactic:
        """Build a Tactic from a contiguous span of frames."""
        span_frames = frames[start:end]
        span_labels = labels[start:end]

        # Most common name in the span
        name_counts: dict[str, int] = {}
        for _, name in span_labels:
            name_counts[name] = name_counts.get(name, 0) + 1
        best_name = max(name_counts, key=name_counts.get)  # type: ignore[arg-type]

        category = span_labels[0][0]

        # Actor: who has control during this span
        control_scores = [f.control_score for f in span_frames if f.control_score is not None]
        actor = None
        if control_scores:
            avg = sum(control_scores) / len(control_scores)
            if avg > 0.15:
                actor = "fighter_a"
            elif avg < -0.15:
                actor = "fighter_b"

        # Confidence from impact presence and keyword match density
        impact_frames = sum(1 for f in span_frames if f.impact)
        confidence = min(1.0, 0.4 + 0.3 * (impact_frames / max(len(span_frames), 1)) + 0.3 * (len(span_frames) / 5))

        return Tactic(
            name=best_name,
            category=category,
            start_time=span_frames[0].timestamp,
            end_time=span_frames[-1].timestamp,
            confidence=round(confidence, 3),
            actor=actor,
            description=span_frames[0].description,
        )

    def _classify_window(
        self,
        tactics: list[Tactic],
        frames: list[FrameAnalysis],
        window_start: float,
        window_end: float,
    ) -> StrategyType:
        """Classify the dominant strategy in a time window."""
        cat_counts: dict[TacticCategory, int] = {}
        for t in tactics:
            cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
        total = sum(cat_counts.values())
        if total == 0:
            return StrategyType.MIXED

        strike_pct = cat_counts.get(TacticCategory.STRIKE, 0) / total
        grapple_pct = cat_counts.get(TacticCategory.GRAPPLE, 0) / total
        defense_pct = cat_counts.get(TacticCategory.DEFENSE, 0) / total

        # Check movement vectors for forward pressure
        window_frames = [f for f in frames if window_start <= f.timestamp < window_end]
        forward_pressure = self._has_forward_pressure(window_frames)

        if strike_pct >= 0.6 and forward_pressure:
            return StrategyType.PRESSURE
        if strike_pct >= 0.6 and not forward_pressure:
            return StrategyType.COUNTER
        if grapple_pct >= 0.4:
            return StrategyType.GRAPPLE_DOMINANT
        if defense_pct >= 0.4 and total <= 3:
            return StrategyType.POINT_FIGHTING
        return StrategyType.MIXED

    def _has_forward_pressure(self, frames: list[FrameAnalysis]) -> bool:
        """Check if the dominant fighter is consistently advancing."""
        if not frames:
            return False
        positive_control = sum(1 for f in frames if f.control_score is not None and abs(f.control_score) > 0.2)
        return positive_control > len(frames) * 0.5

    def _compute_strategy_confidence(self, tactics: list[Tactic], strategy_type: StrategyType) -> float:
        """Compute confidence for a strategy classification."""
        if not tactics:
            return 0.0
        avg_tactic_conf = sum(t.confidence for t in tactics) / len(tactics)
        # Boost if strategy is non-mixed (clearer signal)
        clarity_bonus = 0.1 if strategy_type != StrategyType.MIXED else 0.0
        return round(min(1.0, avg_tactic_conf + clarity_bonus), 3)

    def _describe_strategy(self, strategy_type: StrategyType, tactics: list[Tactic]) -> str:
        """Generate a human-readable strategy description."""
        tactic_names = list({t.name for t in tactics})
        names_str = ", ".join(tactic_names[:4])

        descriptions = {
            StrategyType.PRESSURE: f"Pressure fighting with {names_str}",
            StrategyType.COUNTER: f"Counter-striking using {names_str}",
            StrategyType.GRAPPLE_DOMINANT: f"Grapple-dominant game plan featuring {names_str}",
            StrategyType.CLINCH_WORK: f"Clinch-based strategy with {names_str}",
            StrategyType.POINT_FIGHTING: f"Point fighting at range using {names_str}",
            StrategyType.MIXED: f"Mixed approach including {names_str}",
        }
        return descriptions.get(strategy_type, f"Strategy with {names_str}")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
