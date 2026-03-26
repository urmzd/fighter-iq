"""Fighter IQ domain model — tactics, strategies, and frame embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Tactics — atomic fighting actions
# ---------------------------------------------------------------------------


class TacticCategory(Enum):
    """High-level category of a fighting action."""

    STRIKE = "strike"
    GRAPPLE = "grapple"
    MOVEMENT = "movement"
    DEFENSE = "defense"
    TRANSITION = "transition"


@dataclass
class Tactic:
    """An atomic fighting action identified across one or more frames.

    Examples: jab, double-leg takedown, slip, angle cut, clinch entry.
    """

    name: str
    category: TacticCategory
    start_time: float
    end_time: float
    confidence: float  # 0.0–1.0
    actor: str | None = None  # "fighter_a", "fighter_b", or None
    description: str = ""


# ---------------------------------------------------------------------------
# Strategies — sequences of tactics forming a game plan
# ---------------------------------------------------------------------------


class StrategyType(Enum):
    """Recognized fighting strategies (game plans)."""

    PRESSURE = "pressure"  # forward pressure, volume striking
    COUNTER = "counter"  # reactive, waiting for openings
    GRAPPLE_DOMINANT = "grapple_dominant"  # takedown-centric, top control
    CLINCH_WORK = "clinch_work"  # inside fighting, dirty boxing
    POINT_FIGHTING = "point_fighting"  # in-and-out, low volume
    MIXED = "mixed"  # no dominant pattern


@dataclass
class Strategy:
    """A coherent game plan inferred from a sequence of tactics."""

    strategy_type: StrategyType
    tactics: list[Tactic]
    start_time: float
    end_time: float
    confidence: float
    description: str = ""


# ---------------------------------------------------------------------------
# Frame embeddings — CLIP vectors for similarity / clustering / search
# ---------------------------------------------------------------------------


@dataclass
class FrameEmbedding:
    """CLIP embedding for a single extracted frame."""

    timestamp: float
    embedding: np.ndarray  # shape (embedding_dim,)
    image_hash: str = ""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def tactic_to_dict(t: Tactic) -> dict:
    return {
        "name": t.name,
        "category": t.category.value,
        "start_time": round(t.start_time, 2),
        "end_time": round(t.end_time, 2),
        "confidence": round(t.confidence, 3),
        "actor": t.actor,
        "description": t.description,
    }


def strategy_to_dict(s: Strategy) -> dict:
    return {
        "strategy_type": s.strategy_type.value,
        "tactics": [tactic_to_dict(t) for t in s.tactics],
        "start_time": round(s.start_time, 2),
        "end_time": round(s.end_time, 2),
        "confidence": round(s.confidence, 3),
        "description": s.description,
    }


def tactic_from_dict(d: dict) -> Tactic:
    return Tactic(
        name=d["name"],
        category=TacticCategory(d["category"]),
        start_time=d["start_time"],
        end_time=d["end_time"],
        confidence=d["confidence"],
        actor=d.get("actor"),
        description=d.get("description", ""),
    )


def strategy_from_dict(d: dict) -> Strategy:
    return Strategy(
        strategy_type=StrategyType(d["strategy_type"]),
        tactics=[tactic_from_dict(t) for t in d["tactics"]],
        start_time=d["start_time"],
        end_time=d["end_time"],
        confidence=d["confidence"],
        description=d.get("description", ""),
    )
