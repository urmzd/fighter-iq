"""Service boundary protocols for the Fighter IQ pipeline."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from fighter_iq import FrameAnalysis
from fighter_iq.models import FrameEmbedding, Strategy, Tactic


@runtime_checkable
class Ingestor(Protocol):
    """Produces frames from a video source."""

    def extract(
        self,
        source: Path,
        interval: float = 1.0,
        max_duration: float | None = None,
    ) -> Generator[tuple[float, Image.Image], None, None]:
        """Yield (timestamp_seconds, PIL.Image) tuples."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Produces vector embeddings from images."""

    def embed_frame(self, image: Image.Image) -> np.ndarray:
        """Return embedding vector for a single image."""
        ...

    def embed_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Return embedding vectors for a batch of images."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


@runtime_checkable
class Agent(Protocol):
    """Analyzes fight frames — detection, description, spatial metrics, summaries."""

    def analyze_frame(
        self,
        image: Image.Image,
        timestamp: float,
        context: dict | None = None,
    ) -> FrameAnalysis:
        """Analyze a single frame, returning structured analysis."""
        ...

    def summarize_segment(self, frames: list[FrameAnalysis]) -> str:
        """Produce a narrative summary from a batch of frame analyses."""
        ...

    def summarize_fight(self, segment_narratives: list[str]) -> str:
        """Produce an overall fight summary from segment narratives."""
        ...


@runtime_checkable
class StrategyService(Protocol):
    """Classifies tactics and strategies from frame analyses and embeddings."""

    def identify_tactics(
        self,
        frames: list[FrameAnalysis],
        embeddings: list[FrameEmbedding],
    ) -> list[Tactic]:
        """Identify individual tactics from analyzed frames."""
        ...

    def classify_strategies(
        self,
        tactics: list[Tactic],
        frames: list[FrameAnalysis],
    ) -> list[Strategy]:
        """Group tactics into higher-level strategies."""
        ...
