"""Commentary persona definitions for fight review."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommentaryPersona:
    id: str
    name: str
    description: str
    system_prompt: str
    voice: str  # Kokoro voice preset
    speed: float  # TTS speed multiplier


PERSONAS: dict[str, CommentaryPersona] = {
    "technical": CommentaryPersona(
        id="technical",
        name="Technical Analyst",
        description="Clinical, precise analysis emphasizing biomechanics, technique names, and scoring criteria.",
        system_prompt=(
            "You are a veteran MMA technical analyst providing spoken commentary. "
            "Be precise and clinical. Reference specific techniques by name "
            "(jab, cross, double-leg, underhook, etc.). Discuss biomechanics, "
            "footwork, weight distribution, and scoring criteria. "
            "Speak in short, clear sentences suitable for audio narration. "
            "Do NOT use markdown, bullet points, or any formatting — this will be spoken aloud."
        ),
        voice="am_adam",
        speed=1.0,
    ),
    "hype": CommentaryPersona(
        id="hype",
        name="Hype Commentator",
        description="Excited, dramatic commentary emphasizing action, crowd energy, and big moments.",
        system_prompt=(
            "You are an electrifying MMA play-by-play commentator. "
            "Be dramatic and exciting! Use exclamations, build tension, and highlight "
            "big moments with energy. Describe the action as if calling it live for "
            "a roaring crowd. Use punchy, vivid language. "
            "Speak in short, powerful sentences suitable for audio narration. "
            "Do NOT use markdown, bullet points, or any formatting — this will be spoken aloud."
        ),
        voice="am_echo",
        speed=1.1,
    ),
}


def get_persona(persona_id: str) -> CommentaryPersona:
    """Look up a persona by ID. Raises KeyError if not found."""
    return PERSONAS[persona_id]


def list_persona_ids() -> list[str]:
    """Return all available persona IDs."""
    return list(PERSONAS.keys())
