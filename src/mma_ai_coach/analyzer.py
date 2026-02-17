"""Per-frame VLM action description using Qwen2.5-VL via mlx-vlm."""

from __future__ import annotations

from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

_VLM_MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

_BASE_PROMPT = (
    "You are an expert martial arts analyst. Describe the action in this frame "
    "from a combat sports perspective (MMA, Muay Thai, or BJJ). Focus on:\n"
    "- Fighter stances and positioning\n"
    "- Any strikes being thrown (punches, kicks, elbows, knees)\n"
    "- Grappling positions (clinch, takedown, guard, mount, side control)\n"
    "- Movement patterns (advancing, retreating, circling)\n"
    "Be concise (2-3 sentences). If no fighters are visible, say so."
)


def _build_frame_prompt(
    fighter_descriptions: list[tuple[str, str]] | None = None,
    referee_detected: bool = False,
) -> str:
    """Build a context-aware VLM prompt with fighter appearance info."""
    parts = [_BASE_PROMPT]

    if fighter_descriptions:
        parts.append("\nFighters in this frame:")
        for label, desc in fighter_descriptions:
            parts.append(f"- {label}: wearing {desc}")
        parts.append(
            "Refer to fighters by their label based on clothing, NOT by left/right position."
        )

    if referee_detected:
        parts.append(
            "A referee has been identified — do NOT describe the referee as a fighter."
        )

    return "\n".join(parts)


def load_vision_model() -> tuple:
    """Load the Qwen2.5-VL-7B vision-language model.

    Returns (model, processor, config).
    """
    model, processor = load(_VLM_MODEL_ID)
    config = load_config(_VLM_MODEL_ID)
    return model, processor, config


def analyze_frame(
    model,
    processor,
    config,
    image: Image.Image,
    timestamp: float,
    fighter_descriptions: list[tuple[str, str]] | None = None,
    referee_detected: bool = False,
) -> str:
    """Generate a martial arts action description for a single frame."""
    prompt = _build_frame_prompt(fighter_descriptions, referee_detected)

    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    result = generate(
        model,
        processor,
        formatted,
        image=[image],
        max_tokens=150,
        verbose=False,
    )

    return result.text.strip()
