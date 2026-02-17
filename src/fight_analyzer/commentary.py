"""Generate spoken commentary text using persona-styled LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from fight_analyzer import AnalysisResult, SegmentSummary
from fight_analyzer.personas import CommentaryPersona
from fight_analyzer.summarizer import _format_frame_for_prompt


@dataclass
class CommentarySegment:
    start_time: float
    end_time: float
    text: str
    segment_index: int


def _build_segment_prompt(
    segment: SegmentSummary,
    frames_in_segment: list,
    segment_index: int,
    total_segments: int,
) -> str:
    """Build the user prompt for commentary generation on one segment."""
    frames_text = "\n".join(_format_frame_for_prompt(f) for f in frames_in_segment)

    return (
        f"You are providing spoken commentary for segment {segment_index + 1} of {total_segments} "
        f"of an MMA fight.\n\n"
        f"Segment narrative summary:\n{segment.narrative}\n\n"
        f"Frame-by-frame data:\n{frames_text}\n\n"
        f"Average control score: {segment.avg_control:.2f}, Impacts: {segment.impacts}\n\n"
        f"Generate 2-4 sentences of natural spoken commentary for this segment. "
        f"Keep it concise and suitable for voice narration."
    )


def generate_commentary(
    model,
    tokenizer,
    result: AnalysisResult,
    persona: CommentaryPersona,
) -> list[CommentarySegment]:
    """Generate spoken commentary segments using the text model and persona."""
    commentary: list[CommentarySegment] = []
    batch_size = result.settings.get("batch_size", 5)

    for seg_idx, segment in enumerate(result.segments):
        # Find frames belonging to this segment
        start_frame_idx = seg_idx * batch_size
        end_frame_idx = min(start_frame_idx + batch_size, len(result.frames))
        frames_in_segment = result.frames[start_frame_idx:end_frame_idx]

        user_prompt = _build_segment_prompt(
            segment, frames_in_segment, seg_idx, len(result.segments)
        )

        messages = [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(temp=0.7)
        response = generate(
            model, tokenizer, prompt=formatted, max_tokens=200, sampler=sampler, verbose=False
        )

        start_time = segment.timestamps[0]
        end_time = segment.timestamps[-1]

        commentary.append(
            CommentarySegment(
                start_time=start_time,
                end_time=end_time,
                text=response.strip(),
                segment_index=seg_idx,
            )
        )

    return commentary
