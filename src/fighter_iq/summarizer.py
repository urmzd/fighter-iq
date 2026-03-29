"""Segment stitching and final summary using Qwen2.5-1.5B via mlx-lm."""

from mlx_lm import generate, load

from fighter_iq import FrameAnalysis, SegmentSummary

_TEXT_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def load_text_model() -> tuple:
    """Load the Qwen2.5-1.5B text model for summarization.

    Returns (model, tokenizer).
    """
    model, tokenizer = load(_TEXT_MODEL_ID)
    return model, tokenizer


def _format_frame_for_prompt(frame: FrameAnalysis) -> str:
    """Format a single frame analysis into a text block for the LLM."""
    impact_str = ""
    if frame.impact:
        impact_str = f" [IMPACT: {frame.impact_type}]"

    incomplete_str = " [INCOMPLETE]" if frame.incomplete else ""

    vectors_str = ", ".join(f"({v[0]:.1f}, {v[1]:.1f})" for v in frame.movement_vectors)

    control_str = "N/A" if frame.control_score is None else f"{frame.control_score:.2f}"

    return (
        f"[{frame.timestamp:.1f}s] {frame.description}{impact_str}{incomplete_str}\n"
        f"  Control: {control_str} | "
        f"Proximity: {', '.join(f'{p:.2f}' for p in frame.proximity_to_center)} | "
        f"Movement: {vectors_str}"
    )


def stitch_segment(model, tokenizer, frames: list[FrameAnalysis]) -> SegmentSummary:
    """Stitch a batch of frame analyses into a coherent segment narrative."""
    frames_text = "\n".join(_format_frame_for_prompt(f) for f in frames)

    prompt = (
        "You are an expert MMA analyst. Given the following frame-by-frame analysis "
        "of a combat sports sequence, write a concise narrative summary (3-5 sentences) "
        "describing the action, who has control, and any significant moments.\n\n"
        f"Frame data:\n{frames_text}\n\n"
        "Narrative summary:"
    )

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate(model, tokenizer, prompt=formatted, max_tokens=250, verbose=False)

    valid_scores = [f.control_score for f in frames if f.control_score is not None]
    avg_control = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    impact_count = sum(1 for f in frames if f.impact)
    incomplete_count = sum(1 for f in frames if f.incomplete)

    return SegmentSummary(
        timestamps=[f.timestamp for f in frames],
        narrative=response.strip(),
        avg_control=round(avg_control, 3),
        impacts=impact_count,
        incomplete_frames=incomplete_count,
    )


def final_summary(model, tokenizer, segments: list[SegmentSummary]) -> str:
    """Generate a final overall analysis from all segment summaries."""
    segments_text = ""
    for i, seg in enumerate(segments):
        time_range = f"{seg.timestamps[0]:.1f}s - {seg.timestamps[-1]:.1f}s"
        segments_text += (
            f"Segment {i + 1} ({time_range}):\n"
            f"  {seg.narrative}\n"
            f"  Avg control: {seg.avg_control:.2f}, Impacts: {seg.impacts}\n\n"
        )

    prompt = (
        "You are an expert MMA analyst. Given these segment summaries of a fight, "
        "write a concise overall analysis (4-6 sentences). Cover:\n"
        "- Who dominated and how\n"
        "- Key moments (strikes, takedowns)\n"
        "- Overall fight dynamics and control shifts\n\n"
        f"{segments_text}"
        "Overall analysis:"
    )

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate(model, tokenizer, prompt=formatted, max_tokens=400, verbose=False)

    return response.strip()
