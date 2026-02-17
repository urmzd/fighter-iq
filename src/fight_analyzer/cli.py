"""CLI entry point for Fight Analyzer."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="fight-analyzer",
    help="Fight Analyzer — Analyze martial arts videos with AI.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Fight Analyzer — Analyze martial arts videos with AI."""


@app.command()
def analyze(
    source: str = typer.Argument(
        ...,
        help="Path to a local video file.",
    ),
    interval: float = typer.Option(
        1.0,
        "--interval",
        "-i",
        help="Seconds between extracted frames.",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        "-b",
        help="Number of frames per segment for stitching.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save analysis to JSON file (default: stdout).",
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        "-d",
        help="Only analyze the first N seconds of the video (minimum 120s).",
    ),
    visualize: bool = typer.Option(
        True,
        "--visualize/--no-visualize",
        help="Show OpenCV window with annotated frames.",
    ),
) -> None:
    """Analyze a martial arts video for control, movement, and impact."""

    video_path = Path(source)
    if not video_path.exists():
        console.print(f"[red]Error:[/] File not found: {video_path}")
        raise typer.Exit(code=1)

    from fight_analyzer.pipeline import MIN_ANALYSIS_DURATION

    if duration is not None and duration < MIN_ANALYSIS_DURATION:
        console.print(
            f"[yellow]Warning:[/] Minimum analysis duration is {MIN_ANALYSIS_DURATION}s "
            f"(requested {duration}s). Clamping to {MIN_ANALYSIS_DURATION}s."
        )
        duration = float(MIN_ANALYSIS_DURATION)

    console.print(f"[bold]Fight Analyzer[/] — Analyzing: [cyan]{video_path}[/]")
    console.print(f"  Interval: {interval}s | Batch size: {batch_size}")
    if duration:
        console.print(f"  Max duration: {duration}s")

    from fight_analyzer.pipeline import run_pipeline

    run_pipeline(
        video_path=video_path,
        interval=interval,
        batch_size=batch_size,
        output_path=output,
        max_duration=duration,
        visualize=visualize,
    )


@app.command()
def review(
    analysis: Path = typer.Option(
        ...,
        "--analysis",
        "-a",
        help="Path to analysis JSON file from the analyze command.",
    ),
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        help="Path to the original video file.",
    ),
    persona: str = typer.Option(
        "technical",
        "--persona",
        "-p",
        help="Commentary persona (technical, hype).",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        help="Port for the review web server.",
    ),
) -> None:
    """Review a fight analysis with annotated video and TTS commentary."""
    import gc
    from fight_analyzer.personas import get_persona, list_persona_ids

    if not analysis.exists():
        console.print(f"[red]Error:[/] Analysis file not found: {analysis}")
        raise typer.Exit(code=1)
    if not video.exists():
        console.print(f"[red]Error:[/] Video file not found: {video}")
        raise typer.Exit(code=1)

    available = list_persona_ids()
    if persona not in available:
        console.print(f"[red]Error:[/] Unknown persona '{persona}'. Available: {', '.join(available)}")
        raise typer.Exit(code=1)

    selected_persona = get_persona(persona)
    console.print(f"[bold]Fight Analyzer[/] — Fight Review")
    console.print(f"  Analysis: [cyan]{analysis}[/]")
    console.print(f"  Video: [cyan]{video}[/]")
    console.print(f"  Persona: [cyan]{selected_persona.name}[/]")

    # 1. Load analysis
    console.print("\n[bold blue]Step 1/4:[/] Loading analysis...")
    from fight_analyzer.pipeline import load_analysis
    result = load_analysis(analysis)
    console.print(f"  Loaded {len(result.frames)} frames, {len(result.segments)} segments.")

    # 2. Render annotated video
    console.print("\n[bold blue]Step 2/4:[/] Rendering annotated video...")
    from fight_analyzer.renderer import render_annotated_video
    output_dir = Path("outputs") / "review"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video_path = output_dir / f"{video.stem}_annotated.mp4"
    render_annotated_video(video, result, annotated_video_path)
    console.print(f"  Saved: [green]{annotated_video_path}[/]")

    # 3. Generate commentary text
    console.print("\n[bold blue]Step 3/4:[/] Generating commentary text...")
    from fight_analyzer.summarizer import load_text_model
    from fight_analyzer.commentary import generate_commentary
    text_model, text_tokenizer = load_text_model()
    commentary_segments = generate_commentary(text_model, text_tokenizer, result, selected_persona)
    del text_model, text_tokenizer
    gc.collect()
    console.print(f"  Generated {len(commentary_segments)} commentary segments.")

    # 4. Synthesize TTS audio
    console.print("\n[bold blue]Step 4/4:[/] Synthesizing TTS audio...")
    from fight_analyzer.tts import load_tts_model, synthesize_continuous
    tts_model = load_tts_model()

    # Calculate total duration from last frame timestamp + interval
    total_duration = result.frames[-1].timestamp + result.settings.get("interval", 1.0) if result.frames else 0
    commentary_audio_path = output_dir / "commentary.wav"
    synthesize_continuous(tts_model, commentary_segments, selected_persona, total_duration, commentary_audio_path)
    del tts_model
    gc.collect()
    console.print(f"  Saved: [green]{commentary_audio_path}[/]")

    # 5. Launch review UI
    console.print(f"\n[bold green]Launching review UI on port {port}...[/]")
    from fight_analyzer.review_ui import launch_review
    launch_review(
        video_path=annotated_video_path,
        audio_path=commentary_audio_path,
        analysis=result,
        commentary_segments=commentary_segments,
        persona=selected_persona,
        port=port,
    )


if __name__ == "__main__":
    app()
