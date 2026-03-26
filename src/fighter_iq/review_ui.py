"""NiceGUI web interface for fight review with annotated video and TTS commentary."""

from __future__ import annotations

from pathlib import Path

from nicegui import app, ui

from fighter_iq import AnalysisResult
from fighter_iq.commentary import CommentarySegment
from fighter_iq.personas import CommentaryPersona


def launch_review(
    video_path: Path,
    audio_path: Path,
    analysis: AnalysisResult,
    commentary_segments: list[CommentarySegment],
    persona: CommentaryPersona,
    port: int = 8080,
) -> None:
    """Start the NiceGUI review server."""
    # Serve local media files
    media_dir = video_path.parent
    app.add_media_files("/media", str(media_dir))

    audio_dir = audio_path.parent
    app.add_media_files("/audio", str(audio_dir))

    video_url = f"/media/{video_path.name}"
    audio_url = f"/audio/{audio_path.name}"

    @ui.page("/")
    def review_page():
        _build_review_page(
            video_url, audio_url, analysis, commentary_segments, persona
        )

    ui.run(port=port, title="Fighter IQ — Fight Review", reload=False)


def _build_review_page(
    video_url: str,
    audio_url: str,
    analysis: AnalysisResult,
    commentary_segments: list[CommentarySegment],
    persona: CommentaryPersona,
) -> None:
    """Build the review page layout."""
    is_playing = {"value": False}

    # --- Header ---
    with ui.header().classes("items-center justify-between"):
        ui.label("Fighter IQ — Fight Review").classes("text-xl font-bold")
        ui.label(f"Persona: {persona.name}").classes("text-sm opacity-75")

    with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):
        # --- Video Player ---
        video = ui.video(video_url).classes("w-full rounded-lg").props("preload=auto")
        video.props("controls=false")

        # --- Hidden Audio ---
        audio = ui.audio(audio_url).classes("hidden").props("preload=auto")

        # --- Transport Controls ---
        with ui.row().classes("items-center gap-4 w-full"):
            play_btn = ui.button("Play", on_click=lambda: _toggle_playback(
                video, audio, play_btn, is_playing
            )).props("icon=play_arrow")

            time_label = ui.label("0:00 / 0:00").classes("font-mono text-sm")

            seek_slider = ui.slider(min=0, max=100, value=0, step=0.1).classes("flex-grow")
            seek_slider.on(
                "update:model-value",
                lambda e: _on_seek(video, audio, e.args),
                throttle=0.3,
            )

        # --- Commentary Text Panel ---
        with ui.card().classes("w-full"):
            ui.label("Commentary").classes("text-sm font-bold opacity-60")
            commentary_text = ui.label("Press play to begin...").classes(
                "text-base leading-relaxed"
            )

        # --- Segment Navigation ---
        if commentary_segments:
            with ui.row().classes("gap-2 flex-wrap"):
                ui.label("Segments:").classes("text-sm font-bold self-center")
                for seg in commentary_segments:
                    ui.button(
                        f"Seg {seg.segment_index + 1}",
                        on_click=lambda s=seg: _jump_to_segment(
                            video, audio, s, commentary_text
                        ),
                    ).props("dense size=sm")

        # --- Segment Summary Table ---
        if analysis.segments:
            with ui.expansion("Segment Details", icon="list").classes("w-full"):
                for i, seg in enumerate(analysis.segments):
                    time_range = f"{seg.timestamps[0]:.1f}s – {seg.timestamps[-1]:.1f}s"
                    with ui.card().classes("w-full mb-2"):
                        ui.label(f"Segment {i + 1} ({time_range})").classes(
                            "text-sm font-bold"
                        )
                        ui.label(seg.narrative).classes("text-sm")
                        with ui.row().classes("gap-4 text-xs opacity-60"):
                            ui.label(f"Control: {seg.avg_control:.2f}")
                            ui.label(f"Impacts: {seg.impacts}")

        # --- Fight Summary ---
        if analysis.summary:
            with ui.expansion("Fight Summary", icon="summarize").classes("w-full"):
                ui.label(analysis.summary).classes("text-sm leading-relaxed")

    # --- Timer to sync UI state ---
    ui.timer(
        0.25,
        lambda: _sync_ui(
            video, audio, time_label, seek_slider, commentary_text,
            commentary_segments, play_btn, is_playing,
        ),
    )


async def _toggle_playback(video, audio, play_btn, is_playing):
    """Toggle play/pause for both video and audio."""
    if is_playing["value"]:
        video.pause()
        audio.pause()
        play_btn.props("icon=play_arrow")
        play_btn.text = "Play"
        is_playing["value"] = False
    else:
        video.play()
        audio.play()
        play_btn.props("icon=pause")
        play_btn.text = "Pause"
        is_playing["value"] = True


async def _on_seek(video, audio, value):
    """Handle seek slider changes."""
    try:
        duration = await ui.run_javascript(
            f'getHtmlElement("{video.id}").duration || 0'
        )
        if duration > 0:
            target = float(value) / 100.0 * duration
            video.seek(target)
            audio.seek(target)
    except Exception:
        pass


async def _jump_to_segment(video, audio, segment: CommentarySegment, commentary_text):
    """Jump playback to a specific segment's start time."""
    video.seek(segment.start_time)
    await ui.run_javascript(
        f'getHtmlElement("{audio.id}").currentTime = {segment.start_time}'
    )
    commentary_text.text = segment.text


async def _sync_ui(video, audio, time_label, seek_slider, commentary_text, segments,
                   play_btn, is_playing):
    """Periodically sync the UI state with video playback position."""
    try:
        current_time = await ui.run_javascript(
            f'getHtmlElement("{video.id}").currentTime || 0'
        )
        duration = await ui.run_javascript(
            f'getHtmlElement("{video.id}").duration || 0'
        )
    except Exception:
        return

    if duration <= 0:
        return

    # Update time label
    ct_min, ct_sec = divmod(int(current_time), 60)
    dur_min, dur_sec = divmod(int(duration), 60)
    time_label.text = f"{ct_min}:{ct_sec:02d} / {dur_min}:{dur_sec:02d}"

    # Update seek slider without triggering seek
    seek_slider.value = current_time / duration * 100

    # Update commentary text based on current time
    active_segment = None
    for seg in segments:
        if seg.start_time <= current_time <= seg.end_time + 2.0:
            active_segment = seg
    if active_segment:
        commentary_text.text = active_segment.text

    # Auto-pause at end and reset Play button
    try:
        is_ended = await ui.run_javascript(
            f'getHtmlElement("{video.id}").ended || false'
        )
    except Exception:
        is_ended = False

    if is_ended and is_playing["value"]:
        video.pause()
        audio.pause()
        play_btn.props("icon=play_arrow")
        play_btn.text = "Play"
        is_playing["value"] = False

    # Sync audio position if drifted > 1s (wider threshold avoids micro-interruptions)
    try:
        audio_time = await ui.run_javascript(
            f'getHtmlElement("{audio.id}").currentTime || 0'
        )
        if abs(audio_time - current_time) > 1.0:
            audio.seek(current_time)
    except Exception:
        pass
