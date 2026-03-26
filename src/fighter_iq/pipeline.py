"""Pipeline orchestration — wires the four services together with live feedback."""

import json
from datetime import datetime
from pathlib import Path

import cv2
from rich.live import Live
from rich.panel import Panel

from fighter_iq import (
    AnalysisResult,
    BBox,
    FighterDetection,
    FighterIdentity,
    FrameAnalysis,
    Keypoint,
    SegmentSummary,
)
from fighter_iq import ui
from fighter_iq.event_stream import EventStream
from fighter_iq.models import (
    FrameEmbedding,
    Strategy,
    Tactic,
    strategy_from_dict,
    strategy_to_dict,
    tactic_from_dict,
    tactic_to_dict,
)
from fighter_iq.shutdown import install_signal_handlers, is_shutdown_requested
from fighter_iq.visualizer import draw_annotations, show_frame

_WINDOW_NAME = "Fighter IQ"
MIN_ANALYSIS_DURATION = 120  # seconds — one standard MMA round


def run_pipeline(
    video_path: Path,
    interval: float = 1.0,
    batch_size: int = 5,
    output_path: Path | None = None,
    max_duration: float | None = None,
    visualize: bool = True,
) -> AnalysisResult:
    """Run the full analysis pipeline: ingest → embed → analyze → strategize."""

    if max_duration is not None and max_duration < MIN_ANALYSIS_DURATION:
        ui.warn(
            f"Minimum analysis duration is {MIN_ANALYSIS_DURATION}s "
            f"(requested {max_duration}s). Clamping to {MIN_ANALYSIS_DURATION}s."
        )
        max_duration = float(MIN_ANALYSIS_DURATION)

    install_signal_handlers()

    result = AnalysisResult(
        video=str(video_path),
        settings={
            "interval": interval,
            "batch_size": batch_size,
            "max_duration": max_duration,
        },
    )

    # ── Create services ───────────────────────────────────────────────

    from fighter_iq.services import CLIPEmbedder, FightAgent, FightStrategyService, VideoIngestor

    ingestor = VideoIngestor()
    embedder = CLIPEmbedder()
    agent = FightAgent()
    strategy_svc = FightStrategyService()

    # ── Phase 1: Per-frame analysis ───────────────────────────────────

    ui.info("Phase 1: Loading models (YOLO + VLM + CLIP)...")
    agent.load_detection_models()
    embedder.load()
    ui.phase_ok("Models loaded")

    embeddings: list[FrameEmbedding] = []
    frame_count = 0
    shutdown = False
    user_quit = False
    stream = EventStream()

    try:
        with Live(stream.get_renderable(), console=ui.console, refresh_per_second=4) as live:
            for timestamp, image in ingestor.extract(video_path, interval, max_duration):
                if is_shutdown_requested():
                    shutdown = True
                    break

                ts_str = f"{timestamp:.1f}s"

                # Embed
                vector = embedder.embed_frame(image)
                embeddings.append(embedder.make_embedding(timestamp, image, vector))

                # Analyze (detection + VLM + spatial)
                frame_analysis = agent.analyze_frame(image, timestamp)
                result.frames.append(frame_analysis)

                # Stream events
                fighters = frame_analysis.fighters
                conf_str = ", ".join(f"{f.confidence:.2f}" for f in fighters)
                stream.add(ts_str, "DETECT", f"{len(fighters)} fighter(s) (conf: {conf_str})")

                if frame_analysis.filtered_referee:
                    stream.add(ts_str, "FILTERED", f"Excluded: referee (conf: {frame_analysis.filtered_referee.confidence:.2f})")

                if frame_analysis.incomplete:
                    stream.add(ts_str, "INCOMPLETE", f"Only {len(fighters)} fighter(s) detected")

                short_desc = frame_analysis.description.replace("\n", " ")
                if len(short_desc) > 120:
                    short_desc = short_desc[:117] + "..."
                stream.add(ts_str, "DESCRIBE", short_desc)

                if frame_analysis.control_score is not None:
                    ctrl = frame_analysis.control_score
                    control_dir = "A" if ctrl >= 0 else "B"
                    ctrl_event = "CONTROL_A" if ctrl >= 0 else "CONTROL_B"
                    stream.add(ts_str, ctrl_event, f"Control: {abs(ctrl):.2f} → {control_dir}")

                if frame_analysis.impact:
                    stream.add(ts_str, "IMPACT", f"{frame_analysis.impact_type or 'unknown'}")

                live.update(stream.get_renderable())

                # Visualization
                if visualize:
                    annotated = draw_annotations(
                        image,
                        fighters,
                        frame_analysis.control_score,
                        frame_analysis.impact,
                        frame_analysis.impact_type,
                        frame_analysis.movement_vectors,
                        None,
                        incomplete=frame_analysis.incomplete,
                        referee=frame_analysis.filtered_referee,
                    )
                    if not show_frame(_WINDOW_NAME, annotated, wait_ms=1):
                        user_quit = True
                        break

                frame_count += 1
    finally:
        if visualize:
            cv2.destroyAllWindows()
        agent.unload_detection_models()
        embedder.unload()

    if shutdown:
        ui.warn(f"Shutdown requested — saving partial results ({frame_count} frames).")
    elif user_quit:
        ui.warn(f"Analysis stopped by user after {frame_count} frames.")

    if not result.frames:
        ui.error("No frames extracted. Check video file.")
        _save_results(result, video_path, output_path)
        return result

    ui.phase_ok("Phase 1 complete", f"{frame_count} frames analyzed")

    # ── Phase 2: Segment stitching + final summary ────────────────────

    tactics: list[Tactic] = []
    strategies: list[Strategy] = []

    if not shutdown and not is_shutdown_requested():
        ui.info("Phase 2: Loading text model (Qwen2.5-1.5B)...")
        agent.load_text_model()
        ui.phase_ok("Text model loaded")

        try:
            for i in range(0, len(result.frames), batch_size):
                if is_shutdown_requested():
                    break
                batch = result.frames[i : i + batch_size]
                segment = agent.stitch_segment(batch)
                result.segments.append(segment)

                time_range = f"{segment.timestamps[0]:.1f}s – {segment.timestamps[-1]:.1f}s"
                ui.console.print(
                    Panel(
                        segment.narrative,
                        title=f"Segment {len(result.segments)} ({time_range})",
                        border_style="cyan",
                    )
                )

            if not is_shutdown_requested():
                ui.console.print()
                narratives = [s.narrative for s in result.segments]
                result.summary = agent.summarize_fight(narratives)
                ui.console.print(
                    Panel(result.summary, title="Final Summary", border_style="green")
                )
        finally:
            agent.unload_text_model()

    # ── Phase 3: Tactic + strategy identification ─────────────────────

    if not shutdown and not is_shutdown_requested() and result.frames:
        ui.info("Phase 3: Identifying tactics and strategies...")
        tactics = strategy_svc.identify_tactics(result.frames, embeddings)
        strategies = strategy_svc.classify_strategies(tactics, result.frames)
        ui.phase_ok("Strategy analysis complete", f"{len(tactics)} tactics, {len(strategies)} strategies")

        for s in strategies:
            ui.console.print(
                Panel(
                    s.description,
                    title=f"{s.strategy_type.value.replace('_', ' ').title()} ({s.start_time:.0f}s–{s.end_time:.0f}s)",
                    border_style="yellow",
                )
            )

    # ── Output ────────────────────────────────────────────────────────

    _save_results(result, video_path, output_path, tactics, strategies)
    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _save_results(
    result: AnalysisResult,
    video_path: Path,
    output_path: Path | None,
    tactics: list[Tactic] | None = None,
    strategies: list[Strategy] | None = None,
) -> None:
    """Serialize and save results to disk."""
    output_data = _serialize_result(result, tactics, strategies)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_path = outputs_dir / f"{video_path.stem}_analysis_{ts}.json"
    auto_path.write_text(json.dumps(output_data, indent=2))
    ui.phase_ok("Analysis saved", str(auto_path))

    if output_path:
        output_path.write_text(json.dumps(output_data, indent=2))
        ui.phase_ok("Also saved", str(output_path))


def load_analysis(path: Path) -> AnalysisResult:
    """Deserialize an analysis JSON file into an AnalysisResult.

    Backward-compatible with older JSON files that lack tactics/strategies.
    """
    data = json.loads(path.read_text())

    frames: list[FrameAnalysis] = []
    for f in data.get("frames", []):
        fighters: list[FighterDetection] = []
        for d in f.get("fighters", []):
            bbox = BBox(*d["bbox"])
            kps = (
                [Keypoint(k["x"], k["y"], k["confidence"]) for k in d["keypoints"]]
                if d.get("keypoints")
                else None
            )
            identity = FighterIdentity(d["identity"]) if d.get("identity") else None
            fighters.append(
                FighterDetection(
                    bbox=bbox,
                    confidence=d["confidence"],
                    keypoints=kps,
                    identity=identity,
                )
            )

        frames.append(
            FrameAnalysis(
                timestamp=f["timestamp"],
                description=f["description"],
                fighters=fighters,
                control_score=f.get("control_score"),
                proximity_to_center=f.get("proximity_to_center", []),
                movement_vectors=[tuple(v) for v in f.get("movement_vectors", [])],
                impact=f.get("impact", False),
                impact_type=f.get("impact_type"),
                incomplete=f.get("incomplete", False),
                fighter_count=f.get("fighter_count", len(fighters)),
                fighter_appearances=f.get("fighter_appearances", {}),
            )
        )

    segments: list[SegmentSummary] = []
    for s in data.get("segments", []):
        segments.append(
            SegmentSummary(
                timestamps=s["timestamps"],
                narrative=s["narrative"],
                avg_control=s["avg_control"],
                impacts=s["impacts"],
                incomplete_frames=s.get("incomplete_frames", 0),
            )
        )

    return AnalysisResult(
        video=data.get("video", ""),
        settings=data.get("settings", {}),
        frames=frames,
        segments=segments,
        summary=data.get("summary", ""),
    )


def _serialize_result(
    result: AnalysisResult,
    tactics: list[Tactic] | None = None,
    strategies: list[Strategy] | None = None,
) -> dict:
    """Convert AnalysisResult + tactics/strategies to a JSON-serializable dict."""
    data = {
        "video": result.video,
        "settings": result.settings,
        "frames": [
            {
                "timestamp": f.timestamp,
                "description": f.description,
                "fighters": [
                    {
                        "bbox": d.bbox.to_list(),
                        "confidence": round(d.confidence, 3),
                        "identity": d.identity.value if d.identity else None,
                        "keypoints": [
                            {"x": round(kp.x, 1), "y": round(kp.y, 1), "confidence": round(kp.confidence, 3)}
                            for kp in d.keypoints
                        ] if d.keypoints else None,
                    }
                    for d in f.fighters
                ],
                "fighter_appearances": f.fighter_appearances,
                "control_score": f.control_score,
                "proximity_to_center": f.proximity_to_center,
                "movement_vectors": [list(v) for v in f.movement_vectors],
                "impact": f.impact,
                "impact_type": f.impact_type,
                "incomplete": f.incomplete,
                "fighter_count": f.fighter_count,
            }
            for f in result.frames
        ],
        "segments": [
            {
                "timestamps": s.timestamps,
                "narrative": s.narrative,
                "avg_control": s.avg_control,
                "impacts": s.impacts,
                "incomplete_frames": s.incomplete_frames,
            }
            for s in result.segments
        ],
        "summary": result.summary,
    }

    if tactics:
        data["tactics"] = [tactic_to_dict(t) for t in tactics]
    if strategies:
        data["strategies"] = [strategy_to_dict(s) for s in strategies]

    return data
