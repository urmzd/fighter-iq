"""Pipeline orchestration: streaming per-frame analysis with real-time output."""

import gc
import json
from datetime import datetime
from pathlib import Path

import cv2
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from mma_ai_coach import (
    AnalysisResult,
    BBox,
    FighterDetection,
    FighterIdentity,
    FighterProfile,
    FrameAnalysis,
    Keypoint,
    PersonRole,
    SegmentSummary,
)
from mma_ai_coach.event_stream import EventStream
from mma_ai_coach.shutdown import install_signal_handlers, is_shutdown_requested
from mma_ai_coach.visualizer import draw_annotations, show_frame

console = Console()

_WINDOW_NAME = "MMA AI Coach"
MIN_ANALYSIS_DURATION = 120  # seconds — one standard MMA round


def run_pipeline(
    video_path: Path,
    interval: float = 1.0,
    batch_size: int = 5,
    output_path: Path | None = None,
    max_duration: float | None = None,
    visualize: bool = True,
) -> AnalysisResult:
    """Run the full analysis pipeline with streaming per-frame output."""

    if max_duration is not None and max_duration < MIN_ANALYSIS_DURATION:
        console.print(
            f"[yellow]Warning:[/] Minimum analysis duration is {MIN_ANALYSIS_DURATION}s "
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

    # ── Phase 1: Per-frame analysis (YOLO + VLM loaded simultaneously) ───

    console.print("\n[bold blue]Phase 1:[/] Loading models (YOLO + VLM)...")
    from mma_ai_coach.detector import (
        load_detector,
        detect_persons,
        detect_fighters,
        filter_spectators,
        filter_referee,
        initialize_profiles,
        update_profile,
        match_profiles,
        filter_referee_with_profiles,
    )
    from mma_ai_coach.analyzer import load_vision_model, analyze_frame
    from mma_ai_coach.spatial import (
        compute_control,
        compute_proximity,
        compute_movement_vectors,
        detect_impact,
    )
    from mma_ai_coach.extractor import extract_frames

    yolo = load_detector()
    vlm_model, vlm_processor, vlm_config = load_vision_model()
    console.print("  Models loaded.\n")

    prev_detections: list[FighterDetection] | None = None
    frame_count = 0
    shutdown = False

    # Fighter identity tracking state
    fighter_profiles: list[FighterProfile] = []
    profiles_initialized: bool = False

    stream = EventStream()
    user_quit = False

    try:
        with Live(stream.get_renderable(), console=console, refresh_per_second=4) as live:
            for timestamp, image in extract_frames(video_path, interval, max_duration):
                if is_shutdown_requested():
                    shutdown = True
                    break

                ts_str = f"{timestamp:.1f}s"

                # --- Detection with profile-based identity tracking ---
                fighter_descriptions: list[tuple[str, str]] | None = None
                fighter_appearances: dict[str, str] = {}

                if profiles_initialized:
                    # Path A: Profiles exist — use appearance matching
                    persons = detect_persons(yolo, image)
                    persons = filter_spectators(persons, image.width, image.height)
                    candidates = [p for p in persons if p.role != PersonRole.SPECTATOR]

                    matched_pairs, unmatched = match_profiles(
                        candidates, fighter_profiles, image
                    )

                    # Build matched index set for referee filtering
                    matched_indices: set[int] = set()
                    for person, _prof in matched_pairs:
                        for i, p in enumerate(persons):
                            if p is person:
                                matched_indices.add(i)
                                break

                    # Filter referee among unmatched (works even with <=2 persons)
                    persons = filter_referee_with_profiles(
                        persons, matched_indices, image.width, image.height
                    )

                    # Update matched profiles
                    for person, prof in matched_pairs:
                        update_profile(prof, person, image)

                    # Build fighters list in stable identity order
                    identity_map: dict[FighterIdentity, FighterDetection] = {}
                    for person, prof in matched_pairs:
                        identity_map[prof.identity] = FighterDetection(
                            bbox=person.bbox,
                            confidence=person.confidence,
                            keypoints=person.keypoints,
                            identity=prof.identity,
                        )

                    fighters = []
                    for ident in [FighterIdentity.FIGHTER_A, FighterIdentity.FIGHTER_B]:
                        if ident in identity_map:
                            fighters.append(identity_map[ident])

                    # Referee and spectators
                    referee: FighterDetection | None = None
                    spectators: list[FighterDetection] = []
                    for p in persons:
                        if p.role == PersonRole.REFEREE:
                            referee = FighterDetection(bbox=p.bbox, confidence=p.confidence)
                        elif p.role == PersonRole.SPECTATOR:
                            spectators.append(FighterDetection(bbox=p.bbox, confidence=p.confidence))

                    # Increment frames_since_last_seen for unmatched profiles
                    matched_profile_ids = {prof.identity for _, prof in matched_pairs}
                    for prof in fighter_profiles:
                        if prof.identity not in matched_profile_ids:
                            prof.frames_since_last_seen += 1

                    # Re-initialize if both profiles go stale (camera cut/replay)
                    if all(p.frames_since_last_seen > 5 for p in fighter_profiles):
                        profiles_initialized = False
                        fighter_profiles = []

                    # Build fighter descriptions for VLM
                    fighter_descriptions = [
                        (f"Fighter {prof.identity.value[-1].upper()}", prof.appearance_description)
                        for prof in fighter_profiles
                    ]
                    fighter_appearances = {
                        prof.identity.value: prof.appearance_description
                        for prof in fighter_profiles
                    }
                else:
                    # Path B: Cold start — original pipeline
                    fighters, referee, spectators = detect_fighters(
                        yolo, image, image.width, image.height
                    )

                    # Initialize profiles when we have 2 fighters AND a referee
                    # (3+ persons seen, referee filtered — prevents bootstrapping from fighter + referee)
                    if len(fighters) == 2 and referee is not None:
                        persons = detect_persons(yolo, image)
                        persons = filter_spectators(persons, image.width, image.height)
                        fighter_persons = [p for p in persons if p.role in (PersonRole.FIGHTER, PersonRole.UNKNOWN)]
                        fighter_persons.sort(key=lambda p: p.bbox.area, reverse=True)
                        fighter_profiles = initialize_profiles(fighter_persons[:2], image)
                        profiles_initialized = True

                        # Assign identities to current fighters
                        for i, f in enumerate(fighters):
                            f.identity = fighter_profiles[i].identity if i < len(fighter_profiles) else None

                        fighter_descriptions = [
                            (f"Fighter {prof.identity.value[-1].upper()}", prof.appearance_description)
                            for prof in fighter_profiles
                        ]
                        fighter_appearances = {
                            prof.identity.value: prof.appearance_description
                            for prof in fighter_profiles
                        }

                incomplete = len(fighters) < 2

                conf_str = ", ".join(f"{f.confidence:.2f}" for f in fighters)
                stream.add(ts_str, "DETECT", f"{len(fighters)} fighter(s) (conf: {conf_str})")
                live.update(stream.get_renderable())

                # Filtered event
                filter_parts: list[str] = []
                if referee:
                    filter_parts.append(f"referee (conf: {referee.confidence:.2f})")
                if spectators:
                    filter_parts.append(f"{len(spectators)} spectator(s)")
                if filter_parts:
                    stream.add(ts_str, "FILTERED", "Excluded: " + ", ".join(filter_parts))
                    live.update(stream.get_renderable())

                # Incomplete event
                if incomplete:
                    stream.add(ts_str, "INCOMPLETE", f"Only {len(fighters)} fighter(s) detected")
                    live.update(stream.get_renderable())

                # Describe with fighter context
                description = analyze_frame(
                    vlm_model, vlm_processor, vlm_config, image, timestamp,
                    fighter_descriptions=fighter_descriptions,
                    referee_detected=(referee is not None),
                )
                short_desc = description.replace("\n", " ")
                if len(short_desc) > 120:
                    short_desc = short_desc[:117] + "..."
                stream.add(ts_str, "DESCRIBE", short_desc)
                live.update(stream.get_renderable())

                # Spatial metrics
                if incomplete:
                    control = None
                    impact, impact_type = False, None
                else:
                    control = compute_control(
                        prev_detections, fighters, image.width, image.height
                    )
                    impact, impact_type = detect_impact(
                        prev_detections, fighters, description
                    )

                # These handle 0-1 fighters gracefully
                proximity = compute_proximity(fighters, image.width, image.height)
                vectors = compute_movement_vectors(prev_detections, fighters)

                # Control event (only for complete frames)
                if control is not None:
                    control_dir = "A" if control >= 0 else "B"
                    ctrl_event = "CONTROL_A" if control >= 0 else "CONTROL_B"
                    stream.add(ts_str, ctrl_event, f"Control: {abs(control):.2f} → {control_dir}")

                # Impact event
                if impact:
                    stream.add(ts_str, "IMPACT", f"{impact_type or 'unknown'}")

                live.update(stream.get_renderable())

                # Visualization
                if visualize:
                    annotated = draw_annotations(
                        image, fighters, control, impact, impact_type,
                        vectors, prev_detections,
                        incomplete=incomplete,
                        referee=referee,
                    )
                    if not show_frame(_WINDOW_NAME, annotated, wait_ms=1):
                        user_quit = True
                        break

                frame_analysis = FrameAnalysis(
                    timestamp=round(timestamp, 2),
                    description=description,
                    fighters=fighters,
                    control_score=round(control, 3) if control is not None else None,
                    proximity_to_center=[round(p, 3) for p in proximity],
                    movement_vectors=[(round(dx, 1), round(dy, 1)) for dx, dy in vectors],
                    impact=impact,
                    impact_type=impact_type,
                    incomplete=incomplete,
                    fighter_count=len(fighters),
                    filtered_referee=referee,
                    filtered_spectators=spectators,
                    fighter_appearances=fighter_appearances,
                )
                result.frames.append(frame_analysis)

                # Only update prev_detections when frame is complete
                if not incomplete:
                    prev_detections = fighters
                frame_count += 1
    finally:
        if visualize:
            cv2.destroyAllWindows()
        del yolo, vlm_model, vlm_processor, vlm_config
        gc.collect()

    if shutdown:
        console.print(f"\n[yellow]Shutdown requested — saving partial results ({frame_count} frames).[/]")
    elif user_quit:
        console.print(f"\n[yellow]Analysis stopped by user after {frame_count} frames.[/]")

    if not result.frames:
        console.print("[red]No frames extracted. Check video file.[/]")
        _save_results(result, video_path, output_path)
        return result

    console.print(
        f"[bold green]Phase 1 complete:[/] {frame_count} frames analyzed.\n"
    )

    # ── Phase 2: Segment stitching + final summary (text model) ──────────

    if not shutdown and not is_shutdown_requested():
        console.print("[bold blue]Phase 2:[/] Loading text model (Qwen2.5-1.5B)...")
        from mma_ai_coach.summarizer import load_text_model, stitch_segment, final_summary

        text_model, text_tokenizer = load_text_model()
        console.print("  Model loaded.\n")

        try:
            for i in range(0, len(result.frames), batch_size):
                if is_shutdown_requested():
                    break
                batch = result.frames[i : i + batch_size]
                segment = stitch_segment(text_model, text_tokenizer, batch)
                result.segments.append(segment)

                time_range = f"{segment.timestamps[0]:.1f}s – {segment.timestamps[-1]:.1f}s"
                console.print(
                    Panel(
                        segment.narrative,
                        title=f"Segment {len(result.segments)} ({time_range})",
                        border_style="blue",
                    )
                )

            if not is_shutdown_requested():
                console.print()
                result.summary = final_summary(text_model, text_tokenizer, result.segments)
                console.print(
                    Panel(result.summary, title="Final Summary", border_style="green")
                )
        finally:
            del text_model, text_tokenizer
            gc.collect()

    # ── Output ───────────────────────────────────────────────────────────

    _save_results(result, video_path, output_path)
    return result


def _save_results(
    result: AnalysisResult, video_path: Path, output_path: Path | None
) -> None:
    """Serialize and save results to disk."""
    output_data = _serialize_result(result)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_path = outputs_dir / f"{video_path.stem}_analysis_{ts}.json"
    auto_path.write_text(json.dumps(output_data, indent=2))
    console.print(f"\n[bold green]Analysis saved to:[/] {auto_path}")

    if output_path:
        output_path.write_text(json.dumps(output_data, indent=2))
        console.print(f"[bold green]Also saved to:[/] {output_path}")


def load_analysis(path: Path) -> AnalysisResult:
    """Deserialize an analysis JSON file into an AnalysisResult.

    Inverse of _serialize_result().
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


def _serialize_result(result: AnalysisResult) -> dict:
    """Convert AnalysisResult to a JSON-serializable dict."""
    return {
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
