# Architecture — Fighter IQ

## 1. System Overview

The pipeline is organized around four service boundaries, each defined by a Protocol in `protocols.py`:

```
Video File
  │
  ▼
┌─── Ingestor (services/ingestor.py) ───────────────────────────────┐
│  Frame Extraction (extractor.py, OpenCV, configurable interval)   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─── Embedding Model (services/embedder.py) ────────────────────────┐
│  CLIP ViT-B/32 → 512-dim vectors for similarity + clustering     │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─── Agent (services/agent.py) ─────────────────────────────────────┐
│  YOLO Pose Detection (detector.py, YOLOv8n-pose)                  │
│    ├─ 3-stage role classification: spectator → referee → fighter   │
│    └─ Color-histogram identity tracking                            │
│  VLM Frame Description (analyzer.py, Qwen2.5-VL-7B-Instruct)     │
│  Spatial Metrics (spatial.py — control, proximity, impact)         │
│  Segment Stitching (summarizer.py, Qwen2.5-1.5B-Instruct)        │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─── Strategy Service (services/strategy.py) ───────────────────────┐
│  Tactic identification (keyword + embedding boundary detection)    │
│  Strategy classification (pressure, counter, grapple-dominant...)  │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
              Commentary + TTS + Rendering + Review UI
```

The pipeline runs in three phases:

- **Phase 1** (per-frame): Ingestor yields frames → Embedder produces CLIP vectors → Agent runs detection, VLM description, spatial metrics. Models loaded: YOLOv8n-pose + Qwen2.5-VL-7B + CLIP ViT-B/32.
- **Phase 2** (batch): Agent stitches frame batches into segment narratives + final summary. Models loaded: Qwen2.5-1.5B.
- **Phase 3** (strategy): Strategy Service identifies tactics from frame analyses + embeddings, classifies strategies from tactic sequences.

A separate `review` command handles commentary generation, TTS, video rendering, and the web UI.

### Domain Model

- **Tactic** (`models.py`): An atomic fighting action (jab, takedown, slip, angle cut) with category, timestamps, confidence, and actor attribution.
- **Strategy** (`models.py`): A sequence of tactics forming a game plan (pressure fighting, counter-striking, grapple-dominant) with type classification and confidence.
- **FrameEmbedding** (`models.py`): CLIP vector for a frame, used for similarity clustering and action boundary detection.

---

## 2. Module Map

### `__init__.py`
Core data models shared across the pipeline. Key types: `PersonRole` and `FighterIdentity` enums, `BBox` (with center/area properties), `Keypoint` (x, y, confidence), `ColorHistogram` (HSV histogram + dominant color), `FighterProfile` (identity + histogram + staleness tracking), `DetectedPerson` (raw YOLO output), `FighterDetection` (pipeline output with optional identity), `FrameAnalysis`, `SegmentSummary`, `AnalysisResult`.

### `models.py`
Fighter IQ domain model. Defines `TacticCategory` and `StrategyType` enums, `Tactic` (atomic action with name, category, timestamps, confidence, actor), `Strategy` (game plan with type, tactic sequence, confidence), `FrameEmbedding` (CLIP vector + timestamp). Includes serialization helpers for JSON round-tripping.

### `protocols.py`
Four `@runtime_checkable` Protocol interfaces defining the service boundaries: `Ingestor` (frame extraction), `EmbeddingModel` (image → vector), `Agent` (frame analysis + summarization), `StrategyService` (tactic/strategy classification).

### `services/ingestor.py`
`VideoIngestor` class wrapping `extractor.py`. Implements the `Ingestor` protocol.

### `services/embedder.py`
`CLIPEmbedder` class using open_clip ViT-B/32 for 512-dim image embeddings. Implements the `EmbeddingModel` protocol. Supports single-frame and batch embedding with cosine similarity helper.

### `services/agent.py`
`FightAgent` class encapsulating detection + VLM + spatial + summarization. Implements the `Agent` protocol. Owns model lifecycle (load/unload for YOLO, VLM, text model) and fighter profile tracking state. Delegates to `detector.py`, `analyzer.py`, `spatial.py`, `summarizer.py`.

### `services/strategy.py`
`FightStrategyService` class for tactic identification and strategy classification. Implements the `StrategyService` protocol. Uses keyword matching on VLM descriptions + CLIP embedding similarity for action boundary detection. Classifies strategies via sliding-window tactic category distributions.

### `extractor.py`
OpenCV-based frame extraction. Reads video at a configurable interval (default 1 fps), yields `(timestamp, PIL.Image)` tuples. Supports `max_duration` cutoff.

### `detector.py`
YOLOv8n-pose detection with a 3-stage role classification pipeline and color-histogram identity tracking. Contains: `detect_persons()` (raw YOLO, conf >= 0.80), `filter_spectators()` (bbox size/position heuristics), `filter_referee()` (aspect ratio + proximity + area scoring), `detect_fighters()` (cold-start orchestrator), `extract_color_histogram()` (lower-50% bbox → HSV histogram), `initialize_profiles()` / `update_profile()` (EMA-blended histogram), `match_profiles()` (appearance + spatial greedy assignment), `filter_referee_with_profiles()` (profile-aware referee scoring).

### `analyzer.py`
Per-frame VLM description using Qwen2.5-VL-7B-Instruct (4-bit, via mlx-vlm). Builds a context-aware prompt with fighter appearance labels and referee presence. Returns 2–3 sentence MMA action descriptions.

### `spatial.py`
Computes four spatial metrics from fighter detections: `compute_control()` (−1 to +1 score from vertical position, forward pressure, cage proximity), `compute_proximity()` (normalized distance to frame center), `compute_movement_vectors()` (dx/dy displacement between frames), `detect_impact()` (bbox IoU + displacement + VLM keyword matching for strike/takedown/knockdown/sweep).

### `visualizer.py`
OpenCV frame annotation. Draws COCO-17 skeleton connections, fighter bounding boxes (green = A, blue = B), referee box (gray), control score bar, impact border (red), movement vector arrows (yellow), and incomplete-frame notices.

### `pipeline.py`
Three-phase orchestrator wiring the four services together. Phase 1: Ingestor yields frames → Embedder produces vectors → Agent analyzes each frame. Phase 2: Agent stitches segments + final summary. Phase 3: Strategy Service identifies tactics and classifies strategies. Handles live event streaming, OpenCV visualization, graceful shutdown (SIGINT/SIGTERM), and JSON serialization including tactics/strategies.

### `summarizer.py`
Qwen2.5-1.5B-Instruct (4-bit, via mlx-lm) for text summarization. `stitch_segment()` takes a batch of `FrameAnalysis` objects and produces a 3–5 sentence narrative. `final_summary()` takes all segment summaries and produces a 4–6 sentence overall analysis.

### `commentary.py`
LLM-driven per-segment spoken commentary. Takes the segment narrative + frame data + persona system prompt, generates 2–4 sentences of natural spoken text per segment using the same Qwen2.5-1.5B model with temperature 0.7.

### `personas.py`
Defines `CommentaryPersona` dataclass (id, name, system prompt, Kokoro voice preset, speed multiplier). Ships two personas: `technical` (clinical biomechanics analysis, voice `am_adam`, speed 1.0) and `hype` (dramatic play-by-play, voice `am_echo`, speed 1.1).

### `tts.py`
Kokoro-82M TTS synthesis via mlx-audio. `synthesize_segment()` produces a single WAV. `synthesize_continuous()` concatenates all segment texts into one TTS call for consistent prosody, then places the speech at the correct timeline offset within a silence-padded WAV matching video duration. Sample rate: 24 kHz.

### `renderer.py`
Pre-renders an annotated MP4 from the original video + analysis JSON. Reads every source frame (at native fps), finds the surrounding analyzed frames, linearly interpolates fighter bboxes and keypoints between them, and writes annotations via `draw_annotations()`.

### `review_ui.py`
NiceGUI web application for fight review. Serves annotated video + commentary audio, provides transport controls (play/pause, seek slider, segment jump buttons), displays live commentary text synced to playback position, and shows expandable segment details and fight summary.

### `cli.py`
Typer CLI with two commands: `analyze` (runs the full detection + VLM + spatial + summarization pipeline) and `review` (loads analysis JSON → renders annotated video → generates commentary → synthesizes TTS → launches web UI).

### `event_stream.py`
Rich Live rolling event table for real-time terminal output during analysis. Displays timestamped events (DETECT, DESCRIBE, IMPACT, CONTROL, FILTERED, INCOMPLETE) with color-coded styling.

### `shutdown.py`
Graceful shutdown via `threading.Event`. Installs SIGINT/SIGTERM handlers that set a flag checked by the pipeline loop, enabling partial-result saving.

---

## 3. Detection & Identity Pipeline (detailed)

### Cold-Start Path (no profiles initialized)

```
detect_fighters() — detector.py:412
  │
  ├─ detect_persons()        → YOLO inference, conf >= 0.80, sorted by confidence
  │                             (detector.py:29)
  │
  ├─ filter_spectators()     → Mark small/peripheral bboxes as SPECTATOR
  │                             Gate: skips if non_spectator_count <= 2  (detector.py:74)
  │
  ├─ filter_referee()        → Score remaining UNKNOWN persons on referee heuristics
  │                             Gate: skips if len(candidates) <= 2  (detector.py:122)
  │                             Heuristics: upright aspect ratio (40%), proximity to
  │                             fighter midpoint (30%), smaller than fighters (30%)
  │
  └─ Build output            → REFEREE → referee slot
                                SPECTATOR → spectators list
                                FIGHTER or UNKNOWN → fighters list, top 2 by confidence
                                (detector.py:427–438)
```

### Profile Path (after bootstrap)

```
FightAgent._detect_with_profiles()  (services/agent.py)
  │
  ├─ detect_persons()        → Same raw YOLO detection
  ├─ filter_spectators()     → Same spectator filtering
  │
  ├─ match_profiles()        → For each (person, profile) pair, compute:
  │                             80% color histogram correlation (cv2.HISTCMP_CORREL)
  │                             20% spatial proximity (bbox center distance, staleness decay)
  │                             Greedy assignment, appearance threshold 0.3
  │                             (detector.py:280)
  │
  ├─ filter_referee_with_profiles() → Score unmatched persons against referee heuristics
  │                                    Same 3 heuristics, but requires score > 0.4
  │                                    Works even with <= 2 persons (no gate)
  │                                    (detector.py:352)
  │
  ├─ update_profile()        → EMA blend (alpha=0.3) of histogram, update bbox/confidence
  │                             (detector.py:259)
  │
  └─ Build identity_map      → Stable FIGHTER_A / FIGHTER_B ordering regardless of
                                bbox position or confidence  (pipeline.py:141–153)
```

### Profile Bootstrap

**Condition** (pipeline.py:192): exactly 2 fighters detected AND referee is not None (i.e., 3+ persons seen and referee successfully identified).

**Process**: Re-run `detect_persons()` + `filter_spectators()`, take top-2 by area as fighter persons, call `initialize_profiles()` which extracts color histograms from the lower 50% of each bbox (shorts region) and assigns FIGHTER_A / FIGHTER_B in area order.

**Staleness**: If both profiles go unseen for >5 frames (pipeline.py:171), profiles are reset and the system falls back to cold-start.

---

## 4. ML Diagnosis — What's Going Wrong

### Problem 1: Referee Classified as Fighter

**Root cause chain:**

1. `filter_referee()` (detector.py:122) has a hard gate: `if len(candidates) <= 2: return` — when YOLO detects only 2–3 people and spectator filtering removes one, only 2 UNKNOWN candidates remain, so the referee filter **never runs**.

2. `filter_spectators()` (detector.py:74) has the same pattern: `if non_spectator_count <= 2: return` — with 2 or fewer non-spectators, nothing gets filtered.

3. In `detect_fighters()` (detector.py:432–434), anyone with role UNKNOWN is treated as a fighter:
   ```python
   # FIGHTER or UNKNOWN (when <=2 persons, they stay UNKNOWN = treated as fighters)
   fighters.append(FighterDetection(bbox=p.bbox, confidence=p.confidence, keypoints=p.keypoints))
   ```

4. Profile bootstrap (pipeline.py:192) requires `len(fighters) == 2 and referee is not None`. If the referee is never identified, profiles never initialize — the system stays on the cold-start path forever, and the referee keeps being treated as a fighter every frame.

**Why this is common in MMA footage:** Standard cage cameras typically see 2 fighters + 1 referee with few or no visible spectators. YOLO detects exactly 3 persons. After spectator filtering (which does nothing since count <= 3), all 3 pass to `filter_referee()`. This works fine. But when one person is briefly occluded or YOLO's 0.80 confidence threshold drops someone, only 2 remain → referee filter is skipped → referee becomes a fighter.

### Problem 2: Skeleton Drawn on Wrong Person (Referee)

**Root cause:** This is a direct consequence of Problem 1. When the referee is misclassified as a fighter, it enters the `fighters[:2]` list returned by `detect_fighters()` (detector.py:438). The visualizer (visualizer.py:72) iterates `fighters[:2]` and draws skeleton + label for each:

```python
for i, fighter in enumerate(fighters[:2]):
    color = colors[i]
    label = labels[i]
```

So the referee gets a "Fighter A" or "Fighter B" skeleton overlay with green/blue coloring. The skeleton keypoints themselves are correct per YOLO's pose estimation — they're just drawn on the referee instead of the actual fighter. Meanwhile the real second fighter either:
- Falls to index 2+ in the fighters list and is discarded by the `[:2]` slice
- Was the person dropped by the 0.80 confidence threshold in `detect_persons()`

### Problem 3: Identity Tracking Uses Only Color, Not Pose

**Root cause:** `match_profiles()` (detector.py:280) computes match scores using:
- **80%** color histogram correlation (lower-half bbox → shorts region)
- **20%** spatial proximity (bbox center distance with staleness decay)

The 17 COCO keypoints are extracted by YOLO and stored on every `DetectedPerson`, but they are **never used for matching**. This means:

- Two fighters with similar shorts colors (both black, both blue) are nearly indistinguishable to the tracker
- The referee (who often wears a solid black shirt) can match a fighter's profile if shorts colors are close
- During clinch/grappling, the lower-50% bbox crop captures both fighters' bodies, contaminating the histogram with the opponent's colors

### Problem 4: Confidence Threshold Too Aggressive

**Root cause:** The 0.80 confidence threshold in `detect_persons()` (detector.py:42) drops valid detections in challenging poses. YOLOv8-nano-pose produces lower confidence scores on:
- Heavily occluded bodies (clinch, grappling scrambles)
- Unusual poses (inverted guard, turtle position, side control)
- Motion blur (especially at the default 1 fps extraction interval)

Dropping from 3 to 2 detected persons triggers the `<= 2` guards in both `filter_spectators()` and `filter_referee()`, causing the cascade failure described in Problem 1.

---

## 5. Improvement Recommendations

### Fix 1 (Critical): Always-On Referee Filtering

Remove the `len(candidates) <= 2` gate in `filter_referee()`. Instead, score **all** non-spectator, non-matched persons against the referee heuristic even when there are only 2. Apply a minimum score threshold (like the 0.4 already used in `filter_referee_with_profiles()` at detector.py:406) so the system won't mark a real fighter as referee when the score is low.

**File:** `src/fighter_iq/detector.py` — `filter_referee()` (line 113) and `detect_fighters()` (line 412)

### Fix 2 (Critical): Lower YOLO Confidence Threshold

Drop from 0.80 → 0.50. YOLOv8 is already filtering to class 0 (person-only); 0.50 retains occluded and unusual-pose detections while still rejecting pure noise. This keeps 3 people visible more consistently, preventing the cascade failure where `<= 2` guards skip both filter stages.

**File:** `src/fighter_iq/detector.py` — `detect_persons()` (line 42)

### Fix 3 (High Value): Pose-Based Referee Detection

Use the 17 COCO keypoints already extracted to distinguish referee from fighters:
- **Arm angle:** Referee typically stands with arms at sides or slightly raised; fighters have guard up (elbows bent, wrists near chin level)
- **Torso angle:** Referee stands upright; fighters adopt stances with wider base and forward lean
- **Body area occupied:** Fighters in stance occupy more horizontal space; referee presents a narrow silhouette

Compute a `pose_referee_score` from keypoints alongside the existing bbox-geometry heuristics. This is more robust than aspect ratio alone because it uses actual body pose rather than bounding box shape.

**File:** `src/fighter_iq/detector.py` — new helper `_compute_pose_referee_score(keypoints)`, integrated into both `filter_referee()` and `filter_referee_with_profiles()`

### Fix 4 (High Value): Keypoint-Based Identity Matching

Add keypoint spatial features to `match_profiles()`:
- **Torso center** (mean of shoulder + hip keypoints) as a more stable spatial anchor than bbox center, which shifts with pose changes
- **Body proportion signature** (shoulder width / hip width, torso length / leg length) as a soft biometric that is more stable than shorts color during clinch

Suggested weight distribution: 60% color histogram, 20% keypoint-spatial proximity, 20% body proportion similarity.

**File:** `src/fighter_iq/detector.py` — `match_profiles()` (line 280)

### Fix 5 (Medium): Temporal Consistency via Kalman Filter

Replace the raw bbox center matching in `match_profiles()` with a Kalman filter per tracked identity. Predict next-frame position from velocity, associate detections to predictions using the Hungarian algorithm, and handle occlusion gracefully by coasting on predictions for several frames. This prevents identity swaps during fast lateral movement or clinch exchanges.

### Fix 6 (Medium): Histogram Crop Masking

During grappling, the lower-50% bbox crop in `extract_color_histogram()` (detector.py:199) often contains both fighters' bodies. Use the keypoint hip positions (indices 11, 12 in the COCO-17 layout) to create a tighter polygonal mask around just the target fighter's shorts/legs region, excluding the opponent's body from the histogram computation.

**File:** `src/fighter_iq/detector.py` — `extract_color_histogram()` (line 199)
