---
name: fighter-iq
description: "AI-powered combat sports strategy analysis. Identifies tactics and strategies from fight video using CLIP embeddings, VLM descriptions, and spatial metrics. Use when working on the analysis pipeline, domain model, or service interfaces."
---

# Fighter IQ

AI-powered strategy analysis for combat sports. Processes fight video through four service stages to identify tactics (atomic actions) and strategies (game plans).

## Domain Model

### Tactics

A **tactic** is an atomic fighting action identified across one or more frames. Each has a category, name, timestamps, confidence score, and optional actor attribution (Fighter A or B).

**Categories:**

| Category | Examples |
|----------|----------|
| **Strike** | jab, cross, hook, uppercut, roundhouse kick, elbow, knee |
| **Grapple** | double-leg takedown, single-leg, clinch entry, mount, guard pass, back take, slam, suplex |
| **Movement** | angle cut, circling, advancing, retreating, lateral footwork |
| **Defense** | slip, block, parry, sprawl, evade, shell up |
| **Transition** | guard to mount, clinch to takedown, scramble to standing |

Tactics are identified by:
1. Keyword classification from VLM frame descriptions
2. CLIP embedding similarity for action boundary detection (similarity < 0.85 = new action)
3. Consecutive same-category frames merged into spans

### Strategies

A **strategy** is a sequence of tactics forming a coherent game plan over a time window (default 30 seconds). Classified by tactic category distribution + movement patterns.

| Strategy | Description | Classification Rule |
|----------|-------------|---------------------|
| **Pressure** | Forward movement + high volume striking | >60% strikes + forward pressure |
| **Counter** | Reactive positioning + exploiting openings | >60% strikes + no forward pressure |
| **Grapple-dominant** | Takedown-centric + top control | >40% grapple tactics |
| **Clinch work** | Inside fighting + dirty boxing | >40% clinch/grapple in close range |
| **Point fighting** | In-and-out + selective engagement | >40% defense + low tactic count |
| **Mixed** | No dominant pattern | Default when no threshold met |

### How Tactics Compose Into Strategies

A pressure fighter might show this tactic sequence:
```
jab → cross → advancing → jab → hook → advancing → clinch entry → knee
```
Categories: STRIKE, STRIKE, MOVEMENT, STRIKE, STRIKE, MOVEMENT, GRAPPLE, STRIKE

With >60% strikes and consistent forward movement vectors, this classifies as **Pressure**.

A counter-striker might show:
```
retreating → slip → cross → circling → block → hook → retreating
```
Categories: MOVEMENT, DEFENSE, STRIKE, MOVEMENT, DEFENSE, STRIKE, MOVEMENT

With strikes present but no forward pressure, this classifies as **Counter**.

## Architecture

### Four Service Boundaries

Each service implements a Protocol defined in `protocols.py`:

**1. Ingestor** (`services/ingestor.py`)
- Input: video file path, interval, max duration
- Output: yields `(timestamp, PIL.Image)` tuples
- Implementation: `VideoIngestor` wrapping OpenCV frame extraction

**2. Embedding Model** (`services/embedder.py`)
- Input: PIL.Image
- Output: 512-dim numpy vector (normalized)
- Implementation: `CLIPEmbedder` using open_clip ViT-B/32
- Used for: frame similarity, action boundary detection, future clustering/search

**3. Agent** (`services/agent.py`)
- Input: PIL.Image + timestamp + context
- Output: `FrameAnalysis` (detections, description, spatial metrics)
- Implementation: `FightAgent` wrapping YOLOv8 + Qwen2.5-VL-7B + spatial computation
- Also handles: segment stitching and fight summarization via Qwen2.5-1.5B
- Stateful: owns fighter profile tracking (color histogram + EMA blending)

**4. Strategy Service** (`services/strategy.py`)
- Input: `list[FrameAnalysis]` + `list[FrameEmbedding]`
- Output: `list[Tactic]` + `list[Strategy]`
- Implementation: `FightStrategyService` using keyword matching + embedding similarity

### Pipeline Phases

```
Phase 1 (per-frame):  Ingestor → Embedder → Agent.analyze_frame()
Phase 2 (batch):      Agent.stitch_segment() → Agent.summarize_fight()
Phase 3 (strategy):   StrategyService.identify_tactics() → classify_strategies()
```

### Key Files

| File | Purpose |
|------|---------|
| `models.py` | Tactic, Strategy, FrameEmbedding, TacticCategory, StrategyType |
| `protocols.py` | Ingestor, EmbeddingModel, Agent, StrategyService protocols |
| `services/ingestor.py` | VideoIngestor |
| `services/embedder.py` | CLIPEmbedder (ViT-B/32, 512-dim) |
| `services/agent.py` | FightAgent (YOLO + VLM + spatial + summarizer) |
| `services/strategy.py` | FightStrategyService (tactic ID + strategy classification) |
| `pipeline.py` | 3-phase orchestrator wiring services together |
| `__init__.py` | Core data models (BBox, FrameAnalysis, etc.) |
| `detector.py` | YOLOv8 pose detection + role filtering + identity tracking |
| `analyzer.py` | Qwen2.5-VL-7B per-frame VLM description |
| `spatial.py` | Control scoring, proximity, movement vectors, impact detection |
| `summarizer.py` | Qwen2.5-1.5B segment stitching + final summary |
