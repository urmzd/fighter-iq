# Fighter IQ

AI-powered strategy analysis for combat sports video.

Fighter IQ watches fight footage and identifies **tactics** (individual actions like jabs, level changes, angle cuts) and **strategies** (game plans like pressure fighting, counter-striking, grapple-dominant play).

## Architecture

Four service boundaries, each defined by a Protocol interface:

```
Video File
  │
  ▼
┌─────────────────────┐
│     Ingestor         │  Frame extraction at configurable intervals
│  (services/ingestor) │  Yields (timestamp, PIL.Image) tuples
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Embedding Model     │  CLIP ViT-B/32 image embeddings (512-dim)
│  (services/embedder) │  Frame similarity, clustering, semantic search
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│       Agent          │  YOLOv8 pose detection + Qwen2.5-VL-7B descriptions
│  (services/agent)    │  Spatial metrics: control, proximity, impact
│                      │  Segment stitching via Qwen2.5-1.5B
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Strategy Service    │  Tactic identification from descriptions + embeddings
│ (services/strategy)  │  Strategy classification (pressure, counter, grapple...)
└─────────────────────┘
```

## Domain Model

**Tactic** — an atomic fighting action spanning one or more frames:
- *Strike*: jab, cross, hook, uppercut, roundhouse, elbow, knee
- *Grapple*: takedown, clinch entry, mount, guard pass, back take
- *Movement*: angle cut, circling, advancing, retreating
- *Defense*: slip, block, parry, sprawl
- *Transition*: guard to mount, clinch to takedown

**Strategy** — a sequence of tactics forming a coherent game plan:
- *Pressure*: constant forward movement + high volume striking
- *Counter*: reactive positioning + exploiting openings
- *Grapple-dominant*: takedown-centric + top control
- *Clinch work*: inside fighting + dirty boxing
- *Point fighting*: in-and-out movement + selective engagement

## Requirements

- Python 3.11+
- Apple Silicon recommended (MLX acceleration for VLM, LLM, and TTS)
- ~16 GB RAM for concurrent model loading

## Installation

```bash
uv sync
```

## Quick Start

Analyze a video:

```bash
fighter-iq analyze path/to/fight.mp4
```

Review results with commentary:

```bash
fighter-iq review --analysis outputs/fight_analysis_*.json --video path/to/fight.mp4
```

See [USAGE.md](USAGE.md) for the full CLI reference and example workflows.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — module map, detection pipeline, ML diagnosis
- [USAGE.md](USAGE.md) — CLI reference and examples

## License

Apache 2.0 — see [LICENSE](LICENSE).
