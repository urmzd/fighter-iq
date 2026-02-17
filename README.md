# Fight Analyzer

AI-powered video analysis pipeline for martial arts and combat sports.

## Features

- **Frame extraction** — configurable-interval sampling from any video file
- **Pose detection** — YOLOv8-nano-pose for 17-keypoint skeleton tracking
- **Fighter recognition** — color-histogram identity tracking with referee/spectator filtering
- **AI frame analysis** — Qwen2.5-VL-7B vision-language descriptions per frame
- **Spatial metrics** — control scoring, proximity, movement vectors, impact detection
- **Segment stitching** — Qwen2.5-1.5B narrative summaries across frame batches
- **Commentary generation** — persona-styled spoken text (technical analyst or hype commentator)
- **TTS synthesis** — Kokoro-82M text-to-speech with timeline-aligned audio
- **Web review UI** — NiceGUI player with synced annotated video, audio, and commentary

## Requirements

- Python 3.11+
- Apple Silicon recommended (MLX acceleration for VLM, LLM, and TTS models)
- ~16 GB RAM for concurrent model loading

## Installation

```bash
uv sync
```

## Quick Start

Analyze a video:

```bash
fight-analyzer analyze path/to/fight.mp4
```

Review the results with commentary:

```bash
fight-analyzer review --analysis outputs/fight_analysis_*.json --video path/to/fight.mp4
```

See [USAGE.md](USAGE.md) for the full CLI reference and example workflows.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map, detection pipeline details, and ML diagnosis notes.

## License

Apache 2.0 — see [LICENSE](LICENSE).
