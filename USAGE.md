# Usage

## Installation

```bash
# Clone and install
git clone <repo-url> && cd fighter-iq
uv sync
```

All ML model weights are downloaded automatically on first run.

## CLI Reference

### `fighter-iq analyze`

Run the full detection + VLM + spatial + summarization pipeline on a video.

```
fighter-iq analyze <SOURCE> [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--interval` | `-i` | `1.0` | Seconds between extracted frames |
| `--batch-size` | `-b` | `5` | Frames per segment for narrative stitching |
| `--output` | `-o` | auto | Save analysis JSON to a specific path |
| `--duration` | `-d` | full video | Only analyze the first N seconds (minimum 120s) |
| `--visualize / --no-visualize` | | `--visualize` | Show live OpenCV window with annotated frames |

### `fighter-iq review`

Load an analysis JSON, render annotated video, generate commentary, synthesize TTS, and launch the web review UI.

```
fighter-iq review [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--analysis` | `-a` | required | Path to analysis JSON from the analyze command |
| `--video` | `-v` | required | Path to the original video file |
| `--persona` | `-p` | `technical` | Commentary persona (`technical` or `hype`) |
| `--port` | | `8080` | Port for the review web server |

## Example Workflows

### Quick sanity test

```bash
fighter-iq analyze inputs/clip.mp4 --duration 10 --no-visualize --output outputs/test.json
fighter-iq review --analysis outputs/test.json --video inputs/clip.mp4
```

### Full analysis with live visualization

```bash
fighter-iq analyze inputs/full_fight.mp4 --interval 0.5 --output outputs/full.json
```

### Review with hype commentator

```bash
fighter-iq review \
  --analysis outputs/full.json \
  --video inputs/full_fight.mp4 \
  --persona hype \
  --port 9090
```

## Output Files

| File | Description |
|------|-------------|
| `outputs/<video>_analysis_<timestamp>.json` | Full analysis with per-frame data, segments, tactics, strategies, and summary |
| `outputs/review/<video>_annotated.mp4` | Rendered video with skeleton overlays, control bars, and impact borders |
| `outputs/review/commentary.wav` | Timeline-aligned TTS audio matching video duration |

## Web Review UI

The review command launches a NiceGUI web application at `http://localhost:<port>` with:

- Annotated video player with synchronized commentary audio
- Play/pause, seek slider, and segment jump buttons
- Live commentary text display synced to playback position
- Expandable segment details with control scores and impact counts
- Overall fight summary panel
