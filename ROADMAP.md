# Roadmap

Upgrade path for Fighter IQ — TTS, analysis, and inference improvements based on the current open-source landscape (Q1 2026).

## TTS Upgrades

Currently using Kokoro-82M via mlx-audio. Three models unlock significant commentary improvements:

### Orpheus TTS (Canopy AI)

- **What**: 150M / 1B / 3B variants built on Llama-3, trained on 100k+ hours
- **Why**: Real-time streaming (~200ms latency), zero-shot voice cloning, and inline emotion tags
- **Impact**: Map `impact=True` frames to excitement tags, `control_score` extremes to tension tags in `commentary.py`. Drop-in upgrade for `tts.py`
- **License**: Open
- **Links**: [GitHub](https://github.com/canopyai/Orpheus-TTS), [HuggingFace](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)

### Dia 1.6B (Nari Labs)

- **What**: Single-pass multi-speaker dialogue generation with `[S1]`/`[S2]` tags
- **Why**: Two-commentator mode — play-by-play + color analyst — without stitching separate audio. Produces nonverbals (gasps, laughter). Our `personas.py` already defines "technical" and "hype" personas that map directly to `[S1]`/`[S2]`
- **Impact**: New TTS backend option in `tts.py`, new `--dual-commentary` flag in `cli.py`
- **Trade-off**: English only, batch generation (no streaming), ~10GB VRAM
- **License**: Apache 2.0
- **Links**: [GitHub](https://github.com/nari-labs/dia), [HuggingFace](https://huggingface.co/nari-labs/Dia-1.6B)

### Zonos 1.6B (Zyphra)

- **What**: Fine-grained emotion sliders (happiness, fear, anger, surprise, sadness) + speaking rate/pitch control
- **Why**: Programmatically adjust emotion intensity based on fight metrics (strike frequency, knockdowns, control shifts)
- **Impact**: Emotion-reactive commentary in `commentary.py` driven by `FrameAnalysis` metrics
- **License**: Apache 2.0
- **Links**: [GitHub](https://github.com/Zyphra/Zonos)

## Pose Estimation Upgrades

Currently using YOLOv8n-pose. Two models address the six detection problems documented in ARCHITECTURE.md:

### ViTPose++ (Offline Analysis)

- **What**: Vision Transformer pose estimation with mixture-of-experts, state-of-the-art accuracy on COCO/CrowdPose
- **Why**: Significantly better than YOLOv8 for grappling/clinch scenarios where fighters overlap. Directly addresses ARCHITECTURE.md problems #3 (identity tracking), #4 (confidence threshold), and #6 (histogram contamination)
- **Impact**: Alternative pose backend in `detector.py` for batch/offline analysis. Keep YOLOv8 for real-time
- **Links**: [GitHub](https://github.com/ViTAE-Transformer/ViTPose), [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/vitpose)

### RTMPose (MMPose — Real-Time)

- **What**: CSPNeXt backbone + SimCC head, classification-based decoding (no heatmaps)
- **Why**: Real-time multi-person pose with better occlusion handling than YOLO. Addresses ARCHITECTURE.md problem #5 (temporal inconsistency)
- **Impact**: Drop-in replacement path for YOLOv8n-pose in `detector.py` with improved multi-person tracking
- **Links**: [MMPose GitHub](https://github.com/open-mmlab/mmpose)

## Action Recognition (New Capability)

Currently relying on VLM description keyword matching for tactic identification. Dedicated action recognition models are faster and more accurate for known action types:

### TimeSformer (Meta)

- **What**: Pure transformer for video classification, no convolutions. 3x faster training than 3D CNNs
- **Why**: Classify fighting actions (strikes, takedowns, clinch entries) directly from video clips instead of parsing VLM text. Higher accuracy, lower latency for known action types
- **Impact**: New action classifier in `services/strategy.py` alongside existing keyword matching. VLM descriptions remain for novel/ambiguous actions
- **Links**: [GitHub](https://github.com/facebookresearch/TimeSformer)

### Multi-Person Physics-Based Pose for Combat Sports

- **What**: Multi-stage, multi-view transformer with epipolar geometry constraints, designed specifically for boxing
- **Why**: Purpose-built for the two-fighter occlusion problem. Released annotated elite boxing video datasets for fine-tuning
- **Impact**: Potential replacement for the entire detection pipeline in `detector.py` for combat sports contexts
- **Links**: [arXiv 2504.08175](https://arxiv.org/html/2504.08175v1)

## Inference Optimization

### TurboQuant (Google, ICLR 2026)

- **What**: KV cache compression to 3 bits via random rotation + residual correction (QJL). 6x memory reduction, up to 8x speedup. Zero accuracy loss, no training required
- **Why**: Qwen2.5-VL-7B is the bottleneck in Phase 1 (per-frame analysis). TurboQuant could enable longer sequences, smaller GPUs, and higher throughput
- **Status**: Paper + pseudocode only. Rust crate `turbo-quant` exists. Full open-source implementations expected ~Q2 2026
- **Impact**: Apply to Qwen2.5-VL-7B and Qwen2.5-1.5B KV caches. Stacks on top of existing 4-bit mlx quantization
- **Links**: [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), [turbo-quant crate](https://lib.rs/crates/turbo-quant)

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Orpheus TTS in `tts.py` | Low | Emotion-tagged commentary |
| 2 | Dia dual-commentator mode | Medium | Two-voice commentary |
| 3 | ViTPose++ offline backend | Medium | Fixes grappling detection |
| 4 | TimeSformer action recognition | Medium | Accurate tactic classification |
| 5 | Zonos emotion-reactive TTS | Low | Metric-driven emotion |
| 6 | RTMPose real-time replacement | Medium | Better multi-person tracking |
| 7 | TurboQuant KV compression | Blocked | Waiting on open-source impl |
| 8 | Combat sports pose paper | High | Purpose-built detection |
