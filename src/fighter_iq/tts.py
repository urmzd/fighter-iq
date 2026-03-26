"""TTS synthesis via mlx-audio Kokoro for commentary narration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from fighter_iq.commentary import CommentarySegment
from fighter_iq.personas import CommentaryPersona

_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
_SAMPLE_RATE = 24000


@dataclass
class AudioSegment:
    path: Path
    start_time: float
    end_time: float
    duration: float
    segment_index: int


def load_tts_model():
    """Load the Kokoro-82M TTS model via mlx-audio."""
    from mlx_audio.tts.utils import load_model

    return load_model(_MODEL_ID)


def synthesize_segment(
    model,
    text: str,
    voice: str,
    speed: float,
    output_path: Path,
) -> Path:
    """Generate a WAV file for a single commentary segment."""
    audio_chunks = []
    for result in model.generate(text=text, voice=voice, speed=speed):
        audio_chunks.append(np.array(result.audio))

    audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    # Ensure 1D
    if audio.ndim > 1:
        audio = audio.squeeze()

    sf.write(str(output_path), audio, _SAMPLE_RATE)
    return output_path


def synthesize_commentary(
    model,
    segments: list[CommentarySegment],
    persona: CommentaryPersona,
    output_dir: Path,
) -> list[AudioSegment]:
    """Synthesize all commentary segments to WAV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_segments: list[AudioSegment] = []

    for seg in segments:
        wav_path = output_dir / f"seg_{seg.segment_index:03d}.wav"
        synthesize_segment(model, seg.text, persona.voice, persona.speed, wav_path)

        info = sf.info(str(wav_path))
        audio_segments.append(
            AudioSegment(
                path=wav_path,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=info.duration,
                segment_index=seg.segment_index,
            )
        )

    return audio_segments


def synthesize_continuous(
    model,
    segments: list[CommentarySegment],
    persona: CommentaryPersona,
    total_duration: float,
    output_path: Path,
) -> Path:
    """Synthesize all commentary as one continuous audio and pad to video length.

    Concatenates all segment texts into a single string, synthesizes once for
    consistent prosody with no inter-segment gaps, then places the speech at the
    first segment's start_time within a silence-padded timeline matching the video
    duration.
    """
    if not segments:
        # No commentary — produce a silent file matching video duration
        total_samples = int(total_duration * _SAMPLE_RATE)
        silence = np.zeros(total_samples, dtype=np.int16)
        sf.write(str(output_path), silence, _SAMPLE_RATE, subtype="PCM_16")
        return output_path

    # Concatenate all commentary text with sentence spacing
    full_text = "  ".join(seg.text for seg in segments)

    # Single TTS call for the entire commentary
    audio_chunks: list[np.ndarray] = []
    for result in model.generate(text=full_text, voice=persona.voice, speed=persona.speed):
        audio_chunks.append(np.array(result.audio))

    speech = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    if speech.ndim > 1:
        speech = speech.squeeze()

    # Build timeline: silence before first segment, then continuous speech, then silence
    total_samples = int(total_duration * _SAMPLE_RATE)
    timeline = np.zeros(total_samples, dtype=np.float32)

    start_sample = int(segments[0].start_time * _SAMPLE_RATE)
    end_sample = min(start_sample + len(speech), total_samples)
    samples_to_copy = end_sample - start_sample
    if samples_to_copy > 0:
        timeline[start_sample:end_sample] = speech[:samples_to_copy]

    timeline_int16 = np.clip(timeline * 32767, -32768, 32767).astype(np.int16)
    sf.write(str(output_path), timeline_int16, _SAMPLE_RATE, subtype="PCM_16")
    return output_path


def concatenate_timeline(
    audio_segments: list[AudioSegment],
    total_duration: float,
    output_path: Path,
) -> Path:
    """Place audio segments at correct timestamps with silence padding.

    Creates a single WAV file where each segment starts at its start_time,
    enabling trivial audio-video sync.
    """
    total_samples = int(total_duration * _SAMPLE_RATE)
    timeline = np.zeros(total_samples, dtype=np.float32)

    for seg in audio_segments:
        data, sr = sf.read(str(seg.path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        start_sample = int(seg.start_time * _SAMPLE_RATE)
        end_sample = min(start_sample + len(data), total_samples)
        samples_to_copy = end_sample - start_sample
        if samples_to_copy > 0:
            timeline[start_sample:end_sample] = data[:samples_to_copy]

    timeline_int16 = np.clip(timeline * 32767, -32768, 32767).astype(np.int16)
    sf.write(str(output_path), timeline_int16, _SAMPLE_RATE, subtype="PCM_16")
    return output_path
