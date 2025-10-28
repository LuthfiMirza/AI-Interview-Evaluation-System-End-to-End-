"""Audio processing utilities leveraging ffmpeg-python."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import ffmpeg

LOGGER = logging.getLogger(__name__)

BASE_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "transcripts"


def extract_audio(video_path: str, output_dir: Optional[Path] = None) -> str:
    """Extract mono 16kHz WAV audio from a video file.

    Args:
        video_path: Path to the source video file.
        output_dir: Optional directory to store the extracted audio.

    Returns:
        Path to the generated audio file as a string.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    target_dir = output_dir or BASE_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_path = target_dir / f"{video_file.stem}_{uuid.uuid4().hex}.wav"

    LOGGER.info("Extracting audio from %s to %s", video_file, audio_path)
    try:
        (
            ffmpeg.input(str(video_file))
            .output(
                str(audio_path),
                acodec="pcm_s16le",
                ac=1,
                ar=16000,
                format="wav",
                loglevel="error",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        LOGGER.exception("ffmpeg failed to extract audio: %s", exc)
        raise RuntimeError("Audio extraction failed") from exc

    return str(audio_path)
