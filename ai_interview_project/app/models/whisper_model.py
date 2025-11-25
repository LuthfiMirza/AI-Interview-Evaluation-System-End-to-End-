"""Whisper speech-to-text model integration."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import whisper

LOGGER = logging.getLogger(__name__)

DEFAULT_STT_CONFIG: Dict[str, Any] = {
    "task": "transcribe",
    "language": "en",
    "temperature": 0.0,
    "best_of": 5,
    "beam_size": 5,
    "patience": 0.2,
    "condition_on_previous_text": True,
}


def _confidence_from_segments(segments: Iterable[Dict[str, Any]]) -> float:
    """Convert Whisper ``avg_logprob`` scores into a probability-style confidence."""
    logprobs: List[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        logprob = float(segment.get("avg_logprob", -5.0))
        logprobs.append(logprob)
    if not logprobs:
        return 0.0
    probs = [math.exp(max(min(lp, 0.0), -5.0)) for lp in logprobs]
    return round(sum(probs) / len(probs), 3)


class WhisperTranscriber:
    """Lightweight wrapper around Whisper for reuse across requests."""

    def __init__(
        self,
        model_size: str = "medium.en",
        device: str | None = None,
        default_config: Dict[str, Any] | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self._model = None
        self.default_config = default_config or DEFAULT_STT_CONFIG

    def load_model(self) -> whisper.Whisper:
        """Load the Whisper model lazily to avoid expensive initialization."""
        if self._model is None:
            LOGGER.info("Loading Whisper model: size=%s device=%s", self.model_size, self.device)
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Transcribe an audio file and return Whisper's structured response."""
        model = self.load_model()
        infer_config = {**self.default_config, **kwargs}
        LOGGER.info("Transcribing audio file with Whisper: %s", audio_path)
        result = model.transcribe(audio_path, **infer_config)
        segments = result.get("segments") or []
        result["confidence"] = _confidence_from_segments(segments)
        return result


@lru_cache(maxsize=2)
def get_transcriber(
    model_size: str = "medium.en",
    device: str | None = None,
) -> WhisperTranscriber:
    """Return a cached `WhisperTranscriber` instance."""
    return WhisperTranscriber(model_size=model_size, device=device)


def transcribe_audio(audio_path: str, model_size: str = "medium.en", **kwargs: Any) -> Dict[str, Any]:
    """Convenience function to perform transcription using a cached transcriber."""
    transcriber = get_transcriber(model_size=model_size, device=kwargs.pop("device", None))
    return transcriber.transcribe(audio_path, **kwargs)
