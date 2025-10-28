"""Whisper speech-to-text model integration."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict

import whisper

LOGGER = logging.getLogger(__name__)


class WhisperTranscriber:
    """Lightweight wrapper around Whisper for reuse across requests."""

    def __init__(self, model_size: str = "small", device: str | None = None) -> None:
        self.model_size = model_size
        self.device = device
        self._model = None

    def load_model(self) -> whisper.Whisper:
        """Load the Whisper model lazily to avoid expensive initialization."""
        if self._model is None:
            LOGGER.info("Loading Whisper model: size=%s device=%s", self.model_size, self.device)
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Transcribe an audio file and return Whisper's structured response."""
        model = self.load_model()
        LOGGER.info("Transcribing audio file with Whisper: %s", audio_path)
        return model.transcribe(audio_path, **kwargs)


@lru_cache(maxsize=2)
def get_transcriber(model_size: str = "small", device: str | None = None) -> WhisperTranscriber:
    """Return a cached `WhisperTranscriber` instance."""
    return WhisperTranscriber(model_size=model_size, device=device)


def transcribe_audio(audio_path: str, model_size: str = "small", **kwargs: Any) -> Dict[str, Any]:
    """Convenience function to perform transcription using a cached transcriber."""
    transcriber = get_transcriber(model_size=model_size, device=kwargs.pop("device", None))
    return transcriber.transcribe(audio_path, **kwargs)
