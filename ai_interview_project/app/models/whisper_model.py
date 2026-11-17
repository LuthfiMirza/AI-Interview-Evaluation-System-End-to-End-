"""Whisper speech-to-text model integration with technical prompting and multilingual support."""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

import whisper

LOGGER = logging.getLogger(__name__)

DEFAULT_TECHNICAL_PROMPT: str = (
    "Wawancara kerja software engineer: React, Next.js, TypeScript, JavaScript, "
    "Component, Props, State Management, Zustand, Redux, Context API, Hooks, "
    "useState, useEffect, Routing, App Router, Pages Router, Data Fetching, "
    "API, REST, SSR, CSR, SSG, Hydration, Tailwind CSS, Git."
)

DEFAULT_STT_CONFIG: Dict[str, Any] = {
    "task": "transcribe",
    "temperature": 0.0,
    "best_of": 5,
    "beam_size": 5,
    "patience": 0.2,
    "condition_on_previous_text": True,
    "initial_prompt": DEFAULT_TECHNICAL_PROMPT,
}


def resolve_model_size(model_size: str, language: Optional[str] = None) -> str:
    """
    Resolve model size for transcription.
    If target language is non-English (e.g. 'id') or auto-detected,
    switch English-only models (e.g. 'base.en', 'medium.en') to multilingual counterparts ('base', 'medium').
    """
    if language is not None and language.lower() not in {"en", "english"}:
        if model_size.endswith(".en"):
            multilingual = model_size[:-3]
            LOGGER.info(
                "Switched English-only Whisper model '%s' to multilingual '%s' for language='%s'",
                model_size,
                multilingual,
                language,
            )
            return multilingual
    return model_size


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
    # Clamp logprobs, exponentiate, then gently boost with sqrt to keep the scale readable.
    probs = [math.exp(max(min(lp, 0.0), -5.0)) for lp in logprobs]
    boosted = [math.sqrt(p) for p in probs]
    avg = sum(boosted) / len(boosted)
    return round(min(1.0, max(0.0, avg)), 3)


class WhisperTranscriber:
    """Lightweight wrapper around Whisper for reuse across requests."""

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        default_config: Dict[str, Any] | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self._model = None
        self._loaded_model_size: str | None = None
        self.default_config = dict(DEFAULT_STT_CONFIG)
        if default_config:
            self.default_config.update(default_config)

    def load_model(self, target_model_size: str | None = None) -> whisper.Whisper:
        """Load the Whisper model lazily to avoid expensive initialization."""
        size_to_load = target_model_size or self.model_size
        if self._model is None or self._loaded_model_size != size_to_load:
            LOGGER.info("Loading Whisper model: size=%s device=%s", size_to_load, self.device)
            self._model = whisper.load_model(size_to_load, device=self.device)
            self._loaded_model_size = size_to_load
        return self._model

    def transcribe(
        self,
        audio_path: str,
        initial_prompt: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe an audio file and return Whisper's structured response."""
        infer_config = {**self.default_config, **kwargs}
        if initial_prompt is not None:
            infer_config["initial_prompt"] = initial_prompt
        if language is not None:
            infer_config["language"] = language

        target_lang = infer_config.get("language")
        resolved_size = resolve_model_size(self.model_size, target_lang)

        model = self.load_model(target_model_size=resolved_size)
        LOGGER.info(
            "Transcribing audio file with Whisper (size=%s, lang=%s): %s",
            resolved_size,
            target_lang,
            audio_path,
        )
        result = model.transcribe(audio_path, **infer_config)
        segments = result.get("segments") or []
        result["confidence"] = _confidence_from_segments(segments)
        return result


@lru_cache(maxsize=4)
def get_transcriber(
    model_size: str = "base",
    device: str | None = None,
) -> WhisperTranscriber:
    """Return a cached `WhisperTranscriber` instance."""
    return WhisperTranscriber(model_size=model_size, device=device)


def transcribe_audio(
    audio_path: str,
    model_size: str | None = None,
    initial_prompt: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to perform transcription using a cached transcriber."""
    size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
    transcriber = get_transcriber(model_size=size, device=kwargs.pop("device", None))
    return transcriber.transcribe(
        audio_path,
        initial_prompt=initial_prompt,
        language=language,
        **kwargs,
    )
