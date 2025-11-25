"""High-level Speech-to-Text service with noise handling and evaluation hooks."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio import functional as AF

try:
    from torchaudio.sox_effects import apply_effects_file

    _SOX_AVAILABLE = True
except (ImportError, RuntimeError):
    apply_effects_file = None  # type: ignore[assignment]
    _SOX_AVAILABLE = False

from app.models.whisper_model import transcribe_audio

LOGGER = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
DEFAULT_SOX_EFFECTS: List[List[str]] = [
    ["gain", "-n"],
    ["highpass", "200"],
    ["lowpass", "3800"],
    ["compand", "0.02,0.20", "6:-70,-60,-20", "-5", "-90", "0.2"],
    ["norm"],
]


class SpeechToTextService:
    """Encapsulates audio pre-processing and Whisper inference."""

    def __init__(
        self,
        model_size: str = "medium.en",
        device: str | None = None,
        sox_effects: List[List[str]] | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.sox_effects = sox_effects or DEFAULT_SOX_EFFECTS

    def _preprocess_audio(self, audio_path: str) -> Path:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if _SOX_AVAILABLE and apply_effects_file is not None:
            LOGGER.info("Cleaning audio signal with SoX effects: %s", self.sox_effects)
            try:
                waveform, sample_rate = apply_effects_file(str(audio_file), effects=self.sox_effects)
            except RuntimeError as exc:
                LOGGER.warning("SoX effects unavailable at runtime (%s); falling back.", exc)
                waveform, sample_rate = self._load_audio(str(audio_file))
                waveform = self._apply_fallback_filters(waveform, sample_rate)
        else:
            LOGGER.info("SoX effects not available; using portable filters instead.")
            waveform, sample_rate = self._load_audio(str(audio_file))
            waveform = self._apply_fallback_filters(waveform, sample_rate)

        if sample_rate != TARGET_SAMPLE_RATE:
            LOGGER.debug("Resampling audio from %s Hz to %s Hz", sample_rate, TARGET_SAMPLE_RATE)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = TARGET_SAMPLE_RATE

        temp_dir = Path(tempfile.mkdtemp(prefix="stt_clean_"))
        cleaned_path = temp_dir / f"{audio_file.stem}_clean.wav"
        self._save_audio(cleaned_path, waveform, sample_rate)
        return cleaned_path

    def _load_audio(self, path: str) -> tuple[torch.Tensor, int]:
        data, sample_rate = sf.read(path, always_2d=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(np.transpose(data))
        return waveform.float(), int(sample_rate)

    def _save_audio(self, path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
        array = waveform.cpu().numpy()
        if waveform.shape[0] > 1:
            array = np.transpose(array)
        else:
            array = array.squeeze(0)
        sf.write(str(path), array, sample_rate)

    def _apply_fallback_filters(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Approximate the SoX chain using portable torchaudio operations."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        try:
            waveform = AF.highpass_biquad(waveform, sample_rate, cutoff_freq=200.0)
            waveform = AF.lowpass_biquad(waveform, sample_rate, cutoff_freq=3800.0)
        except RuntimeError as exc:  # pragma: no cover - guard for unsupported ops
            LOGGER.warning("Biquad filter failed, keeping original waveform: %s", exc)
        rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
        waveform = waveform / max(rms, torch.tensor(1e-4, device=waveform.device))
        return waveform.clamp(min=-1.0, max=1.0)

    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        """Run Whisper transcription on a cleaned audio signal."""
        cleaned_path = self._preprocess_audio(audio_path)
        try:
            result = transcribe_audio(
                str(cleaned_path),
                model_size=self.model_size,
                device=self.device,
                **kwargs,
            )
        finally:
            try:
                cleaned_path.unlink(missing_ok=True)
            finally:
                shutil.rmtree(cleaned_path.parent, ignore_errors=True)
        return result


def _auto_device() -> str | None:
    env_device = os.getenv("WHISPER_DEVICE")
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    return None


stt_service = SpeechToTextService(
    model_size=os.getenv("WHISPER_MODEL_SIZE", "medium.en"),
    device=_auto_device(),
)
