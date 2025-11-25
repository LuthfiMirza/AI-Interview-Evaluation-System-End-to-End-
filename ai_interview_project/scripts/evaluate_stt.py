"""Evaluate STT accuracy on a labeled dataset to ensure ≥90% accuracy."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Tuple

from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer

from app.services.stt_service import SpeechToTextService

NORMALIZE = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ],
)


def _iter_dataset(dataset_dir: Path, limit: int | None = None) -> Iterable[Tuple[Path, str]]:
    audio_exts = {".wav", ".mp3", ".m4a"}
    count = 0
    for audio_path in sorted(dataset_dir.iterdir()):
        if audio_path.suffix.lower() not in audio_exts:
            continue
        transcript_path = audio_path.with_suffix(".txt")
        if not transcript_path.exists():
            continue
        reference = transcript_path.read_text(encoding="utf-8").strip()
        if not reference:
            continue
        yield audio_path, reference
        count += 1
        if limit is not None and count >= limit:
            break


def evaluate(dataset_dir: Path, model_size: str, device: str | None, limit: int | None) -> dict:
    service = SpeechToTextService(model_size=model_size, device=device)
    total_words = 0
    total_errors = 0
    sample_scores: List[float] = []
    samples: List[dict] = []

    for audio_path, reference in _iter_dataset(dataset_dir, limit):
        result = service.transcribe(str(audio_path))
        hypothesis = result.get("text", "").strip()

        reference_norm = NORMALIZE(reference)
        hypothesis_norm = NORMALIZE(hypothesis)

        sample_wer = wer(reference_norm, hypothesis_norm)
        words = max(len(reference_norm.split()), 1)
        errors = int(round(sample_wer * words))
        total_words += words
        total_errors += errors
        sample_scores.append(1.0 - sample_wer)
        samples.append(
            {
                "file": str(audio_path.name),
                "reference": reference,
                "transcript": hypothesis,
                "confidence": result.get("confidence"),
                "accuracy": round(1.0 - sample_wer, 4),
            }
        )

    overall_accuracy = 1.0 if total_words == 0 else 1.0 - (total_errors / total_words)
    return {
        "dataset": str(dataset_dir),
        "samples_evaluated": len(samples),
        "overall_accuracy": round(overall_accuracy, 4),
        "median_accuracy": round(statistics.median(sample_scores), 4) if samples else 0.0,
        "min_accuracy": round(min(sample_scores), 4) if samples else 0.0,
        "max_accuracy": round(max(sample_scores), 4) if samples else 0.0,
        "details": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Whisper STT pipeline on a labeled dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing <sample>.wav audio files with matching <sample>.txt transcripts.",
    )
    parser.add_argument("--model-size", default="medium.en", help="Whisper checkpoint to use.")
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cuda'.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N files (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args.dataset_dir, args.model_size, args.device, args.limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    accuracy = report["overall_accuracy"]
    if accuracy < 0.9:
        raise SystemExit(
            f"Accuracy threshold of 0.90 not met. Current accuracy: {accuracy:.3f}. "
            "Improve dataset quality or adjust the model configuration.",
        )


if __name__ == "__main__":
    main()
