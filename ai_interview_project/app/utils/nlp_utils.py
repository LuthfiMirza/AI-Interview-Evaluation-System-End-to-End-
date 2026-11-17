"""Utility helpers for NLP-based scoring, concept coverage, and summarization."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import pipeline

from app.models.nlp_model import NLPScoringModel

LOGGER = logging.getLogger(__name__)


def generate_summary(
    transcript: str,
    max_sentences: int = 3,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
) -> str:
    """Summarize the transcript using a HuggingFace abstractive model."""
    if not transcript or len(transcript.strip().split()) < 15:
        return transcript.strip()

    LOGGER.info("Generating summary with model: %s", model_name)
    try:
        summarizer = pipeline("summarization", model=model_name)
        result = summarizer(transcript, max_length=130, min_length=25, do_sample=False)
        summary_text = result[0]["summary_text"]
        sentences = summary_text.split(". ")
        clipped = ". ".join(sentences[:max_sentences]).strip()
        return clipped.rstrip(".") + "."
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Summarization fallback due to exception: %s", exc)
        # Fallback to simple first sentence extraction
        first_sentence = transcript.strip().split(".")[0]
        return (first_sentence + ".").strip()


def score_transcript(
    transcript: str,
    expected_answer: str,
    key_concepts: Optional[List[str]] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    duration_seconds: Optional[float] = None,
    scoring_model: Optional[NLPScoringModel] = None,
    include_summary: bool = False,
    summarizer_model: str = "facebook/bart-base",
) -> Dict[str, Any]:
    """
    Compute NLP scores, acoustic fluency breakdown, concept coverage, and optional summary.
    """
    scoring_model = scoring_model or NLPScoringModel()
    scores = scoring_model.score(
        candidate_text=transcript,
        reference_text=expected_answer,
        key_concepts=key_concepts,
        segments=segments,
        duration_seconds=duration_seconds,
    )

    summary = ""
    if include_summary and transcript.strip():
        try:
            summary = generate_summary(transcript, model_name=summarizer_model)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to generate summary: %s", exc)
            summary = ""

    scores["summary"] = summary
    return scores
