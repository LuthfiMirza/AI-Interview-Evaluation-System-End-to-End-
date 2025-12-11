"""Utility helpers for NLP-based scoring and summarization."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from transformers import pipeline

from app.models.nlp_model import NLPScoringModel

LOGGER = logging.getLogger(__name__)


def generate_summary(transcript: str, max_sentences: int = 3, model_name: str = "sshleifer/distilbart-cnn-12-6") -> str:
    """Summarize the transcript using a HuggingFace abstractive model."""
    LOGGER.info("Generating summary with model: %s", model_name)
    summarizer = pipeline("summarization", model=model_name)
    result = summarizer(transcript, max_length=130, min_length=30, do_sample=False)
    summary_text = result[0]["summary_text"]
    sentences = summary_text.split(". ")
    clipped = ". ".join(sentences[:max_sentences]).strip()
    return clipped.rstrip(".") + "."


def score_transcript(
    transcript: str,
    expected_answer: str,
    scoring_model: Optional[NLPScoringModel] = None,
    include_summary: bool = False,
    summarizer_model: str = "facebook/bart-base",
) -> Dict[str, Any]:
    """Compute NLP scores and optional summary for a transcript."""
    scoring_model = scoring_model or NLPScoringModel()
    scores = scoring_model.score(candidate_text=transcript, reference_text=expected_answer)

    summary = ""
    if include_summary:
        try:
            summary = generate_summary(transcript, model_name=summarizer_model)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to generate summary: %s", exc)
            summary = ""

    scores["summary"] = summary
    return scores
