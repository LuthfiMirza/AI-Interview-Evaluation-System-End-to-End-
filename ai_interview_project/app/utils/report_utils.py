"""Aggregation utilities to combine STT + NLP results with acoustic & concept details."""

from __future__ import annotations

import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def weighted_average(*values: float, weights: tuple[float, ...]) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("Values and weights must be the same length and non-empty.")
    numerator = sum(value * weight for value, weight in zip(values, weights))
    denominator = sum(weights)
    return numerator / denominator if denominator else 0.0


def _extract_confidence(stt_result: Dict[str, Any]) -> float:
    segments = stt_result.get("segments") or []
    if not segments:
        return 0.0
    last_segment = segments[-1]
    if isinstance(last_segment, dict):
        return float(last_segment.get("avg_logprob", 0.0))
    return 0.0


def aggregate_results(
    stt_result: Dict[str, Any],
    nlp_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine STT + NLP outputs into a unified interview report."""
    verbal_score = float(nlp_result.get("overall_score", 0.0))
    fluency_score = float(nlp_result.get("fluency", verbal_score))
    relevance_score = float(nlp_result.get("relevance", verbal_score))
    confidence = float(stt_result.get("confidence") or _extract_confidence(stt_result))
    final_score = verbal_score  # with no vision features, use verbal score as the final score

    report = {
        "verbal_score": round(verbal_score, 3),
        "fluency_score": round(fluency_score, 3),
        "relevance_score": round(relevance_score, 3),
        "confidence": round(confidence, 3),
        "final_score": round(final_score, 3),
        "summary": nlp_result.get("summary", ""),
        "transcript": stt_result.get("text", ""),
        "details": nlp_result.get("details", {}),
    }
    LOGGER.info("Aggregated interview report generated.")
    return report
