"""Aggregation utilities to combine NLP, vision, and STT results."""

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
    vision_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine subsystem outputs into a unified interview report."""
    verbal_score = float(nlp_result.get("overall_score", 0.0))
    non_verbal_score = 1.0 - float(vision_result.get("cheating_score", 0.0))
    confidence = float(stt_result.get("confidence") or _extract_confidence(stt_result))
    final_score = weighted_average(
        verbal_score,
        non_verbal_score,
        weights=(0.65, 0.35),
    )

    report = {
        "verbal_score": round(verbal_score, 3),
        "non_verbal_score": round(non_verbal_score, 3),
        "confidence": round(confidence, 3),
        "final_score": round(final_score, 3),
        "summary": nlp_result.get("summary", ""),
        "vision_metrics": vision_result,
        "transcript": stt_result.get("text", ""),
    }
    LOGGER.info("Aggregated interview report generated.")
    return report
