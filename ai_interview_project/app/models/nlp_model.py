"""NLP scoring models for interview evaluation with acoustic fluency and concept-based relevance."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)

# Common filler words and verbal tics across Indonesian and English interview contexts
DEFAULT_FILLER_WORDS: Set[str] = {
    # Indonesian fillers
    "ehm", "eh", "uhm", "uh", "um", "ah", "anu", "apa ya", "apaya",
    "kayak", "kayaknya", "gitu", "gitulah", "mungkin", "terus",
    "jadi kayak", "nah", "kan", "dong", "sih",
    # English fillers
    "like", "you know", "basically", "actually", "literally",
    "sort of", "kind of", "i mean", "so yeah", "well",
}


def _mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Perform mean pooling on transformer token embeddings weighted by attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


@dataclass
class EmbeddingModel:
    """Generate sentence embeddings for semantic similarity scoring using Sentence-Transformers."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[int | str] = None

    def __post_init__(self) -> None:
        LOGGER.info("Loading embedding model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if self.device is not None:
            self.model.to(self.device)

    def embed(self, text: str) -> Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def similarity(self, reference: str, candidate: str) -> float:
        if not reference.strip() or not candidate.strip():
            return 0.0
        ref_emb = self.embed(reference)
        cand_emb = self.embed(candidate)
        sim = torch.nn.functional.cosine_similarity(ref_emb, cand_emb).item()
        # Scale negative cosine similarity gracefully to 0.0 - 1.0 range
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))


@dataclass
class AcousticFluencyScorer:
    """
    Evaluates candidate speech fluency using acoustic flow and speech rate metrics
    (Words Per Minute, filler word density, and pause frequency).
    
    This replaces English-only grammar classifiers (like CoLA) with an objective,
    multilingual evaluation of communication pace and hesitation.
    """

    filler_words: Set[str] = field(default_factory=lambda: DEFAULT_FILLER_WORDS)
    ideal_wpm_min: float = 110.0
    ideal_wpm_max: float = 165.0

    def analyze_pace(self, text: str, duration_seconds: float) -> Tuple[float, float, str]:
        """Compute WPM and normalized pace score."""
        words = re.findall(r"\b[\w'-]+\b", text)
        word_count = len(words)
        if duration_seconds <= 0.5:
            # Fallback estimation if duration is unknown: assume standard 130 WPM
            wpm = 130.0
            return 1.0, wpm, "optimal"

        wpm = (word_count / duration_seconds) * 60.0

        if self.ideal_wpm_min <= wpm <= self.ideal_wpm_max:
            pace_score = 1.0
            status = "optimal"
        elif wpm < self.ideal_wpm_min:
            # Below ideal: smoothly scale down
            if wpm >= 70.0:
                pace_score = 0.65 + 0.35 * ((wpm - 70.0) / (self.ideal_wpm_min - 70.0))
            else:
                pace_score = max(0.35, 0.65 * (wpm / 70.0))
            status = "slow"
        else:
            # Above ideal: too fast
            if wpm <= 210.0:
                pace_score = 0.70 + 0.30 * ((210.0 - wpm) / (210.0 - self.ideal_wpm_max))
            else:
                pace_score = max(0.40, 0.70 - (wpm - 210.0) * 0.003)
            status = "fast"

        return round(pace_score, 3), round(wpm, 1), status

    def analyze_fillers(self, text: str) -> Tuple[float, int, List[str], float]:
        """Detect filler words and compute penalty score."""
        words = re.findall(r"\b[\w'-]+\b", text.lower())
        total_words = max(len(words), 1)
        lower_text = f" {text.lower()} "

        found_fillers: List[str] = []
        filler_count = 0

        # Check multi-word and single-word fillers
        for filler in sorted(self.filler_words, key=len, reverse=True):
            pattern = rf"\b{re.escape(filler)}\b"
            matches = re.findall(pattern, lower_text)
            if matches:
                count = len(matches)
                filler_count += count
                found_fillers.extend([filler] * count)
                # Remove matched filler to prevent double counting
                lower_text = re.sub(pattern, " ", lower_text)

        filler_rate = filler_count / total_words

        # Scoring curve: <= 2% is ideal, > 8% drops score
        if filler_rate <= 0.02:
            filler_score = 1.0
        elif filler_rate <= 0.08:
            filler_score = 1.0 - ((filler_rate - 0.02) / 0.06) * 0.35
        else:
            filler_score = max(0.35, 0.65 - ((filler_rate - 0.08) * 2.5))

        return round(filler_score, 3), filler_count, list(set(found_fillers)), round(filler_rate, 4)

    def analyze_pauses(self, segments: Optional[List[Dict[str, Any]]]) -> Tuple[float, int]:
        """Analyze silence gaps between consecutive Whisper transcription segments."""
        if not segments or len(segments) < 2:
            return 1.0, 0

        long_pauses = 0
        for i in range(len(segments) - 1):
            prev_end = float(segments[i].get("end", 0.0))
            next_start = float(segments[i + 1].get("start", prev_end))
            gap = next_start - prev_end
            if gap >= 2.5:
                long_pauses += 1

        if long_pauses == 0:
            pause_score = 1.0
        elif long_pauses <= 2:
            pause_score = 0.85
        elif long_pauses <= 5:
            pause_score = 0.70
        else:
            pause_score = max(0.40, 0.70 - (long_pauses - 5) * 0.05)

        return round(pause_score, 3), long_pauses

    def score(
        self,
        text: str,
        segments: Optional[List[Dict[str, Any]]] = None,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute composite fluency score based on acoustic flow metrics."""
        if not text.strip():
            return {
                "score": 0.0,
                "wpm": 0.0,
                "wpm_status": "empty",
                "filler_count": 0,
                "filler_words": [],
                "filler_rate": 0.0,
                "long_pauses": 0,
            }

        # Determine speech duration from segments if not provided
        if duration_seconds is None and segments:
            try:
                start_time = float(segments[0].get("start", 0.0))
                end_time = float(segments[-1].get("end", 0.0))
                duration_seconds = max(0.0, end_time - start_time)
            except (IndexError, TypeError, ValueError):
                duration_seconds = 0.0

        duration_seconds = duration_seconds or 0.0

        pace_score, wpm, wpm_status = self.analyze_pace(text, duration_seconds)
        filler_score, filler_count, filler_words, filler_rate = self.analyze_fillers(text)
        pause_score, long_pauses = self.analyze_pauses(segments)

        composite_fluency = (0.45 * pace_score) + (0.35 * filler_score) + (0.20 * pause_score)
        composite_fluency = max(0.0, min(1.0, composite_fluency))

        return {
            "score": round(composite_fluency, 3),
            "wpm": wpm,
            "wpm_status": wpm_status,
            "filler_count": filler_count,
            "filler_words": filler_words,
            "filler_rate": filler_rate,
            "long_pauses": long_pauses,
        }


@dataclass
class FluencyModel:
    """Backwards-compatible wrapper that aliases AcousticFluencyScorer."""

    scorer: AcousticFluencyScorer = field(default_factory=AcousticFluencyScorer)

    def score(self, text: str) -> float:
        return self.scorer.score(text)["score"]


class NLPScoringModel:
    """
    High-level interface to compute NLP scores for transcribed interview responses.
    Evaluates both concept coverage / semantic relevance and acoustic fluency.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        fluency_scorer: Optional[AcousticFluencyScorer] = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.fluency_scorer = fluency_scorer or AcousticFluencyScorer()

    def relevance(
        self,
        candidate_text: str,
        reference_text: str,
        key_concepts: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate relevance based on key concept checklist coverage and semantic similarity.
        """
        if not candidate_text.strip():
            return 0.0, {
                "matched_concepts": [],
                "missing_concepts": key_concepts or [],
                "concept_coverage": 0.0,
                "semantic_similarity": 0.0,
            }

        # 1. Semantic Embedding Similarity
        semantic_sim = self.embedding_model.similarity(reference_text, candidate_text)

        # 2. Key Concept Checklist Coverage
        matched_concepts: List[str] = []
        missing_concepts: List[str] = []
        candidate_lower = candidate_text.lower()

        if key_concepts:
            for concept in key_concepts:
                # Break concept into constituent terms or regex match
                terms = [t.strip().lower() for t in re.split(r"[/,]", concept) if t.strip()]
                concept_matched = False
                for term in terms:
                    # Match exact term with punctuation flexibility (e.g. next.js, react.js, c++)
                    escaped_term = re.escape(term)
                    # Also support variant without dot (e.g. nextjs for next.js)
                    alt_term = re.escape(term.replace(".", ""))
                    pattern = rf"(\b{escaped_term}\b|\b{alt_term}\b)"
                    if re.search(pattern, candidate_lower):
                        concept_matched = True
                        break

                if concept_matched:
                    matched_concepts.append(concept)
                else:
                    missing_concepts.append(concept)

            concept_coverage = len(matched_concepts) / len(key_concepts)
            overall_relevance = (0.55 * concept_coverage) + (0.45 * semantic_sim)
        else:
            concept_coverage = 1.0 if semantic_sim >= 0.70 else semantic_sim
            overall_relevance = semantic_sim

        overall_relevance = max(0.0, min(1.0, overall_relevance))
        details = {
            "matched_concepts": matched_concepts,
            "missing_concepts": missing_concepts,
            "concept_coverage": round(concept_coverage, 3),
            "semantic_similarity": round(semantic_sim, 3),
        }
        return round(overall_relevance, 3), details

    def fluency(
        self,
        text: str,
        segments: Optional[List[Dict[str, Any]]] = None,
        duration_seconds: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        fluency_info = self.fluency_scorer.score(text, segments=segments, duration_seconds=duration_seconds)
        return fluency_info["score"], fluency_info

    def score(
        self,
        candidate_text: str,
        reference_text: str,
        key_concepts: Optional[List[str]] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute composite scores and detailed breakdown for transcript."""
        relevance_score, relevance_details = self.relevance(
            candidate_text=candidate_text,
            reference_text=reference_text,
            key_concepts=key_concepts,
        )
        fluency_score, fluency_details = self.fluency(
            text=candidate_text,
            segments=segments,
            duration_seconds=duration_seconds,
        )

        overall = (relevance_score * 0.6) + (fluency_score * 0.4)
        LOGGER.info(
            "Computed NLP scores -- relevance: %.3f, fluency: %.3f, overall: %.3f",
            relevance_score,
            fluency_score,
            overall,
        )

        return {
            "fluency": round(fluency_score, 3),
            "relevance": round(relevance_score, 3),
            "overall_score": round(overall, 3),
            "details": {
                "wpm": fluency_details.get("wpm", 0.0),
                "wpm_status": fluency_details.get("wpm_status", "unknown"),
                "filler_count": fluency_details.get("filler_count", 0),
                "filler_words_found": fluency_details.get("filler_words", []),
                "filler_rate_pct": round(fluency_details.get("filler_rate", 0.0) * 100, 1),
                "long_pauses": fluency_details.get("long_pauses", 0),
                "matched_concepts": relevance_details.get("matched_concepts", []),
                "missing_concepts": relevance_details.get("missing_concepts", []),
                "concept_coverage_pct": round(relevance_details.get("concept_coverage", 0.0) * 100, 1),
                "semantic_similarity": relevance_details.get("semantic_similarity", 0.0),
            },
        }
