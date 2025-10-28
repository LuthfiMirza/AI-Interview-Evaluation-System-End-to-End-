"""NLP scoring models built on top of HuggingFace Transformers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, pipeline

LOGGER = logging.getLogger(__name__)


def _mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


@dataclass
class EmbeddingModel:
    """Generate sentence embeddings for semantic similarity scoring."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[int] = None

    def __post_init__(self) -> None:
        LOGGER.info("Loading embedding model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if self.device is not None:
            self.model.to(self.device)

    def embed(self, text: str) -> Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        if self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def similarity(self, reference: str, candidate: str) -> float:
        ref_emb = self.embed(reference)
        cand_emb = self.embed(candidate)
        return torch.nn.functional.cosine_similarity(ref_emb, cand_emb).item()


@dataclass
class FluencyModel:
    """Classifier that approximates fluency and grammar quality."""

    model_name: str = "textattack/bert-base-uncased-CoLA"
    device: Optional[int] = None

    def __post_init__(self) -> None:
        LOGGER.info("Loading fluency model: %s", self.model_name)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device if self.device is not None else -1,
        )

    def score(self, text: str) -> float:
        result = self.pipeline(text, truncation=True)[0]
        return float(result.get("score", 0.0))


class NLPScoringModel:
    """High-level interface to compute NLP scores for transcribed answers."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        fluency_model: Optional[FluencyModel] = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.fluency_model = fluency_model or FluencyModel()

    def relevance(self, candidate_text: str, reference_text: str) -> float:
        return self.embedding_model.similarity(reference_text, candidate_text)

    def fluency(self, text: str) -> float:
        return self.fluency_model.score(text)

    def score(self, candidate_text: str, reference_text: str) -> dict:
        relevance = self.relevance(candidate_text, reference_text)
        fluency = self.fluency(candidate_text)
        overall = (relevance * 0.6) + (fluency * 0.4)
        LOGGER.info(
            "Computed NLP scores -- relevance: %.3f, fluency: %.3f, overall: %.3f",
            relevance,
            fluency,
            overall,
        )
        return {
            "fluency": round(fluency, 3),
            "relevance": round(relevance, 3),
            "overall_score": round(overall, 3),
        }
