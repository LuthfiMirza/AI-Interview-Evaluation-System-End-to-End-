"""SQLAlchemy models mirroring the Prisma schema."""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db import Base


class InterviewRecord(Base):
    __tablename__ = "Interview"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    candidateId: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str] = mapped_column(String, default="en", nullable=False)
    status: Mapped[str] = mapped_column(String, default="processing", nullable=False)
    createdAt: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updatedAt: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    verbalScore: Mapped[Optional[float]] = mapped_column(Float)
    nonVerbalScore: Mapped[Optional[float]] = mapped_column(Float)
    cheatingScore: Mapped[Optional[float]] = mapped_column(Float)
    finalScore: Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    summary: Mapped[Optional[str]] = mapped_column(Text)

    transcript: Mapped["TranscriptRecord"] = relationship(
        "TranscriptRecord",
        back_populates="interview",
        uselist=False,
        cascade="all, delete-orphan",
    )
    nlp: Mapped["NLPScoreRecord"] = relationship(
        "NLPScoreRecord",
        back_populates="interview",
        uselist=False,
        cascade="all, delete-orphan",
    )
    vision: Mapped["VisionMetricsRecord"] = relationship(
        "VisionMetricsRecord",
        back_populates="interview",
        uselist=False,
        cascade="all, delete-orphan",
    )


class TranscriptRecord(Base):
    __tablename__ = "Transcript"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: uuid.uuid4().hex,
    )
    interviewId: Mapped[str] = mapped_column(
        String,
        ForeignKey("Interview.id"),
        unique=True,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    segments: Mapped[Optional[dict]] = mapped_column(JSON)
    createdAt: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    interview: Mapped[InterviewRecord] = relationship("InterviewRecord", back_populates="transcript")


class NLPScoreRecord(Base):
    __tablename__ = "NLPScore"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: uuid.uuid4().hex,
    )
    interviewId: Mapped[str] = mapped_column(
        String,
        ForeignKey("Interview.id"),
        unique=True,
        nullable=False,
    )
    fluency: Mapped[float] = mapped_column(Float, nullable=False)
    relevance: Mapped[float] = mapped_column(Float, nullable=False)
    overall: Mapped[float] = mapped_column(Float, nullable=False)
    createdAt: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    interview: Mapped[InterviewRecord] = relationship("InterviewRecord", back_populates="nlp")


class VisionMetricsRecord(Base):
    __tablename__ = "VisionMetrics"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: uuid.uuid4().hex,
    )
    interviewId: Mapped[str] = mapped_column(
        String,
        ForeignKey("Interview.id"),
        unique=True,
        nullable=False,
    )
    eyeContactRatio: Mapped[float] = mapped_column(Float, nullable=False)
    phoneDetected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    multiPerson: Mapped[bool] = mapped_column(Boolean, nullable=False)
    cheatingScore: Mapped[float] = mapped_column(Float, nullable=False)
    createdAt: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    interview: Mapped[InterviewRecord] = relationship("InterviewRecord", back_populates="vision")
