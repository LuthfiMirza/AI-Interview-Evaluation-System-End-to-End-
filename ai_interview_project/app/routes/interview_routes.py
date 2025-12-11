"""API routes handling interview video ingestion and evaluation."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.db import session_scope
from app.models.db_models import InterviewRecord, NLPScoreRecord, TranscriptRecord
from app.utils.audio_utils import extract_audio
from app.utils.nlp_utils import score_transcript
from app.utils.report_utils import aggregate_results
from app.services import stt_service

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/interviews", tags=["interviews"])

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RESULT_STORE: Dict[str, Dict] = {}


def _persist_processing_record(interview_id: str, candidate_identifier: str, language: str = "en") -> None:
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if interview is None:
            interview = InterviewRecord(id=interview_id, candidateId=candidate_identifier, language=language)
            session.add(interview)
        interview.candidateId = candidate_identifier
        interview.language = language
        interview.status = "processing"
        interview.summary = None
        interview.verbalScore = None
        interview.nonVerbalScore = None
        interview.finalScore = None
        interview.cheatingScore = None
        interview.confidence = None


def _persist_success(
    interview_id: str,
    candidate_identifier: str,
    report: Dict[str, float | str | Dict],
    stt_result: Dict,
    nlp_result: Dict,
) -> None:
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if interview is None:
            interview = InterviewRecord(id=interview_id, candidateId=candidate_identifier, language="en")
            session.add(interview)

        interview.candidateId = candidate_identifier
        interview.status = "completed"
        interview.verbalScore = float(report.get("verbal_score") or 0.0)
        interview.nonVerbalScore = None
        interview.finalScore = float(report.get("final_score") or 0.0)
        interview.cheatingScore = None
        interview.confidence = float(report.get("confidence") or 0.0)
        interview.summary = report.get("summary") or ""

        transcript_text = stt_result.get("text", "")
        segments = stt_result.get("segments")

        if interview.transcript:
            interview.transcript.text = transcript_text
            interview.transcript.segments = segments
        else:
            interview.transcript = TranscriptRecord(
                interviewId=interview_id,
                text=transcript_text,
                segments=segments,
            )

        if interview.nlp:
            interview.nlp.fluency = float(nlp_result.get("fluency") or 0.0)
            interview.nlp.relevance = float(nlp_result.get("relevance") or 0.0)
            interview.nlp.overall = float(nlp_result.get("overall_score") or 0.0)
        else:
            interview.nlp = NLPScoreRecord(
                interviewId=interview_id,
                fluency=float(nlp_result.get("fluency") or 0.0),
                relevance=float(nlp_result.get("relevance") or 0.0),
                overall=float(nlp_result.get("overall_score") or 0.0),
            )


def _persist_failure(interview_id: str, candidate_identifier: Optional[str], error_message: str) -> None:
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if interview is None:
            interview = InterviewRecord(id=interview_id, candidateId=candidate_identifier or interview_id, language="en")
            session.add(interview)
        interview.candidateId = candidate_identifier or interview.candidateId
        interview.status = "failed"
        interview.summary = error_message


def _process_interview(
    interview_id: str,
    candidate_identifier: str,
    video_path: Path,
    expected_answer: str,
) -> None:
    LOGGER.info("Processing interview %s", interview_id)
    audio_path: Path | None = None
    try:
        audio_path = Path(extract_audio(str(video_path)))
        stt_result = stt_service.transcribe(str(audio_path))
        transcript_text = stt_result.get("text", "")
        if not transcript_text.strip():
            raise ValueError("Transkripsi kosong; pastikan video memiliki suara yang jelas.")
        nlp_result = score_transcript(transcript_text, expected_answer or transcript_text)
        report = aggregate_results(stt_result, nlp_result)
        RESULT_STORE[interview_id] = {
            "candidate_id": candidate_identifier,
            "status": "completed",
            "report": report,
        }
        _persist_success(interview_id, candidate_identifier, report, stt_result, nlp_result)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to process interview %s: %s", interview_id, exc)
        RESULT_STORE[interview_id] = {
            "candidate_id": candidate_identifier,
            "status": "failed",
            "error": str(exc),
        }
        _persist_failure(interview_id, candidate_identifier, str(exc))
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
        try:
            video_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to clean up temporary files for %s", interview_id)


@router.post("/upload")
async def upload_interview(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    candidate_id: str | None = Form(None),
    expected_answer: str = Form(""),
) -> Dict[str, str]:
    interview_id = f"INTV-{uuid.uuid4().hex[:8]}"
    candidate_identifier = candidate_id or f"CAND-{uuid.uuid4().hex[:6]}"
    temp_video_path = UPLOAD_DIR / f"{interview_id}_{file.filename}"

    LOGGER.info("Received upload for candidate %s storing at %s", candidate_identifier, temp_video_path)
    with temp_video_path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)
    await file.close()

    RESULT_STORE[interview_id] = {"candidate_id": candidate_identifier, "status": "processing"}
    _persist_processing_record(interview_id, candidate_identifier)

    background_tasks.add_task(_process_interview, interview_id, candidate_identifier, temp_video_path, expected_answer)
    return {"interview_id": interview_id, "status": "processing"}


@router.get("/result/{interview_id}")
async def get_result(interview_id: str) -> Dict:
    result = RESULT_STORE.get(interview_id)

    # If in-memory state is missing or still processing, check the database for fresher status.
    if not result or result.get("status") == "processing":
        db_result = _load_result_from_db(interview_id)
        if db_result:
            RESULT_STORE[interview_id] = db_result  # cache for subsequent calls
            result = db_result

    if not result:
        raise HTTPException(status_code=404, detail="Interview not found")
    return result


def _load_result_from_db(interview_id: str) -> Optional[Dict]:
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if interview is None:
            return None

        response: Dict[str, Dict | str] = {
            "candidate_id": interview.candidateId,
            "status": interview.status,
        }

        if interview.status == "completed":
            transcript = interview.transcript
            report = {
                "verbal_score": interview.verbalScore,
                "confidence": interview.confidence,
                "final_score": interview.finalScore,
                "summary": interview.summary or "",
                "transcript": transcript.text if transcript else "",
            }
            response["report"] = report
        elif interview.status == "failed":
            response["error"] = interview.summary or "Processing failed"

        return response
