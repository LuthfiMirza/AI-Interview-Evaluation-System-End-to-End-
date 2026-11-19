"""API routes handling interview video ingestion, evaluation, and HR candidate management."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import or_, select

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


class HRMetadataUpdate(BaseModel):
    stage: Optional[str] = Field(None, description="Candidate ATS stage: INBOX, SHORTLISTED, TECHNICAL_TEST, USER_INTERVIEW, OFFERED, REJECTED")
    isStarred: Optional[bool] = Field(None, description="Toggle favorite / star status")
    recruiterNotes: Optional[str] = Field(None, description="Internal recruiter notes")
    roleLevel: Optional[str] = Field(None, description="JUNIOR, MID, SENIOR")


class BulkActionRequest(BaseModel):
    interview_ids: List[str]
    action: str = Field(..., description="Action: set_stage, toggle_star, delete")
    stage: Optional[str] = None
    is_starred: Optional[bool] = None


def _generate_insights(report: Dict[str, Any], nlp_result: Dict[str, Any]) -> tuple[list[str], list[str]]:
    details = nlp_result.get("details", {})
    matched = details.get("matched_concepts", [])
    missing = details.get("missing_concepts", [])
    wpm = details.get("wpm", 0.0)
    wpm_status = details.get("wpm_status", "unknown")
    filler_rate = details.get("filler_rate_pct", 0.0)

    strengths: list[str] = []
    if matched:
        strengths.append(f"Menguasai konsep teknis: {', '.join(matched[:3])}")
    if wpm_status == "optimal":
        strengths.append(f"Tempo bicara stabil & artikulatif ({wpm:.0f} WPM, zona optimal)")
    elif 100 <= wpm <= 170:
        strengths.append(f"Penyampaian cukup lancar ({wpm:.0f} WPM)")
    if filler_rate <= 2.0:
        strengths.append(f"Komunikasi lugas dengan rasio kata jeda sangat rendah ({filler_rate}%)")
    elif filler_rate <= 3.5:
        strengths.append("Artikulasi relatif baik dan minim keraguan vokal")

    if not strengths:
        strengths.append("Menyelesaikan seluruh pertanyaan wawancara dengan suara terdengar jelas")

    growth: list[str] = []
    if missing:
        growth.append(f"Perlu klarifikasi pemahaman konsep: {', '.join(missing[:3])}")
    if wpm < 105 and wpm > 0:
        growth.append(f"Tempo bicara cenderung lambat ({wpm:.0f} WPM), perlu dorongan agar lebih percaya diri")
    elif wpm > 175:
        growth.append(f"Tempo bicara tergesa-gesa ({wpm:.0f} WPM), disarankan lebih terstruktur")
    if filler_rate > 3.0:
        growth.append(f"Frekuensi kata pengisi ('ehm', 'anu', dsb) mencapai {filler_rate}%, perlu latihan bicara lugas")

    if not growth:
        growth.append("Pertahankan penguasaan teknis dan konsistensi tempo bicara pada interview lanjutan")

    return strengths[:3], growth[:2]


def _persist_processing_record(
    interview_id: str,
    candidate_identifier: str,
    language: str = "en",
    role_level: str = "MID",
) -> None:
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if interview is None:
            interview = InterviewRecord(
                id=interview_id,
                candidateId=candidate_identifier,
                language=language,
                roleLevel=role_level,
                stage="INBOX",
                isStarred=False,
            )
            session.add(interview)
        interview.candidateId = candidate_identifier
        interview.language = language
        interview.roleLevel = role_level
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
    role_level: str = "MID",
) -> None:
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if interview is None:
            interview = InterviewRecord(
                id=interview_id,
                candidateId=candidate_identifier,
                language="en",
                roleLevel=role_level,
                stage="INBOX",
                isStarred=False,
            )
            session.add(interview)

        interview.candidateId = candidate_identifier
        interview.roleLevel = role_level
        interview.status = "completed"
        interview.verbalScore = float(report.get("verbal_score") or 0.0)
        interview.nonVerbalScore = None
        interview.finalScore = float(report.get("final_score") or 0.0)
        interview.cheatingScore = None
        interview.confidence = float(report.get("confidence") or 0.0)
        interview.summary = report.get("summary") or ""

        # Auto-generate strengths & growth areas
        strengths, growth = _generate_insights(report, nlp_result)
        interview.keyStrengths = {"items": strengths}
        interview.growthAreas = {"items": growth}

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
    role_level: str = "MID",
) -> None:
    LOGGER.info("Processing interview %s", interview_id)
    audio_path: Path | None = None
    try:
        audio_path = Path(extract_audio(str(video_path)))
        stt_result = stt_service.transcribe(str(audio_path))
        transcript_text = stt_result.get("text", "")
        segments = stt_result.get("segments") or []
        if not transcript_text.strip():
            raise ValueError("Transkripsi kosong; pastikan video memiliki suara yang jelas.")
        nlp_result = score_transcript(
            transcript=transcript_text,
            expected_answer=expected_answer or transcript_text,
            segments=segments,
        )
        report = aggregate_results(stt_result, nlp_result)
        RESULT_STORE[interview_id] = {
            "candidate_id": candidate_identifier,
            "status": "completed",
            "report": report,
        }
        _persist_success(interview_id, candidate_identifier, report, stt_result, nlp_result, role_level=role_level)
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
    role_level: str = Form("MID"),
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
    _persist_processing_record(interview_id, candidate_identifier, role_level=role_level)

    background_tasks.add_task(
        _process_interview,
        interview_id,
        candidate_identifier,
        temp_video_path,
        expected_answer,
        role_level,
    )
    return {"interview_id": interview_id, "status": "processing"}


@router.get("")
async def list_interviews(
    stage: Optional[str] = Query(None),
    role_level: Optional[str] = Query(None),
    is_starred: Optional[bool] = Query(None),
    q: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List interviews with optional filter by stage, level, star, or text search."""
    with session_scope() as session:
        query = select(InterviewRecord).order_by(InterviewRecord.updatedAt.desc())
        if stage and stage.upper() != "ALL":
            query = query.where(InterviewRecord.stage == stage.upper())
        if role_level and role_level.upper() != "ALL":
            query = query.where(InterviewRecord.roleLevel == role_level.upper())
        if is_starred is not None:
            query = query.where(InterviewRecord.isStarred == is_starred)

        interviews = session.scalars(query).all()
        results = []

        q_lower = q.strip().lower() if q else None
        for item in interviews:
            transcript = item.transcript
            transcript_text = transcript.text if transcript else ""

            # If text search query is given, filter candidateId, summary, or transcript
            if q_lower:
                matches_id = q_lower in item.candidateId.lower()
                matches_summary = q_lower in (item.summary or "").lower()
                matches_transcript = q_lower in transcript_text.lower()
                if not (matches_id or matches_summary or matches_transcript):
                    continue

            nlp = item.nlp
            key_strengths = item.keyStrengths.get("items", []) if isinstance(item.keyStrengths, dict) else []
            growth_areas = item.growthAreas.get("items", []) if isinstance(item.growthAreas, dict) else []

            results.append({
                "id": item.id,
                "candidateId": item.candidateId,
                "status": item.status,
                "stage": item.stage,
                "isStarred": item.isStarred,
                "roleLevel": item.roleLevel,
                "recruiterNotes": item.recruiterNotes,
                "finalScore": item.finalScore,
                "verbalScore": item.verbalScore,
                "confidence": item.confidence,
                "summary": item.summary,
                "keyStrengths": key_strengths,
                "growthAreas": growth_areas,
                "transcript": transcript_text,
                "fluencyScore": nlp.fluency if nlp else None,
                "relevanceScore": nlp.relevance if nlp else None,
                "updatedAt": item.updatedAt.isoformat() if item.updatedAt else None,
                "createdAt": item.createdAt.isoformat() if item.createdAt else None,
            })
        return results


@router.patch("/{interview_id}/hr-metadata")
async def update_hr_metadata(interview_id: str, payload: HRMetadataUpdate) -> Dict[str, Any]:
    """Update candidate HR metadata: pipeline stage, star flag, notes, or role level."""
    with session_scope() as session:
        interview = session.get(InterviewRecord, interview_id)
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")

        if payload.stage is not None:
            interview.stage = payload.stage.upper()
        if payload.isStarred is not None:
            interview.isStarred = payload.isStarred
        if payload.recruiterNotes is not None:
            interview.recruiterNotes = payload.recruiterNotes
        if payload.roleLevel is not None:
            interview.roleLevel = payload.roleLevel.upper()

        session.flush()
        return {
            "status": "success",
            "interview_id": interview.id,
            "stage": interview.stage,
            "isStarred": interview.isStarred,
            "recruiterNotes": interview.recruiterNotes,
            "roleLevel": interview.roleLevel,
        }


@router.post("/bulk")
async def bulk_action(payload: BulkActionRequest) -> Dict[str, Any]:
    """Perform bulk operations across multiple candidate interviews."""
    with session_scope() as session:
        updated_count = 0
        for i_id in payload.interview_ids:
            interview = session.get(InterviewRecord, i_id)
            if not interview:
                continue

            if payload.action == "set_stage" and payload.stage:
                interview.stage = payload.stage.upper()
                updated_count += 1
            elif payload.action == "toggle_star":
                interview.isStarred = not interview.isStarred
                updated_count += 1
            elif payload.action == "set_star" and payload.is_starred is not None:
                interview.isStarred = payload.is_starred
                updated_count += 1
            elif payload.action == "delete":
                session.delete(interview)
                updated_count += 1

        session.flush()
        return {
            "status": "success",
            "action": payload.action,
            "affected_count": updated_count,
        }


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
            "stage": interview.stage,
            "is_starred": interview.isStarred,
            "role_level": interview.roleLevel,
            "recruiter_notes": interview.recruiterNotes,
        }

        if interview.status == "completed":
            transcript = interview.transcript
            nlp = interview.nlp
            key_strengths = interview.keyStrengths.get("items", []) if isinstance(interview.keyStrengths, dict) else []
            growth_areas = interview.growthAreas.get("items", []) if isinstance(interview.growthAreas, dict) else []
            report = {
                "verbal_score": interview.verbalScore,
                "fluency_score": nlp.fluency if nlp else interview.verbalScore,
                "relevance_score": nlp.relevance if nlp else interview.verbalScore,
                "confidence": interview.confidence,
                "final_score": interview.finalScore,
                "summary": interview.summary or "",
                "transcript": transcript.text if transcript else "",
                "key_strengths": key_strengths,
                "growth_areas": growth_areas,
            }
            response["report"] = report
        elif interview.status == "failed":
            response["error"] = interview.summary or "Processing failed"

        return response

