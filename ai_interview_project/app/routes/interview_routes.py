"""API routes handling interview video ingestion and evaluation."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.models.whisper_model import transcribe_audio
from app.utils.audio_utils import extract_audio
from app.utils.nlp_utils import score_transcript
from app.utils.report_utils import aggregate_results
from app.utils.vision_utils import analyze_video

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/interviews", tags=["interviews"])

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RESULT_STORE: Dict[str, Dict] = {}


def _process_interview(interview_id: str, video_path: Path, expected_answer: str) -> None:
    LOGGER.info("Processing interview %s", interview_id)
    audio_path: Path | None = None
    try:
        audio_path = Path(extract_audio(str(video_path)))
        stt_result = transcribe_audio(str(audio_path))
        transcript_text = stt_result.get("text", "")
        nlp_result = score_transcript(transcript_text, expected_answer or transcript_text)
        vision_result = analyze_video(str(video_path))
        report = aggregate_results(stt_result, nlp_result, vision_result)
        RESULT_STORE[interview_id] = {
            "candidate_id": interview_id,
            "status": "completed",
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to process interview %s: %s", interview_id, exc)
        RESULT_STORE[interview_id] = {
            "candidate_id": interview_id,
            "status": "failed",
            "error": str(exc),
        }
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

    background_tasks.add_task(_process_interview, interview_id, temp_video_path, expected_answer)
    return {"interview_id": interview_id, "status": "processing"}


@router.get("/result/{interview_id}")
async def get_result(interview_id: str) -> Dict:
    result = RESULT_STORE.get(interview_id)
    if not result:
        raise HTTPException(status_code=404, detail="Interview not found")
    return result
