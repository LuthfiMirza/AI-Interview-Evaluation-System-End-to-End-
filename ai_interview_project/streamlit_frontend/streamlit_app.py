"""Streamlit prototype dashboard for uploading interviews and viewing reports."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import httpx
import streamlit as st

# Ensure project root is on sys.path so `app` package is importable when running from subdir.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.models.nlp_model import NLPScoringModel  # noqa: E402
from app.models.whisper_model import get_transcriber  # noqa: E402
from app.utils.audio_utils import extract_audio  # noqa: E402
from app.utils.nlp_utils import score_transcript  # noqa: E402
from app.utils.report_utils import aggregate_results  # noqa: E402

MAX_UPLOAD_MB = int(os.getenv("STREAMLIT_MAX_UPLOAD_MB", "200"))

# Simple role/level templates for expected answers & pass thresholds
ROLE_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Frontend Engineer": {
        "junior": {
            "expected_answer": (
                "Jelaskan pengalaman membangun UI dengan React/Next, manajemen state (hooks/RTK), "
                "testing dasar, dan perhatian ke aksesibilitas serta performa render."
            ),
            "threshold": 0.78,
        },
        "mid": {
            "expected_answer": (
                "Bahas arsitektur frontend (componentization, state global), optimasi (memo, lazy load), "
                "testing (RTL/Jest), aksesibilitas, dan kolaborasi dengan backend/desain."
            ),
            "threshold": 0.82,
        },
        "senior": {
            "expected_answer": (
                "Strategi skalabilitas frontend, design system, performance profiling, testing strategy, "
                "observability (logging/metrics), dan pengalaman memimpin delivery."
            ),
            "threshold": 0.86,
        },
    },
    "Backend Engineer": {
        "junior": {
            "expected_answer": (
                "Bangun API REST dasar, ORM/SQL, auth sederhana, logging, dan penanganan error."
            ),
            "threshold": 0.76,
        },
        "mid": {
            "expected_answer": (
                "Desain API, transaksi database, caching, background jobs, testing, dan monitoring."
            ),
            "threshold": 0.8,
        },
        "senior": {
            "expected_answer": (
                "Arsitektur layanan, skalabilitas, observability, security (authz/authn), "
                "reliability, dan review desain."
            ),
            "threshold": 0.85,
        },
    },
    "Android Engineer": {
        "junior": {
            "expected_answer": (
                "Pengalaman dengan Kotlin/Java, activity/fragment, layout, basic MVVM, "
                "networking sederhana, dan testing dasar."
            ),
            "threshold": 0.75,
        },
        "mid": {
            "expected_answer": (
                "Arsitektur (MVVM/Clean), DI (Hilt/Koin), coroutines/Flow, offline first, "
                "modularisasi, testing (unit/instrumentation)."
            ),
            "threshold": 0.8,
        },
        "senior": {
            "expected_answer": (
                "Arsitektur skala besar, performance, release management, observability, "
                "security/privacy, dan kolaborasi lintas tim."
            ),
            "threshold": 0.85,
        },
    },
    "Product Marketing": {
        "junior": {
            "expected_answer": (
                "Riset persona dasar, channel awareness, konten, dan contoh eksekusi kampanye sederhana."
            ),
            "threshold": 0.72,
        },
        "mid": {
            "expected_answer": (
                "Strategi go-to-market, messaging, channel mix, A/B testing, dan pembacaan metrik (CAC/LTV)."
            ),
            "threshold": 0.78,
        },
        "senior": {
            "expected_answer": (
                "Perencanaan GTM end-to-end, segmentasi, positioning, budget pacing, "
                "eksperimen growth, dan koordinasi lintas fungsi."
            ),
            "threshold": 0.82,
        },
    },
    "Curriculum/Content": {
        "junior": {
            "expected_answer": (
                "Pengalaman membuat materi pembelajaran, struktur modul, contoh evaluasi, dan umpan balik peserta."
            ),
            "threshold": 0.74,
        },
        "mid": {
            "expected_answer": (
                "Desain kurikulum, alignment dengan standar industri, rubrik penilaian, "
                "dan iterasi berdasarkan data completion/feedback."
            ),
            "threshold": 0.79,
        },
        "senior": {
            "expected_answer": (
                "Strategi kurikulum skala besar, quality assurance konten, kolaborasi dengan ahli, "
                "dan roadmap pembelajaran."
            ),
            "threshold": 0.83,
        },
    },
}


def _base_template(role: str, level: str) -> Tuple[str, float]:
    role_data = ROLE_TEMPLATES.get(role) or {}
    level_key = level.lower()
    data = role_data.get(level_key) or {}
    expected = data.get("expected_answer", "")
    threshold = float(data.get("threshold", 0.8))
    return expected, threshold


def _get_template(role: str, level: str, use_overrides: bool = True) -> Tuple[str, float]:
    """Fetch template, applying any session overrides when present."""
    expected, threshold = _base_template(role, level)
    if use_overrides and hasattr(st, "session_state"):
        overrides = st.session_state.get("role_overrides", {})
        role_over = overrides.get(role, {})
        level_key = level.lower()
        level_over = role_over.get(level_key) or role_over.get(level)
        if level_over:
            expected = level_over.get("expected_answer", expected) or expected
            threshold = float(level_over.get("threshold", threshold))
    return expected, threshold


def _default_api_base() -> str:
    # Use env first, then Streamlit secrets (if available), fallback to localhost for local dev.
    env_val = os.getenv("NEXT_PUBLIC_API_BASE_URL")
    if env_val:
        return env_val
    try:
        return st.secrets.get("NEXT_PUBLIC_API_BASE_URL", "http://localhost:8000/api")  # type: ignore[attr-defined]
    except Exception:
        return "http://localhost:8000/api"


@st.cache_resource(show_spinner=False)
def _cached_transcriber() -> Any:
    """Load Whisper once per session to avoid repeated downloads."""
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
    device = os.getenv("WHISPER_DEVICE")
    return get_transcriber(model_size=model_size, device=device)


@st.cache_resource(show_spinner=False)
def _cached_nlp_model() -> NLPScoringModel:
    """Load NLP scoring models once per session."""
    return NLPScoringModel()


def upload_video(
    api_base: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str | None,
    candidate_id: str | None,
    expected_answer: str,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"expected_answer": expected_answer}
    if candidate_id:
        data["candidate_id"] = candidate_id

    files = {"file": (filename, file_bytes, mime_type or "application/octet-stream")}
    with httpx.Client(timeout=120) as client:
        response = client.post(f"{api_base}/interviews/upload", data=data, files=files)
        response.raise_for_status()
        return response.json()


def fetch_result(api_base: str, interview_id: str) -> Dict[str, Any]:
    with httpx.Client(timeout=60) as client:
        response = client.get(f"{api_base}/interviews/result/{interview_id}")
        response.raise_for_status()
        return response.json()


def poll_until_complete(api_base: str, interview_id: str, timeout: int = 900, interval: int = 5) -> Dict[str, Any]:
    start = time.monotonic()
    with st.spinner("Processing interview…"):
        while True:
            result = fetch_result(api_base, interview_id)
            status = result.get("status")
            if status != "processing":
                return result
            if time.monotonic() - start > timeout:
                return result
            time.sleep(interval)


def _recommendation_label(score: float, threshold: float) -> Tuple[str, str]:
    """Return (label, color) for recommendation."""
    if score >= threshold + 0.05:
        return "Layak dipertimbangkan", "green"
    if score >= threshold - 0.05:
        return "Perlu interview lanjutan", "orange"
    return "Kurang sesuai", "red"


def _competency_rows(score: float) -> list[tuple[str, float, str]]:
    """Use verbal score as proxy for all competencies (current pipeline is text-only)."""
    proxy = max(0.0, min(score, 1.0))
    items = [
        ("Komunikasi & Klaritas", proxy),
        ("Problem Solving", proxy),
        ("Teamwork", proxy - 0.05),
        ("Cultural Fit", proxy - 0.03),
        ("Penguasaan Teknis", proxy + 0.02),
    ]
    rows: list[tuple[str, float, str]] = []
    for name, val in items:
        v = max(0.0, min(val, 1.0))
        desc = "Baik" if v >= 0.8 else "Cukup" if v >= 0.65 else "Perlu perbaikan"
        rows.append((name, v, desc))
    return rows


def _extract_quotes(transcript: str, max_quotes: int = 2) -> list[str]:
    sentences = [s.strip() for s in transcript.replace("\n", " ").split(".") if s.strip()]
    return sentences[:max_quotes]


def _process_locally(
    uploaded_file: Any,
    expected_answer: str,
    role: str,
    level: str,
    threshold: float,
    candidate_id: str | None,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Run inference locally (Streamlit-only) using Whisper + NLP models."""
    size_mb = (uploaded_file.size or 0) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise RuntimeError(f"File terlalu besar ({size_mb:.1f} MB). Batas {MAX_UPLOAD_MB} MB.")

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    temp_video.write(uploaded_file.getbuffer())
    temp_video.flush()
    temp_video.close()

    audio_path = None
    try:
        with st.spinner("Menyiapkan audio (ffmpeg)…"):
            audio_path = extract_audio(temp_video.name)

        with st.spinner("Memuat model Whisper & transcribing… (pertama kali agak lama)"):
            stt_result = _cached_transcriber().transcribe(audio_path)

        transcript = (stt_result.get("text") or "").strip()
        with st.spinner("Skoring NLP (relevance + fluency)…"):
            nlp_scores = score_transcript(
                transcript=transcript,
                expected_answer=expected_answer or "",
                scoring_model=_cached_nlp_model(),
                include_summary=False,
            )

        report = aggregate_results(stt_result, nlp_scores)
        interview_id = f"local-{uuid4().hex[:8]}"
        meta = {
            "candidate_id": candidate_id or "",
            "role": role,
            "level": level,
            "threshold": threshold,
            "expected_answer": expected_answer,
        }
        result = {"status": "completed", "report": report, "candidate_id": candidate_id or ""}
        return result, interview_id, meta
    finally:
        for path in [audio_path, temp_video.name]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except OSError:
                    pass


def render_report(result: Dict[str, Any], interview_id: str | None = None, meta: Dict[str, Any] | None = None) -> None:
    status = result.get("status")
    st.subheader("Processing Status")
    st.info(f"Interview status: **{status}**")

    if status == "failed":
        st.error(result.get("error", "Processing failed"))
        return

    report = result.get("report") or {}
    if not report:
        st.warning("Report not available yet. Try fetching again in a moment.")
        return

    verbal = float(report.get("verbal_score") or 0.0)
    final_score = float(report.get("final_score") or verbal)
    confidence = float(report.get("confidence") or 0.0)
    role = meta.get("role") if meta else None
    level = meta.get("level") if meta else None
    threshold = float(meta.get("threshold", 0.8) if meta else 0.8)
    expected_prompt = meta.get("expected_answer") if meta else ""
    candidate_id = result.get("candidate_id") or (meta.get("candidate_id") if meta else "")

    rec_label, rec_color = _recommendation_label(final_score, threshold)

    # Header card
    with st.container():
        st.markdown("### Laporan Kandidat")
        col_a, col_b = st.columns([3, 1])
        title_parts = [candidate_id or "Kandidat", f"– {role}" if role else "", f"({level})" if level else ""]
        col_a.markdown(f"**{' '.join([p for p in title_parts if p]).strip()}**")
        badge = f":{rec_color}[{rec_label}]"
        col_a.markdown(f"{badge} &nbsp;&nbsp; Skor: **{final_score*100:.0f}/100**")
        info = []
        if interview_id:
            info.append(f"ID: {interview_id}")
        info.append(f"Tanggal: n/a")
        info.append("Durasi: n/a")
        info.append("Tipe: Video Recording")
        col_a.markdown(" · ".join(info))
        col_b.metric("Confidence", f"{confidence:.3f}")

    st.divider()

    # Ringkasan eksekutif
    with st.container():
        st.markdown("#### Ringkasan AI")
        col1, col2 = st.columns([2, 1])
        col1.write(f"Rekomendasi: **{rec_label}** (threshold {threshold:.2f})")
        col1.write(f"Confidence transkripsi: **{confidence:.3f}**")
        if expected_prompt:
            with col1.expander("Kisi-kisi / Expected Answer"):
                col1.markdown(expected_prompt)
        strengths = []
        areas = []
        if verbal >= 0.85:
            strengths.append("Relevansi jawaban tinggi terhadap prompt.")
        else:
            areas.append("Perlu peningkatan relevansi contoh jawaban.")
        if confidence < 0.7:
            areas.append("Perbaiki kualitas audio atau artikulasi untuk meningkatkan confidence.")
        else:
            strengths.append("Artikulasi cukup jelas berdasarkan confidence.")
        col1.markdown("**Kekuatan utama**")
        col1.markdown("\n".join([f"- {s}" for s in strengths]) or "- (tidak ada)")
        col1.markdown("**Area pengembangan**")
        col1.markdown("\n".join([f"- {a}" for a in areas]) or "- (tidak ada)")
        col1.markdown("**Catatan penting**")
        col1.markdown(f"- Ringkasan: {report.get('summary') or 'n/a'}")
        col2.metric("Final Score", f"{final_score:.2f}")
        col2.metric("Verbal Score", f"{verbal:.2f}")

    st.divider()

    # Skor per kompetensi (proxy)
    with st.container():
        st.markdown("#### Evaluasi Kompetensi (proxy verbal)")
        rows = _competency_rows(verbal)
        for name, val, desc in rows:
            st.progress(val, text=f"{name} — {val*5:.1f}/5 ({desc})")

    # Detail per pertanyaan (single accordion as placeholder)
    with st.container():
        st.markdown("#### Detail Jawaban")
        with st.expander("Transkrip & Ringkasan"):
            st.markdown(f"**Ringkasan**: {report.get('summary') or 'n/a'}")
            transcript = report.get("transcript") or ""
            if transcript:
                st.text_area("Transkrip Lengkap", value=transcript, height=200)
            else:
                st.info("Transkrip tidak tersedia.")
            st.caption("Catatan: detail per-pertanyaan belum tersedia di pipeline ini.")

    # Analisis audio & delivery (proxy from confidence)
    with st.container():
        st.markdown("#### Analisis Audio & Delivery (proxy)")
        speed = "Normal"  # tidak dianalisis saat ini
        articulation = "Jelas" if confidence >= 0.7 else "Perlu perbaikan"
        filler = "Sedang"  # placeholder
        emotion = "Tidak dianalisis"
        consistency = "Tinggi" if confidence >= 0.7 else "Sedang"
        st.markdown(
            f"- Kecepatan bicara: {speed}\n"
            f"- Artikulasi: {articulation}\n"
            f"- Filler words: {filler}\n"
            f"- Emosi: {emotion}\n"
            f"- Konsistensi jawaban: {consistency}\n"
            "_Catatan: metrik ini bersifat placeholder, pipeline saat ini fokus ke STT + NLP._"
        )

    # Key quotes
    with st.container():
        st.markdown("#### Kutipan Kunci")
        quotes = _extract_quotes(report.get("transcript") or "")
        if not quotes:
            st.info("Belum ada kutipan yang dapat ditarik dari transkrip.")
        else:
            for idx, q in enumerate(quotes, 1):
                st.markdown(f"> {q}")
                if idx >= 3:
                    break

    # Rekomendasi tindak lanjut
    with st.container():
        st.markdown("#### Rekomendasi Tindak Lanjut")
        if rec_label == "Layak dipertimbangkan":
            action = "Shortlist / lanjut interview user"
        elif rec_label == "Perlu interview lanjutan":
            action = "Jadwalkan interview lanjutan"
        else:
            action = "Pertimbangkan kandidat lain"
        st.markdown(f"Rekomendasi: **{action}**")
        st.markdown("- Topik digali: contoh konkret kolaborasi tim, problem solving, dan alignment dengan peran.")

    # Transparansi AI
    st.markdown("---")
    st.caption(
        "Laporan dihasilkan otomatis dari transkrip (Whisper) dan evaluasi NLP. "
        "Keputusan akhir tetap di tangan tim HR."
    )



def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f8fb;
            --card: #ffffff;
            --sidebar: #061525;
            --sidebar-soft: #071827;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --text: #0f172a;
            --muted: #64748b;
            --muted-soft: #94a3b8;
            --border: #e5e7eb;
            --green: #10b981;
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            --radius: 20px;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}
        .stApp {background: var(--bg); color: var(--text);}
        .block-container {padding: 2rem 2.4rem 3rem; max-width: 1440px;}

        /* Persistent mini rail when native sidebar is collapsed */
        .stApp::before {
            content: "";
            position: fixed;
            left: 0;
            top: 0;
            width: 3.75rem;
            height: 100vh;
            background: linear-gradient(180deg, #061525 0%, #071827 100%);
            border-right: 1px solid rgba(96, 165, 250, 0.22);
            z-index: 0;
            pointer-events: none;
        }
        .stApp::after {
            content: "AI";
            position: fixed;
            left: 0.72rem;
            top: 4.1rem;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 13px;
            display: grid;
            place-items: center;
            background: rgba(37, 99, 235, 0.22);
            border: 1px solid rgba(96, 165, 250, 0.35);
            color: #dbeafe;
            font-weight: 900;
            z-index: 0;
            pointer-events: none;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }
        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #061525 0%, #071827 100%);
            padding: 1.15rem 1.05rem 1.8rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            visibility: visible !important;
            height: 3.1rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.6rem 0.2rem;
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]::after {
            content: "AI Interview Assessment";
            color: #f8fafc;
            font-size: 0.86rem;
            font-weight: 850;
            letter-spacing: -0.02em;
            white-space: nowrap;
            opacity: 0.96;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button {
            visibility: visible !important;
            color: #dbeafe !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #dbeafe !important;
        }
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.78) !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 12px !important;
            color: #f8fafc !important;
        }
        .sidebar-logo {padding: 0.15rem 0 1.25rem; border-bottom: 1px solid rgba(148, 163, 184, 0.18); margin-bottom: 1.15rem;}
        .sidebar-logo h2 {color: #f8fafc; font-size: 1.45rem; margin: 0; letter-spacing: -0.035em;}
        .sidebar-logo p {color: #93c5fd; margin: 0.18rem 0 0; font-size: 0.88rem;}
        .sidebar-section-title {font-size: 0.72rem; font-weight: 850; color: #67e8f9; letter-spacing: 0.14em; margin: 0.55rem 0 1rem;}
        .sidebar-status-card {background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.30); border-radius: 18px; padding: 1rem; margin-top: 1.25rem;}
        .status-line {display: flex; align-items: center; gap: 0.55rem; color: #ecfdf5; font-weight: 850;}
        .green-dot {width: 9px; height: 9px; border-radius: 999px; background: var(--green); box-shadow: 0 0 0 5px rgba(16, 185, 129, 0.13); display: inline-block; flex: 0 0 auto;}
        .sidebar-status-card p {color: #a7f3d0 !important; font-size: 0.84rem; margin: 0.35rem 0 0 1.45rem;}
        .sidebar-footer {color: #94a3b8; font-size: 0.75rem; margin-top: 2.1rem; line-height: 1.5;}

        /* Native collapsed/open sidebar controls: keep visible and obvious */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        button[aria-label*="sidebar" i] {
            visibility: visible !important;
            position: fixed !important;
            left: 0.72rem !important;
            top: 0.78rem !important;
            z-index: 999999 !important;
            background: #061525 !important;
            border: 1px solid rgba(96, 165, 250, 0.45) !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.20) !important;
            color: #dbeafe !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button,
        button[aria-label*="sidebar" i] svg {
            color: #dbeafe !important;
            fill: #dbeafe !important;
            opacity: 1 !important;
        }
        .floating-open-sidebar {margin: -0.75rem 0 1rem 0; max-width: 190px;}
        .floating-open-sidebar div.stButton > button {
            background: #061525 !important;
            color: #dbeafe !important;
            border: 1px solid rgba(96, 165, 250, 0.45) !important;
            border-radius: 999px !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16) !important;
            padding: 0.5rem 0.9rem !important;
        }

        /* Main cards */
        .hero-card, .card, .step {
            background: var(--card);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }
        .hero-card {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: 1.4rem;
            border-radius: 24px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .hero-kicker {display: inline-flex; align-items: center; gap: 0.4rem; background: #dbeafe; color: #1d4ed8; border-radius: 999px; padding: 0.35rem 0.75rem; font-size: 0.8rem; font-weight: 850; margin-bottom: 0.85rem;}
        .hero-card h1 {font-size: 2.35rem; line-height: 1.05; color: var(--text); margin: 0 0 0.65rem; letter-spacing: -0.055em;}
        .hero-card p {font-size: 1rem; color: var(--muted); margin: 0; max-width: 760px; line-height: 1.6;}
        .hero-visual {width: 108px; height: 108px; border-radius: 26px; background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%); display: grid; place-items: center; color: white; box-shadow: 0 18px 38px rgba(37, 99, 235, 0.26); font-size: 2.35rem;}
        .stepper {display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.85rem; margin: 0 0 20px;}
        .step {border-radius: 18px; padding: 0.85rem 1rem; color: var(--muted); font-weight: 850; display: flex; align-items: center; min-height: 54px;}
        .step.active {background: #eff6ff; color: var(--primary); border-color: #93c5fd;}
        .step span {display: inline-flex; width: 28px; height: 28px; border-radius: 999px; align-items: center; justify-content: center; margin-right: 0.6rem; background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569; font-size: 0.82rem; font-weight: 900;}
        .step.active span {background: var(--primary); border-color: var(--primary); color: white;}
        .card {border-radius: var(--radius); padding: 24px; margin-bottom: 20px;}
        .card-title {display: flex; align-items: center; gap: 0.6rem; color: var(--text); font-size: 1.08rem; font-weight: 850; margin-bottom: 1rem;}
        .card-title-row {display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-bottom: 1rem;}
        .support-note {background: #f8fafc; border: 1px dashed #bfdbfe; border-radius: 16px; padding: 0.85rem 1rem; margin-bottom: 0.95rem; color: var(--muted); font-size: 0.88rem; line-height: 1.45;}

        /* Light main form controls */
        .main .stTextInput input,
        .main .stNumberInput input,
        .main textarea,
        .main div[data-baseweb="select"] > div,
        .main div[data-testid="stFileUploader"] section {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        .main input::placeholder, .main textarea::placeholder {color: var(--muted-soft) !important;}
        .main label, .main p, .main span {color: var(--text);}
        .main small, .main .stCaptionContainer, .main [data-testid="stMarkdownContainer"] p {color: var(--muted);}
        .main div[data-baseweb="select"] span {color: var(--text) !important;}
        .main div[data-testid="stFileUploader"] section {background: #f8fbff !important; border: 1.5px dashed #93c5fd !important; padding: 0.65rem !important;}
        .main div[data-testid="stFileUploader"] section * {color: var(--text) !important;}
        div.stButton > button, div.stFormSubmitButton > button {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.78rem 1rem;
            font-weight: 850;
            box-shadow: 0 12px 22px rgba(37, 99, 235, 0.20);
        }
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {background: var(--primary-hover); color: white; border: none;}

        /* Right panel */
        .status-badge {display: inline-flex; align-items: center; gap: 0.45rem; background: #ecfdf5; color: #047857; border: 1px solid #bbf7d0; border-radius: 999px; padding: 0.35rem 0.75rem; font-size: 0.8rem; font-weight: 850;}
        .status-subtitle {font-size: 0.82rem; color: var(--muted); margin-top: 0.45rem; text-align: right;}
        .metric-grid {display: grid; grid-template-columns: 1.25fr 1fr 1fr; gap: 0.75rem; margin-top: 1rem;}
        .metric-card {background: #f8fafc; border: 1px solid var(--border); border-radius: 18px; padding: 0.95rem; min-height: 106px;}
        .metric-card span {display: block; color: var(--muted); font-size: 0.74rem; font-weight: 850; text-transform: uppercase; letter-spacing: 0.04em;}
        .metric-card strong {display: block; color: var(--text); font-size: 1.28rem; margin: 0.28rem 0 0.08rem; letter-spacing: -0.03em;}
        .metric-card.primary strong {font-size: 1.58rem;}
        .metric-card em {font-style: normal; color: #059669; font-size: 0.78rem; font-weight: 850;}
        .summary-text {background: #f8fafc; border: 1px solid var(--border); border-radius: 18px; padding: 1rem; color: #334155; line-height: 1.7; font-size: 0.95rem;}
        .transcript-line {background: #f8fafc; border: 1px solid var(--border); border-radius: 14px; padding: 0.72rem 0.85rem; margin-bottom: 0.55rem; color: #334155; font-size: 0.9rem; line-height: 1.5;}
        .tiny-button {border: 1px solid #bfdbfe; color: var(--primary); background: #eff6ff; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.78rem; font-weight: 850; white-space: nowrap;}

        @media (max-width: 980px) {
            .block-container {padding: 1.2rem 1rem 2rem 4.7rem;}
            .hero-card {grid-template-columns: 1fr;}
            .hero-visual {display: none;}
            .stepper, .metric-grid {grid-template-columns: 1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_header() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
            <h2>AI Interview</h2>
            <p>Assessment MVP</p>
        </div>
        <div class="sidebar-section-title">CONTROL PANEL</div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_status(local_mode: bool) -> None:
    title = "Local inference ready" if local_mode else "FastAPI ready"
    subtitle = "Streamlit model mode enabled" if local_mode else "All systems operational"
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status-card">
            <div class="status-line"><span class="green-dot"></span>{title}</div>
            <p>{subtitle}</p>
        </div>
        <div class="sidebar-footer">
            © 2024 AI Interview Assessment<br/>
            MVP Prototype
        </div>
        """,
        unsafe_allow_html=True,
    )

def request_sidebar_state(is_open: bool) -> None:
    st.session_state["sidebar_open"] = is_open
    st.rerun()

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div>
                <div class="hero-kicker">● MVP Demo</div>
                <h1>AI Interview Assessment</h1>
                <p>Upload interview videos, trigger the FastAPI pipeline, atau jalankan inference lokal di Streamlit untuk demo cepat.</p>
            </div>
            <div class="hero-visual">▶︎</div>
        </div>
        <div class="stepper">
            <div class="step active"><span>1</span>Setup</div>
            <div class="step"><span>2</span>Upload</div>
            <div class="step"><span>3</span>Evaluate</div>
            <div class="step"><span>4</span>Review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _score_to_label(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Good"
    if score > 0:
        return "Needs Review"
    return "Pending"

def _latest_report() -> tuple[Dict[str, Any] | None, str | None, Dict[str, Any] | None]:
    result = st.session_state.get("latest_result")
    interview_id = st.session_state.get("last_interview_id")
    meta = st.session_state.get("meta", {}).get(interview_id) if interview_id else None
    if isinstance(result, dict):
        return result, interview_id, meta
    return None, interview_id, meta

def render_status_cards() -> None:
    result, _, _ = _latest_report()
    report = (result or {}).get("report") or {}
    status = (result or {}).get("status") or "ready"
    verbal = float(report.get("verbal_score") or 0.78)
    final_score = float(report.get("final_score") or 0.82)
    relevance = float(report.get("relevance") or report.get("overall_score") or 0.86)
    subtitle = "Report ready" if report else "Waiting for upload"
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title-row">
                <div class="card-title">📊 Evaluation Status</div>
                <div>
                    <div class="status-badge"><span class="green-dot"></span>{status.title() if status != 'ready' else 'Ready'}</div>
                    <div class="status-subtitle">{subtitle}</div>
                </div>
            </div>
            <div class="metric-grid">
                <div class="metric-card primary"><span>Final Score</span><strong>{final_score*100:.0f}/100</strong><em>{_score_to_label(final_score)}</em></div>
                <div class="metric-card"><span>Fluency</span><strong>{verbal*100:.0f}/100</strong><em>{_score_to_label(verbal)}</em></div>
                <div class="metric-card"><span>Relevance</span><strong>{relevance*100:.0f}/100</strong><em>{_score_to_label(relevance)}</em></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_summary_card() -> None:
    result, _, _ = _latest_report()
    report = (result or {}).get("report") or {}
    summary = report.get("summary") or (
        "Kandidat menunjukkan pemahaman yang baik tentang React/Next dan manajemen state dengan hooks/RTK. "
        "Penjelasan cukup jelas dan terstruktur dengan contoh yang relevan. "
        "Perhatian terhadap aksesibilitas dan performa menunjukkan awareness yang baik."
    )
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">✨ AI Summary</div>
            <div class="summary-text">{escape(summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_transcript_card() -> None:
    result, _, _ = _latest_report()
    report = (result or {}).get("report") or {}
    transcript = (report.get("transcript") or "").strip()
    if transcript:
        sentences = [line.strip() for line in transcript.replace("\n", " ").split(".") if line.strip()]
        lines = [f"{idx:02d}:00 {sentence[:150]}..." for idx, sentence in enumerate(sentences[:4], 1)]
    else:
        lines = [
            "00:12 Saya biasanya menggunakan React dengan Next.js untuk proyek produksi...",
            "00:28 Untuk state management, saya lebih sering pakai Redux Toolkit...",
            "00:45 Dalam hal testing, saya menulis unit test dengan Jest dan React Testing Library...",
        ]
    st.markdown('<div class="card"><div class="card-title-row"><div class="card-title">📝 Transcript Preview</div><div class="tiny-button">View full transcript</div></div>', unsafe_allow_html=True)
    for line in lines[:5]:
        st.markdown(f'<div class="transcript-line">{escape(line)}</div>', unsafe_allow_html=True)
    if transcript:
        with st.expander("View full transcript"):
            st.text_area("Full transcript", transcript, height=220, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

def main() -> None:
    initial_sidebar_state = "expanded" if st.session_state.get("sidebar_open", True) else "collapsed"
    st.set_page_config(
        page_title="AI Interview Assessment",
        layout="wide",
        initial_sidebar_state=initial_sidebar_state,
    )
    inject_dashboard_css()

    if "last_interview_id" not in st.session_state:
        st.session_state["last_interview_id"] = None
    if "selected_role" not in st.session_state:
        st.session_state["selected_role"] = list(ROLE_TEMPLATES.keys())[0]
    if "selected_level" not in st.session_state:
        st.session_state["selected_level"] = "Junior"
    if "_prev_role" not in st.session_state:
        st.session_state["_prev_role"] = st.session_state["selected_role"]
    if "_prev_level" not in st.session_state:
        st.session_state["_prev_level"] = st.session_state["selected_level"]
    if "role_overrides" not in st.session_state:
        st.session_state["role_overrides"] = {}
    if "meta" not in st.session_state:
        st.session_state["meta"] = {}
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "sidebar_open" not in st.session_state:
        st.session_state["sidebar_open"] = True
    if "expected_answer_text" not in st.session_state or "current_threshold" not in st.session_state:
        default_exp, default_thr = _get_template(st.session_state["selected_role"], st.session_state["selected_level"])
        st.session_state["expected_answer_text"] = default_exp
        st.session_state["current_threshold"] = default_thr

    render_sidebar_header()
    if st.sidebar.button("‹ Collapse panel", use_container_width=True):
        request_sidebar_state(False)

    if not st.session_state.get("sidebar_open", True):
        st.markdown('<div class="floating-open-sidebar">', unsafe_allow_html=True)
        if st.button("☰ Open controls", key="open_sidebar_button"):
            request_sidebar_state(True)
        st.markdown('</div>', unsafe_allow_html=True)

    local_mode = st.sidebar.checkbox(
        "Run locally (tanpa FastAPI)",
        value=False,
        help="Streamlit memuat model Whisper + NLP langsung. Cocok untuk demo/traffic kecil.",
    )

    if local_mode:
        st.sidebar.info("Local mode aktif: upload diproses di Streamlit, tidak memanggil API.")
        api_base = ""
        poll_enabled = False
        poll_timeout = 0
    else:
        api_base = st.sidebar.text_input("FastAPI Base URL", _default_api_base())
        poll_enabled = st.sidebar.checkbox("Auto-poll result", value=True)
        poll_timeout = st.sidebar.number_input(
            "Auto-poll timeout (seconds)",
            min_value=60,
            max_value=1800,
            step=60,
            value=900,
            help="Perpanjang jika unduhan model pertama kali memakan waktu lama.",
        )

    with st.sidebar.expander("Role Templates / Overrides", expanded=False):
        override_role = st.selectbox(
            "Role (override)",
            options=list(ROLE_TEMPLATES.keys()),
            index=list(ROLE_TEMPLATES.keys()).index(st.session_state["selected_role"]),
            key="override_role",
        )
        override_level = st.selectbox(
            "Level (override)",
            options=["Junior", "Mid", "Senior"],
            index=["Junior", "Mid", "Senior"].index(st.session_state["selected_level"]),
            key="override_level",
        )
        current_expected, current_thr = _get_template(override_role, override_level)
        override_expected = st.text_area(
            "Expected Answer Override",
            value=current_expected,
            key="override_expected_text",
            help="Isi untuk mengganti kisi-kisi role/level ini.",
        )
        override_thr = st.number_input(
            "Threshold Override",
            min_value=0.5,
            max_value=1.0,
            step=0.01,
            value=float(current_thr),
            key="override_threshold_value",
        )
        if st.button("Simpan Override", use_container_width=True):
            role_overrides = st.session_state.get("role_overrides", {})
            role_overrides.setdefault(override_role, {})
            role_overrides[override_level.lower()] = {
                "expected_answer": override_expected,
                "threshold": override_thr,
            }
            st.session_state["role_overrides"] = role_overrides
            if (
                override_role == st.session_state.get("selected_role")
                and override_level.lower() == st.session_state.get("selected_level", "").lower()
            ):
                st.session_state["expected_answer_text"] = override_expected
                st.session_state["current_threshold"] = float(override_thr)
                st.session_state["_prev_role"] = override_role
                st.session_state["_prev_level"] = override_level
            st.success(f"Override disimpan untuk {override_role} ({override_level}).")
    render_sidebar_status(local_mode)

    render_hero()

    prev_role = st.session_state.get("_prev_role")
    prev_level = st.session_state.get("_prev_level")

    left_col, right_col = st.columns([1.18, 0.82], gap="large")

    with left_col:
        st.markdown('<div class="card"><div class="card-title">⚙️ Interview Setup</div>', unsafe_allow_html=True)
        role_col, level_col = st.columns(2)
        with role_col:
            role = st.selectbox(
                "Role",
                options=list(ROLE_TEMPLATES.keys()),
                index=list(ROLE_TEMPLATES.keys()).index(st.session_state["selected_role"]),
                key="selected_role",
            )
        with level_col:
            level = st.selectbox(
                "Level",
                options=["Junior", "Mid", "Senior"],
                index=["Junior", "Mid", "Senior"].index(st.session_state["selected_level"]),
                key="selected_level",
            )
        if role != prev_role or level != prev_level:
            expected_default, threshold = _get_template(role, level)
            st.session_state["expected_answer_text"] = expected_default
            st.session_state["current_threshold"] = threshold
            st.session_state["_prev_role"] = role
            st.session_state["_prev_level"] = level
        threshold = st.session_state.get("current_threshold", 0.8)
        st.caption(f"Evaluation threshold: {threshold:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">🎥 Upload Interview Video</div>'
            '<div class="support-note"><strong>Upload via Browser</strong><br/>'
            'Drag & drop atau pilih video interview. Limit 200MB per file • MP4, WEBM, MOV, MKV, MPEG4.</div>',
            unsafe_allow_html=True,
        )
        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Select interview video",
                type=["mp4", "webm", "mov", "mkv", "mpeg4"],
                help="Maximum recommended duration 5 minutes.",
            )
            candidate_id = st.text_input("Candidate ID (optional)")
            expected_prefill = st.session_state.get("expected_answer_text", "")
            expected_answer = st.text_area(
                "Expected Answer / Prompt",
                value=expected_prefill,
                key="expected_answer_text",
                height=135,
                help="Digunakan untuk menghitung relevansi jawaban terhadap kisi-kisi role.",
            )
            submit = st.form_submit_button("▶ Start Evaluation", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if submit:
            if not uploaded_file:
                st.error("Please attach a video file first.")
            else:
                try:
                    if local_mode:
                        result, interview_id, meta = _process_locally(
                            uploaded_file=uploaded_file,
                            expected_answer=expected_answer or st.session_state.get("expected_answer_text", ""),
                            role=role,
                            level=level,
                            threshold=threshold,
                            candidate_id=candidate_id or None,
                        )
                        st.session_state["last_interview_id"] = interview_id
                        st.session_state.setdefault("meta", {})[interview_id] = meta
                        st.session_state["latest_result"] = result
                        st.success("Proses lokal selesai.")
                    else:
                        st.info("Uploading video…")
                        response = upload_video(
                            api_base=api_base,
                            file_bytes=uploaded_file.read(),
                            filename=uploaded_file.name,
                            mime_type=uploaded_file.type,
                            candidate_id=candidate_id or None,
                            expected_answer=expected_answer or st.session_state.get("expected_answer_text", ""),
                        )
                        interview_id = response["interview_id"]
                        st.session_state["last_interview_id"] = interview_id
                        st.session_state.setdefault("meta", {})[interview_id] = {
                            "candidate_id": candidate_id or "",
                            "role": role,
                            "level": level,
                            "threshold": threshold,
                            "expected_answer": expected_answer or st.session_state.get("expected_answer_text", ""),
                        }
                        st.success(f"Job queued successfully! Interview ID: {interview_id}")
                        if poll_enabled:
                            result = poll_until_complete(api_base, interview_id, timeout=int(poll_timeout))
                            st.session_state["latest_result"] = result
                except httpx.HTTPStatusError as exc:
                    st.error(f"Upload failed: {exc.response.text}")
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

        st.markdown('<div class="card"><div class="card-title">🔎 Fetch Latest Report</div>', unsafe_allow_html=True)
        if local_mode:
            st.info("Local mode aktif. Bagian fetch hanya untuk skenario FastAPI.")
        else:
            interview_id_input = st.text_input(
                "Interview ID",
                value=st.session_state.get("last_interview_id") or "",
                help="Use the ID returned after an upload.",
            )
            if st.button("Fetch Result", use_container_width=True):
                if not interview_id_input:
                    st.warning("Provide an interview ID first.")
                else:
                    try:
                        result = fetch_result(api_base, interview_id_input)
                        st.session_state["last_interview_id"] = interview_id_input
                        st.session_state["latest_result"] = result
                        st.success("Report fetched successfully.")
                    except httpx.HTTPStatusError as exc:
                        st.error(f"Backend error: {exc.response.text}")
                    except Exception as exc:  # noqa: BLE001
                        st.exception(exc)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        render_status_cards()
        render_summary_card()
        render_transcript_card()

    latest_result, latest_interview_id, latest_meta = _latest_report()
    if latest_result and latest_result.get("report"):
        with st.expander("Detailed AI Evaluation Report", expanded=False):
            render_report(latest_result, interview_id=latest_interview_id, meta=latest_meta)

if __name__ == "__main__":
    main()
