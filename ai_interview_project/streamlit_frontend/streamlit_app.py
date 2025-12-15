"""Streamlit prototype dashboard for uploading interviews and viewing reports."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import httpx
import streamlit as st

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
    # Use env first, then Streamlit secrets, fallback to localhost for local dev
    return (
        os.getenv("NEXT_PUBLIC_API_BASE_URL")
        or (getattr(st, "secrets", {}) or {}).get("NEXT_PUBLIC_API_BASE_URL")
        or "http://localhost:8000/api"
    )


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


def main() -> None:
    st.set_page_config(page_title="AI Interview Assessment (Prototype)", layout="wide")
    st.title("AI Interview Assessment – MVP (Streamlit Prototype)")
    st.write(
        "Upload interview videos, trigger the FastAPI pipeline, and inspect the autogenerated report "
        "without leaving this prototype."
    )

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
    if "expected_answer_text" not in st.session_state or "current_threshold" not in st.session_state:
        default_exp, default_thr = _get_template(st.session_state["selected_role"], st.session_state["selected_level"])
        st.session_state["expected_answer_text"] = default_exp
        st.session_state["current_threshold"] = default_thr

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

    # Sidebar overrides for role templates
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
            # Jika override ini untuk role/level yang sedang dipilih, langsung terapkan ke prompt/form
            if (
                override_role == st.session_state.get("selected_role")
                and override_level.lower() == st.session_state.get("selected_level", "").lower()
            ):
                st.session_state["expected_answer_text"] = override_expected
                st.session_state["current_threshold"] = float(override_thr)
                st.session_state["_prev_role"] = override_role
                st.session_state["_prev_level"] = override_level
            st.success(f"Override disimpan untuk {override_role} ({override_level}).")

    # Role & Level selectors outside the form so changes apply immediately
    st.subheader("Role & Level")
    prev_role = st.session_state.get("_prev_role")
    prev_level = st.session_state.get("_prev_level")
    role = st.selectbox(
        "Role",
        options=list(ROLE_TEMPLATES.keys()),
        index=list(ROLE_TEMPLATES.keys()).index(st.session_state["selected_role"]),
        key="selected_role",
    )
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

    with st.form("upload_form", clear_on_submit=True):
        st.subheader("1) Upload Video")
        uploaded_file = st.file_uploader(
            "Select interview video",
            type=["mp4", "webm", "mov", "mkv"],
            help="Maximum recommended duration 5 minutes.",
        )
        candidate_id = st.text_input("Candidate ID (optional)")
        # Keep the expected answer pre-filled with the active role template even after form clears.
        expected_prefill = st.session_state.get("expected_answer_text", "")
        expected_answer = st.text_area(
            "Expected Answer / Prompt",
            value=expected_prefill,
            key="expected_answer_text",
            help="Digunakan untuk menghitung relevansi jawaban terhadap kisi-kisi role.",
        )
        submit = st.form_submit_button("Start Evaluation", use_container_width=True)

    if submit:
        if not uploaded_file:
            st.error("Please attach a video file first.")
        else:
            try:
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
                    meta = st.session_state.get("meta", {}).get(interview_id)
                    render_report(result, interview_id=interview_id, meta=meta)
            except httpx.HTTPStatusError as exc:
                st.error(f"Upload failed: {exc.response.text}")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    st.divider()
    st.subheader("2) Fetch Latest Report")
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
                meta = st.session_state.get("meta", {}).get(interview_id_input) if "meta" in st.session_state else None
                render_report(result, interview_id=interview_id_input, meta=meta)
            except httpx.HTTPStatusError as exc:
                st.error(f"Backend error: {exc.response.text}")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)


if __name__ == "__main__":
    main()
