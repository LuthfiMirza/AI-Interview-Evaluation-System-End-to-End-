# AI Interview Evaluation System

Full-stack AI interview analysis system powered by Whisper Speech-to-Text, zero-bias acoustic speech rate metrics, concept-matching NLP scoring, PostgreSQL persistence, and dual recruiter portals (**Next.js 14 TailAdmin Analytics Dashboard** + **Streamlit Studio**).

The system accepts recorded interview videos, extracts high-fidelity audio, transcribes speech with jargon-aware OpenAI Whisper, evaluates technical qualification through acoustic fluency and semantic concept coverage (tailored for Frontend React & Next.js roles), persists records in PostgreSQL, and delivers recruiter-friendly insights.

> **Note on Experimental Extensions:** Vision-based/non-verbal analysis and cheating detection are present as experimental extension modules and dependencies, while the active core pipeline focuses on speech acoustics and verbal competency.

---

## Architecture Overview

![System Architecture Overview](docs/images/architecture.png)

```mermaid
flowchart TD
  A["Candidate Video (.mp4 / .webm)"] --> B["FastAPI Backend (:9001)"]
  B --> C["FFmpeg Audio Extract (16kHz Mono)"]
  C --> D["Whisper STT (Tech Jargon Bias)"]
  D --> E["Transcript Segments & Timestamps"]
  E --> F["Acoustic Fluency Scorer<br/>(WPM, Fillers, Pauses)"]
  E --> G["Concept Relevance Scorer<br/>(MiniLM + Rubric Checklist)"]
  F --> H["Final Verbal Scoring Aggregator<br/>(60% Relevance + 40% Fluency)"]
  G --> H
  H --> I[("PostgreSQL Database")]
  I --> J["Next.js 14 TailAdmin Dashboard (:3000)"]
  I --> K["Streamlit Rapid Studio (:8501)"]
```

---

## Demo Preview

### 1. Next.js 14 TailAdmin Modern Dashboard (Primary Portal)

Built with **Next.js 14 App Router**, **Tailwind CSS**, and inspired by **TailAdmin**, the primary dashboard provides a multi-page, role-driven workspace for technical recruiters and engineering managers.

#### A. Executive Overview & Live Hub
Presents high-level hiring metrics, average verbal scores, speech pace (WPM), candidate throughput, and quick links to active evaluation tools.

![Next.js Executive Overview](docs/images/nextjs-dashboard-overview.png)

#### B. Video Assessment Studio
Interactive ingestion panel allowing recruiters to select target qualification levels (Junior, Mid, Senior Frontend Engineer) with automated rubric concept mapping, video drag-and-drop, and real-time processing status.

![Next.js Video Assessment Studio](docs/images/nextjs-video-assessment.png)

#### C. React & Next.js Rubric & Criteria
Comprehensive technical qualification rubrics defining expected answer targets and key concept checklists (App Router, Zustand, Rendering Strategies, Web Vitals, Clean Architecture) alongside Whisper tech-jargon vocabulary injections.

![Next.js Rubrics and Criteria](docs/images/nextjs-rubrics.png)

#### D. Speech & NLP Metrics Analytics
Transparent metric breakdown illustrating acoustic pacing benchmarks (110–165 WPM optimal), bilingual filler word detection (`"ehm"`, `"uhm"`, `"anu"`, `"apa ya"`), and the exact verbal formula weighting.

![Next.js Speech and NLP Analytics](docs/images/nextjs-nlp-analytics.png)

#### E. Candidate Directory & Database
Structured candidate directory backed by PostgreSQL and Prisma ORM, offering status filtering, score breakdowns, AI summaries, and audit histories.

![Next.js Candidate Directory](docs/images/nextjs-candidate-directory.png)

---

### 2. Streamlit Prototype (Rapid Assessment Studio)

The companion Streamlit application provides a rapid recruiter-facing interface for ad-hoc evaluations, local model verification, and deep-dive transcript inspections.

#### Upload & Setup
Clean setup interface supporting candidate ID configuration, target role selection, and API endpoint routing.

![Streamlit upload and setup screen](docs/images/streamlit-upload.png)

#### Evaluation Summary
Aggregated score cards displaying verbal score, fluency, relevance, and actionable recruiter feedback.

![Streamlit evaluation summary screen](docs/images/streamlit-report.png)

#### Detailed Report & Segments
Drill-down view of timestamped Whisper segments, acoustic pacing notes, and transcript inspection.

![Streamlit detailed report screen](docs/images/streamlit-detailed-report.png)

---

## What This Project Does

- **Video Ingestion:** Uploads recorded interview videos through FastAPI REST API, Next.js TailAdmin dashboard, or Streamlit prototype.
- **Audio Preprocessing:** Extracts 16 kHz mono audio cleanly from video formats using `ffmpeg`.
- **Jargon-Biased STT:** Transcribes speech using OpenAI Whisper models (`medium`, `base`), enhanced with an `initial_prompt` covering modern software terms (React, Next.js, App Router, Zustand, TypeScript).
- **Acoustic Fluency Scoring:** Evaluates speaking rate (Words Per Minute), detects bilingual filler words, and measures pause gaps without English grammar bias.
- **Concept Relevance Scoring:** Matches candidate responses against role-specific key concepts (55% weight) combined with `all-MiniLM-L6-v2` cosine semantic similarity (45% weight).
- **PostgreSQL Persistence:** Stores interviews, transcripts, and detailed NLP evaluation metrics using SQLAlchemy and Prisma ORM.
- **Dual Recruiter Frontends:** Full-featured Next.js 14 TailAdmin dashboard for enterprise analytics and Streamlit for rapid verification.
- **STT Accuracy Validation:** Includes a dedicated evaluation script for benchmarking WER/accuracy across test datasets.

---

## Tech Stack

| Layer | Technologies |
| --- | --- |
| **Backend Engine** | FastAPI, Uvicorn, SQLAlchemy |
| **Speech-to-Text (STT)** | OpenAI Whisper (`medium`, `base`), FFmpeg, Software Jargon Prompts |
| **NLP & Acoustic Scoring** | PyTorch, `sentence-transformers` (`all-MiniLM-L6-v2`), `AcousticFluencyScorer`, `ConceptRelevanceScorer` |
| **Database & ORM** | PostgreSQL, SQLAlchemy, Prisma ORM |
| **Primary Dashboard** | Next.js 14 (App Router), React 18, Tailwind CSS v3, TailAdmin UI, Lucide Icons |
| **Rapid Prototype** | Streamlit, Pandas, Altair |
| **Tooling & Infrastructure** | Docker, TypeScript, ESLint, Python Dotenv |
| **Experimental Extensions** | YOLOv8, MediaPipe, OpenCV, MinIO/S3 |

---

## Repository Layout

```text
ai_interview_project/
├── app/                         # FastAPI application
│   ├── main.py                  # App entrypoint, CORS, health checks, DB init
│   ├── db.py                    # SQLAlchemy engine & session management
│   ├── routes/interview_routes.py # REST endpoints (upload, polling, results)
│   ├── models/                  # DB entities + NLP/Whisper model wrappers
│   │   ├── nlp_model.py         # AcousticFluencyScorer & ConceptRelevanceScorer
│   │   └── whisper_model.py     # Whisper wrapper with technical jargon bias
│   ├── services/stt_service.py  # Audio extraction & transcription pipeline
│   └── utils/                   # Audio, NLP, report, and metrics utilities
├── frontend/                    # Next.js 14 TailAdmin Analytics Dashboard
│   ├── src/app/
│   │   ├── page.tsx             # Executive Overview
│   │   ├── assessment/page.tsx  # Video Assessment Studio
│   │   ├── candidates/page.tsx  # Candidate Directory
│   │   ├── rubrics/page.tsx     # React & Next.js Evaluation Rubrics
│   │   ├── analytics/page.tsx   # Speech & NLP Metrics Analytics
│   │   └── components/          # Sidebar, Header, Shell, Metrics, Modals
│   ├── tailwind.config.js       # TailAdmin design tokens and palettes
│   └── prisma/schema.prisma     # Prisma database schema definition
├── streamlit_frontend/          # Streamlit rapid evaluation prototype
│   └── streamlit_app.py         # Streamlit UI with React/Next.js role templates
├── scripts/                     # STT benchmarking and evaluation tools
├── docs/images/                 # Architecture diagram and high-res UI previews
├── requirements.txt             # Backend and ML dependencies
├── Dockerfile                   # Production-ready backend container
├── .env.example                 # Backend environment variable template
└── README.md
```

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- `ffmpeg` installed and available on system `PATH`
- PostgreSQL database instance
- Optional: Docker Desktop for containerized deployment

---

### 1. Backend API (FastAPI)

```bash
cd ai_interview_project
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 9001 --reload
```

*Swagger API Documentation is accessible at: `http://127.0.0.1:9001/docs`*

---

### 2. Next.js 14 TailAdmin Dashboard (Primary Portal)

```bash
cd ai_interview_project/frontend
cp .env.example .env
npm install
npx prisma generate
npm run dev
```

*Open your browser at: `http://localhost:3000`*

Available pages:
- **Overview:** `http://localhost:3000/`
- **Video Assessment:** `http://localhost:3000/assessment`
- **Candidate Directory:** `http://localhost:3000/candidates`
- **Evaluation Rubrics:** `http://localhost:3000/rubrics`
- **Speech & NLP Analytics:** `http://localhost:3000/analytics`

---

### 3. Streamlit Prototype (Companion Studio)

```bash
cd ai_interview_project
source .venv/bin/activate
streamlit run streamlit_frontend/streamlit_app.py
```

*Open your browser at: `http://localhost:8501`*

---

### 4. Docker Backend Deployment

```bash
cd ai_interview_project
docker build -t ai-interview-api .
docker run --rm -p 9001:9001 --env-file .env ai-interview-api
```

---

## API Endpoints

Swagger UI: `http://127.0.0.1:9001/docs`

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health status, GPU availability, and model readiness |
| `POST` | `/api/interviews/upload` | Upload video (`file`), `candidate_id`, `role`, and `expected_answer` |
| `GET` | `/api/interviews/result/{interview_id}` | Poll processing status or fetch full scoring report |

---

## Environment Variables

### Backend `.env`:

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | Yes | PostgreSQL connection string (`postgresql://user:pass@host:5432/dbname`) |
| `LOG_LEVEL` | No | Python logging level (default: `INFO`) |
| `WHISPER_MODEL_SIZE` | No | Whisper model size (default: `base`, options: `medium`, `small`, `tiny`) |
| `WHISPER_DEVICE` | No | Device target override (`cuda`, `mps`, or `cpu`) |

### Frontend `.env`:

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | Yes | PostgreSQL connection string used by Prisma |
| `NEXT_PUBLIC_API_BASE_URL` | No | FastAPI backend URL (default: `http://127.0.0.1:9001/api`) |

---

## STT Accuracy Validation

Benchmark Whisper Speech-to-Text accuracy against labeled audio samples:

```bash
cd ai_interview_project
python -m scripts.evaluate_stt --dataset-dir ./data/stt_eval --model-size medium
```

*The evaluation script verifies transcription Word Error Rate (WER) and outputs metric reports.*

---

## Portfolio Assets Checklist

- [x] `docs/images/architecture.png` — complete system technical blueprint.
- [x] `docs/images/nextjs-dashboard-overview.png` — executive overview with live metrics and recent evaluations.
- [x] `docs/images/nextjs-video-assessment.png` — video upload studio with level-based rubrics.
- [x] `docs/images/nextjs-rubrics.png` — React & Next.js qualification criteria & key concept checklists.
- [x] `docs/images/nextjs-nlp-analytics.png` — speech rate (WPM), bilingual filler words, and scoring breakdown.
- [x] `docs/images/nextjs-candidate-directory.png` — candidate directory & PostgreSQL records.
- [x] `docs/images/streamlit-upload.png` — Streamlit upload & setup screen.
- [x] `docs/images/streamlit-report.png` — Streamlit evaluation summary screen.
- [x] `docs/images/streamlit-detailed-report.png` — Streamlit detailed report/transcript view.

---

## GitHub About Suggestion

**Description:**
```text
Full-stack AI interview evaluation system using Whisper STT, acoustic speech rate analytics, concept-matching NLP, PostgreSQL, Next.js 14 TailAdmin dashboard, and Streamlit.
```

**Topics:**
```text
fastapi, whisper, nlp, speech-to-text, interview-analysis, nextjs, tailwindcss, tailadmin, streamlit, postgresql, sqlalchemy, prisma, machine-learning, ai-recruitment
```

---

## Roadmap

- [ ] Celery / Redis durable task queue for asynchronous batch processing.
- [ ] Integration of multi-speaker diarization for interviewer-candidate conversation separation.
- [ ] Role-Based Access Control (RBAC) & NextAuth integration for multi-tenant HR organizations.
- [ ] Cloud object storage integration (AWS S3 / Google Cloud Storage) for uploaded media artifacts.
- [ ] Automated end-to-end testing pipeline with GitHub Actions CI/CD.
