# AI Interview Evaluation System (End-to-End)

Automated evaluation pipeline that ingests recorded interview videos, transcribes speech with Whisper, analyses content with Transformer models, and consolidates the findings into an HR-friendly dashboard. Vision-based cheating detection is currently disabled to focus on STT + NLP quality.

---

## 🔧 Tech Stack

| Layer              | Tools / Frameworks                                   | Notes |
| ------------------ | ---------------------------------------------------- | ----- |
| Backend / API      | FastAPI, Uvicorn, SQLAlchemy (planned)               | Async job orchestration & REST endpoints |
| Speech-to-Text     | OpenAI Whisper (local)                               | 16 kHz mono audio pipeline |
| NLP Scoring        | HuggingFace Transformers, Sentence Transformers      | Fluency, relevance, summarisation |
| Vision / Cheating  | (Temporarily disabled)                               | Focus is on speech + NLP only |
| Storage            | PostgreSQL, MinIO/S3 (future)                        | Transcripts, scores, raw media |
| Dashboard          | Next.js 14 (App Router), Prisma ORM, React           | Real-time result visualisation |
| Tooling            | Docker, ffmpeg, python-dotenv, Typescript, ESLint    | Deployability & DX |

---

## 📁 Repository Layout

```
ai_interview_project/
├── app/                    # FastAPI application
│   ├── main.py             # Service factory + CORS + health
│   ├── routes/             # REST endpoints (upload/result)
│   ├── models/             # ML model wrappers (Whisper/NLP/YOLO)
│   ├── utils/              # Audio, NLP, vision, aggregation helpers
│   └── outputs/            # Generated transcripts (gitignored)
├── frontend/               # Next.js + Prisma dashboard
│   ├── prisma/schema.prisma
│   ├── src/app/page.tsx    # Interview overview table
│   └── src/lib/prisma.ts   # Singleton Prisma client
├── requirements.txt        # Backend dependencies
├── Dockerfile              # Backend container image
├── .env.example            # Sample environment configuration
└── README.md               # Project documentation
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (ships with npm)
- ffmpeg available on `$PATH`
- PostgreSQL 14+ (local instance or Docker)
- (Optional) Docker Desktop for container workflows

pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

### 2. Backend Setup (FastAPI)
```bash
cd ai_interview_project
cp .env.example .env          # adjust DB + storage creds
python3 -m venv .venv
source .venv/bin/activate
.\.venv310\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
uvicorn app.main:app --reload
```

Key endpoints (Swagger available at `http://localhost:8000/docs`):
- `GET /health` – service heartbeat
- `POST /api/interviews/upload` – multipart upload (`file`, optional `candidate_id`, optional `expected_answer`)
- `GET /api/interviews/result/{interview_id}` – poll for aggregated report

### 3. Frontend Setup (Next.js + Prisma)
```bash
cd frontend
cp .env.example .env          # set DATABASE_URL & NEXT_PUBLIC_API_BASE_URL
npm install
npx prisma migrate dev --name init
npx prisma generate
npm run dev
```

Open `http://localhost:3000` to explore the dashboard. It queries PostgreSQL for processed interviews and highlights the FastAPI Swagger link for quick navigation.

### 4. Docker Build (backend only)
```bash
docker build -t ai-interview-api .
docker run --rm -p 8000:8000 --env-file .env ai-interview-api
```

### 5. Streamlit Prototype (optional)
```bash
cd ai_interview_project
streamlit run streamlit_app.py
```

Set the FastAPI base URL in the sidebar (defaults to `http://localhost:8000/api`), upload a video, and the page will poll until the report is ready while displaying confidence and summary.

---

## 🎯 STT MVP & Accuracy Validation

### Performance tips (quality vs speed)
- Default `WHISPER_MODEL_SIZE=base.en` in `.env` balances accuracy and size; use `medium.en` for higher accuracy (slower) or `tiny.en` for fastest download.
- If you have a GPU, set `WHISPER_DEVICE=cuda` to accelerate Whisper.
- Summaries are disabled by default to reduce NLP latency; enable by calling `score_transcript(..., include_summary=True)` if needed.

The MVP now focuses on delivering ≥90 % accuracy for English speech-to-text via a noise-robust Whisper pipeline. Key components:

- SoX-based denoising (high-pass, low-pass, companding, normalization) before inference.
- Whisper `medium.en` checkpoint with beam-search + temperature 0 to reduce hallucinations.
- Confidence score derived from segment log-probabilities.

### 1. Prepare a labeled dataset
Create a folder containing audio clips (`.wav`, `.mp3`, `.m4a`) and matching ground-truth transcripts (`sample.wav` + `sample.txt`). Short, clean sentences (~30–90 s) across several speakers work best for smoke tests; expand with noisier data for robustness checks.

### 2. Run the evaluation harness
```bash
cd ai_interview_project
python -m scripts.evaluate_stt --dataset-dir ./data/stt_eval --model-size medium.en --device cuda
```

- The script prints per-sample accuracy plus aggregate stats (overall, median, min/max).
- It exits with a non-zero status if overall accuracy drops below 0.90, prompting you to inspect audio quality or adjust preprocessing/model size.
- Use `--limit N` for quick regression tests on a subset.

### 3. Operational tips
- For noisy sources, tweak `WHISPER_MODEL_SIZE`/`WHISPER_DEVICE` in `.env` and rerun.
- Keep evaluation datasets versioned to monitor regressions as you iterate on preprocessing or model checkpoints.

---

## 🧠 Data Pipeline Overview

1. **Upload Service** – receives interview video, persists temp media.
2. **Audio Extraction** – ffmpeg converts to 16 kHz mono WAV (`audio_utils.extract_audio`).
3. **Speech-to-Text** – Whisper generates transcription segments + confidences.
4. **NLP Scoring** – transformer embeddings evaluate relevance vs. expected answer, BERT classifies fluency, summariser creates HR digest.
5. **Aggregation Layer** – blend of verbal scores plus metadata; stored and exposed via API.
6. **Dashboard** – Prisma pulls scores for an overview table; future work includes charts and drill-downs.

---

## 🔐 Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (backend + frontend) |
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `S3_BUCKET_NAME` | Future S3/MinIO storage config |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | Development object storage |
| `LOG_LEVEL` | Python log level (`INFO` default) |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend base URL for FastAPI (e.g. `http://localhost:8000/api`) |

Store secrets in `.env` (backend) and `frontend/.env` (for local development); never commit these files.

---

## ✅ Development Checklist

- [ ] Start PostgreSQL locally (`localhost:5432`) with matching credentials.
- [ ] Apply Prisma migrations whenever the schema changes.
- [ ] Exercise the upload & result endpoints for regression testing.
- [ ] Record and attach sample interview videos for load testing.
- [ ] Configure logging and storage settings before production deployment.

---

## 🛣️ Roadmap & Ideas

- Integrate asynchronous task queue (Celery / Dramatiq / RQ) for large jobs.
- Persist transcripts & reports via SQLAlchemy models instead of in-memory store.
- Support language detection and multi-language scoring.
- Deploy Streamlit prototype or fully-fledged Next.js visualisations (charts, timelines).
- Add authentication & RBAC for HR stakeholders.
- Automate builds with CI/CD (GitHub Actions) and container registry pushes.

---

## 🤝 Contributing

1. Fork the repository and create a feature branch.
2. Run linting/tests where available (`pytest` for backend modules, `npm run lint` for frontend).
3. Submit a PR describing motivation, testing, and screenshots (if UI changes).

---

## 📄 License

This project inherits the license of the hosting repository. Ensure compliance with Whisper and HuggingFace model licenses when redistributing model weights or building commercial solutions.

---

Happy building! 🚀
