# AI Interview Evaluation System (End-to-End)

Automated evaluation pipeline that ingests recorded interview videos, transcribes speech with Whisper, analyses content with Transformer models, observes non-verbal behaviour using YOLOv8 + Mediapipe, detects cheating cues, and consolidates the findings into an HR-friendly dashboard.

---

## 🔧 Tech Stack

| Layer              | Tools / Frameworks                                   | Notes |
| ------------------ | ---------------------------------------------------- | ----- |
| Backend / API      | FastAPI, Uvicorn, SQLAlchemy (planned)               | Async job orchestration & REST endpoints |
| Speech-to-Text     | OpenAI Whisper (local)                               | 16 kHz mono audio pipeline |
| NLP Scoring        | HuggingFace Transformers, Sentence Transformers      | Fluency, relevance, summarisation |
| Vision / Cheating  | Ultralytics YOLOv8, Mediapipe, OpenCV                | Eye contact, phone detection, multi-person |
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

### 2. Backend Setup (FastAPI)
```bash
cd ai_interview_project
cp .env.example .env          # adjust DB + storage creds
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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

---

## 🧠 Data Pipeline Overview

1. **Upload Service** – receives interview video, persists temp media.
2. **Audio Extraction** – ffmpeg converts to 16 kHz mono WAV (`audio_utils.extract_audio`).
3. **Speech-to-Text** – Whisper generates transcription segments + confidences.
4. **NLP Scoring** – transformer embeddings evaluate relevance vs. expected answer, BERT classifies fluency, summariser creates HR digest.
5. **Vision Analysis** – YOLOv8 detects people/phones, Mediapipe estimates eye contact; metrics converted into cheating score.
6. **Aggregation Layer** – weighted blend of verbal/non-verbal scores plus metadata; stored and exposed via API.
7. **Dashboard** – Prisma pulls scores for an overview table; future work includes charts and drill-downs.

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
- [ ] Confirm YOLO model weights (`yolov8n.pt` by default) are reachable.
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

This project inherits the license of the hosting repository. Ensure compliance with Whisper, YOLOv8, and HuggingFace model licenses when redistributing model weights or building commercial solutions.

---

Happy building! 🚀
