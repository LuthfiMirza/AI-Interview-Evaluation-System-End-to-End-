# AI Interview Evaluation System

Automated pipeline that scores interview videos by combining speech-to-text, NLP content analysis, and vision-based cheating detection. The system exposes a FastAPI backend and supports a dashboard prototype for HR teams.

## Features
- Video ingestion service that extracts audio (ffmpeg) and transcribes speech with Whisper.
- NLP scoring using Transformer models for relevance, fluency, and summarisation.
- Cheating detection leveraging YOLOv8 and Mediapipe-based gaze estimation.
- Aggregated reporting with configurable weighting and storage hooks.
- REST API endpoints for upload, result retrieval, and health checks.
- Container-ready deployment with Docker and requirements for optional Streamlit dashboard.

## Project Structure
```
ai_interview_project/
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── interview_routes.py
│   ├── utils/
│   │   ├── audio_utils.py
│   │   ├── nlp_utils.py
│   │   ├── vision_utils.py
│   │   └── report_utils.py
│   ├── models/
│   │   ├── whisper_model.py
│   │   ├── nlp_model.py
│   │   └── yolo_model.py
│   └── outputs/
│       └── transcripts/
├── frontend/
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## Getting Started
1. **Clone and configure**
   ```bash
   cp .env.example .env
   ```
   Update secrets and database credentials as required.

2. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Upload interview video**
   - Endpoint: `POST /api/interviews/upload`
   - Form fields:
     - `file`: interview video (mp4)
     - `candidate_id` (optional)
     - `expected_answer` (reference text for NLP relevance scoring)

5. **Check results**
   - Endpoint: `GET /api/interviews/result/{interview_id}`

## Docker
```bash
docker build -t ai-interview-api .
docker run -p 8000:8000 --env-file .env ai-interview-api
```

## Next.js Dashboard (Prisma)
The `frontend/` workspace hosts a Next.js 14 dashboard that reads interview metrics from PostgreSQL using Prisma.

1. **Install dependencies**
   ```bash
   cd frontend
   cp .env.example .env
   npm install
   ```
2. **Run migrations and generate the Prisma client**
   ```bash
   npx prisma migrate dev --name init
   npx prisma generate
   ```
3. **Start the dashboard**
   ```bash
   npm run dev
   ```

Visit `http://localhost:3000` to review scores, cheating indicators, and summaries once the FastAPI backend has persisted results. The page highlights the API base URL so you can jump to Swagger docs quickly.

## Future Enhancements
- Persist interview metadata and reports using PostgreSQL via SQLAlchemy.
- Integrate MinIO/S3 storage for raw media and generated artefacts.
- Add authentication, rate limiting, and job queue for large-scale processing.
- Expand KPI visualisations and HR workflow tooling in the frontend.

## Next Steps Checklist
- `pip install -r requirements.txt` (or build the Docker image) to install backend dependencies.
- Launch the API with `uvicorn app.main:app --reload` and test `/api/interviews/upload` plus `/api/interviews/result/{id}`.
- Populate PostgreSQL via the processing pipeline, then run `npm run dev` inside `frontend/` to explore the Prisma-powered Next.js dashboard.
