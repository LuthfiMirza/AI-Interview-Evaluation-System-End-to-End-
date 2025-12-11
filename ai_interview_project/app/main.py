"""FastAPI application entrypoint for the AI Interview Evaluation System."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routes.interview_routes import router as interview_router

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Interview Evaluation System",
        version="0.1.0",
        description="Automated evaluation of interview videos focusing on speech-to-text and NLP scoring.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    async def health() -> dict:
        return {"status": "ok"}

    @app.on_event("startup")
    def _initialize_database() -> None:
        init_db()

    app.include_router(interview_router, prefix="/api")

    return app


app = create_app()
