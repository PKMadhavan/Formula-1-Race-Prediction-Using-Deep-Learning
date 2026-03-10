"""
FastAPI application entry point.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_loader import store
from src.api.routers import health, predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup
    store.load_all()
    yield
    # (cleanup on shutdown if needed)


app = FastAPI(
    title="F1 Race Predictor API",
    description="Live prediction API for lap times, pit stops, and final positions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "F1 Race Predictor API",
        "docs": "/docs",
        "health": "/health",
    }
