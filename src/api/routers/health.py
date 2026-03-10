from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.api.model_loader import store

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok",
        models_loaded=store.models_status,
        version="1.0.0",
    )
