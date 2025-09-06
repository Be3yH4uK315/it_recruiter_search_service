from fastapi import APIRouter, Body
from typing import Dict, Any
from app.services.search_logic import search_engine

router = APIRouter()

@router.post("/")
def search_candidates_endpoint(filters: Dict[str, Any] = Body(...)):
    results = search_engine.search_candidates(filters)
    return {"results": results}