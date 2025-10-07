from fastapi import APIRouter, BackgroundTasks, Depends
from app.ml_models import ML_MODELS, indexer
from app.services.search_logic import SearchEngine
from app.models.search import SearchFilters

router = APIRouter()

def get_search_engine() -> SearchEngine:
    return ML_MODELS["search_engine"]

@router.post("/")
async def search_candidates_endpoint(
    filters: SearchFilters,
    engine: SearchEngine = Depends(get_search_engine)
):
    filters_dict = filters.model_dump(exclude_none=True)
    results = await engine.hybrid_search(filters=filters_dict)
    return {"results": results}

@router.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    background_tasks.add_task(indexer.run_full_reindex)
    return {"message": "Full re-indexation process has been started in the background."}