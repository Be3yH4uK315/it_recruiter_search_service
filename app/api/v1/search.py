import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from app.ml_models import ML_MODELS, indexer
from app.services.search_logic import SearchEngine
from app.models.search import SearchFilters

logger = logging.getLogger(__name__)
router = APIRouter()

def get_search_engine() -> SearchEngine:
    return ML_MODELS["search_engine"]

@router.post("/")
async def search_candidates_endpoint(
    filters: SearchFilters,
    engine: SearchEngine = Depends(get_search_engine)
):
    try:
        logger.info(f"Search request with filters: {filters.model_dump()}")
        filters_dict = filters.model_dump(exclude_none=True)
        results = await engine.hybrid_search(filters=filters_dict)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    logger.info(f"Starting rebuild with task_id: {task_id}")
    background_tasks.add_task(indexer.run_full_reindex)
    return {"message": "Full re-indexation started", "task_id": task_id}