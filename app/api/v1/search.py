from fastapi import APIRouter, BackgroundTasks
from app.services.search_logic import search_engine
from app.services.indexer import indexer
from app.models.search import SearchFilters

router = APIRouter()

@router.post("/")
def search_candidates_endpoint(filters: SearchFilters):
    results = search_engine.search_candidates(
        filters=filters.model_dump(exclude_none=True),
        exclude_ids=filters.exclude_ids
    )
    return {"results": results}

@router.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    background_tasks.add_task(indexer.run_full_reindex)
    return {"message": "Full re-indexation process has been started in the background."}