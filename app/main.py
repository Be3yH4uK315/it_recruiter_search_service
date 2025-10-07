import logging
from fastapi import FastAPI
from app.api.v1.search import router as search_router
from app.services.consumer import consumer

logger = logging.getLogger(__name__)
app = FastAPI(title="Search Service")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    await consumer.connect()
    consumer.start_consuming()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    await consumer.close()

app.include_router(search_router, prefix="/v1/search")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Search Service"}
