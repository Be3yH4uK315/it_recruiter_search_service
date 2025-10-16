import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from elasticsearch import AsyncElasticsearch

from app.api.v1.search import router as search_router
from app.services.milvus_client import milvus_client
from app.services.consumer import consumer
from app.core.config import settings
from app.services.indexer import CANDIDATE_ALIAS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def ensure_es_alias_exists():
    """
    Проверяет при старте, существует ли алиас. Если нет, создает
    пустой индекс и направляет алиас на него.
    """
    es_client = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    try:
        alias_exists = await es_client.indices.exists_alias(name=CANDIDATE_ALIAS)
        if not alias_exists:
            logger.warning(f"Alias '{CANDIDATE_ALIAS}' not found. Creating initial index and alias.")
            initial_index = f"{CANDIDATE_ALIAS}-initial"
            
            if not await es_client.indices.exists(index=initial_index):
                 await es_client.indices.create(index=initial_index)

            await es_client.indices.put_alias(index=initial_index, name=CANDIDATE_ALIAS)
            logger.info(f"Successfully created alias '{CANDIDATE_ALIAS}' pointing to '{initial_index}'.")
    finally:
        await es_client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    await ensure_es_alias_exists()
    await consumer.connect()
    consumer.start_consuming()
    yield
    logger.info("Application shutdown...")
    await consumer.close()
    milvus_client.disconnect()

app = FastAPI(title="Search Service", lifespan=lifespan)

app.include_router(search_router, prefix="/v1/search")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Search Service"}

@app.get("/health")
async def health_check():
    try:
        es_client = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
        if not await es_client.ping():
            raise Exception("Elasticsearch down")
        if not milvus_client.has_collection():
            raise Exception("Milvus collection missing")
        if not await consumer.check_connection():
            raise Exception("RabbitMQ connection failed")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
    finally:
        await es_client.close()