import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from pymilvus import (
    MilvusException, connections, utility,
    Collection, CollectionSchema, FieldSchema, DataType
)
from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "candidates_embeddings"
DIMENSION = 768
HOST = getattr(settings, 'MILVUS_HOST', 'localhost')
PORT = getattr(settings, 'MILVUS_PORT', '19530')
INDEX_PARAMS = getattr(settings, 'MILVUS_INDEX_PARAMS', {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
})

class MilvusClient:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self._connect_with_retry()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _connect_with_retry(self):
        connections.connect("default", host=self.host, port=self.port)
        logger.info(f"Connected to Milvus on {self.host}:{self.port}")

    def disconnect(self):
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")

    def has_collection(self):
        return utility.has_collection(COLLECTION_NAME)

    def create_collection_if_not_exists(self):
        if self.has_collection():
            logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
            collection = Collection(COLLECTION_NAME)
        else:
            candidate_id = FieldSchema(name="candidate_id", dtype=DataType.VARCHAR, is_primary=True, max_length=36)
            embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)

            schema = CollectionSchema(fields=[candidate_id, embedding], description="Candidate profile embeddings")
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            logger.info(f"Created new collection '{COLLECTION_NAME}'.")

        try:
            if not collection.indexes:
                logger.warning(f"No index found for collection '{COLLECTION_NAME}'. Creating one...")
                collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
                logger.info("Index created for embedding field.")
            else:
                logger.info(f"Index already exists for collection '{COLLECTION_NAME}'.")

            collection.load()
            logger.info(f"Collection '{COLLECTION_NAME}' loaded after creation/check.")
        except Exception as e:
            logger.error(f"Failed to create or check index: {e}")
            raise RuntimeError("Index creation failed")

        return collection

    def insert_vectors(self, collection: Collection, ids: list, vectors: list):
        if not ids or not vectors:
            return None

        try:
            logger.info(f"Inserting {len(ids)} vectors into Milvus collection...")
            mr = collection.insert([ids, vectors])
            collection.flush()

            try:
                progress = utility.loading_progress(COLLECTION_NAME)
                if progress.get('loading_progress', '0%') != '100%':
                    logger.info(f"Collection not fully loaded ({progress['loading_progress']}). Loading now...")
                    collection.load()
            except MilvusException as me:
                if me.code == 101:
                    logger.warning("Collection not loaded yet. Loading now...")
                    collection.load()
                else:
                    raise

            return mr
        except Exception as e:
            logger.error(f"Error during insert_vectors: {e}")
            raise

milvus_client = MilvusClient()