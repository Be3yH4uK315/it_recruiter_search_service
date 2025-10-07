import logging
from pymilvus import (
    connections, utility,
    Collection, CollectionSchema, FieldSchema, DataType
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "candidates_embeddings"
DIMENSION = 768
INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

class MilvusClient:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        connections.connect("default", host=self.host, port=self.port)
        logger.info(f"Connected to Milvus on {host}:{port}")

    def has_collection(self):
        return utility.has_collection(COLLECTION_NAME)

    def create_collection_if_not_exists(self):
        if self.has_collection():
            logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
            return Collection(COLLECTION_NAME)

        candidate_id = FieldSchema(name="candidate_id", dtype=DataType.VARCHAR, is_primary=True, max_length=36)
        embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)

        schema = CollectionSchema(fields=[candidate_id, embedding], description="Candidate profile embeddings")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        logging(f"Creating collection '{COLLECTION_NAME}'...")
        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
        logger.info("Index created for embedding field.")
        return collection

    def insert_vectors(self, collection: Collection, ids: list, vectors: list):
        if not ids or not vectors:
            return None

        logger.info(f"Inserting {len(ids)} vectors into Milvus collection...")
        mr = collection.insert([ids, vectors])
        collection.load()
        return mr

milvus_client = MilvusClient()