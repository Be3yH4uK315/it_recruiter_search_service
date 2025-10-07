import httpx
import logging
from elasticsearch import AsyncElasticsearch, helpers
from sentence_transformers import SentenceTransformer
from pymilvus import utility

from app.services.milvus_client import milvus_client, COLLECTION_NAME

logger = logging.getLogger(__name__)

CANDIDATE_INDEX = "candidates"

class Indexer:
    def __init__(self, model: SentenceTransformer, candidate_api_url: str, es_url: str):
        self.es_client = AsyncElasticsearch(es_url)
        self.candidate_api_url = candidate_api_url
        self.milvus_collection = milvus_client.create_collection_if_not_exists()
        self.model = model

    def _format_candidate_for_es(self, candidate: dict) -> dict:
        skills_list = [skill["skill"].lower() for skill in candidate.get("skills", [])]
        work_modes_list = candidate.get("work_modes", [])

        return {
            "id": candidate["id"],
            "telegram_id": candidate["telegram_id"],
            "headline_role": candidate.get("headline_role"),
            "experience_years": candidate.get("experience_years"),
            "location": candidate.get("location"),
            "work_modes": work_modes_list,
            "skills": skills_list,
        }

    async def _get_all_candidates(self) -> list:
        async with httpx.AsyncClient(
                http2=False, trust_env=False, timeout=10.0
        ) as client:
            try:
                response = await client.get(f"{self.candidate_api_url}/candidates/all")
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"Error fetching candidates: {e}")
                return []
            except httpx.HTTPStatusError as e:
                logger.error(f"Error response {e.response.status_code}: {e.response.text}")
                return []

    def _create_candidate_document_for_ml(self, candidate: dict) -> str:
        """Собирает все важные текстовые поля кандидата в один документ для эмбеддинга."""
        skills = [skill['skill'] for skill in candidate.get('skills', [])]
        projects = [f"{p.get('title', '')}: {p.get('description', '')}" for p in candidate.get('projects', [])]
        
        text_parts = [
            candidate.get('headline_role', ''),
            candidate.get('headline_role', ''),
            candidate.get('headline_role', ''),
            candidate.get('display_name', ''),
            *skills,
            *projects
        ]
        return ". ".join(filter(None, text_parts))

    def _create_es_actions(self, candidates: list):
        for candidate in candidates:
            source_data = self._format_candidate_for_es(candidate)
            yield {
                "_index": CANDIDATE_INDEX,
                "_id": candidate["id"],
                "_source": source_data,
            }

    async def index_document(self, candidate_data: dict):
        doc_id = candidate_data["id"]
        body = self._format_candidate_for_es(candidate_data)
        await self.es_client.index(index=CANDIDATE_INDEX, id=doc_id, document=body)
        logger.info(f"Indexed/Updated document with ID: {doc_id}")

    async def delete_document(self, candidate_id: str):
        try:
            await self.es_client.delete(index=CANDIDATE_INDEX, id=candidate_id)
            logger.info(f"Deleted document with ID: {candidate_id}")
        except Exception as e:
            logger.error(f"Could not delete document {candidate_id}: {e}")

    def upsert_vector(self, candidate_data: dict):
        """Создает или обновляет вектор для одного кандидата в Milvus."""
        try:
            doc_id = candidate_data["id"]
            text_doc = self._create_candidate_document_for_ml(candidate_data)
            vector = self.model.encode(text_doc)
            self.milvus_collection.upsert([[doc_id], [vector]])
            logger.info(f"Upserted vector for ID: {doc_id}")
        except Exception as e:
            logger.error(f"Could not upsert vector for {doc_id}: {e}")

    def delete_vector(self, candidate_id: str):
        """Удаляет вектор из Milvus по ID."""
        try:
            expr = f'candidate_id in ["{candidate_id}"]'
            self.milvus_collection.delete(expr)
            logger.info(f"Deleted vector with ID: {candidate_id}")
        except Exception as e:
            logger.error(f"Could not delete vector {candidate_id}: {e}")

    async def run_full_reindex(self):
        logger.info("Starting full re-indexation process...")
        if await self.es_client.indices.exists(index=CANDIDATE_INDEX):
            await self.es_client.indices.delete(index=CANDIDATE_INDEX)
        await self.es_client.indices.create(index=CANDIDATE_INDEX)

        if milvus_client.has_collection():
            utility.drop_collection(COLLECTION_NAME)
        self.milvus_collection = milvus_client.create_collection_if_not_exists()

        candidates_data = await self._get_all_candidates()
        if not candidates_data:
            logger.warning("No candidates found to index.")
            return {"status": "success", "indexed": 0}

        logger.info(f"Processing {len(candidates_data)} candidates for embedding and indexing...")
        
        texts_for_ml = [self._create_candidate_document_for_ml(c) for c in candidates_data]
        vectors = self.model.encode(texts_for_ml, show_progress_bar=True)
        
        milvus_ids = [c['id'] for c in candidates_data]
        
        es_actions = self._create_es_actions(candidates_data)
        es_success, es_failed = await helpers.async_bulk(self.es_client, es_actions)
        logger.info(f"Elasticsearch: Successfully indexed: {es_success}, Failed: {es_failed}")

        milvus_client.insert_vectors(self.milvus_collection, milvus_ids, list(vectors))
        logger.info(f"Milvus: Inserted {len(milvus_ids)} vectors.")
        
        return {"status": "success", "indexed": es_success, "failed": es_failed}