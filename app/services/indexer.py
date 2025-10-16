import asyncio
import time
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from elasticsearch import AsyncElasticsearch, NotFoundError, helpers
from sentence_transformers import SentenceTransformer
from pymilvus import utility

from app.core.config import settings
from app.services.milvus_client import milvus_client, COLLECTION_NAME

logger = logging.getLogger(__name__)

CANDIDATE_ALIAS = getattr(settings, 'CANDIDATE_ALIAS', 'candidates')
BATCH_SIZE = getattr(settings, "BATCH_SIZE", 500)

class Indexer:
    def __init__(self, model: SentenceTransformer, candidate_api_url: str, es_url: str):
        self.es_client = AsyncElasticsearch(es_url)
        self.candidate_api_url = candidate_api_url
        self.milvus_collection = milvus_client.create_collection_if_not_exists()
        self.model = model
        self.executor = asyncio.get_event_loop().run_in_executor

    def _format_candidate_for_es(self, candidate: dict) -> dict:
        if "id" not in candidate:  # IMPROVED: Валидация
            raise ValueError("Candidate data missing 'id'")
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_candidates_batch(self, limit: int, offset: int) -> list:
        url = f"{self.candidate_api_url}/candidates/?limit={limit}&offset={offset}"
        async with httpx.AsyncClient(http2=False, trust_env=False, timeout=20.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"Error fetching candidates batch: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Error response {e.response.status_code}: {e.response.text}")
                raise

    def _create_candidate_document_for_ml(self, candidate: dict) -> str:
        """Создает структурированный документ с префиксами для лучшего понимания моделью."""
        text_parts = []
        
        if role := candidate.get('headline_role'):
            text_parts.append(f"Должность: {role}")
            
        if skills := [skill['skill'] for skill in candidate.get('skills', [])]:
            text_parts.append(f"Навыки: {', '.join(skills)}")
            
        if projects := candidate.get('projects', []):
            projects_text = ". ".join([f"{p.get('title', '')}: {p.get('description', '')}" for p in projects])
            text_parts.append(f"Проекты: {projects_text}")
        
        if experiences := candidate.get('experiences', []):
            exp_text = ". ".join([f"{exp.get('position', '')} в {exp.get('company', '')}: {exp.get('responsibilities', '')}" for exp in experiences])
            text_parts.append(f"Опыт: {exp_text}")

        return ". ".join(filter(None, text_parts)) + "."

    def _create_es_actions(self, candidates: list, index_name: str):
        for candidate in candidates:
            source_data = self._format_candidate_for_es(candidate)
            yield {"_index": index_name, "_id": candidate["id"], "_source": source_data}

    async def index_document(self, candidate_data: dict):
        doc_id = candidate_data["id"]
        body = self._format_candidate_for_es(candidate_data)
        await self.es_client.index(index=CANDIDATE_ALIAS, id=doc_id, document=body)
        logger.info(f"Indexed/Updated document with ID: {doc_id} via alias")

    async def delete_document(self, candidate_id: str):
        try:
            await self.es_client.delete_by_query(index=CANDIDATE_ALIAS, query={"term": {"_id": candidate_id}})
            logger.info(f"Deleted document with ID: {candidate_id} via alias")
        except Exception as e:
            logger.error(f"Could not delete document {candidate_id}: {e}")

    async def upsert_vector(self, candidate_data: dict):
        try:
            doc_id = candidate_data["id"]
            text_doc = self._create_candidate_document_for_ml(candidate_data)
            vector = await self.executor(None, self.model.encode, text_doc)
            exists = self.milvus_collection.query(expr=f'candidate_id == "{doc_id}"', output_fields=["candidate_id"])
            if exists:
                logger.info(f"Updating existing vector for {doc_id}")
            await self.executor(None, self.milvus_collection.upsert, [[doc_id], [vector.tolist()]])
            self.milvus_collection.flush()
            logger.info(f"Upserted vector for ID: {doc_id}")
        except Exception as e:
            logger.error(f"Could not upsert vector for {doc_id}: {e}")

    async def delete_vector(self, candidate_id: str):
        try:
            expr = f'candidate_id in ["{candidate_id}"]'
            await self.executor(None, self.milvus_collection.delete, expr)
            self.milvus_collection.flush()
            logger.info(f"Deleted vector with ID: {candidate_id}")
        except Exception as e:
            logger.error(f"Could not delete vector {candidate_id}: {e}")

    async def run_full_reindex(self):
        """
        Переиндексация без простоя с использованием алиасов.
        """
        logger.info("Starting zero-downtime re-indexation process...")
        
        new_index_name = f"{CANDIDATE_ALIAS}-{int(time.time())}"
        logger.info(f"Creating new index: {new_index_name}")
        await self.es_client.indices.create(index=new_index_name)

        offset = 0
        total_indexed = 0
        while True:
            candidates_batch = await self._get_candidates_batch(limit=BATCH_SIZE, offset=offset)
            if not candidates_batch:
                break
            
            texts_for_ml = [self._create_candidate_document_for_ml(c) for c in candidates_batch]
            vectors = await self.executor(None, self.model.encode, texts_for_ml)
            
            es_actions = self._create_es_actions(candidates_batch, new_index_name)
            es_success, es_failed = await helpers.async_bulk(self.es_client, es_actions)

            if offset == 0:
                 if milvus_client.has_collection():
                    utility.drop_collection(COLLECTION_NAME)
                 self.milvus_collection = milvus_client.create_collection_if_not_exists()
            
            milvus_ids = [c['id'] for c in candidates_batch]
            await self.executor(None, milvus_client.insert_vectors, self.milvus_collection, milvus_ids, list(vectors))

            total_indexed += es_success
            offset += BATCH_SIZE
            logger.info(f"Batch processed. Total indexed so far: {total_indexed}")

        logger.info(f"Successfully indexed {total_indexed} documents into '{new_index_name}'.")

        logger.info(f"Switching alias '{CANDIDATE_ALIAS}' to point to '{new_index_name}'")
        try:
            old_indices = await self.es_client.indices.get_alias(name=CANDIDATE_ALIAS)
        except NotFoundError:
            old_indices = {}

        actions = {"actions": [{"add": {"index": new_index_name, "alias": CANDIDATE_ALIAS}}]}
        for old_index in old_indices.keys():
            actions["actions"].append({"remove": {"index": old_index, "alias": CANDIDATE_ALIAS}})
        
        await self.es_client.indices.update_aliases(body=actions)
        logger.info("Alias switched successfully.")

        for old_index in old_indices.keys():
            logger.info(f"Deleting old index: {old_index}")
            await self.es_client.indices.delete(index=old_index)

        return {"status": "success", "total_indexed": total_indexed, "active_index": new_index_name}
