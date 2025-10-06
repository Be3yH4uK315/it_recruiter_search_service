import httpx
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from pymilvus import utility

from app.core.config import ELASTICSEARCH_URL, CANDIDATE_API_URL
from app.services.milvus_client import milvus_client, COLLECTION_NAME

CANDIDATE_INDEX = "candidates"

class Indexer:
    def __init__(self):
        self.es_client = Elasticsearch(ELASTICSEARCH_URL)
        self.candidate_api_url = CANDIDATE_API_URL
        self.milvus_collection = milvus_client.create_collection_if_not_exists()
        
        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        print("Model loaded successfully.")

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
                response = await client.get(f"{CANDIDATE_API_URL}/candidates/all")
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                print(f"Error fetching candidates: {e}")
                return []
            except httpx.HTTPStatusError as e:
                print(f"Error response {e.response.status_code}: {e.response.text}")
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

    def index_document(self, candidate_data: dict):
        doc_id = candidate_data["id"]
        body = self._format_candidate_for_es(candidate_data)
        self.es_client.index(index=CANDIDATE_INDEX, id=doc_id, document=body)
        print(f"Indexed/Updated document with ID: {doc_id}")

    def delete_document(self, candidate_id: str):
        try:
            self.es_client.delete(index=CANDIDATE_INDEX, id=candidate_id)
            print(f"Deleted document with ID: {candidate_id}")
        except Exception as e:
            print(f"Could not delete document {candidate_id}: {e}")

    def upsert_vector(self, candidate_data: dict):
        """Создает или обновляет вектор для одного кандидата в Milvus."""
        try:
            doc_id = candidate_data["id"]
            text_doc = self._create_candidate_document_for_ml(candidate_data)
            vector = self.model.encode(text_doc)
            self.milvus_collection.upsert([[doc_id], [vector]])
            print(f"Upserted vector for ID: {doc_id}")
        except Exception as e:
            print(f"Could not upsert vector for {doc_id}: {e}")

    def delete_vector(self, candidate_id: str):
        """Удаляет вектор из Milvus по ID."""
        try:
            expr = f'candidate_id in ["{candidate_id}"]'
            self.milvus_collection.delete(expr)
            print(f"Deleted vector with ID: {candidate_id}")
        except Exception as e:
            print(f"Could not delete vector {candidate_id}: {e}")

    async def run_full_reindex(self):
        print("Starting full re-indexation process...")
        if self.es_client.indices.exists(index=CANDIDATE_INDEX):
            self.es_client.indices.delete(index=CANDIDATE_INDEX)
        self.es_client.indices.create(index=CANDIDATE_INDEX)

        if milvus_client.has_collection():
            utility.drop_collection(COLLECTION_NAME)
        self.milvus_collection = milvus_client.create_collection_if_not_exists()

        candidates_data = await self._get_all_candidates()
        if not candidates_data:
            print("No candidates found to index.")
            return {"status": "success", "indexed": 0}

        print(f"Processing {len(candidates_data)} candidates for embedding and indexing...")
        
        es_actions = []
        milvus_ids = []
        milvus_vectors = []

        for candidate in candidates_data:
            text_doc = self._create_candidate_document_for_ml(candidate)
            vector = self.model.encode(text_doc)
            milvus_ids.append(candidate['id'])
            milvus_vectors.append(vector)
            es_actions.append(next(self._create_es_actions([candidate])))

        es_success, es_failed = helpers.bulk(self.es_client, es_actions)
        print(f"Elasticsearch: Successfully indexed: {es_success}, Failed: {es_failed}")

        milvus_client.insert_vectors(self.milvus_collection, milvus_ids, milvus_vectors)
        print(f"Milvus: Inserted {len(milvus_ids)} vectors.")
        
        return {"status": "success", "indexed": es_success, "failed": es_failed}

indexer = Indexer()

if __name__ == "__main__":
    import asyncio
    asyncio.run(indexer.run_full_reindex())