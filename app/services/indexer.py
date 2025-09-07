import httpx
from elasticsearch import Elasticsearch, helpers
from app.core.config import ELASTICSEARCH_URL, CANDIDATE_API_URL

CANDIDATE_INDEX = "candidates"

CANDIDATE_MAPPING = {
    "properties": {
        "id": {"type": "keyword"},
        "telegram_id": {"type": "long"},
        "headline_role": {"type": "text", "analyzer": "standard"},
        "experience_years": {"type": "float"},
        "location": {"type": "keyword"},
        "work_modes": {"type": "keyword"},
        "skills": {"type": "keyword"},
    }
}


class Indexer:
    def __init__(self):
        self.es_client = Elasticsearch(ELASTICSEARCH_URL)
        self.candidate_api_url = CANDIDATE_API_URL

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

    def _create_es_actions(self, candidates: list):
        for candidate in candidates:
            skills_list = [skill["skill"].lower() for skill in candidate.get("skills", [])]
            work_modes_list = candidate.get("work_modes", [])

            yield {
                "_index": CANDIDATE_INDEX,
                "_id": candidate["id"],
                "_source": {
                    "id": candidate["id"],
                    "telegram_id": candidate["telegram_id"],
                    "headline_role": candidate.get("headline_role"),
                    "experience_years": candidate.get("experience_years"),
                    "location": candidate.get("location"),
                    "work_modes": work_modes_list,
                    "skills": skills_list,
                },
            }

    async def run_full_reindex(self):
        print("Starting full re-indexation process...")
        if self.es_client.indices.exists(index=CANDIDATE_INDEX):
            print(f"Deleting old index '{CANDIDATE_INDEX}'...")
            self.es_client.indices.delete(index=CANDIDATE_INDEX)

        print(f"Creating new index '{CANDIDATE_INDEX}' with mapping...")
        self.es_client.indices.create(index=CANDIDATE_INDEX, mappings=CANDIDATE_MAPPING)

        candidates_data = await self._get_all_candidates()
        if not candidates_data:
            print("No candidates found to index.")
            return {"status": "success", "indexed": 0}

        print(f"Indexing {len(candidates_data)} candidates...")
        success, failed = helpers.bulk(self.es_client, self._create_es_actions(candidates_data))
        print(f"Successfully indexed: {success}, Failed: {failed}")
        return {"status": "success", "indexed": success, "failed": failed}

indexer = Indexer()