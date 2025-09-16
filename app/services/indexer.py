import httpx
from elasticsearch import Elasticsearch, helpers
from app.core.config import ELASTICSEARCH_URL, CANDIDATE_API_URL

CANDIDATE_INDEX = "candidates"

CANDIDATE_MAPPING_WITH_ANALYZER = {
    "settings": {
        "analysis": {
            "filter": {
                "english_stemmer_filter": {
                    "type": "stemmer",
                    "language": "english"
                },
                "synonym_filter": {
                    "type": "synonym_graph",
                    "synonyms": [
                        "js, javascript",
                        "c#, csharp",
                        "dev, developer, development",
                        "lead, manager, head"
                    ]
                }
            },
            "analyzer": {
                "custom_text_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "synonym_filter",
                        "english_stemmer_filter"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "telegram_id": {"type": "long"},
            "headline_role": {
                "type": "text",
                "analyzer": "custom_text_analyzer"
            },
            "experience_years": {"type": "float"},
            "location": {"type": "keyword"},
            "work_modes": {"type": "keyword"},
            "skills": {
                "type": "text",
                "analyzer": "custom_text_analyzer"
            },
        }
    }
}

class Indexer:
    def __init__(self):
        self.es_client = Elasticsearch(ELASTICSEARCH_URL)
        self.candidate_api_url = CANDIDATE_API_URL

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

    async def run_full_reindex(self):
        print("Starting full re-indexation process...")
        if self.es_client.indices.exists(index=CANDIDATE_INDEX):
            print(f"Deleting old index '{CANDIDATE_INDEX}'...")
            self.es_client.indices.delete(index=CANDIDATE_INDEX)

        print(f"Creating new index '{CANDIDATE_INDEX}' with mapping...")
        self.es_client.indices.create(
            index=CANDIDATE_INDEX,
            settings=CANDIDATE_MAPPING_WITH_ANALYZER["settings"],
            mappings=CANDIDATE_MAPPING_WITH_ANALYZER["mappings"]
        )
        candidates_data = await self._get_all_candidates()
        if not candidates_data:
            print("No candidates found to index.")
            return {"status": "success", "indexed": 0}

        print(f"Indexing {len(candidates_data)} candidates...")
        success, failed = helpers.bulk(self.es_client, self._create_es_actions(candidates_data))
        print(f"Successfully indexed: {success}, Failed: {failed}")
        return {"status": "success", "indexed": success, "failed": failed}

indexer = Indexer()

if __name__ == "__main__":
    import asyncio
    asyncio.run(indexer.run_full_reindex())