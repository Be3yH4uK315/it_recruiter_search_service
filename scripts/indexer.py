import httpx
import asyncio
from elasticsearch import Elasticsearch, helpers
import os
from dotenv import load_dotenv

load_dotenv()

ELASTIC_URL = os.getenv("ELASTICSEARCH_URL")
CANDIDATE_API_URL = os.getenv("CANDIDATE_SERVICE_URL")
CANDIDATE_INDEX = "candidates"

es_client = Elasticsearch(ELASTIC_URL)

async def get_all_candidates():
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

def create_es_actions(candidates):
    for candidate in candidates:
        skills_list = [skill["skill"] for skill in candidate.get("skills", [])]
        yield {
            "_index": CANDIDATE_INDEX,
            "_id": candidate["id"],
            "_source": {
                "id": candidate["id"],
                "telegram_id": candidate["telegram_id"],
                "headline_role": candidate.get("headline_role"),
                "experience_years": candidate.get("experience_years"),
                "location": candidate.get("location"),
                "skills": skills_list,
            },
        }

async def main():
    print("Starting indexation process...")

    if not es_client.indices.exists(index=CANDIDATE_INDEX):
        print(f"Creating index '{CANDIDATE_INDEX}'...")
        es_client.indices.create(index=CANDIDATE_INDEX)

    candidates_data = await get_all_candidates()

    if not candidates_data:
        print("No candidates found to index. Exiting.")
        return

    print(f"Indexing {len(candidates_data)} candidates...")
    try:
        success, failed = helpers.bulk(es_client, create_es_actions(candidates_data))
        print(f"Successfully indexed: {success}, Failed: {failed}")
    except Exception as e:
        print(f"An error occurred during bulk indexing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
