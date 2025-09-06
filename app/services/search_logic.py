from elasticsearch import Elasticsearch
from app.core.config import ELASTICSEARCH_URL


class SearchEngine:
    def __init__(self):
        self.es_client = Elasticsearch(ELASTICSEARCH_URL)
        self.index_name = "candidates"

    def search_candidates(self, filters: dict) -> list:

        must_queries = []
        should_queries = []

        if filters.get("role"):
            must_queries.append({"match": {"headline_role": filters["role"]}})
        if filters.get("experience_min"):
            must_queries.append({"range": {"experience_years": {"gte": filters["experience_min"]}}})

        if filters.get("must_skills"):
            for skill in filters["must_skills"]:
                must_queries.append({"match": {"skills": skill}})

        if filters.get("nice_skills"):
            for skill in filters["nice_skills"]:
                should_queries.append({"match": {"skills": skill}})

        query = {
            "bool": {
                "must": must_queries,
                "should": should_queries,
                "minimum_should_match": 0 if not should_queries else 1
            }
        }

        try:
            response = self.es_client.search(
                index=self.index_name,
                query=query,
                size=20
            )
            results = [
                {"candidate_id": hit["_source"]["id"], "score": hit["_score"]}
                for hit in response["hits"]["hits"]
            ]
            return sorted(results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            print(f"Error during Elasticsearch search: {e}")
            return []


search_engine = SearchEngine()