from elasticsearch import Elasticsearch
from app.core.config import ELASTICSEARCH_URL
from typing import List


class SearchEngine:
    def __init__(self):
        self.es_client = Elasticsearch(ELASTICSEARCH_URL)
        self.index_name = "candidates"

    def search_candidates(self, filters: dict, exclude_ids: List[str] = None) -> list:
        must_queries = []
        should_queries = []
        must_not_queries = []

        if filters.get("role"):
            must_queries.append({
                "match": {
                    "headline_role": {
                        "query": filters["role"],
                        "boost": 2.0,
                        "fuzziness": "AUTO"
                    }
                }
            })

        experience_range = {}
        if filters.get("experience_min"):
            experience_range["gte"] = filters["experience_min"]
        if filters.get("experience_max"):
            experience_range["lte"] = filters["experience_max"]
        if experience_range:
            must_queries.append({"range": {"experience_years": experience_range}})

        if filters.get("location"):
            must_queries.append({"match": {"location": filters["location"]}})

        if filters.get("work_modes"):
            must_queries.append({"terms": {"work_modes": filters["work_modes"]}})

        if filters.get("must_skills"):
            for skill in filters["must_skills"]:
                must_queries.append({"term": {"skills": skill.lower()}})

        if filters.get("nice_skills"):
            for skill in filters["nice_skills"]:
                should_queries.append({"term": {"skills": {"value": skill.lower(), "boost": 1.5}}})

        if exclude_ids:
            must_not_queries.append({"ids": {"values": exclude_ids}})

        query = {
            "bool": {
                "must": must_queries,
                "should": should_queries,
                "must_not": must_not_queries,
                "minimum_should_match": 0
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