import asyncio
import logging
from functools import lru_cache
from collections import defaultdict
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from app.core.config import settings
from app.services.indexer import CANDIDATE_ALIAS
from app.services.milvus_client import milvus_client

logger = logging.getLogger(__name__)
RRF_K = getattr(settings, 'RRF_K', 60)

class SearchEngine:
    def __init__(self, model: SentenceTransformer):
        self.es_client = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
        self.es_index_name = CANDIDATE_ALIAS
        self.milvus_collection = milvus_client.create_collection_if_not_exists()
        self.model = model
        self.encode_query = lru_cache(maxsize=1024)(self.model.encode)
        self.executor = asyncio.get_event_loop().run_in_executor

    async def _filter_candidates_with_elasticsearch(self, filters: dict) -> List[Dict[str, float]]:
        """
        Возвращает список ID кандидатов, подходящих под точные критерии.
        """
        must_queries = []
        should_queries = []
        must_not_queries = []

        experience_range = {}
        if filters.get("experience_min"):
            experience_range["gte"] = filters["experience_min"]

        if filters.get("experience_max"):
            experience_range["lte"] = filters["experience_max"]

        if experience_range:
            must_queries.append({"range": {"experience_years": experience_range}})

        if filters.get("location"):
            must_queries.append({"match": {"location": {"query": filters["location"], "fuzziness": "AUTO"}}})

        if filters.get("must_skills"):
            for skill in filters["must_skills"]:
                must_queries.append({"match": {"skills": {"query": skill.lower(), "fuzziness": "AUTO"}}})
        
        if filters.get("nice_skills"):
            for skill in filters["nice_skills"]:
                should_queries.append({"match": {"skills": {"query": skill.lower(), "fuzziness": "AUTO"}}})

        if filters.get("work_modes"):
            must_queries.append({"terms": {"work_modes": filters["work_modes"]}})
        
        if filters.get("exclude_ids"):
            must_not_queries.append({"ids": {"values": filters["exclude_ids"]}})

        es_query = {
            "bool": {
                "must": must_queries,
                "should": should_queries,
                "must_not": must_not_queries
            }
        } if must_queries or should_queries or must_not_queries else {"match_all": {}}

        try:
            response = await self.es_client.search(
                index=self.es_index_name,
                query=es_query,
                size=500,
                _source=["id"]
            )
            
            ranked_candidates = [
                {"candidate_id": hit["_source"]["id"], "score": hit.get("_score", 0)}
                for hit in response["hits"]["hits"]
            ]
            ranked_candidates.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Elasticsearch filtered and ranked {len(ranked_candidates)} candidates.")
            return ranked_candidates
        except Exception as e:
            logger.error(f"Error during Elasticsearch filtering: {e}")
            return []

    def _rank_candidates_with_milvus(self, query_text: str, candidate_ids: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Ищет в Milvus среди отфильтрованных ID самые близкие по смыслу.
        """
        if not candidate_ids or not query_text:
            return []
            
        query_vector = self.encode_query(query_text)
        
        id_filter_expression = f"candidate_id in {candidate_ids}".replace("'", '"')

        logger.info(f"Searching in Milvus among {len(candidate_ids)} candidates for query: '{query_text}'")
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        results = self.milvus_collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=id_filter_expression
        )
        
        ranked_results = [{"candidate_id": hit.id, "score": hit.distance} for hit in results[0]]
        
        logger.info(f"Milvus returned {len(ranked_results)} ranked candidates.")
        return ranked_results

    async def hybrid_search(self, filters: dict) -> List[Dict[str, Any]]:
        """
        Основной метод гибридного поиска.
        """
        semantic_parts = []
        if filters.get("role"):
            semantic_parts.append(filters["role"])
        if filters.get("nice_skills"):
            semantic_parts.extend(filters["nice_skills"])

        semantic_query_text = ", ".join(semantic_parts)
        
        es_results = await self._filter_candidates_with_elasticsearch(filters)
        if not es_results:
            return []

        filtered_ids = [res['candidate_id'] for res in es_results]

        milvus_results = await asyncio.get_event_loop().run_in_executor(
            None, self._rank_candidates_with_milvus, semantic_query_text, filtered_ids
        )

        rrf_scores = defaultdict(float)
        for rank, doc in enumerate(es_results):
            rrf_scores[doc['candidate_id']] += 1 / (RRF_K + rank + 1)

        for rank, doc in enumerate(milvus_results):
            rrf_scores[doc["candidate_id"]] += 1 / (RRF_K + rank + 1)

        if not rrf_scores:
            return []
            
        final_results = [{"candidate_id": doc_id, "score": score} for doc_id, score in rrf_scores.items()]
        final_results.sort(key=lambda x: x['score'], reverse=True)

        return final_results
