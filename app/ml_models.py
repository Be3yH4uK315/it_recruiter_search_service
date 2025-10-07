import logging
from sentence_transformers import SentenceTransformer
from app.services.search_logic import SearchEngine
from app.services.indexer import Indexer
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading Sentence Transformer model...")
SENTENCE_MODEL = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
logger.info("Sentence Transformer model loaded successfully.")

search_engine_instance = SearchEngine(model=SENTENCE_MODEL)
indexer_instance = Indexer(
    model=SENTENCE_MODEL,
    candidate_api_url=settings.CANDIDATE_API_URL,
    es_url=settings.ELASTICSEARCH_URL
)

ML_MODELS = {
    "search_engine": search_engine_instance,
    "indexer": indexer_instance,
}

indexer = indexer_instance