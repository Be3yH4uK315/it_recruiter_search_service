import logging
from sentence_transformers import SentenceTransformer
from app.services.search_logic import SearchEngine
from app.services.indexer import Indexer
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LazySentenceTransformer:
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.model_name = getattr(settings, 'SENTENCE_MODEL_NAME', model_name)
        self._model = None

    def __getattr__(self, name):
        if self._model is None:
            try:
                logger.info(f"Lazy loading Sentence Transformer model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                self._model.encode("Dummy text for warm-up.")
                logger.info("Model loaded and warmed up.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError("Model loading failed")
        return getattr(self._model, name)

SENTENCE_MODEL = LazySentenceTransformer()

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