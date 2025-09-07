import os
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
CANDIDATE_API_URL = os.getenv("CANDIDATE_SERVICE_URL")