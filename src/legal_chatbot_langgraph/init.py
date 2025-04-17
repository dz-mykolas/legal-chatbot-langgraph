# init.py

import logging
import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

client = None
embeddings = None

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    # You might want to ping the client here to ensure connection
    client.get_collections() # Example check

    print("Qdrant client and embeddings initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize Qdrant client or embeddings: {e}")
    # Decide how to handle this - maybe exit or raise the exception
    raise RuntimeError(f"Qdrant initialization failed: {e}") from e
