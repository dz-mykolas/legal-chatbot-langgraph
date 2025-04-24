import json
from typing import Any, Dict, Literal, Optional
from langchain_core.tools import tool
from qdrant_client import models
from legal_chatbot_langgraph.qdrant_config import logger, client, embeddings
import json
from typing import Literal, Optional, Dict, Any, List
from langchain_core.tools import tool
from qdrant_client import models

@tool
def retrieve_relevant_chunks(query: str, k: int = 5) -> str:
    """
    Searches the 'my_chunks' collection based on a text query using qdrant-client.
    Returns a list of chunk details, including their IDs, content, score,
    and full metadata payload.

    Args:
        query: The search query string.
        k: The number of chunk results to retrieve.
    """
    collection_name = "my_chunks"
    logger.info(f"--- Tool Call: retrieve_relevant_chunks ---")
    logger.info(f"Query: {query}")
    logger.info(f"K: {k}")
    logger.info(f"---------------------------------------------")

    # Ensure client and embeddings are available
    if not client or not embeddings:
         error_msg = "Qdrant client or embeddings not available for tool execution."
         logger.error(error_msg)
         return json.dumps([{"error": error_msg}])

    try:
        # 1. Embed the search query
        logger.info("Embedding the search query for chunks...")
        try:
            query_vector = embeddings.embed_query(query)
            logger.info("Chunk query embedding successful.")
        except Exception as e:
            logger.error(f"Failed to embed query for chunks: {e}", exc_info=True)
            return json.dumps([{"error": f"Failed to embed query for chunks: {e}"}])

        # 2. Perform the search using qdrant-client in 'my_chunks'
        logger.info(f"Performing search in '{collection_name}' using qdrant-client...")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            # No filter needed for initial chunk search usually
            query_filter=None,
            limit=k,
            with_payload=True,  # Get the payload
            with_vectors=False
        )
        logger.info(f"Chunk search completed via qdrant-client. Found {len(search_result)} results.")

        # 3. Handle case where no chunks are found
        if not search_result:
            info_msg = f"No chunks found via qdrant-client in '{collection_name}' for query '{query}'."
            logger.info(info_msg)
            return json.dumps([{"info": info_msg}])

        # 4. Structure the results
        structured_results = []
        for i, hit in enumerate(search_result):
            payload = hit.payload or {}
            structured_results.append({
                "chunk_id": hit.id, # Return the specific chunk ID
                "score": float(hit.score),
                "content": payload.get("content", ""), # Extract content
                "chunk_metadata": payload # Full payload of the chunk
            })
            logger.debug(f"Processed chunk hit {i+1}: score={hit.score:.4f}, id={hit.id}")

        logger.info(f"Successfully retrieved and formatted {len(structured_results)} chunks.")
        return json.dumps(structured_results, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"Error during chunk retrieval: {e}"
        logger.error(error_msg, exc_info=True)
        return json.dumps([{"error": error_msg}])

@tool
def retrieve_documents_by_chunk_ids(
    chunk_ids: List[str],
    k: int = 10
    ) -> str:
    """
    Retrieves documents or sections from the specified collection that contain
    ANY of the provided chunk_ids in their payload field (expected key: 'chunk_ids').
    Uses a qdrant-client scroll filter.

    Args:
        chunk_ids: A list of chunk IDs found from the retrieve_relevant_chunks tool.
        k: Maximum number of parent documents/sections to retrieve.
    """
    collection_name = "my_documents"

    logger.info(f"--- Tool Call: retrieve_documents_by_chunk_ids ---")
    logger.info(f"Chunk IDs to find parents for: {chunk_ids}")
    logger.info(f"Target Document Collection: {collection_name}")
    logger.info(f"Max parent results (k): {k}")
    logger.info(f"---------------------------------------------")

    # Ensure client is available (embeddings not needed here)
    if not client:
         error_msg = "Qdrant client not available for tool execution."
         logger.error(error_msg)
         return json.dumps([{"error": error_msg}])

    if not chunk_ids:
        logger.warning("No chunk IDs provided to retrieve_documents_by_chunk_ids.")
        return json.dumps([{"info": "No chunk IDs provided."}])

    try:
        # 1. Construct the filter to find parents
        #    -> ASSUMPTION: Parent docs/sections have a field named 'chunk_ids'
        #       containing a list of their child chunk IDs. Adjust key if needed.
        parent_filter = models.Filter(
            should=[ # Find docs/sections matching ANY of the chunk IDs
                models.FieldCondition(
                    key="all_chunk_ids",
                    match=models.MatchAny(any=chunk_ids)
                )
            ]
        )
        logger.info(f"Constructed parent filter: {parent_filter}")

        # 2. Use scroll API to retrieve all matching parent items up to limit k
        #    We use scroll because we're filtering, not ranking by similarity.
        logger.info(f"Scrolling '{collection_name}' to find parents...")
        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=parent_filter,
            limit=k, # Apply limit
            with_payload=True,
            with_vectors=False
        )
        # scroll_result is List[models.Record]
        logger.info(f"Parent document scroll completed. Found {len(scroll_result)} results.")

        # 3. Handle case where no matching parents are found
        if not scroll_result:
             info_msg = f"No matching document found in '{collection_name}' for the provided chunk IDs."
             logger.info(info_msg)
             return json.dumps([{"info": info_msg}])

        # 4. Structure the results
        structured_results = []
        for i, record in enumerate(scroll_result):
            # record is models.Record (id, payload, vector)
            payload = record.payload or {}
            structured_results.append({
                "document_id": record.id, # ID of the doc/section itself
                "document_collection": collection_name,
                "document_payload": payload # Full payload of the doc/section
            })
            logger.debug(f"Processed documents record {i+1}: id={record.id}")

        logger.info(f"Successfully retrieved and formatted {len(structured_results)} document items from {collection_name}.")
        return json.dumps(structured_results, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"Error during documents retrieval by chunk IDs: {e}"
        logger.error(error_msg, exc_info=True)
        return json.dumps([{"error": error_msg}])
    
@tool
def retrieve_chunks_by_ids(chunk_ids: List[str]) -> str:
    """
    Retrieves specific chunks from the 'my_chunks' collection by their exact IDs.
    Uses the qdrant-client retrieve method. Returns full chunk details.

    Args:
        chunk_ids: A list of chunk IDs to retrieve.
    """
    collection_name = "my_chunks" # This tool specifically targets the chunks collection
    logger.info(f"--- Tool Call: retrieve_chunks_by_ids ---")
    logger.info(f"Chunk IDs to retrieve: {chunk_ids}")
    logger.info(f"Target Collection: {collection_name}")
    logger.info(f"-----------------------------------------")

    # Ensure client is available (embeddings not needed)
    if not client:
         error_msg = "Qdrant client not available for tool execution."
         logger.error(error_msg)
         return json.dumps([{"error": error_msg}])

    if not chunk_ids:
        logger.warning("No chunk IDs provided to retrieve_chunks_by_ids.")
        return json.dumps([{"info": "No chunk IDs provided."}])

    try:
        # 1. Use the retrieve API to fetch points by ID
        logger.info(f"Retrieving chunks by ID from '{collection_name}' using qdrant-client...")
        # Note: client.retrieve returns a list of models.Record
        # It will only return points that actually exist for the given IDs.
        retrieved_records = client.retrieve(
            collection_name=collection_name,
            ids=chunk_ids,
            with_payload=True,  # Get the payload
            with_vectors=False # Usually vectors aren't needed here
        )
        logger.info(f"Chunk retrieval by ID completed. Found {len(retrieved_records)} matching chunks.")

        # 2. Handle case where no chunks are found for the given IDs
        if not retrieved_records:
             info_msg = f"No chunks found in '{collection_name}' for the provided IDs: {chunk_ids}."
             logger.info(info_msg)
             return json.dumps([{"info": info_msg}])

        # 3. Structure the results (similar to retrieve_relevant_chunks, but no score)
        structured_results = []
        for i, record in enumerate(retrieved_records):
            # record is models.Record (id, payload, vector - if requested)
            payload = record.payload or {}
            structured_results.append({
                "chunk_id": record.id, # Return the specific chunk ID
                # No score is returned by client.retrieve
                "content": payload.get("content", ""), # Extract content
                "chunk_metadata": payload # Full payload of the chunk
            })
            logger.debug(f"Processed retrieved chunk {i+1}: id={record.id}")

        logger.info(f"Successfully retrieved and formatted {len(structured_results)} chunks by ID.")
        return json.dumps(structured_results, ensure_ascii=False, indent=2)

    except Exception as e:
        # Catch potential errors from qdrant-client or elsewhere
        error_msg = f"Error during chunk retrieval by IDs: {e}"
        logger.error(error_msg, exc_info=True)
        return json.dumps([{"error": error_msg}])
    