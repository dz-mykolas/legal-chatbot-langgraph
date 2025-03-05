# src/legal_chatbot_langgraph/utils/tools.py
import os
from langchain_community.tools import tool
from qdrant_client import models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Then replace hardcoded values with environment variables
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    url=os.getenv("QDRANT_URL"),
    content_payload_key=os.getenv("QDRANT_CONTENT_KEY"),
)

query_cache = {}
@tool
def qdrant_retriever(
    query: str, 
    initial_k=int(os.getenv("INITIAL_K", "5")), 
    max_k=int(os.getenv("MAX_K", "20")), 
    min_score_threshold=float(os.getenv("MIN_SCORE_THRESHOLD", "0.60"))
):
    """Search Tauragė city documents with dynamic k based on similarity threshold."""
    if query in query_cache:
        print("Using cached results")
        return query_cache[query]
    
    retrieved_docs = []
    retrieved_ids = set()
    current_k = initial_k
    
    while current_k <= max_k:
        # Filter out previously retrieved documents
        filter_condition = None
        if retrieved_ids:
            filter_condition = models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="_id",
                        match=models.MatchAny(any=list(retrieved_ids))
                    )
                ]
            )
        
        # Get next batch of results
        results = vector_store.similarity_search_with_score(
            query=query,
            k=current_k,
            filter=filter_condition,
            score_threshold=min_score_threshold
        )
        
        print(f"Result: {results}")

        # Process and filter results
        for doc, score in results:
            if doc.metadata.get("_id") not in retrieved_ids:  # Skip if somehow we get duplicates
                retrieved_ids.add(doc.metadata.get("_id"))
                retrieved_docs.append((doc, score))
                print(f"Score: {score}, Content: {doc.page_content[:100]}")
        
        # Check if we've reached our threshold
        if retrieved_docs and retrieved_docs[-1][1] < min_score_threshold:
            break
            
        # Increase k for next iteration
        print(f"Current k: {current_k}, Increasing k by {initial_k}")
        current_k += initial_k
    
    # Sort by similarity score and prepare return value
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)
    retrieved_text = "\n".join(doc.page_content for doc, _ in retrieved_docs)
    query_cache[query] = retrieved_text
    
    return retrieved_text
