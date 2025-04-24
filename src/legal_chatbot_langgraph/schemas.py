# legal_chatbot_langgraph/schemas.py

import json
from typing import Literal, TypedDict, Annotated, Sequence, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages

# --- Tool Argument Schemas ---

class RetrieveRelevantChunksArgs(BaseModel):
    query: str = Field(..., description="The search query string for finding relevant text chunks.")
    k: int = Field(5, description="Number of chunk results to retrieve.")

class RetrieveDocumentsByChunkIdsArgs(BaseModel):
    chunk_ids: List[str] = Field(..., description="List of chunk IDs whose parent documents/sections are sought.")
    k: int = Field(10, description="Maximum number of parent documents/sections to retrieve.")

class RetrieveChunksByIdsArgs(BaseModel):
    chunk_ids: List[str] = Field(..., description="A list of exact chunk IDs to retrieve.")

# --- Retriever Action Schemas ---

class ToolCallRequest(BaseModel):
    """Specifies a single tool call to be executed."""
    tool_name: Literal[
        "retrieve_relevant_chunks",
        "retrieve_documents_by_chunk_ids",
        "retrieve_chunks_by_ids"
    ] = Field(..., description="The name of the specific tool to call.")
    arguments: Dict[str, Any] = Field(..., description="The arguments dictionary for the chosen tool, matching its specific requirements.")

    @field_validator('arguments', mode='before')
    @classmethod
    def parse_arguments_string(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Arguments field is a string but not valid JSON: {value}")
        return value

class RetrieverAction(BaseModel):
    """
    The decision made by the retriever LLM, including which specific tool calls to perform
    or indicating that retrieval is complete.
    """
    retrieval_calls: List[ToolCallRequest] = Field(
        description="List of specific tool calls to perform. Leave empty if retrieval is complete.",
        default_factory=list
    )
    reasoning: str = Field(..., description="Brief explanation for the chosen tool calls or the decision to stop searching.")

# --- State Schemas ---

# Input Schema: What the API user MUST provide
class RetrievalStateInput(TypedDict):
    messages: Sequence[BaseMessage]

# Main Graph State
class RetrievalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_data: List[Dict[str, Any]]
    retriever_decision: Optional[RetrieverAction]

# Output Schema: What the graph returns
class RetrievalStateOutput(TypedDict):
    messages: Sequence[BaseMessage]
    # You could add other final fields if needed
    # retrieved_data: List[Dict[str, Any]]
    