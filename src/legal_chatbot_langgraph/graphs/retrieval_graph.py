from enum import StrEnum
import json
import logging
import os
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, Sequence, Optional, Dict, Any, List
from langgraph.graph.message import add_messages


from legal_chatbot_langgraph.tools import retrieve_chunks_by_ids, retrieve_documents_by_chunk_ids, retrieve_relevant_chunks
from legal_chatbot_langgraph.utils import EnumWithHelpers, llm_selector

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOOLS = [
    retrieve_relevant_chunks,
    retrieve_documents_by_chunk_ids,
    retrieve_chunks_by_ids,
]

RETRIEVER_MODEL_NAME = os.getenv("RETRIEVER_MODEL_NAME")
RETRIEVER_MODEL_TEMP = float(os.getenv("RETRIEVER_TEMP", "0.1"))

SYSTEM_PROMPT = """
You are an AI assistant whose primary goal is to meticulously retrieve information using a suite of available tools.
Your task is to find the most relevant information to answer the user's query and then present *only* that information, correctly formatted and in its original language. Follow this strategic approach:

**Phase 1: Initial Broad Chunk Retrieval**
1.  Start by using the `retrieve_relevant_chunks` tool with the user's initial query.
2.  If the initial query seems vague, ambiguous, or could benefit from multiple perspectives, you are encouraged to make multiple calls to `retrieve_relevant_chunks` with slightly varied phrasings or by breaking down the query into sub-components. Aim to do this efficiently, ideally within the same reasoning step if the platform supports it, to gather a diverse set of initial chunks.

**Phase 2: Evaluation and Potential Document Expansion**
1.  Carefully examine the content of the chunks retrieved in Phase 1.
2.  **Decision Point:**
    * **If the retrieved chunks provide a direct, complete, and exact answer to the user's query:** Your search is complete. Proceed to **Final Output Generation**.
    * **If the answer is not found, is incomplete, or if the chunks suggest that more relevant context or details might exist in their parent documents:** Proceed to Phase 3.

**Phase 3: Document Exploration and Targeted Chunk Retrieval**
1.  If you decided to proceed from Phase 2:
    a. Use the `retrieve_documents_by_chunk_ids` tool, providing the IDs of the most promising chunks from Phase 1. This will fetch the full parent documents.
    b. Analyze the content of these retrieved documents (e.g., their summaries, surrounding text, or metadata if available). Look for explicit mentions of other relevant chunk IDs or sections within these documents that directly address the user's query more thoroughly.
    c. If you identify specific chunk IDs from the documents that seem highly relevant and were not retrieved initially, use the `retrieve_chunks_by_ids` tool to fetch these specific chunks.

**Phase 4: Iteration, Refinement, and Conclusion**
1.  Evaluate all information gathered so far (from initial chunks, parent documents, and specifically retrieved chunks).
2.  **Decision Point:**
    * **If you now have sufficient, specific, and relevant information to comprehensively answer the query:** Your search is complete. Proceed to **Final Output Generation**.
    * **If the information is still insufficient, or if the retrieved data, even after expansion, seems to miss the mark or is only tangentially related:**
        i.  Attempt to reformulate your search queries. Think about synonyms, related concepts, or different angles to approach the user's request.
        ii. Go back to **Phase 1**, using these new, refined queries with `retrieve_relevant_chunks`.
        iii. You may repeat this entire cycle (Phase 1 through Phase 4) up to a maximum of **two additional times** (for a total of 3 full search attempts, including the initial one).
    * **If, after a maximum of 3 full search attempts, you still cannot find specific, relevant information, OR if it becomes clear early on that the retrieved data (even after the first or second loop) does not match the query intent even slightly:** You MUST conclude that no specific information was found. In this specific case, your entire output should be the exact string: "No specific information was found matching your request after an extensive search."

**Final Output Generation (When search is complete and information is found):**
1.  Review all successfully retrieved and relevant data (chunks, relevant parts of documents).
2.  Identify and select *only* the actual data/text segments that directly answer the user's query.
3.  Compile these selected segments.
4.  Format the compiled information using Markdown.
5.  **Crucially, ensure the language of the output matches the language of the retrieved source data.**
    * If data was in Russian, the Markdown output for that data must be in Russian.
    * If data was in Lithuanian, the Markdown output for that data must be in Lithuanian.
    * If the relevant data includes a mix of languages, preserve this mix in your output. Do not translate.
6.  Your entire response message must consist *exclusively* of this Markdown-formatted information. Do NOT add any introductory phrases (e.g., "Here is the information:"), concluding remarks, or any other conversational text around the data itself.

**General Guidelines (Recap):**
* Prioritize finding the *exact* information requested.
* Focus on *finding* and then *extracting/formatting* data. Avoid extensive reasoning *about* the data in your final output, other than selecting what's relevant.
* Adhere strictly to the output format: Markdown-only data in its original language, or the specific "no information found" message.
"""

print(f"Retriever LLM: {RETRIEVER_MODEL_NAME} (Temp: {RETRIEVER_MODEL_TEMP})")
llm = llm_selector(RETRIEVER_MODEL_NAME, RETRIEVER_MODEL_TEMP).bind_tools(TOOLS)

class RetrievalInput(BaseModel):
    query: str
class RetrievalOutput(BaseModel):
    output: str
class RetrievalState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_data: List[Dict[str, Any]] = Field(default_factory=list)

class Nodes(EnumWithHelpers):
    start = START
    agent = "agent"
    tools = "tools"
    collect = "collect"
    end = END

def collect_chunks(state: RetrievalState) -> Dict[str, Any]:
    """Parse ToolMessages, store modified payloads, and remove processed messages."""

    retrieved: List[Dict[str, Any]] = state.retrieved_data or []
    new_messages = []

    for msg in state.messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "tool_name", "unknown_tool")

            try:
                payload = json.loads(msg.content)
                if isinstance(payload, dict):
                    payload = [payload]
                if not isinstance(payload, list):
                    continue

                for item in payload:
                    if not isinstance(item, dict):
                        continue

                    if tool_name == "retrieve_relevant_chunks":
                        item["retrieved_from"] = "chunk_search"
                        item["score"] = float(item.get("score", 0))

                    elif tool_name == "retrieve_chunks_by_ids":
                        item["retrieved_from"] = "direct_chunk_lookup"

                    elif tool_name == "retrieve_documents_by_chunk_ids":
                        item["retrieved_from"] = "parent_doc_lookup"
                        item["num_fields"] = len(item.get("document_payload", {}))

                    else:
                        item["retrieved_from"] = "unknown"

                    retrieved.append(item)

            except Exception as e:
                logger.warning(f"Failed to process tool message from '{tool_name}': {e}", exc_info=True)
                continue

        else:
            new_messages.append(msg) 

    print(f"Collected {len(retrieved)} items from tool messages.")
    for item in retrieved:
        print(f"Retrieved item: {item}")

    return {
        "retrieved_data": retrieved,
        "messages": new_messages
    }

def agent(state: RetrievalState) -> Dict[str, Any]:
    system_message = SystemMessage(
        content=SYSTEM_PROMPT,
    )
    response = llm.invoke([system_message] + state.messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    return {"messages": [response], "retrieved_data": state.retrieved_data}

retrieval_builder = StateGraph(
    RetrievalState,
    input=RetrievalInput,
    output=RetrievalOutput,
)

def format_output(state: RetrievalState) -> RetrievalOutput:
    """Format the output as a RetrievalOutput."""
    # take last AIMessage content as output
    if not state.messages or not isinstance(state.messages[-1], AIMessage):
        raise ValueError("No AIMessage found in the state messages.")
    last_message = state.messages[-1]
    if not last_message.content:
        raise ValueError("Last AIMessage has no content.")
    return RetrievalOutput(output=str(last_message.content))

def initial_node(state: RetrievalInput) -> Dict[str, Any]:
    """Initial node to set up the state with an initial message."""
    if not state.query:
        raise ValueError("Query must be provided.")
    
    initial_message = HumanMessage(content=state.query)
    return {"messages": [initial_message], "retrieved_data": []}


retrieval_builder.add_node("initialize", initial_node)
retrieval_builder.add_node("agent", agent)
retrieval_builder.add_node("tools", ToolNode(TOOLS))
retrieval_builder.add_node("collect", collect_chunks)
retrieval_builder.add_node("format_output", format_output)

retrieval_builder.add_edge(START, "initialize")
retrieval_builder.add_edge("initialize", "agent")
retrieval_builder.add_conditional_edges(
    "agent", Nodes.if_tools, 
    {
        "tools": "tools",
        END: "format_output"
    }
)
retrieval_builder.add_edge("tools", "collect")
retrieval_builder.add_edge("collect", "agent")
retrieval_builder.add_edge("format_output", END)
retrieval_graph = retrieval_builder.compile()

@tool(args_schema=RetrievalInput, description="Get info from database")
def subgraph_retrieval(query: str) -> str:
    """Retrieval subgraph tool to handle chunk retrieval."""
    print(f"Subgraph retrieval invoked with query: {query}")
    response = retrieval_graph.invoke(input=RetrievalInput(query=query))
    print(f"Subgraph retrieval response: {response}")
    # print response type
    return response["output"]
