import json
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Literal, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Define state with messages and a flag for completion
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    complete: bool = False

# Then replace hardcoded values with environment variables
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    url=os.getenv("QDRANT_URL"),
    content_payload_key=os.getenv("QDRANT_CONTENT_KEY"),
)

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

query_cache = {}
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"), 
    temperature=float(os.getenv("LLM_TEMPERATURE", "0"))
).bind_tools([qdrant_retriever])

def call_model(state: AgentState):
    messages = state["messages"]

    system_message = SystemMessage(content="""
    You are a meticulous Tauragė city assistant that exhaustively explores all aspects of user queries. When answering questions. Don't hesitate to ask user for more details if you feel like you need them, but if the first question is clear enough, then do the search first:

    **Enhanced Reasoning Process:**
    1. Initial Exploration:
    - Start with 2-3 broad queries covering different angles of the request
    - Analyze results for mentioned organizations, services, or programs

    2. Recursive Deep Dive:
    - For every mentioned entity/service, generate follow-up queries (these are examples, not exhaustive, in different contexts it would have different queries):
        * "[Organization Name] finansinė parama"
        * "[Service Name] reikalavimai"
        * "[Program] finansavimo suma"
    - Specifically look for numerical data, eligibility criteria, and temporal aspects

    3. Contextual Verification:
    - If documents mention support types (financial, legal), verify amounts/durations
    - Check multiple sources for conflicting information
    - Cross-reference contact details with official directories
    
    **Tool Call Strategy:**
    - Again, these are just examples and not exhaustive, the example is provided in support/aid context. The actual queries would depend on the context of the user query and the information available in the documents.
    - First Iteration Example:
    {"tool_calls": [
        {"name": "qdrant_retriever", "args": {"query": "mokymosi stipendijos studentams 2024"}},
        {"name": "qdrant_retriever", "args": {"query": "socialinė pagalba jaunimui Tauragė"}},
        {"name": "qdrant_retriever", "args": {"query": "jaunimo centras finansavimas"}}
    ]}
    - Subsequent Iteration Example (after finding some kind of institution or program):
    {"tool_calls": [
        {"name": "qdrant_retriever", "args": {"query": "Tauragės jaunimo draugija parama"}},
        {"name": "qdrant_retriever", "args": {"query": "V. Kudirkos g. 9 paramos formos jaunimui"}}
    ]}

    **Enhanced Answer Requirements:**
    - Always include numerical data if available (sums, percentages, durations)
    - List all possible support channels even if user didn't explicitly ask
    - Structure information hierarchically:
    1. Immediate financial support
    2. Long-term assistance programs
    3. Specialized services
    4. Community organizations

    **Example Final Output:**
    {
    "answer": "Tauragėje jaunimas ir studentai gali gauti:<br>
    **1. Finansinė parama:**<br>
    - Jaunimo reikalų departamentas: Iki 250€/mėn studijų stipendija (2024m. duomenys)<br>
    - Tauragės jaunimo draugija: Vienkartinė 200€ parama projektams<br>

    **2. Ilgalaikės programos:**<br>
    - Mentorystės programa (8 mėn. trukmė)<br>
    - Kvalifikacijos kėlimo kursai (finansuojama 75% kainos)<br>

    **3. Specializuotos paslaugos:**<br>
    - Nemokami karjeros konsultacijų paketai (3 individualios, 2 grupinės)<br>
    - Laikino apgyvendinimo galimybės (iki 6 mėn.)<br>

    **4. Bendruomeninės organizacijos:**<br>
    - Jaunimo erdvė V. Kudirkos g. 9: Nemokama įranga projektams, darbo vietos<br>
    - Savanorystės centras: Sertifikuotos programos su stipendija<br>

    Ar norėtumėte sužinoti apie konkrečius paraiškų pildymo terminus ar papildomas paramos programas jaunimui?",
    "tool_calls": []
    }
    """)

    # Prepend the system message to the messages if it's not already there
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages

    response = llm.invoke(messages)
    print("LLM Response:", response)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1]
    outputs = []
    
    for tool_call in last_msg.tool_calls:
        if tool_call["name"] == "qdrant_retriever":
            query = tool_call["args"]["query"]
            result = qdrant_retriever.invoke(query)
            outputs.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )
            print(f"Retrieved data for: {query}")
    
    return {"messages": outputs}

def should_continue(state: AgentState):
    if state.get("complete"):
        return "end"
    
    last_msg = state["messages"][-1]
    print("Tool Calls in should_continue:", last_msg.tool_calls)
    
    # If there are no tool calls, this is the final message going to the user
    if not last_msg.tool_calls:
        # Check if the message content is in JSON format and clean it up if needed
        content = last_msg.content
        try:
            # Try to parse it as JSON
            parsed_json = json.loads(content)
            if "answer" in parsed_json:
                # Extract just the answer part and update the message
                last_msg.content = parsed_json["answer"]
                print("Reformatted final message from JSON to plain text")
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON or not a string, leave it as is
            pass
        
        return "end"
    else:
        return "continue"

# Build the ReAct workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END}
)

workflow.add_edge("tools", "agent")
graph = workflow.compile()
