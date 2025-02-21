# graph.py
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# --- Step 1: Set up the Qdrant vector store for document retrieval ---
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    collection_name="langgraph_demo",
    url="http://localhost:6333/"  # Adjust the URL as needed
)
retriever = vector_store.as_retriever()

# --- Step 2: Set up the LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Node: Qdrant Lookup ---
def qdrant_node(state: MessagesState) -> MessagesState:
    """
    This node extracts the last human message, uses it as a query to look up
    relevant texts from Qdrant, and appends the retrieved info as a SystemMessage.
    """
    query = ""
    # Find the most recent HumanMessage from the state
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if query:
        results = retriever.get_relevant_documents(query)
        # Combine the page_content of all retrieved documents.
        retrieved_text = "\n".join([doc.page_content for doc in results])
    else:
        retrieved_text = "No query provided."

    # Append a SystemMessage with the retrieved info
    state["messages"].append(SystemMessage(content=f"Retrieved info:\n{retrieved_text}"))
    return state

# --- Node: LLM Agent ---
def agent_node(state: MessagesState) -> MessagesState:
    """
    This node invokes the LLM using the current list of messages (which now includes
    the Qdrant retrieval result) and appends the LLM's response to the messages.
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    state["messages"].append(response)
    return state

# --- Step 3: Build and compile the LangGraph ---
def create_graph():
    builder = StateGraph(MessagesState)
    
    # Add our two nodes: one for retrieval and one for LLM response.
    builder.add_node("qdrant", qdrant_node)
    builder.add_node("agent", agent_node)
    
    # Chain the nodes: START -> qdrant -> agent -> END
    builder.add_edge(START, "qdrant")
    builder.add_edge("qdrant", "agent")
    builder.add_edge("agent", END)
    
    graph = builder.compile()
    return graph

graph = create_graph()
