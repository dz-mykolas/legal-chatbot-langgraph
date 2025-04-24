# legal_chatbot_langgraph/graph_builder.py

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Sequence
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Import state definition and node functions
from legal_chatbot_langgraph.schemas import RetrievalState, RetrievalStateInput, RetrievalStateOutput
from legal_chatbot_langgraph.graph_nodes import (
    prepare_initial_state,
    call_retriever_model,
    execute_tools_and_accumulate,
    should_continue_retrieval, # Use the simplified version
    call_synthesizer_model,   # Use the updated version
    # handle_general_question is removed
)

# Define the state graph
workflow = StateGraph(
    RetrievalState,
    input=RetrievalStateInput,
    output=RetrievalStateOutput
)

# Correctly apply add_messages to the 'messages' field in RetrievalState
RetrievalState.__annotations__['messages'] = Annotated[
    Sequence[BaseMessage], add_messages
]

# Add nodes to the graph
workflow.add_node("prepare_state", prepare_initial_state)
workflow.add_node("retriever", call_retriever_model)
workflow.add_node("tools", execute_tools_and_accumulate)
workflow.add_node("synthesizer", call_synthesizer_model) # This node now handles both paths
# No "general_handler" node anymore

# Define the graph's flow (edges)

# Start at the preparation step
workflow.add_edge(START, "prepare_state")

# After preparation, call the retriever
workflow.add_edge("prepare_state", "retriever")

# After the retriever decides, check if we should continue retrieval or synthesize
workflow.add_conditional_edges(
    "retriever",
    should_continue_retrieval, # Use the simplified function
    {
        "continue_retrieval": "tools",      # Go to tools if retrieval needed
        "synthesize": "synthesizer"         # Go directly to synthesizer otherwise
    }
)

# After executing tools, go back to the retriever to re-evaluate
workflow.add_edge("tools", "retriever")

# After synthesizing (for any reason), end the process
workflow.add_edge("synthesizer", END)

# Compile the graph
graph = workflow.compile()
