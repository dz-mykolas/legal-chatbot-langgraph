from dotenv import load_dotenv

load_dotenv()
import os

from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Annotated, Sequence, Optional, Dict, Any, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model

from legal_chatbot_langgraph.utils import EnumWithHelpers, llm_selector
from legal_chatbot_langgraph.graphs.retrieval_graph import subgraph_retrieval
from legal_chatbot_langgraph.tools import get_current_time

MAIN_MODEL_NAME = os.getenv("RETRIEVER_MODEL_NAME")
MAIN_MODEL_TEMP = float(os.getenv("MAIN_MODEL_TEMP", "0.1"))

TOOLS = [
    subgraph_retrieval,
    get_current_time,
]

print(f"Retriever LLM: {MAIN_MODEL_NAME} (Temp: {MAIN_MODEL_TEMP})")
# llm = llm_selector(MAIN_MODEL_NAME, MAIN_MODEL_TEMP).bind_tools(TOOLS)
llm = init_chat_model("google_genai:gemini-2.0-flash").bind_tools(TOOLS)

class Nodes(EnumWithHelpers):
    get_current_time = "get_current_time"
    agent = "agent"
    tools = "tools"
    subgraph_retrieval = "subgraph_retrieval"
    start = START
    end = END

class GraphState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]

graph_builder = StateGraph(
    GraphState,
)

def agent(state: GraphState) -> Dict[str, Any]:
    system_message = SystemMessage(
        content="""
        If calling a tool, examine the receive data and format it in markdown for user to read. Format data in the same language as the user query.
        """
    )
    response = llm.invoke([system_message] + state.messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    response.name = "Chatbot"

    return {"messages": [response]}

graph_builder.add_node(Nodes.agent, agent)
graph_builder.add_node(Nodes.tools, ToolNode(TOOLS))

graph_builder.add_edge(Nodes.start, Nodes.agent)
graph_builder.add_conditional_edges(
    Nodes.agent, Nodes.if_tools,
    Nodes.as_dict(Nodes.tools, Nodes.end)
)
graph_builder.add_edge(Nodes.tools, Nodes.agent)
graph_builder.add_edge(Nodes.agent, Nodes.end)

graph = graph_builder.compile()
