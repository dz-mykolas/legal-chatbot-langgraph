import json
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Literal, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from legal_chatbot_langgraph.utils.tools import qdrant_retriever

# Define state with messages and a flag for completion
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    complete: bool = False

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
    - Again, these are just examples and not exhaustive, the example is provided in support/aid context. The actual queries would depend on the context of the user query and the information available in the documents. DO NOT USE THESE ANSWERS FOR ACTUAL QUERIES.
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
