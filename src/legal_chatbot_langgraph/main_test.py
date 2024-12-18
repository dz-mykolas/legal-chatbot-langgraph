from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio
from colorama import Fore, Style
import operator
import os
from typing import Dict, List, TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.messages import HumanMessage
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation, ToolNode

from legal_chatbot_langgraph.utils.tools import (
    calculate_legal_fee,
    fetch_recent_case_summaries,
    fetch_and_extract_social_support_data,
)

# Define the tools your assistant can use
tools = [
    calculate_legal_fee,
    fetch_recent_case_summaries,
    fetch_and_extract_social_support_data,
]

# State representation
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    thread_id: Optional[str]

# Nodes
def assistant_node(state: AgentState, agent_runnable: OpenAIAssistantRunnable):
    messages = state["messages"]
    thread_id = state["thread_id"]
    
    response = agent_runnable.invoke(messages, config={"configurable": {"thread_id": thread_id}})
    return {"messages": response}

# Updated tool_node to use ToolNode
def tool_node(state: AgentState, tool_node_instance: ToolNode):
    messages = state["messages"]
    # Filter out ToolInvocations and ToolMessages
    filtered_messages = [msg for msg in messages if not isinstance(msg, (ToolInvocation, ToolMessage))]
    last_message = filtered_messages[-1]

    # Check if the last message is an AIMessage and has tool calls
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("tool_calls"):
        tool_calls = last_message.additional_kwargs["tool_calls"]
        tool_outputs = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_input_str = tool_call.get("function", {}).get("arguments", "")

            action = ToolInvocation(
                tool=tool_name,
                tool_input=tool_input_str,
            )

            # Execute the tool using ToolNode and append the result to tool_outputs
            try:
                response = tool_node_instance.invoke(action)
                tool_outputs.append(response)
            except Exception as e:
                print(Fore.RED + f"Error during tool execution: {e}")
                # Handle the error as needed, possibly re-raise or return a default error message
                raise

        # Return the tool outputs
        return {"messages": tool_outputs, "next": "assistant"}

    else:
        # If no tool calls were found, just return, possibly ending the conversation
        return {"messages": [], "next": "end"}
        
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are tool calls, continue
    print(last_message)
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get(
        "tool_calls"
    ):
        return "call_tool"
    # Otherwise, end
    else:
        return "end"

def print_messages(messages: List[BaseMessage]):
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(Fore.GREEN + f"You: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(Fore.BLUE + Style.BRIGHT + f"AI: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(Fore.YELLOW + f"Tool Result: {msg.content}")
        else:
            print(f"{type(msg).__name__}: {msg.content}")
    print(Style.RESET_ALL)

# Graph
def create_graph(assistant_id: str):
    agent_runnable = OpenAIAssistantRunnable(
        assistant_id=assistant_id
    )
    
    tool_node_instance = ToolNode(tools)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", lambda state: assistant_node(state, agent_runnable))
    workflow.add_node("call_tool", lambda state: tool_node(state, tool_node_instance))
    workflow.add_conditional_edges("assistant", should_continue, {"call_tool": "call_tool", "end": END})
    workflow.add_edge("call_tool", "assistant")
    workflow.set_entry_point("assistant")
    return workflow.compile()

async def main():
    print(Fore.CYAN + "Start chatting with the AI (type 'exit' to stop):")

    assistant_id = os.environ.get("ASSISTANT_ID")

    # Create the graph ONCE outside the loop
    graph = create_graph(assistant_id)

    while True:
        thread_id = None # Initialize thread ID for each conversation
        messages: List[BaseMessage] = [] # Initialize messages for each conversation
        user_input = input(Fore.GREEN + "You: ")
        if user_input.lower() == "exit":
            print(Fore.RED + "Ending program.")
            break

        messages.append(HumanMessage(content=user_input))

        config = {}

        try:
            for event in graph.stream(
                {"messages": messages, "thread_id": thread_id},
                config=config
            ):
                if "___END___" not in event:
                    print(event)
                
                if "___END___" in event:
                    result = event["___END___"]
                    print_messages(result["messages"])
                    thread_id = result.get("thread_id")
                    messages = result["messages"]
                    print(f"Thread ID: {thread_id}")

        except Exception as e:
            print(Fore.RED + f"Error in conversation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
