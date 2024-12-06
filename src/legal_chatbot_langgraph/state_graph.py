import json
from langgraph.graph import StateGraph, END
from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage, AIMessage
import openai
from pydantic import BaseModel
import logging
from langchain.agents.openai_assistant.base import OpenAIAssistantFinish

from legal_chatbot_langgraph.tools import fetch_social_support_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(BaseModel):
    messages: List[BaseMessage]
    thread_id: Optional[str] = None
    next_node: Optional[str] = None

def create_langgraph_workflow(assistant):

    def should_continue(state: State) -> State:
        logger.info(f"Checking if conversation should continue. Current messages: {state.messages}")
        last_message = state.messages[-1] if state.messages else None

        if last_message is None:
            logger.warning("Last message is None in should_continue.")
            return State(messages=state.messages, thread_id = state.thread_id, next_node="wait_for_user")

        if isinstance(last_message, AIMessage):
            tool_calls = getattr(last_message, 'tool_calls', None) or last_message.additional_kwargs.get('tool_calls', [])
            if tool_calls:
                logger.info("Tool calls found. Going to call_tool node.")
                return State(messages=state.messages, thread_id=state.thread_id, next_node="call_tool")
            else:
                logger.info("No tool calls found. Ready for next user input.")
                return State(messages=state.messages, thread_id=state.thread_id, next_node="wait_for_user")
        else:
            logger.info("Last message is not from AI. Going to assistant node.")
            return State(messages=state.messages, thread_id=state.thread_id, next_node="assistant")

    def assistant_node(state: State) -> State:
        logger.info(f"Calling assistant with messages: {state.messages} and thread_id: {state.thread_id}")
        try:
            last_message = state.messages[-1] if state.messages else None
            if not last_message:
                raise ValueError("No messages found in state")

            if state.thread_id is None:
                logger.info("Creating new thread as it does not exist.")
                new_thread = openai.beta.threads.create().id
                state.thread_id = new_thread.id
                logger.info(f"New thread created with id: {state.thread_id}")

            # Add the user message to the thread
            assistant.add_message(last_message, thread_id=state.thread_id)

            # Run the assistant, it will automatically use the thread history
            response = assistant.invoke({"thread_id": state.thread_id})

            if isinstance(response, OpenAIAssistantFinish):
                new_message = AIMessage(
                    content=response.return_values.get('output', ''),
                    additional_kwargs={
                        'thread_id': response.thread_id,
                        'run_id': response.run_id
                    }
                )
                new_messages = state.messages + [new_message]
            else:
                error_message = AIMessage(content=f"Unexpected response format from assistant: {type(response)}")
                logger.error(f"Unexpected response format from assistant: {type(response)}")
                new_messages = state.messages + [error_message]
                return State(messages=new_messages, thread_id=state.thread_id, next_node="wait_for_user")

            return State(messages=new_messages, thread_id=state.thread_id, next_node="should_continue")
        except Exception as e:
            logger.error(f"Error in assistant node: {e}", exc_info=True)
            error_message = AIMessage(content=f"Error in assistant: {str(e)}. Please try rephrasing your request.")
            return State(messages=state.messages + [error_message], thread_id = state.thread_id, next_node="wait_for_user")

    def call_tool(state: State) -> State:
        logger.info(f"Calling tool with state: {state}")
        messages: List[BaseMessage] = state.messages
        last_message = messages[-1]
        tool_calls = last_message.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            logger.warning("No tool calls found in last message. Going to wait for user input.")
            return State(messages=messages, thread_id = state.thread_id, next_node="wait_for_user")

        all_updated_messages = list(messages)

        for tool_call in tool_calls:
            if tool_call['function']['name'] == "fetch_social_support_data":
                arguments = json.loads(tool_call['function']['arguments'])
                url = arguments.get('url')
                if url:
                    try:
                        observation = fetch_social_support_data.invoke({"url": url})
                        logger.info(f"Tool Observation: {observation}")
                        all_updated_messages.append(
                            AIMessage(content="", tool_calls=[{"tool_call_id": tool_call["id"], "function": {"name": "fetch_social_support_data", "arguments": json.dumps({"result": observation})}}])
                        )
                    except Exception as e:
                        logger.error(f"Error in tool execution: {e}")
                        all_updated_messages.append(
                            AIMessage(content="", tool_calls=[{"tool_call_id": tool_call["id"], "function": {"name": "fetch_social_support_data", "arguments": json.dumps({"error": f"Error in tool execution: {str(e)}"})}}])
                        )
            else:
                logger.warning(f"Unknown tool requested: {tool_call['function']['name']}")
                all_updated_messages.append(
                    AIMessage(content="", tool_calls=[{"tool_call_id": tool_call["id"], "function": {"name": tool_call['function']['name'], "arguments": json.dumps({"error": f"Unknown tool requested: {tool_call['function']['name']}"})}}])
                )

                return State(messages=all_updated_messages, thread_id=state.thread_id, next_node="assistant")

    def wait_for_user(state: State) -> State:
        logger.info("Waiting for user input.")
        return state
    
    workflow = StateGraph(State)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("call_tool", call_tool)
    workflow.add_node("should_continue", should_continue)
    workflow.add_node("wait_for_user", wait_for_user)

    workflow.add_edge("assistant", "should_continue")
    workflow.add_conditional_edges(
        "should_continue",
        lambda x: x.next_node,
        {
            "assistant": "assistant",
            "call_tool": "call_tool",
            "wait_for_user": "wait_for_user",
            END: END
        },
    )
    workflow.add_edge("call_tool", "assistant")
    workflow.set_entry_point("assistant")

    return workflow