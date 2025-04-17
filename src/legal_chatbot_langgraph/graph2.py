from dotenv import load_dotenv
load_dotenv()

import json
import os
from typing import Literal, TypedDict, Annotated, Sequence, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from legal_chatbot_langgraph.tools import retrieve_chunks_by_ids, retrieve_documents_by_chunk_ids, retrieve_relevant_chunks

class RetrieveRelevantChunksArgs(BaseModel):
    query: str = Field(..., description="The search query string for finding relevant text chunks.")
    k: int = Field(5, description="Number of chunk results to retrieve.")

class RetrieveDocumentsByChunkIdsArgs(BaseModel):
    chunk_ids: List[str] = Field(..., description="List of chunk IDs whose parent documents/sections are sought.")
    k: int = Field(10, description="Maximum number of parent documents/sections to retrieve.")

class RetrieveChunksByIdsArgs(BaseModel):
    chunk_ids: List[str] = Field(..., description="A list of exact chunk IDs to retrieve.")

# Model for the arguments expected by our specific tool
class ToolCallRequest(BaseModel):
    """Specifies a single tool call to be executed."""
    tool_name: Literal[
        "retrieve_relevant_chunks",
        "retrieve_documents_by_chunk_ids",
        "retrieve_chunks_by_ids"
    ] = Field(..., description="The name of the specific tool to call.")
    arguments: Dict[str, Any] = Field(..., description="The arguments dictionary for the chosen tool, matching its specific requirements.")

    # --- ADD VALIDATOR HERE ---
    @field_validator('arguments', mode='before')
    @classmethod
    def parse_arguments_string(cls, value):
        if isinstance(value, str):
            try:
                # If the input is a string, attempt to parse it as JSON
                return json.loads(value)
            except json.JSONDecodeError:
                # If parsing fails, raise a ValueError or handle as needed
                # Pydantic will catch this and report a validation error
                raise ValueError(f"Arguments field is a string but not valid JSON: {value}")
        # If it's already a dict (or other type), let Pydantic handle standard validation
        return value
    # --- END VALIDATOR ---

# Model for the overall decision/action the retriever LLM makes
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


# Input Schema: What the API user MUST provide
class RetrievalStateInput(TypedDict):
    messages: Sequence[BaseMessage]

class RetrievalState(TypedDict):
    # Required fields populated during the graph run
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str # Will be populated by prepare_initial_state
    retrieved_data: List[Dict[str, Any]] # Will be initialized by prepare_initial_state

    # Optional fields managed by the logic
    retriever_decision: Optional[RetrieverAction]

class RetrievalStateOutput(TypedDict):
    # Typically the final message containing the answer
    messages: Sequence[BaseMessage]
    # You could add other final fields if needed, e.g., all sources
    # retrieved_data: List[Dict[str, Any]]

class RetrievalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    retrieved_data: List[Dict[str, Any]]
    retriever_decision: Optional[RetrieverAction]

def prepare_initial_state(state: RetrievalState):
    """
    Takes the initial state (from Input schema) and prepares the full state
    by extracting the original query and initializing retrieved_data.
    THIS IS THE NEW FIRST STEP.
    """
    print("--- Running prepare_initial_state ---")
    # LangGraph passes the input dict here initially.
    # We need to be careful as 'original_query' etc. won't exist yet.
    messages = state['messages'] # This comes from the Input schema

    # Find the first HumanMessage to use as the original query
    first_human_message = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            first_human_message = msg
            break

    if not first_human_message:
        print("Error: No HumanMessage found in initial input.")
        # You might want to return an error message in the state or raise
        # For now, let's create a placeholder query
        original_query = "Error: No user query provided."
        # Add an error message to the state
        error_message = AIMessage(content="Could not process request: No user query found.")
        return {
            "messages": [error_message], # Overwrite initial messages with error
            "original_query": original_query,
            "retrieved_data": [] # Initialize anyway
        }


    original_query = first_human_message.content
    print(f"Extracted original_query: '{original_query}'")

    # Return the fields to add/initialize in the state
    # Note: We don't return 'messages' here because the initial message
    # is already in the state via the Input schema and add_messages accumulator.
    return {
        "original_query": original_query,
        "retrieved_data": [] # Initialize as empty list
    }

# --- LLM Configuration ---
def llm_selector(model_name: str, temperature: float = 0):
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model prefix for model: {model_name}")

# Load from environment or hardcoded fallback
retriever_model = os.getenv("LLM_MODEL_RETRIEVER", "gpt-4o-mini")
synthesizer_model = os.getenv("LLM_MODEL_SYNTHESIZER", "gemini-2.0-flash")
retriever_temp = float(os.getenv("LLM_TEMPERATURE_RETRIEVER", "0.1"))
synthesizer_temp = float(os.getenv("LLM_TEMPERATURE_SYNTHESIZER", "0"))

# Instantiate with smart detection
retriever_llm = llm_selector(retriever_model, retriever_temp)
structured_retriever_llm = retriever_llm.with_structured_output(RetrieverAction)
synthesizer_llm = llm_selector(synthesizer_model, synthesizer_temp)


def call_retriever_model(state: RetrievalState):
    """
    Invokes the LLM to decide the next retrieval actions using specific tools.
    """
    print(f"\n--- Entering call_retriever_model ---")
    original_query = state["original_query"]
    retrieved_so_far = state["retrieved_data"]
    history_messages = state["messages"]

    # --- History Summary ---
    history_summary_lines = []
    last_decision = state.get("retriever_decision")
    if last_decision and last_decision.retrieval_calls:
         history_summary_lines.append("Last search attempts:")
         for call in last_decision.retrieval_calls:
             args_str = str(call.arguments)
             if len(args_str) > 150: args_str = args_str[:150] + "..."
             history_summary_lines.append(f"- Called tool '{call.tool_name}' with args: {args_str}")

    actual_chunk_ids_found = None
    if retrieved_so_far:
        history_summary_lines.append("\nSummary of data found so far:")
        actual_chunk_ids_found = [item['chunk_id'] for item in retrieved_so_far if isinstance(item, dict) and 'chunk_id' in item]
        doc_ids_found = {item['document_id'] for item in retrieved_so_far if isinstance(item, dict) and 'document_id' in item} # These now only come from my_documents

        history_summary_lines.append(f"- Total items retrieved: {len(retrieved_so_far)}")
        history_summary_lines.append(f"- Found {len(actual_chunk_ids_found)} chunks (from 'my_chunks').")
        if actual_chunk_ids_found:
            example_ids = actual_chunk_ids_found[:3]
            history_summary_lines.append(f"- Example Chunk IDs found: {example_ids}")
            history_summary_lines.append(f"  (Use these exact string IDs when referring to specific chunks).")
        if doc_ids_found:
             history_summary_lines.append(f"- Related Document IDs found (from 'my_documents'): {list(doc_ids_found)}")
    else:
        history_summary_lines.append("No data retrieved yet.")

    history_summary = "\n".join(history_summary_lines)

    # --- REVISED Prompt ---
    prompt_context = f"""
Original User Query: "{original_query}"

Available Tools:
1.  `retrieve_relevant_chunks`: Finds relevant text chunks from the 'my_chunks' collection based on semantic similarity.
    - Args Schema: `{{"query": "string", "k": "integer (optional, default 5)"}}`
2.  `retrieve_documents_by_chunk_ids`: Finds parent documents in the 'my_documents' collection that contain ANY of the specified chunk IDs. Useful for getting broader document context.
    - Args Schema: `{{"chunk_ids": ["list", "of", "string IDs"], "k": "integer (optional, default 10)"}}` # Simplified schema
3.  `retrieve_chunks_by_ids`: Retrieves specific chunks from 'my_chunks' using their exact IDs.
    - Args Schema: `{{"chunk_ids": ["list", "of", "string IDs"]}}`

*** CRITICAL INSTRUCTIONS FOR TOOL CALLS ***
When you decide to call a tool, structure the `arguments` field as a JSON object (a dictionary) where keys exactly match the argument names shown in the 'Args Schema'.

*   **Example for `retrieve_relevant_chunks`:**
    `{{"tool_name": "retrieve_relevant_chunks", "arguments": {{"query": "search phrase", "k": 5}}}}`
*   **Example for `retrieve_documents_by_chunk_ids`:** (Note: no collection_name)
    `{{"tool_name": "retrieve_documents_by_chunk_ids", "arguments": {{"chunk_ids": ["{actual_chunk_ids_found[0] if actual_chunk_ids_found else 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'}", "...other IDs..."], "k": 10}}}}`
*   **Example for `retrieve_chunks_by_ids`:**
    `{{"tool_name": "retrieve_chunks_by_ids", "arguments": {{"chunk_ids": ["{actual_chunk_ids_found[0] if actual_chunk_ids_found else 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'}", "...other IDs..."]}}}}`

*   **Mandatory:** The value for `arguments` MUST be a dictionary `{...}`.
*   **Mandatory:** When using `chunk_ids`, provide the exact, full string UUIDs obtained from previous `retrieve_relevant_chunks` calls. Do NOT use numerical indices or simplified IDs.

Your Goal: Plan the next actions to gather information to answer the user query using the available tools and the data hierarchy ('my_documents' contains 'my_chunks').

General Strategy:
1.  Initial Chunk Search: Call `retrieve_relevant_chunks` with arguments like `{{"query": "...", "k": ...}}`.
2.  Analyze Chunks: Look at the `chunk_id` (e.g., '{actual_chunk_ids_found[0] if actual_chunk_ids_found else 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'}') and `chunk_metadata` (especially any `doc_id` fields within the metadata).
3.  Get Document Context: If needed for broader context, call `retrieve_documents_by_chunk_ids` with arguments like `{{"chunk_ids": ["exact_id_1", "exact_id_2"], "k": ...}}`. This searches 'my_documents'.
4.  Retrieve Specific Chunks: If specific detailed text passages are needed again, call `retrieve_chunks_by_ids` with arguments like `{{"chunk_ids": ["exact_id_1", "exact_id_3"]}}`.
5.  Iterate: Repeat analysis and tool calls as needed.
6.  Complete: When sufficient info is gathered, or searches yield no new relevant data, provide an empty `retrieval_calls` list.

Previous Actions & Findings Summary:
{history_summary if history_summary else "This is the first retrieval step."}

Task: Decide the next set of specific tool calls. Populate the `retrieval_calls` list. For each call, specify `tool_name` and construct the `arguments` **as a dictionary** with the correct keys and values (using full UUIDs for `chunk_ids`). Provide reasoning.
"""
    print(f"--- Calling Retriever LLM ({retriever_llm.model}) ---") # Adjust model attribute if needed
    messages_for_llm = [HumanMessage(content=prompt_context)]

    # LLM call and rest of the function...
    response_structured: RetrieverAction = structured_retriever_llm.invoke(messages_for_llm)

    # Keep the detailed logging
    if response_structured.retrieval_calls:
        print("--- LLM Generated Tool Call Arguments (Post-Validation Attempt) ---")
        for i, call in enumerate(response_structured.retrieval_calls):
             print(f"Call {i+1}: Tool='{call.tool_name}', Args='{call.arguments}' (Type: {type(call.arguments)})")
             if isinstance(call.arguments, dict) and 'chunk_ids' in call.arguments:
                 ids = call.arguments['chunk_ids']
                 print(f"  - Generated chunk_ids type: {type(ids)}")
                 if isinstance(ids, list) and ids:
                     print(f"  - First generated chunk_id example: {ids[0]} (Type: {type(ids[0])})")
        print("-----------------------------------------------------------------")

    # Generate summary message
    ai_summary = f"Retriever Decision: {response_structured.reasoning}. "
    if response_structured.retrieval_calls:
        call_summaries = [f"{call.tool_name}" for call in response_structured.retrieval_calls]
        if call_summaries:
             ai_summary += f"Planning {len(response_structured.retrieval_calls)} call(s): {', '.join(call_summaries)}."
        else:
             ai_summary += "No valid tool calls planned despite reasoning."
    else:
        ai_summary += "Stopping retrieval."

    # Return updated state
    return {
        "messages": [AIMessage(content=ai_summary)],
        "retriever_decision": response_structured,
    }


available_tools = {
    "retrieve_relevant_chunks": retrieve_relevant_chunks,
    "retrieve_documents_by_chunk_ids": retrieve_documents_by_chunk_ids,
    "retrieve_chunks_by_ids": retrieve_chunks_by_ids,
}

def execute_tools_and_accumulate(state: RetrievalState):
    """Executes specific tools based on the structured decision and accumulates results."""
    decision = state.get("retriever_decision")
    # Get the current list OR initialize if it's the first run and missing
    accumulated_data = state.get("retrieved_data", [])

    if not decision or not decision.retrieval_calls:
        print("--- No tools to execute ---")
        return {
            "messages": [],
            "retrieved_data": accumulated_data, # Ensure data is passed even if no tools run
            "retriever_decision": decision
        }

    tool_outputs = [] # To store ToolMessages for history
    print(f"--- Executing {len(decision.retrieval_calls)} Tool Call(s) from Structured Decision ---")

    # Iterate through the specific tool calls requested by the LLM
    for planned_call in decision.retrieval_calls:
        tool_name = planned_call.tool_name
        arguments = planned_call.arguments

        # Find the corresponding tool function
        tool_function = available_tools.get(tool_name)

        if not tool_function:
            print(f"Error: Tool '{tool_name}' specified by LLM is not available.")
            # Add an error message? Skip? For now, just log and skip.
            result_str = json.dumps({"error": f"Tool '{tool_name}' not found."})
            tool_call_id = f"error-{tool_name}" # Create a placeholder ID
        else:
            import uuid
            tool_call_id = str(uuid.uuid4())
            print(f"Executing tool: {tool_name} (ID: {tool_call_id}) with args: {arguments}")
            try:
                # Invoke the specific tool with its arguments
                result_str = tool_function.invoke(arguments)

                # Process and accumulate results (handle potential JSON errors)
                try:
                    structured_result = json.loads(result_str)
                    # Check if the result is a list (most tools return lists)
                    if isinstance(structured_result, list):
                         # Filter out potential error objects within the list
                         valid_results = [
                             item for item in structured_result
                             if isinstance(item, dict) and "error" not in item
                         ]
                         print(f"Tool '{tool_name}' returned {len(valid_results)} valid items.")
                         accumulated_data.extend(valid_results) # Append valid items
                    # Handle single dict result (less common but possible)
                    elif isinstance(structured_result, dict) and "error" not in structured_result:
                         print(f"Tool '{tool_name}' returned 1 valid item.")
                         accumulated_data.append(structured_result)
                    elif isinstance(structured_result, dict) and "error" in structured_result:
                         print(f"Tool '{tool_name}' returned an error: {structured_result['error']}")
                         # Don't add error dicts to accumulated_data unless desired
                    else:
                         print(f"Warning: Tool '{tool_name}' returned unexpected JSON structure: {type(structured_result)}")
                         # Add as raw string maybe? For now, we only add valid dicts/lists of dicts.

                except json.JSONDecodeError:
                    print(f"Warning: Tool '{tool_name}' did not return valid JSON: {result_str[:100]}...")
                    # Store the raw non-JSON string as an error in the ToolMessage
                    result_str = json.dumps({"error": "Tool failed to return valid JSON", "output": result_str[:200]})
                except Exception as e:
                    print(f"Error processing result from tool '{tool_name}': {e}")
                    result_str = json.dumps({"error": f"Failed to process tool result: {e}"})

            except Exception as e:
                print(f"Error executing tool '{tool_name}': {e}")
                # Capture the execution error in the ToolMessage
                result_str = json.dumps({"error": f"Tool execution failed: {e}"})
                # Ensure tool_call_id exists even if invocation fails early
                if 'tool_call_id' not in locals():
                     tool_call_id = f"exec-error-{tool_name}-{uuid.uuid4()}"


        # Always append a ToolMessage, even if it contains an error report
        tool_outputs.append(ToolMessage(content=result_str, tool_call_id=tool_call_id, name=tool_name))
        print(f"Tool {tool_name} (id: {tool_call_id}) processing complete.")

    print(f"Total accumulated data points after execution: {len(accumulated_data)}")

    return {
        "messages": tool_outputs,
        "retrieved_data": accumulated_data, # Return the potentially modified list
        "retriever_decision": decision # Pass along the decision that was acted upon
    }

def should_continue_retrieval(state: RetrievalState):
    """Determines if the retrieval loop should continue based on structured decision."""
    print("--- Checking Should Continue Retrieval ---")
    # Decision might be None if there was an error earlier, treat as stop
    decision = state.get("retriever_decision")

    if decision and decision.retrieval_calls:
        print(f"Decision: Continue Retrieval ({len(decision.retrieval_calls)} calls planned)")
        return "continue_retrieval"
    else:
        print("Decision: End Retrieval, Proceed to Synthesis")
        return "synthesize"

def call_synthesizer_model(state: RetrievalState):
    """Invokes the LLM dedicated to synthesizing the final answer."""
    print(f"--- Calling Synthesizer LLM ({synthesizer_llm.name}) ---") # Adjusted attribute name
    original_query = state["original_query"]
    retrieved_data = state["retrieved_data"]

    if not retrieved_data:
        print("Synthesizer: No data retrieved. Generating acknowledgement.")
        final_answer_content = "Atsiprašome, bet nepavyko rasti konkrečios informacijos pagal jūsų užklausą."
        # Return only the final message for the output state
        return {"messages": [AIMessage(content=final_answer_content)]}

    # --- (Synthesizer Prompting Logic remains the same) ---
    context_str = "\n\n---\n\n".join([
        f"Source Collection: {item.get('collection', 'N/A')}\n"
        f"Content: {item.get('content', 'N/A')}\n"
        f"Metadata: {json.dumps(item.get('metadata', {}))}"
        for item in retrieved_data if isinstance(item, dict) # Add type check
    ])

    synthesis_prompt = f"""
You are a helpful assistant for Tauragė city...

Original User Query:
"{original_query}"

Retrieved Context Data:
--- START CONTEXT ---
{context_str}
--- END CONTEXT ---

Instructions: ... (Your detailed instructions) ...
"""
    final_response = synthesizer_llm.invoke([HumanMessage(content=synthesis_prompt)])
    print(f"Synthesizer LLM Response: {final_response.content}")

    # Return only the final message for the output state
    return {"messages": [final_response]}


# --- Build the Graph ---

# IMPORTANT: Define the graph with the Input and Output schemas
workflow = StateGraph(
    RetrievalState,
    input=RetrievalStateInput,
    output=RetrievalStateOutput
)

# Add nodes
workflow.add_node("prepare_state", prepare_initial_state) # New first node
workflow.add_node("retriever", call_retriever_model)
workflow.add_node("tools", execute_tools_and_accumulate)
workflow.add_node("synthesizer", call_synthesizer_model)

# Set entry point using START keyword linked to the preprocessor
workflow.add_edge(START, "prepare_state")

# Add edge from the prepare node to the retriever node
workflow.add_edge("prepare_state", "retriever")

# Add conditional edges from the retriever
workflow.add_conditional_edges(
    "retriever",
    should_continue_retrieval,
    {
        "continue_retrieval": "tools",
        "synthesize": "synthesizer"
    }
)

# Add edge from tools back to retriever
workflow.add_edge("tools", "retriever")

# Add edge from synthesizer to the end
workflow.add_edge("synthesizer", END)

# Compile the graph
graph = workflow.compile()
