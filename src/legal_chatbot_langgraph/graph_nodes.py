# legal_chatbot_langgraph/graph_nodes.py

import json
import uuid
import traceback # For detailed error logging
from typing import Dict, Any, List, Literal, Sequence # Added Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

# Assuming these are correctly imported and configured elsewhere
from legal_chatbot_langgraph.schemas import RetrievalState, RetrieverAction
from legal_chatbot_langgraph.llm_config import (
    synthesizer_llm,
    structured_retriever_llm
)
# Tool functions are needed for the available_tools mapping
from legal_chatbot_langgraph.tools import (
    retrieve_relevant_chunks,
    retrieve_documents_by_chunk_ids,
    retrieve_chunks_by_ids
)

# --- Helper Functions ---

def format_message_history_for_prompt(messages: Sequence[BaseMessage]) -> str:
    """Formats the message history for inclusion in LLM prompts."""
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else \
               "Assistant" if isinstance(msg, AIMessage) else \
               "Tool Result" if isinstance(msg, ToolMessage) else "System"
        content = msg.content
        # Truncate long tool results for clarity in prompt
        if isinstance(msg, ToolMessage) and len(str(content)) > 500:
            content = str(content)[:500] + "... (truncated)"
        lines.append(f"{role}: {content}")
    # Limit overall history length if necessary (e.g., last N messages)
    # history_limit = 10 # Example limit
    # if len(lines) > history_limit:
    #     lines = lines[-history_limit:]
    return "\n".join(lines)

def format_retrieved_data_for_prompt(retrieved_data: List[Dict[str, Any]]) -> str:
    """Formats the retrieved data list into a string for the synthesizer prompt."""
    context_parts = []
    if not retrieved_data:
        return "No specific context documents were retrieved."

    for i, item in enumerate(retrieved_data):
        if isinstance(item, dict):
            content = item.get('content', 'N/A')
            metadata = item.get('metadata', {})
            # Ensure metadata is serializable and keep it concise
            try:
                metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
                if len(metadata_str) > 300: # Limit metadata string length
                     metadata_str = metadata_str[:300] + "... (truncated metadata)"
            except Exception:
                metadata_str = str(metadata)[:300] + "... (truncated metadata)" # Fallback

            source_id = item.get('chunk_id') or item.get('document_id', f'item_{i+1}')
            context_parts.append(
                f"Šaltinis [{source_id}]:\n"
                f"Turinys: {content}\n"
                f"Metaduomenys: {metadata_str}"
            )
        else:
            context_parts.append(f"Šaltinis [item_{i+1}]:\nNeteisingas duomenų formatas: {str(item)[:200]}")
    return "\n\n---\n\n".join(context_parts)


# --- Node: Prepare Initial State ---

def prepare_initial_state(state: RetrievalState) -> Dict[str, Any]:
    """
    Extracts the original query from the initial message list, stores it for reference,
    and initializes the retrieved_data list.
    """
    print("--- Running prepare_initial_state ---")
    messages = state['messages']

    # Ensure messages is a list/sequence before proceeding
    if not isinstance(messages, Sequence) or not messages:
         print("Error: Initial state has no messages.")
         # Return a state that indicates an error, preventing further processing
         return {
             "messages": [AIMessage(content="System Error: No initial message provided.")],
             "original_query": "Error: No initial message.",
             "retrieved_data": [],
             "retriever_decision": RetrieverAction(retrieval_calls=[], reasoning="Error: No initial message.")
         }

    first_human_message = next((msg for msg in messages if isinstance(msg, HumanMessage)), None)

    if not first_human_message:
        print("Error: No HumanMessage found in initial input.")
        error_message = AIMessage(content="Could not process request: No user query found.")
        # Still set original_query to avoid downstream errors, though it's an error state
        return {
            "messages": [error_message],
            "original_query": "Error: No user query provided.",
            "retrieved_data": [],
            # Add a decision to stop immediately
            "retriever_decision": RetrieverAction(retrieval_calls=[], reasoning="Error: No user query found in initial input.")
        }

    original_query = first_human_message.content
    print(f"Extracted original_query (for reference): '{original_query}'")

    # Initialize state for the graph run
    return {
        "original_query": original_query, # Keep for reference/logging
        "retrieved_data": [], # Initialize as empty list
        "retriever_decision": None # Initialize decision as None
        # messages are passed through and handled by add_messages
    }

# --- Node: Call Retriever LLM (Handles Conversation History) ---

def call_retriever_model(state: RetrievalState) -> Dict[str, Any]:
    """
    Invokes the LLM to decide retrieval actions or identify general questions,
    considering the full conversation history and focusing on the latest user query.
    """
    print(f"\n--- Entering call_retriever_model ---")
    messages = state['messages']
    retrieved_so_far = state.get("retrieved_data", [])
    last_decision = state.get("retriever_decision")

    # Find the latest user query
    latest_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if not latest_human_message:
        print("Error: No HumanMessage found in history for retriever.")
        # Stop retrieval if no user query can be identified
        return {
            "messages": [AIMessage(content="System Error: Cannot identify the latest user query.")],
            "retriever_decision": RetrieverAction(retrieval_calls=[], reasoning="Error: No HumanMessage found in history."),
            # Pass other state fields through
             "retrieved_data": retrieved_so_far,
             "original_query": state.get("original_query", "N/A")
        }
    current_user_query = latest_human_message.content
    print(f"Latest User Query for Retriever: '{current_user_query}'")

    # --- Build Retrieval History Summary (Focus on past *retrieval* actions/results) ---
    history_summary_lines = []
    actual_chunk_ids_found = []

    # Summarize previous tool calls if they exist from the *last* decision
    if last_decision and last_decision.retrieval_calls:
         history_summary_lines.append("Last Retrieval Attempts:")
         for call in last_decision.retrieval_calls:
             try:
                 args_str = json.dumps(call.arguments) if isinstance(call.arguments, dict) else str(call.arguments)
             except Exception:
                 args_str = str(call.arguments)
             if len(args_str) > 150: args_str = args_str[:150] + "..."
             history_summary_lines.append(f"- Called tool '{call.tool_name}' with args: {args_str}")

    # Summarize data found *so far in this session*
    if retrieved_so_far:
        history_summary_lines.append("\nSummary of Data Found So Far:")
        actual_chunk_ids_found = [item['chunk_id'] for item in retrieved_so_far if isinstance(item, dict) and 'chunk_id' in item]
        doc_ids_found = {item['document_id'] for item in retrieved_so_far if isinstance(item, dict) and 'document_id' in item}

        history_summary_lines.append(f"- Total items retrieved: {len(retrieved_so_far)}")
        if actual_chunk_ids_found:
            history_summary_lines.append(f"- Found {len(actual_chunk_ids_found)} specific chunks.")
            example_ids = actual_chunk_ids_found[:3]
            history_summary_lines.append(f"- Example Chunk IDs: {example_ids}")
            history_summary_lines.append(f"  (Use these exact IDs when calling tools like retrieve_documents_by_chunk_ids).")
        if doc_ids_found:
             history_summary_lines.append(f"- Related Document IDs: {list(doc_ids_found)}")
    else:
        history_summary_lines.append("No data retrieved yet in this session.")

    retrieval_history_summary = "\n".join(history_summary_lines)
    example_id_for_prompt = actual_chunk_ids_found[0] if actual_chunk_ids_found else 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'

    # Format the full conversation history for the prompt
    conversation_history_str = format_message_history_for_prompt(messages)

    # --- Create Prompt for Retriever LLM (Includes Conversation History) ---
    prompt_context = f"""
You are an AI assistant analyzing a conversation to decide the next step: either answer directly, retrieve information using tools, or confirm retrieval is complete.

Conversation History:
--- START HISTORY ---
{conversation_history_str}
--- END HISTORY ---

Latest User Query: "{current_user_query}"

Available Tools for Information Retrieval:
1.  `retrieve_relevant_chunks`: Finds relevant text chunks from 'my_chunks'. Use for initial search or finding specific passages based on the latest query and history. Schema: `{{"query": "string", "k": "integer (optional, default 5)"}}`
2.  `retrieve_documents_by_chunk_ids`: Finds parent documents in 'my_documents' containing specified chunk IDs. Use for getting broader context after finding relevant chunks. Schema: `{{"chunk_ids": ["list", "of", "string IDs"], "k": "integer (optional, default 10)"}}`
3.  `retrieve_chunks_by_ids`: Retrieves specific chunks from 'my_chunks' using exact IDs. Use if you need the precise text of previously identified chunks again. Schema: `{{"chunk_ids": ["list", "of", "string IDs"]}}`

*** CRITICAL INSTRUCTIONS ***
1.  **Analyze the Latest User Query** within the context of the **Conversation History**.
2.  **Decide the Action:**
    *   **General Question:** If the latest query is conversational (e.g., "thanks", "who are you?", "what about X mentioned before?") and **does not** require searching new document information, set `retrieval_calls = []` and start `reasoning` with "GENERAL_QUESTION:".
    *   **Retrieval Needed:** If the latest query requires searching documents (or refining a previous search based on conversation), plan tool calls. Populate `retrieval_calls` with the correct `tool_name` and `arguments` (as a dictionary). Provide detailed `reasoning` (NOT starting with "GENERAL_QUESTION:"). Use the `query` argument in `retrieve_relevant_chunks` based on the latest user query and relevant history. Use exact `chunk_ids` found previously if needed.
    *   **Retrieval Complete:** If enough information has likely been gathered from previous retrieval steps (summarized below) to answer the latest query, set `retrieval_calls = []` and explain completion in `reasoning` (NOT starting with "GENERAL_QUESTION:").

*** TOOL CALL FORMATTING (If Retrieval Needed) ***
*   `arguments` MUST be a dictionary `{...}`.
*   Use exact string UUIDs for `chunk_ids` (like '{example_id_for_prompt}').

Summary of Previous Retrieval Actions & Findings (in this session):
{retrieval_history_summary if retrieval_history_summary else "This is the first retrieval step for this query."}

Task: Based on the **Latest User Query**, **Conversation History**, and **Retrieval Findings Summary**, decide the next action (General Question, Plan Retrieval, or Complete Retrieval). Fill the `retrieval_calls` list and provide `reasoning`.
"""
    llm_name = getattr(structured_retriever_llm, 'model_name', getattr(structured_retriever_llm, 'model', 'structured_retriever_llm'))
    print(f"--- Calling Retriever LLM ({llm_name}) ---")
    # Use a HumanMessage containing the full context for the LLM
    messages_for_llm = [HumanMessage(content=prompt_context)]

    # --- Invoke LLM and Process Response ---
    try:
        response_structured: RetrieverAction = structured_retriever_llm.invoke(messages_for_llm)

        print("--- LLM Decision (Post Pydantic Validation) ---")
        if response_structured.retrieval_calls:
            print(f"Planning {len(response_structured.retrieval_calls)} tool call(s):")
            # (Logging logic for calls remains the same)
            for i, call in enumerate(response_structured.retrieval_calls):
                args_log = call.arguments if isinstance(call.arguments, dict) else str(call.arguments)
                print(f"  Call {i+1}: Tool='{call.tool_name}', Args='{args_log}' (Type: {type(call.arguments)})")
                if isinstance(call.arguments, dict) and 'chunk_ids' in call.arguments:
                    ids = call.arguments['chunk_ids']
                    print(f"    - Generated chunk_ids type: {type(ids)}")
                    if isinstance(ids, list) and ids:
                        print(f"    - First generated chunk_id example: {ids[0]} (Type: {type(ids[0])})")
        elif response_structured.reasoning.startswith("GENERAL_QUESTION:"):
             print("LLM identified as a General Question for the latest query. No tool calls planned.")
        else:
            print("LLM decided to stop retrieval for the latest query (no tool calls planned).")

        print(f"Reasoning: {response_structured.reasoning}")
        print("-----------------------------------------------------------------")

    except Exception as e:
        print(f"\n!!! Error invoking or parsing Retriever LLM response: {e} !!!")
        print(traceback.format_exc())
        print("--- Returning empty decision to stop retrieval due to error ---")
        response_structured = RetrieverAction(
            retrieval_calls=[],
            reasoning=f"Error during retriever LLM call or output parsing: {e}. Stopping retrieval."
        )

    # --- Generate Summary Message for History ---
    # This AI message summarizes the *decision* made in this step for the history
    ai_summary = f"Retriever Decision: {response_structured.reasoning}. "
    if response_structured.retrieval_calls:
        call_summaries = [f"{call.tool_name}" for call in response_structured.retrieval_calls]
        ai_summary += f"Planning {len(call_summaries)} call(s): {', '.join(call_summaries)}."
    elif response_structured.reasoning.startswith("GENERAL_QUESTION:"):
         ai_summary += "Identified as a general question. Proceeding to final response generation."
    else:
        ai_summary += "Stopping retrieval process. Proceeding to final response generation."

    # Return updated state components
    return {
        "messages": [AIMessage(content=ai_summary)], # Add decision summary to history
        "retriever_decision": response_structured,
        # Pass through existing retrieved data and original query
        "retrieved_data": retrieved_so_far,
        "original_query": state.get("original_query", "N/A") # Keep original query for reference if needed
    }


# --- Node: Execute Tools ---

# Map tool names to actual functions (ensure these are imported)
available_tools = {
    "retrieve_relevant_chunks": retrieve_relevant_chunks,
    "retrieve_documents_by_chunk_ids": retrieve_documents_by_chunk_ids,
    "retrieve_chunks_by_ids": retrieve_chunks_by_ids,
}

def execute_tools_and_accumulate(state: RetrievalState) -> Dict[str, Any]:
    """Executes tools based on the retriever's decision and accumulates results."""
    decision = state.get("retriever_decision")
    # Get current data, default to empty list if not present or None
    accumulated_data = state.get("retrieved_data") if state.get("retrieved_data") is not None else []

    # Check if a decision exists and has calls
    if not decision or not decision.retrieval_calls:
        print("--- No tools to execute ---")
        # Return no new messages and the existing data
        return {"messages": [], "retrieved_data": accumulated_data}

    tool_outputs: List[ToolMessage] = []
    print(f"--- Executing {len(decision.retrieval_calls)} Tool Call(s) ---")

    for planned_call in decision.retrieval_calls:
        tool_name = planned_call.tool_name
        # Arguments should be a dict due to Pydantic validation in RetrieverAction
        arguments = planned_call.arguments if isinstance(planned_call.arguments, dict) else {}
        tool_function = available_tools.get(tool_name)
        # Generate a unique ID for this specific tool call invocation
        tool_call_id = f"{tool_name}-{uuid.uuid4()}"

        if not tool_function:
            print(f"Error: Tool '{tool_name}' not found.")
            result_str = json.dumps({"error": f"Tool '{tool_name}' not found."})
            result_data = [{"error": f"Tool '{tool_name}' not found."}] # Represent error in data
        else:
            print(f"Executing tool: {tool_name} (ID: {tool_call_id}) with args: {arguments}")
            try:
                # Invoke the tool function with the arguments dictionary
                # Assuming tools return structured data (list of dicts, or dict)
                result_data = tool_function.invoke(arguments)

                # Ensure result_data is always a list for consistent processing
                if isinstance(result_data, dict):
                    result_data = [result_data] # Wrap single dict in a list
                elif not isinstance(result_data, list):
                     print(f"Warning: Tool '{tool_name}' returned unexpected type {type(result_data)}. Attempting to wrap.")
                     result_data = [{"warning": "Non-standard tool output", "output": str(result_data)[:500]}]

                # Process and accumulate results (filtering out potential errors within the list)
                valid_results = [item for item in result_data if isinstance(item, dict) and "error" not in item]
                error_results = [item for item in result_data if isinstance(item, dict) and "error" in item]

                print(f"Tool '{tool_name}' returned {len(valid_results)} valid item(s) and {len(error_results)} error item(s).")
                accumulated_data.extend(valid_results) # Add only valid results to state

                # Convert the *entire* tool output (including errors) to JSON string for ToolMessage
                result_str = json.dumps(result_data, default=str) # Use default=str for safety

            except Exception as e:
                print(f"Error executing or processing tool '{tool_name}': {e}")
                print(traceback.format_exc())
                result_str = json.dumps({"error": f"Tool execution/processing failed: {e}"})
                # Add error representation to accumulated data? Optional, depends on desired behavior.
                # accumulated_data.append({"error": f"Tool execution failed: {tool_name}", "details": str(e)})

        # Append ToolMessage with results (or errors) for the conversation history
        tool_outputs.append(ToolMessage(content=result_str, tool_call_id=tool_call_id, name=tool_name))
        print(f"Tool {tool_name} (id: {tool_call_id}) processing complete.")

    print(f"Total accumulated data points after execution: {len(accumulated_data)}")

    # Return tool messages to be added to history and the updated data
    return {
        "messages": tool_outputs,
        "retrieved_data": accumulated_data,
    }


# --- Conditional Edge Logic ---

def should_continue_retrieval(state: RetrievalState) -> Literal["continue_retrieval", "synthesize"]:
    """
    Determines if the retrieval loop should continue (if calls planned)
    or stop and proceed to synthesis (no calls planned, general question, or error).
    """
    print("--- Checking Should Continue Retrieval ---")
    decision = state.get("retriever_decision")

    # If a decision exists AND it has retrieval calls planned, continue.
    if decision and decision.retrieval_calls:
        print(f"Decision: Continue Retrieval ({len(decision.retrieval_calls)} calls planned)")
        return "continue_retrieval"
    else:
        # Otherwise (no decision, no calls, general question identified, error), stop retrieval loop.
        reason = "No calls planned"
        if decision and decision.reasoning.startswith("GENERAL_QUESTION:"):
            reason = "General question identified"
        elif not decision:
             reason = "No retriever decision found (error or initial state)"
        elif decision and "Error" in decision.reasoning:
             reason = "Error occurred in retriever"

        print(f"Decision: Stop Retrieval & Synthesize ({reason})")
        return "synthesize"


# --- Node: Call Synthesizer LLM (Handles Conversation History) ---

def call_synthesizer_model(state: RetrievalState) -> Dict[str, Any]:
    """
    Generates the final response via LLM, considering conversation history.
    Uses different prompts based on whether it's handling a general question
    or synthesizing from retrieved data (or lack thereof) for the latest query.
    """
    model_name = getattr(synthesizer_llm, 'model', 'Synthesizer LLM')
    print(f"--- Entering Synthesizer Node ({model_name}) ---")

    messages = state['messages']
    retrieved_data = state.get("retrieved_data", [])
    last_retriever_decision = state.get("retriever_decision")
    reasoning = last_retriever_decision.reasoning if last_retriever_decision else ""

    # Find the latest user query to ensure the response addresses it
    latest_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if not latest_human_message:
       print("Error: No HumanMessage found in history for synthesizer.")
       # Provide an error message as the final output
       return {"messages": [AIMessage(content="System Error: Cannot identify the latest user query to respond to.")]}
    current_user_query = latest_human_message.content
    print(f"Latest User Query for Synthesizer: '{current_user_query}'")

    # Format conversation history and retrieved data for the prompt
    conversation_history_str = format_message_history_for_prompt(messages)
    context_str = format_retrieved_data_for_prompt(retrieved_data) # Helper handles empty case

    synthesis_prompt_content = ""
    prompt_type = "unknown"

    # --- Determine Prompt Type based on Retriever's Decision and Data ---

    # Case 1: General Question (Identified by Retriever for the *latest* query)
    if reasoning.startswith("GENERAL_QUESTION:"):
        prompt_type = "general_question"
        print("Synthesizer: Handling identified general question via LLM.")
        synthesis_prompt_content = f"""
Jūs esate AI asistentas, dirbantis Tauragės miesto savivaldybėje. Atsakykite į paskutinį vartotojo klausimą mandagiai ir tiesiogiai lietuvių kalba, atsižvelgdami į pokalbio istoriją. Neminite dokumentų paieškos, nebent klausimas yra būtent apie jūsų galimybes ieškoti informacijos.

Pokalbio Istorija:
--- ISTORIJOS PRADŽIA ---
{conversation_history_str}
--- ISTORIJOS PABAIGA ---

Paskutinis Vartotojo Klausimas: "{current_user_query}"

Užduotis: Pateikite tiesioginį, pokalbio stiliaus atsakymą į "Paskutinį Vartotojo Klausimą", naudodami istoriją kontekstui suprasti.
"""

    # Case 2: Retrieval was attempted for the latest query, but no relevant data was found
    elif not retrieved_data:
        prompt_type = "no_data_found"
        print("Synthesizer: No data retrieved for the latest query. Generating acknowledgement via LLM.")
        synthesis_prompt_content = f"""
Jūs esate AI asistentas, dirbantis Tauragės miesto savivaldybėje. Vartotojas uždavė klausimą, bet atlikus paiešką jūsų turimuose dokumentuose, nebuvo rasta jokios relevantiškos informacijos jo paskutinei užklausai atsakyti. Atsižvelkite į pokalbio istoriją.

Pokalbio Istorija:
--- ISTORIJOS PRADŽIA ---
{conversation_history_str}
--- ISTORIJOS PABAIGA ---

Paskutinis Vartotojo Klausimas: "{current_user_query}"

Užduotis: Mandagiai informuokite vartotoją lietuvių kalba, kad pagal jo paskutinę užklausą "{current_user_query}" konkrečios informacijos jūsų turimuose šaltiniuose rasti nepavyko. Nesiūlykite ieškoti kitur, tiesiog konstatuokite faktą.
"""

    # Case 3: Retrieval was successful, synthesize answer from data for the latest query
    else:
        prompt_type = "synthesis_with_data"
        print("Synthesizer: Proceeding with synthesis based on retrieved data via LLM.")
        synthesis_prompt_content = f"""
Jūs esate naudingas asistentas, dirbantis Tauragės miesto savivaldybėje. Jūsų užduotis yra atsakyti į PASKUTINĮ vartotojo klausimą remiantis TIK PATEIKTA informacija iš dokumentų ir atsižvelgiant į pokalbio istoriją kontekstui.

Pokalbio Istorija:
--- ISTORIJOS PRADŽIA ---
{conversation_history_str}
--- ISTORIJOS PABAIGA ---

Paskutinis Vartotojo Klausimas: "{current_user_query}"

Pateikta Informacija Atsakymui (Kontekstas):
--- KONTEKSTO PRADŽIA ---
{context_str}
--- KONTEKSTO PABAIGA ---

Instrukcijos:
1.  Sutelpkite dėmesį į atsakymą į **Paskutinį Vartotojo Klausimą**.
2.  Naudokite **Pokalbio Istoriją** tik kontekstui suprasti (pvz., įvardžiams paaiškinti).
3.  Formuluokite atsakymą **GRIEŽTAI REMIANTIS** pateikta informacija **Kontekste**.
4.  Jei Kontekste esanti informacija yra nepakankama atsakyti į paskutinį klausimą, aiškiai tai nurodykite (pvz., "Pateiktuose dokumentuose nėra informacijos apie X."). **NEGALIMA IŠGALVOTI INFORMACIJOS.**
5.  Jei įmanoma ir prasminga, nurodykite šaltinius (pvz., `Šaltinis [chunk_id]`), kuriais rėmėtės.
6.  Atsakykite aiškiai, konkrečiai ir mandagiai Į LIETUVIŲ KALBĄ.
7.  Būkite objektyvus ir nešališkas.

Užduotis: Parašykite galutinį atsakymą į "Paskutinį Vartotojo Klausimą", remdamiesi TIK pateiktu Kontekstu ir atsižvelgdami į Istoriją.
"""

    # --- Invoke the Synthesizer LLM ---
    print(f"--- Sending prompt (type: {prompt_type}) to Synthesizer LLM ---")
    # print(synthesis_prompt_content) # Uncomment for full prompt debugging

    try:
        # Use invoke for single-turn generation
        final_response_message = synthesizer_llm.invoke([HumanMessage(content=synthesis_prompt_content)])
        # Ensure the response is AIMessage
        if not isinstance(final_response_message, AIMessage):
             final_response_message = AIMessage(content=str(final_response_message))

        final_answer_content = final_response_message.content
        print(f"Synthesizer LLM Response: {final_answer_content}")

    except Exception as e:
         print(f"!!! Error during synthesizer LLM call: {e} !!!")
         print(traceback.format_exc())
         final_answer_content = f"Atsiprašome, įvyko techninė klaida generuojant atsakymą. Klaida: {e}"
         # Ensure the error message is wrapped in AIMessage for consistent output type
         final_response_message = AIMessage(content=final_answer_content)

    # Return *only* the final AI message for the graph output
    # The state's 'messages' field is automatically updated by add_messages with this
    return {"messages": [final_response_message]}
