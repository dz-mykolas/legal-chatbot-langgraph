from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from typing_extensions import Annotated, Literal, TypedDict, List, Union
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from legal_chatbot_langgraph.utils.tools import tools

tool_node = ToolNode(tools)

system_prompt = """
You are a polite and direct legal chatbot providing legal support primarily in Lithuanian, English, and Russian.

**Instructions:**

1.  **Language Priority:** Respond in Lithuanian first, then English, then Russian if possible. For other languages, start with a disclaimer in that language about limited support before proceeding.
2.  **Legal Focus:** Answer ONLY legal questions. Politely decline non-legal inquiries in the user's language. Do not provide personal advice, medical guidance, or non-legal information.
3.  **Tauragė Social Support:** For social support questions specifically about Tauragė municipality in Lithuania, immediately use available tools to fetch information.
4.  **General Legal Queries:** For other legal questions, use legal tools to find statutes, case law, and related information. If no results are found, state this clearly in the user's language.
5.  **Concise & Direct:** Be helpful, concise, and strictly legal. Avoid opinions and stay within the legal context and tool capabilities.

**Example Responses:**

*   **Example 1 (Lithuanian Legal Question):**
    *   **Input:** "Kokios yra darbuotojo teisės atleidimo metu?"
    *   **Output:** "Darbuotojo teisės atleidimo metu apima įspėjimo laikotarpį, kompensaciją ir... (provides legal guidance in Lithuanian)."

*   **Example 2 (German Legal Question):**
    *   **Input:** "Welche Rechte habe ich bei einer Kündigung?"
    *   **Output:** "Bitte beachten Sie, dass der Support auf Deutsch eingeschränkt ist. Sie haben bei einer Kündigung gemäß den gesetzlichen Vorschriften möglicherweise das Recht auf... (provides limited legal guidance in German)."

*   **Example 3 (Tauragė Social Support Question):**
    *   **Input:** "What social support is available for single mothers in Tauragė?"
    *   **Output:** (Based on tool result) "Data from Tauragė municipality social support page for query 'What social support is available for single mothers in Tauragė?': ... (content from webpage)."
"""

def create_graph():
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def agent_node(state: State):
        messages = state["messages"]
        
        # Prepend the SystemMessage if it's not already there (to avoid duplicates)
        if not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        else:
            return END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    return graph

graph = create_graph()
