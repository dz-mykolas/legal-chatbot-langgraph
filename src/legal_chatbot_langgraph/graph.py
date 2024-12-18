from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, Literal, TypedDict, List, Union
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

members = ["summarizer", "searcher"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    f"""
    Manage tasks by deciding whether to complete the task yourself or delegate it to one of the workers in the {members} list. Assess the task request critically and choose the most appropriate next step.
    Consider if the given task could be efficiently handled alone or if it requires delegation to a specific worker. Provide reasoning for your decision.
    
    # Steps
    1. **Analyze the Task:** Understand the user's request in detail.
    2. **Evaluate Options:** Determine if you can complete the task independently or whether it should be delegated to a worker from {members}.
    3. **Decide on Next Step:** Choose to either complete the task yourself or select a specific worker from {members} to handle the task.
    4. **Provide Reasoning:** Offer a brief explanation of your decision-making process.

    # Output Format
    Respond with a JSON object structured as follows:
    - `next_node`: Either one of the {members} or 'FINISHED' if the task is independently completed.
    - `content`: A concise explanation of your reasoning or the completed task.

    # Examples

    **Example 1**
    _Input:_
    User requests technical assistance with setting up a development environment.
    _Output:_
    ```json
    {{
    "next_node": "FINISHED",
    "content": "The task is completed by providing detailed setup instructions suitable for beginners."
    }}
    ```

    **Example 2**
    _Input:_
    User needs a comprehensive financial report analysis.
    _Output:_
    ```json
    {{
    "next_node": "financial_analyst",
    "content": "The task requires detailed financial analysis, best delegated to the financial analyst for specialized expertise."
    }}
    ```

    # Notes
    - Consider the complexity and specialization required for the task.
    - Always ensure the next step chosen maximizes efficiency and effectiveness in task completion.
    - Provide clear and concise reasoning regardless of the decision to handle the task or delegate it.
    """
)
system_prompt = SystemMessage(content=system_prompt)

class Router(TypedDict):
    response: Annotated[str, ..., "A conversational response to the user's query or explanation of taken action"]
    next: Literal[*options]

# class ConversationResponse(BaseModel):
#     response: Annotated[str, ..., "A conversational response to the user's query"]
#     next: Literal[*options]

# class FinalResponse(BaseModel):
#     final_output: Union[Router, ConversationResponse]

def create_graph():
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    llm = ChatOpenAI(model="gpt-4o-mini")

    def supervisor_node(state: State) -> Command[Literal[*members, END]]:
        response = llm.with_structured_output(Router).invoke([system_prompt] + state["messages"])
        print("Response from AI: ")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)
    
    def summarize_node(state: State) -> Command[Literal["supervisor"]]:
        result = llm.invoke(state["messages"])

        return Command(
            update={
                "messages": [
                    AIMessage(content=result.content)
                ]
            },
            goto="supervisor",
        )
    
    def search_node(state: State) -> Command[Literal["supervisor"]]:
        result = llm.invoke(state["messages"])

        return Command(
            update={
                "messages": [
                    AIMessage(content=result.content, name="searcher")
                ]
            },
            goto="supervisor",
        )

    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("summarizer", summarize_node)
    builder.add_node("searcher", search_node)

    graph = builder.compile()

    return graph
    # initial_messages = [
    #     # HumanMessage(content="Summarize this: Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."),
    #     # HumanMessage(content="Search about the recent cases on discrimination."),
    #     HumanMessage(content="Help me with calculating a legal fee"),
    #     HumanMessage(content="end"),
    # ]

    # for initial_message in initial_messages:
    #     print(f"\n--- Starting new conversation with: {initial_message.content} ---")
    #     result = graph.invoke({"messages": [initial_message]})
    #     print(f"Final Result: {result}")
