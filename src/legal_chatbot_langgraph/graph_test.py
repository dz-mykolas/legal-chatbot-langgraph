from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict, Annotated, List, Literal, Optional, Callable
from langchain.schema.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import inspect

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_hop: Optional[str]
    iteration_count: int  # Added iteration counter to state

def node_a(state: State) -> Command[Literal["b", END]]:
    print("Node A executed")
    next_node = "b"
    return Command(
        update={
            "messages": [AIMessage(content="Result from Node A")],
            "next_hop": next_node,
            "iteration_count": state.get("iteration_count", 0) + 1, # Increment counter
        },
        goto="conditional_router"
    )

def node_b(state: State) -> Command[Literal["a", END]]:
    print("Node B executed")
    next_node = "a"
    return Command(
        update={
            "messages": [AIMessage(content="Result from Node B")],
            "next_hop": next_node,
            "iteration_count": state.get("iteration_count", 0) + 1, # Increment counter
        },
        goto="conditional_router"
    )

def conditional_router(state: State) -> Command[Literal["a", "b", END]]:
    next_hop_decision = state.get("next_hop")
    iteration_count = state.get("iteration_count", 0)

    print(f"Conditional Router deciding next hop: {next_hop_decision}, Iteration: {iteration_count}")

    if iteration_count >= 5:  # Stop after 5 iterations (example limit)
        print("Iteration limit reached, going to END")
        return Command(goto=END, update={"next_hop": None})
    elif next_hop_decision == "b":
        return Command(goto="b")
    elif next_hop_decision == "a":
        return Command(goto="a")
    else:
        return Command(goto=END)


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("conditional_router", conditional_router)
builder.add_edge(START, "a")

builder.add_conditional_edges(
    "conditional_router",
    lambda state: state.get("next_hop"),
    {
        "b": "b",
        "a": "a",
        None: END,
    }
)
builder.add_edge("a", "conditional_router")
builder.add_edge("b", "conditional_router")


graph = builder.compile()

result = graph.invoke({"messages": [], "iteration_count": 0})
print("Graph finished")
print(f"Final result: {result}")
