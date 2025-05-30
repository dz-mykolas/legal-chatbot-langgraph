from enum import Enum, StrEnum
import os
from typing import Callable, Dict, Optional, Type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END
from pydantic import BaseModel

def llm_selector(model_name: str, temperature: float = 0):
    """Selects and initializes a LangChain ChatModel based on the model name."""

    def gpt_constructor(name: str, temp: float):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return ChatOpenAI(model=name, temperature=temp)

    def gemini_constructor(name: str, temp: float):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return ChatGoogleGenerativeAI(model=name, temperature=temp)

    # Registry of model prefixes and corresponding factory functions
    LLM_REGISTRY: Dict[str, Callable[[str, float], object]] = {
        "gpt": gpt_constructor,
        "gemini": gemini_constructor,
    }

    for prefix, constructor in LLM_REGISTRY.items():
        if model_name.startswith(prefix):
            return constructor(model_name, temperature)

    raise ValueError(f"Unknown model prefix for model: {model_name}")

class EnumWithHelpers(StrEnum):
    @classmethod
    def as_dict(cls, *members):
        return {m: m for m in members}

    @classmethod
    def if_tools(cls, state: BaseModel):
        last = state.messages[-1]
        return cls.tools if getattr(last, "tool_calls", None) else cls.end
    