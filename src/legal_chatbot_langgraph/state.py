from typing import List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class State(BaseModel):
    """
    Represents the state of the conversation.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: List[BaseMessage] = Field(default=[], extra="allow")
    next_node: Optional[str] = Field(default=None, extra="allow")
    