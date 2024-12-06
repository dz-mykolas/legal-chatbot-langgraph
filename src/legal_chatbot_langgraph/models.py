import openai
from langchain_openai import ChatOpenAI
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from langchain_core.messages import HumanMessage

def initialize_assistant():
    assistant_id = "asst_I4EEDWIC0ZeemsBu58F5olOE"  # Your assistant ID
    return OpenAIAssistantRunnable(assistant_id=assistant_id, as_agent=True)

def create_new_thread():
    """Creates a new thread using the OpenAI API and returns its ID."""
    try:
        response = openai.beta.threads.create()
        return response.id
    except Exception as e:
        print(f"Error creating thread: {e}")
        return None
    