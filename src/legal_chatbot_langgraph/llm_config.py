# legal_chatbot_langgraph/llm_config.py

import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the action schema needed for structured output
from legal_chatbot_langgraph.schemas import RetrieverAction

# Check API key environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

def llm_selector(model_name: str, temperature: float = 0):
    """Selects and initializes a LangChain ChatModel."""
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("gemini-"):
        # Ensure API key is configured via env vars like GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model prefix for model: {model_name}")

# Load environment variables (ensure .env is loaded before this file is imported)
# Typically done in main.py or the entry point script
# from dotenv import load_dotenv
# load_dotenv() # Load here or in main.py

# Load model names and temperatures from environment or use fallbacks
RETRIEVER_MODEL_NAME = os.getenv("LLM_MODEL_RETRIEVER", "gemini-2.0-flash")
SYNTHESIZER_MODEL_NAME = os.getenv("LLM_MODEL_SYNTHESIZER", "gemini-2.0-flash") # Updated Gemini model name
RETRIEVER_TEMP = float(os.getenv("LLM_TEMPERATURE_RETRIEVER", "0.1"))
SYNTHESIZER_TEMP = float(os.getenv("LLM_TEMPERATURE_SYNTHESIZER", "0.0"))

# Instantiate LLMs
retriever_llm = llm_selector(RETRIEVER_MODEL_NAME, RETRIEVER_TEMP)
synthesizer_llm = llm_selector(SYNTHESIZER_MODEL_NAME, SYNTHESIZER_TEMP)

# Instantiate the structured LLM for the retriever
# Ensure the schema passed matches the expected output structure
structured_retriever_llm = retriever_llm.with_structured_output(RetrieverAction)

print(f"Retriever LLM: {RETRIEVER_MODEL_NAME} (Temp: {RETRIEVER_TEMP})")
print(f"Synthesizer LLM: {SYNTHESIZER_MODEL_NAME} (Temp: {SYNTHESIZER_TEMP})")
print(f"Structured Retriever LLM ready (Output Schema: {RetrieverAction.__name__}).")
