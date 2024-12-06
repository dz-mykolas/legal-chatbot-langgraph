import asyncio
from colorama import Fore, Style, init
import os

from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import requests
os.environ["USER_AGENT"] = "LegalChatbot/1.0 (Windows NT 10.0; Win64; x64) LangGraph"

from langchain_core.tools import tool
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from openai import OpenAI

from langchain.agents.openai_assistant.base import OpenAIAssistantAction
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Define Tools with Descriptions
@tool
def calculate_legal_fee(case_type: str, hours: float) -> str:
    """
    Calculates the estimated legal fee based on the type of case and number of hours.
    """
    print("Calculating legal fee...")
    fee_per_hour = {
        "civil": 150,
        "criminal": 200,
        "family": 180,
        "corporate": 250,
    }
    rate = fee_per_hour.get(case_type.lower(), 150)
    estimated_fee = rate * hours
    return f"The estimated legal fee for a {case_type} case over {hours} hours is ${estimated_fee:.2f}."

@tool
def fetch_recent_case_summaries(topic: str) -> str:
    """
    Fetches recent case summaries based on the provided topic.
    """
    print("Fetching recent case summaries...")
    case_summaries = {
        "unemployment": "In recent cases regarding unemployment benefits, courts have upheld extended benefits for those impacted by economic disruptions.",
        "discrimination": "Recent rulings on discrimination have reinforced the importance of workplace equality and compensation for impacted employees.",
        "contract": "Recent contract cases have emphasized strict adherence to written clauses, especially in financial agreements.",
    }
    return case_summaries.get(
        topic.lower(), "No recent case summaries available for the given topic."
    )

@tool
def fetch_and_extract_social_support_data(query: str) -> str:
    """
    Fetches and analyzes social support data from a URL using LangChain and OpenAI API.
    """
    try:
        # Step 1: Fetch content from the URL
        url = "https://taurage.lt/veiklos-sritys/socialine-parama/"
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to fetch the URL content. Status Code: {response.status_code}"
        
        # Parse the content
        soup = BeautifulSoup(response.content, "html.parser")
        extracted_text = soup.get_text()

        # Trim content to a manageable size
        content_to_analyze = extracted_text

        # Step 2: Initialize OpenAI LLM with LangChain
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "OpenAI API key not found in environment variables."
        
        
        content = (
            "The following is a summary of social support available in Tauragė: " + content_to_analyze
        )

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(
            [{"role": "user", "content": content}, {"role": "user", "content": query}],
        )


        return response

    except Exception as e:
        return f"An error occurred: {e}"
    
# Step 2: Initialize Your Custom Assistant using OpenAIAssistantRunnable
def initialize_assistant():
    # Initialize the OpenAI client
    client = OpenAI()
     # Define the instructions for the assistant
    instructions = """
    Create a legal chatbot that accurately answers user questions related to legal issues and refuses topics outside of the legal context. The chatbot should provide support primarily in Lithuanian, English, and Russian, in that order. If a question is asked in any other language, include a disclaimer specifying that support in that language is limited, and provide it using the language of the question.

    Ensure that the chatbot responds exclusively from a legal perspective, avoiding personal advice, medical guidance, or any areas beyond its expertise.
    
    If the user's query is related to social support specifically within the Tauragė municipality in Lithuania, use the `fetch_and_extract_social_support_data` tool to retrieve information from the relevant webpage. The query should be passed as a parameter to this tool.
    For example, if the user asks "What social support is available for single mothers in Tauragė?", use the tool with that query. If the user's question is about general legal matters or social support in other regions, do not use this tool.

    # Steps

    1. **Identify the Language and Legal Context**:
       - Identify the language used (Lithuanian, English, Russian, or other).
       - Determine if the content is within a legal context.
         - If it pertains to legal assistance, proceed with an answer.
         - If it falls outside legal topics, kindly indicate that the chatbot can only answer legal queries, using the language of the question.
         - If the question is asked in a language other than Lithuanian, English, or Russian, provide a disclaimer that the support in that language may be limited. The disclaimer must be in the language used in the question.

    2. **Provide a Legal Answer** (if applicable):
       - Answer the question accurately and strictly within the legal context.
       - Avoid adding any non-legal advice or suggestions.

    # Output Format
    - Responses should be formulated in the same language as the question.
    - If the language is unsupported (non-Lithuanian, English, Russian), include a disclaimer in the appropriate language about the limited support available.

    # Examples

    **Example 1:**
    - **Input:** (Lithuanian) "Kokios yra darbuotojo teisės atleidimo metu?"
    - **Output:** (Lithuanian) "Darbuotojo teisės atleidimo metu apima įspėjimo laikotarpį, kompensaciją ir... (provides the specific legal guidance strictly related to the topic)."

    **Example 2:**
    - **Input:** (German) "Welche Rechte habe ich bei einer Kündigung?"
    - **Output:** (German) "Bitte beachten Sie, dass der Support auf Deutsch eingeschränkt ist. Sie haben bei einer Kündigung gemäß den gesetzlichen Vorschriften möglicherweise das Recht auf... (provides the best possible limited legal guidance)."

    # Notes
    - The chatbot should always be polite, direct, and refrain from opinions.
    - Ensure disclaimers are respectful and communicate the limitation of support clearly.
    - Disclaimers must be included for languages other than Lithuanian, English, and Russian, tailored to the language used.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_legal_fee",
                "description": "Calculate legal fees based on case type and hours.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "case_type": {"type": "string", "description": "The type of legal case."},
                        "hours": {"type": "number", "description": "The number of hours the case is expected to take."}
                    },
                    "required": ["case_type", "hours"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_recent_case_summaries",
                "description": "Fetch summaries of recent legal cases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "The topic to search for case summaries."}
                    },
                    "required": ["topic"]
                }
            }
        },
         {
            "type": "function",
            "function": {
                 "name": "fetch_and_extract_social_support_data",
                 "description": "Fetch social support data from a URL.",
                 "parameters": {
                     "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to search for in the social support data."}
                    },
                    "required": ["query"]
                 }
             }
         }
    ]

    # Create the assistant using the OpenAI client
    assistant = client.beta.assistants.create(
        name="Legal Assistant",
        instructions=instructions,
        tools=tools,
        model="gpt-4o-2024-11-20"
    )
    assistant_id=assistant.id
    assistant_runnable = OpenAIAssistantRunnable(
        assistant_id=assistant_id,
        tools=[calculate_legal_fee,fetch_recent_case_summaries, fetch_and_extract_social_support_data],
        as_agent=True,
    )

    return assistant_runnable, assistant_id, assistant

# Step 3: Define the Chat Loop (Asynchronous)
async def continuous_chat():
    print(Fore.CYAN + "Start chatting with the AI (type 'exit' to stop):")
    assistant_runnable, assistant_id, assistant = initialize_assistant()
    thread_id = None

    while True:
        user_input = input(Fore.GREEN + "You: ")
        if user_input.lower() == "exit":
            print(Fore.RED + "Ending conversation.")
            break

        try:
            # Prepare input data with content and optional thread_id
            input_data = {"content": user_input}
            if thread_id:
                input_data["thread_id"] = thread_id

            # Invoke the assistant
            response = await assistant_runnable.ainvoke(input_data)

            # Process the response
            if isinstance(response, list):
                for step in response:
                    if isinstance(step, OpenAIAssistantAction):
                        # Handle tool calls
                        tool_name = step.tool
                        tool_args = step.tool_input
                        
                        if tool_name == "calculate_legal_fee":
                            result = await calculate_legal_fee.ainvoke(tool_args)
                            print(Fore.YELLOW + f"Tool Result: {result}")
                        
                        elif tool_name == "fetch_recent_case_summaries":
                            result = await fetch_recent_case_summaries.ainvoke(tool_args)
                            print(Fore.YELLOW + f"Tool Result: {result}")
                            
                        elif tool_name == "fetch_and_extract_social_support_data":
                            result = await fetch_and_extract_social_support_data.ainvoke(tool_args)
                            print(Fore.YELLOW + f"Tool Result: {result}")
                    
                    elif hasattr(step, 'return_values'):
                        if 'output' in step.return_values:
                            print(Fore.BLUE + Style.BRIGHT + f"AI: {step.return_values['output']}")
                        if 'thread_id' in step.return_values:
                            thread_id = step.return_values['thread_id']
            
            else:
                # Handle single response
                if isinstance(response, OpenAIAssistantAction):
                    tool_name = response.tool
                    tool_args = response.tool_input
                    
                    if tool_name == "calculate_legal_fee":
                        result = await calculate_legal_fee.ainvoke(tool_args)
                        print(Fore.YELLOW + f"Tool Result: {result}")
                    
                    elif tool_name == "fetch_recent_case_summaries":
                        result = await fetch_recent_case_summaries.ainvoke(tool_args)
                        print(Fore.YELLOW + f"Tool Result: {result}")
                        
                    elif tool_name == "fetch_and_extract_social_support_data":
                        result = await fetch_and_extract_social_support_data.ainvoke(tool_args)
                        print(Fore.YELLOW + f"Tool Result: {result}")
                
                elif hasattr(response, 'return_values'):
                    if 'output' in response.return_values:
                        print(Fore.BLUE + Style.BRIGHT + f"AI: {response.return_values['output']}")
                    if 'thread_id' in response.return_values:
                        thread_id = response.return_values['thread_id']

        except Exception as e:
            print(Fore.RED + f"Error: {e}")
            if hasattr(e, 'response') and e.response:
                print(Fore.RED + f"  Response Status Code: {e.response.status_code}")
                print(Fore.RED + f"  Response Content: {e.response.text}")
                
# Step 4: Run the Chat
if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(continuous_chat())
    