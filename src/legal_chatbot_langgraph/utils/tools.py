import json
from langchain_community.tools import tool
from langchain_community.utilities import RequestsWrapper
import requests
from bs4 import BeautifulSoup

# Legal Fee Calculator Tool
@tool
def calculate_legal_fee(case_type: str, hours: float) -> dict:
    """
    Calculates the estimated legal fee based on the type of case and number of hours.
    """
    fee_per_hour = {
        "civil": 150,
        "criminal": 200,
        "family": 180,
        "corporate": 250,
    }
    rate = fee_per_hour.get(case_type.lower(), 150)
    estimated_fee = rate * hours
    # Return a dictionary containing the result
    return {"result": f"The estimated legal fee for a {case_type} case over {hours} hours is ${estimated_fee:.2f}."}

# Recent Case Summary Tool
@tool
def fetch_recent_case_summaries(topic: str) -> dict:
    """
    Fetches recent case summaries based on the provided topic.
    """
    case_summaries = {
        "unemployment": "In recent cases regarding unemployment benefits, courts have upheld extended benefits for those impacted by economic disruptions.",
        "discrimination": "Recent rulings on discrimination have reinforced the importance of workplace equality and compensation for impacted employees.",
        "contract": "Recent contract cases have emphasized strict adherence to written clauses, especially in financial agreements.",
    }
    # Return a dictionary containing the result
    summary = case_summaries.get(
        topic.lower(), "No recent case summaries available for the given topic."
    )
    return {"result": summary}

# Requests Tool for Social Support Data
@tool
def fetch_social_support_data(url: str) -> dict:
    """
    Fetches and extracts meaningful social support information from the provided URL.
    """
    try:
        response = requests.get(url)
        # print(f"Fetching data from URL: {url}")
        
        # # Parse the HTML content
        # soup = BeautifulSoup(response, 'html.parser')
        
        # # Extract specific data (customize this part based on the webpage structure)
        # main_content = soup.find('div', class_='elementor-widget-wrap')  # Replace 'content' with the appropriate class or ID
        
        # if main_content:
        #     # Extract text without unnecessary HTML tags
        #     text_content = main_content.get_text(separator='\n', strip=True)
        return {"result": response}
        # else:
        #     return {"error": "Failed to locate the relevant content on the page."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}


@tool
def statute_api_tool(query: str) -> str:
    """Searches for legal statutes based on the user query and returns JSON data."""
    print(f"Simulating API call to Statute Search API for query: '{query}'")
    mock_statute_api_response = {
        "query": query,
        "results": [
            {
                "statute_name": "Defamation Act 2009 (Ireland)",
                "jurisdiction": "Ireland",
                "sections": [
                    {"section_number": "Section 2", "title": "Defamation defined", "summary": "Defamation consists of the publication, in spoken or written form, of an imputation that tends to injure a person’s reputation in the eyes of reasonable members of society."},
                    {"section_number": "Section 6", "title": "Fair and reasonable publication on a matter of public interest", "summary": "It is a defence to a defamation action for the defendant to prove that— (a)the defendant was justified in publishing the statement complained of, and (b)the statement was of public interest and was published for the benefit of the public."}
                ]
            }
        ]
    }
    api_response_json = json.dumps(mock_statute_api_response, indent=2)
    return api_response_json

@tool
def case_law_api_tool(query: str) -> str:
    """Searches for case law based on the user query and returns JSON data."""
    print(f"Simulating API call to Case Law Search API for query: '{query}'")
    mock_case_law_api_response = {
        "query": query,
        "results": [
            {
                "case_name": "Miranda v. Arizona, 384 U.S. 436 (1966)",
                "court": "Supreme Court of the United States",
                "summary": "In Miranda v. Arizona (1966), the Supreme Court ruled that criminal suspects must be informed of their constitutional rights, including the right to consult with an attorney and the right to remain silent, prior to police interrogation. This became known as 'Miranda rights' or 'Miranda warning.'"
            },
            {
                "case_name": "Gideon v. Wainwright, 372 U.S. 335 (1963)",
                "court": "Supreme Court of the United States",
                "summary": "The Supreme Court in Gideon v. Wainwright (1963) ruled that states are required to provide legal counsel to indigent defendants in criminal cases. This case significantly expanded the right to counsel, ensuring that even those who cannot afford an attorney receive legal representation."
            }
        ]
    }
    api_response_json = json.dumps(mock_case_law_api_response, indent=2)
    return api_response_json

tools = [
    statute_api_tool,
    case_law_api_tool,
    fetch_recent_case_summaries,
    calculate_legal_fee,
    fetch_social_support_data,
]
