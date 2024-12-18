from langchain_core.messages import AIMessage
from langchain_community.tools import tool
from langchain_community.utilities import RequestsWrapper

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
    rate = fee_per_hour.get(case_type.lower(), 150)  # Default to civil if not found
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
    Fetches the latest information about social support from the provided URL.
    """
    requests_wrapper = RequestsWrapper()
    try:
        response = requests_wrapper.get(url)
        if response.status_code == 200:
            # Return a dictionary containing the result
            return {"result": response.text}
        else:
            return {"error": f"Failed to fetch data. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
