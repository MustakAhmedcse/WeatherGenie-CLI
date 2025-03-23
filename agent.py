# agent.py
import re
from weather_api import get_live_weather
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI  # Using the updated package for chat models
from dotenv import load_dotenv

load_dotenv()

# Initialize the ChatOpenAI model (e.g., using GPT-4o-mini)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

def is_bangla(text: str) -> bool:
    """
    Checks if the given text contains any Bengali characters.
    """
    # Bengali Unicode range: U+0980 to U+09FF
    return bool(re.search(r'[\u0980-\u09FF]', text))

def extract_location(query: str) -> str:
    """
    Attempts to extract the location from the query.
    First, it uses a regex to look for patterns like 'in [Location]'.
    If that fails (e.g., for queries in Bangla), it falls back to using the LLM.
    """
    # Regex for English queries (e.g., "in Dhaka today")
    match = re.search(r"in\s+([A-Za-z\s]+?)(\s+(today|now))?[?\.]?$", query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Fallback: use the LLM to extract the location
        fallback_prompt = (
            f"Extract the location (preferably in English) from the following weather query:\n\n"
            f"'{query}'\n\n"
            "Just provide the location name."
        )
        response = llm([HumanMessage(content=fallback_prompt)])
        location = response.content.strip()
        return location

def decide_response_type(query: str) -> str:
    """
    Uses the LLM to decide if the query requires a live weather API call or a generic response.
    Returns 'live', 'generic', or 'out-of-domain'.
    If the query is in Bangla, instruct the LLM to answer in Bangla.
    """
    decision_prompt = (
        "You are a weather assistant. Given the following query, decide if it requires a live weather API call (reply with 'live'), "
        "a generic weather advice response (reply with 'generic'), or if it is out of your domain (reply with 'out-of-domain').\n\n"
        f"Query: {query}\n\n"
    )
    if is_bangla(query):
        decision_prompt += "Please respond in Bangla and not used নমস্কার! in your aswer"
    else:
        decision_prompt += "Your answer:"
        
    response = llm([HumanMessage(content=decision_prompt)])
    decision = response.content.strip().lower()
    return decision

def generate_final_response(query: str) -> str:
    """
    Based on the decision, either fetch live weather data or generate a generic weather message.
    The final message is composed by the LLM and, if the input is in Bangla, the response will be in Bangla.
    """
    decision = decide_response_type(query)
    print("\n\n OpenAI API Decision : " + decision+" \n\n")

    if decision == "live":
        location = extract_location(query)
        if not location:
            return "I couldn't determine the location from your query. Please try again."
            
        weather_data = get_live_weather(location)
        if "error" in weather_data:
            return weather_data["error"]
        final_prompt = (
            f"The live weather data for {location} is as follows: {weather_data}. "
            "Compose a friendly final response that includes these details and offers advice."
        )
        if is_bangla(query):
            final_prompt += " Please answer in Bangla."
        final_response = llm([HumanMessage(content=final_prompt)])
        return final_response.content.strip()
    
    elif decision == "generic":
        final_prompt = (
            f"Compose a friendly weather advice for the following query without using live data: {query}."
        )
        if is_bangla(query):
            final_prompt += " Please answer in Bangla."
        final_response = llm([HumanMessage(content=final_prompt)])
        return final_response.content.strip()
    
    else:
        return "I apologize for not answering your question. I am only serve weather related questions."

# # Optional testing
# if __name__ == "__main__":
#     # Example Bangla query
#     sample_query = "তুমি কি আমাকে বাংলাদেশের আবহাওয়া বলতে পারো?"
#     print(generate_final_response(sample_query))
