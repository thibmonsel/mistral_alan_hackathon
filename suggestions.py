import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

# Retrieve the API key from environment variables
API_KEY = "your api key"
API_URL = "https://api.mistral.ai/v1/chat/completions"

if not API_KEY:
    raise ValueError("API key not found. Please set it in the .env file.")

# Function to build prompt for suggested questions
def generate_question_prompt(conversation_history, context):
    """
    Creates the prompt to be used for question generation based on the conversation and medical context.
    The prompt is specifically asking for a list of 4 questions that the patient might ask.
    """
    prompt = f"""
    You are an assistant helping a cancer patient. Based on the conversation, suggest 4 relevant and helpful questions that the patient can ask their doctor or that can help clarify their situation.

    Conversation history:
    {conversation_history}

    Context (if available):
    {context}

    Provide the 4 questions in a numbered list, separated by new lines, without any other text.
    """
    return prompt

# Function to interact with Mistral Small model and retrieve question suggestions
def get_suggested_questions(conversation_history, context):
    """
    Calls the Mistral Small API to generate 4 questions based on the conversation and context.
    """
    prompt = generate_question_prompt(conversation_history, context)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,  # Increase max tokens to accommodate 4 questions
        "temperature": 0.1
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to load context from a text file
def load_context_from_file(file_path):
    """
    Loads the context from a .txt file.
    """
    try:
        with open(file_path, 'r') as file:
            context = file.read()
        return context
    except FileNotFoundError:
        return "Context file not found."

# Example usage
if __name__ == "__main__":
    # Load the context from a .txt file
    context_file_path = "ehr_context.txt"  
    context = load_context_from_file(context_file_path)
    
    conversation_history = """
    Patient: I’ve been feeling very tired lately and I’m not sure if it’s a side effect of the treatment or something else.
    Assistant: Have you noticed any other new symptoms?
    Patient: No, just the fatigue.
    """
    
    suggested_question = get_suggested_questions(conversation_history, context)
    print(f"Suggested question: {suggested_question}")

