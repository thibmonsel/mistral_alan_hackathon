import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

# Retrieve the API key from environment variables
API_KEY = "6zF6BXi6P9k4Cz6rQQAJynGZCVrM9shk"
API_URL = "https://api.mistral.ai/v1/chat/completions"

if not API_KEY:
    raise ValueError("API key not found. Please set it in the .env file.")

# Function to build the prompt for classification based on question and EHR
def generate_classification_prompt(question, ehr_context):
    """
    Creates the prompt to classify the patient's question as simple (0) or complex (1).
    The classification is based on whether the question can be answered using the EHR context or general medical knowledge.
    """
    prompt = f"""
    You are an assistant helping a doctor. Classify the following patient question based on whether it is simple or complex:
    
    - Simple (0): Questions that can be answered using general medical knowledge or from the patient's EHR.
    - Complex (1): Questions that require detailed medical explanations, such as specific treatments, side effects, or guidelines beyond the EHR.

    Patient's question: {question}

    EHR context: {ehr_context}

    Your answer should be 0 if it's simple and 1 if it's complex. Only respond with 0 or 1, without any other text.
    """
    return prompt

# Function to classify the patient's question using the EHR context
def classify_patient_question_with_ehr(question, ehr_context):
    """
    Calls the Mistral Small API to classify a patient's question as simple (0) or complex (1)
    based on both the question and the EHR context.
    """
    prompt = generate_classification_prompt(question, ehr_context)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.2  # Lower temperature for deterministic output
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content'].strip()
        if result in ['0', '1']:
            return int(result)
        else:
            return f"Unexpected response: {result}"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to load EHR context from a text file
def load_ehr_context_from_file(file_path):
    """
    Loads the EHR context from a .txt file.
    """
    try:
        with open(file_path, 'r') as file:
            ehr_context = file.read()
        return ehr_context
    except FileNotFoundError:
        return "EHR context file not found."

# Example usage
if __name__ == "__main__":
    # Load the EHR context from a .txt file
    ehr_file_path = "ehr_context.txt"  # Update with your actual file path
    ehr_context = load_ehr_context_from_file(ehr_file_path)
    
    # Example patient question
    patient_question = "why do i have diarrhea, is it related to my treatment?"

    # Classify the question
    classification = classify_patient_question_with_ehr(patient_question, ehr_context)
    
    if classification == 0:
        print("The question is classified as Simple (0).")
    elif classification == 1:
        print("The question is classified as Complex (1).")
    else:
        print(f"Classification result: {classification}")