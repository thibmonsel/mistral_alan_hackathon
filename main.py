import os

from llm_chatbot import MistralChatbot
from utils import _set_env


if __name__ == "__main__":
    _set_env("MISTRAL_API_KEY")
    _set_env("TAVILY_API_KEY")
    _set_env("HF_TOKEN")
    _set_env("LANGCHAIN_API_KEY")

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "pr-standard-godfather-11"

    os.environ["USER_AGENT"] = "myagent"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    chatbot = MistralChatbot(
        db_patient_path="rag_dataset_patient.json",
        db_doctor_path="rag_dataset_doctor.json",
        ehr_path="ehr_context.txt",
    )
    out = chatbot.run_once("How do I treat a patient with stage 3 breast cancer?")
    print(out)
