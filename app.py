import os

import gradio as gr
from llm_chatbot import MistralChatbot
from utils import _set_env


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
    debug=True,
)


def respond(message):
    print(message)
    chat_response = chatbot.run_once(message)
    return chat_response


# Create buttons for quick responses
suggestion1 = "Are there any support services or resources you recommend for managing stress and anxiety related to my treatment?"
suggestion2 = "Who is Michael E. Jordan?"
suggestion3 = " How will my current medications, especially Lisinopril for hypertension and Ondansetron for nausea, be managed around the time of surgery?"
suggestion4 = "Quelle est la capitale de la France ?"
buttons = [suggestion1, suggestion2, suggestion3, suggestion4]

with gr.Blocks() as iface:
    textbox = gr.Textbox(label="Input", placeholder="Type your message here...")
    output_textbox = gr.Textbox(label="Response", interactive=False)

    # Create and add buttons
    for button_value in buttons:
        button = gr.Button(button_value)
        # Use the textbox input as the input for the respond function and set the button's value
        button.click(
            fn=respond, inputs=gr.Textbox(value=button_value), outputs=output_textbox
        )

    # Initially set up the function to respond to the input box
    textbox.submit(fn=respond, inputs=textbox, outputs=output_textbox)

iface.launch()
