import json
from typing import Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import BaseModel, Field


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    explanation: str = Field(description="Explain the reasoning for the score")


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["simple", "complex"] = Field(
        ...,
        description="""Given a user question choose to route it to LLM to one branch where 
        we concatenate EHR data with the prompt if the question is deemed simple and if 
        its complication a RAG is performed and the information retrieved concatenated to the prompt.""",
    )


class MistralChatbot:

    """ """

    rag_prompt = """
    You are an assistant helping a cancer patient. Based on the conversation, suggest 4 relevant and helpful questions that the patient can ask their doctor or that can help clarify their situation.
    Please answer the following questions:
    {questions}

    Context (if available):
    {context}

    After answering the patient's question please provide the 4 questions in a numbered list, separated by new lines.
    """

    router_prompt = """You are an expert at routing a user question to answer either very precise and complex questions or everything else.
    The vectorstore contains documents related to related oncology data, medical guidelines, cancer symptoms...                                
    Use the vectorstore for questions on these topics. For all else, concatenate the EHR text to the prompt."""

    simple_prompt = """ You are an assistant helping a cancer patient. 

    Based on the conversation, suggest 4 relevant and helpful questions that the patient can ask their doctor or that can help clarify their situation.
    
    The patient's Electronic Health Record : {ehr} 
    
    His question: {question}

    Please answer his question given his personal health information.
    Answer:
    """

    hallucination_grader_instructions = """You are a doctor double checking the consistency and truth of your students answer. 

        You will be given GUIDELINE and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) Ensure the STUDENT ANSWER is grounded in the GUIDELINE. 

        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the GUIDELINE.

        Score:

        A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 

        A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    hallucination_grader_prompt = (
        "FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}"
    )

    simple_question_prompt = """ You are an assistant helping a cancer patient. 

    Based on the conversation, suggest 4 relevant and helpful questions

    that the patient can ask their doctor or that can help clarify their situation.

    The patient's Electronic Health Record : {ehr} 

    His question: {question}

    Please answer his question given his personal health information.

    Answer:"""

    def __init__(
        self,
        db_patient_path,
        db_doctor_path,
        ehr_path,
        mistral_model="mistral-large-latest",
        temperature=0.0,
    ) -> None:
        # QUESTION CAN YOU CHANGE THE TEMPERATURE DEPENDING ON THE FLOW ?
        self.llm = ChatMistralAI(model=mistral_model, temperature=temperature)
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.structured_llm_hallucination_grader = self.llm.with_structured_output(
            GradeHallucinations
        )

        self.db_patient_path = db_patient_path
        self.db_doctor_path = db_doctor_path
        self.ehr_path = ehr_path

        # setting up database
        self.setup_rag_db()

    def setup_rag_db(self):
        """
        This initializes the RAG database with the doctor and patient dataset.
        """

        # Load data from JSON file
        with open(self.db_patient_path, "r") as file:
            data = json.load(file)

        # Create documents from JSON data
        docs_list = [
            Document(page_content=item["text"], metadata={"source": item["url"]})
            for item in data.values()
        ]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=MistralAIEmbeddings(),
        )

        # Create retriever
        self.retriever_patient = vectorstore.as_retriever()

    def route_query(self, query: str) -> str:
        """
        This is the first step in our solution. We route the patient's question depending on the complexity of the question (query).
        """
        return self.structured_llm_router.invoke(
            [SystemMessage(content=self.router_prompt)] + [HumanMessage(content=query)]
        )

    def answer_complex_question(self, question):
        """
        If the question is deemed hard the RAG will help to provide the answer.
        """
        docs = self.retriever_patient.invoke(question)
        docs_txt = self.format_docs(docs)
        rag_prompt_formatted = self.rag_prompt.format(
            context=docs_txt, questions=question
        )
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return generation

    def answer_simple_question(self, question):
        """
        If the question is deemed easy just the patient's EHR will help provide context.
        """
        ehr = open(self.ehr_path, "r").read()
        question = "How do I come down im stressed about my surgery ?"
        simple_prompt_formatted = self.simple_question_prompt.format(
            ehr=ehr, question=question
        )
        generation = self.llm.invoke([HumanMessage(content=simple_prompt_formatted)])
        return generation

    def grade_hallucinations(self, question, generation):
        """
        This function grades the hallucinations in the generation answer.
        """
        docs = self.retriever_patient.invoke(question)
        hallucination_grader_prompt_formatted = self.hallucination_grader_prompt.format(
            documents=docs, generation=generation
        )
        return self.structured_llm_hallucination_grader.invoke(
            [SystemMessage(content=self.hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_once(self, question):
        """
        This function runs the chatbot once.
        """
        route = self.route_query(question)
        if route.datasource == "simple":
            generation = self.answer_simple_question(question)
        else:
            generation = self.answer_complex_question(question)
        if self.grade_hallucinations(question, generation):
            return generation.content
        else:
            return "The answer is hallucinated"


import os


# _set_env("MISTRAL_API_KEY")
# _set_env("TAVILY_API_KEY")
# _set_env("HF_TOKEN")
# _set_env("LANGCHAIN_API_KEY")

os.environ["HF_TOKEN"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["MISTRAL_API_KEY"] = "oh8TV1P4NXoafrmMt89YTO1j6rThgrY4"
os.environ["LANGCHAIN_API_KEY"] = "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"

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
