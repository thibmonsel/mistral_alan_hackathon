# mistral_alan_hackathon


# To contribute code

Please checkout `CONTRIBUTING.md`.

# Installation

Please run 
```bash 
pip install .
```

# To download/generate RAG dataset

The RAG dataset is based on guidelines from this website [website](https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1418) and you must create an account to get the pdf data. For this project several cancer guidelines were manually selected but this could have been scraped with more sophisticated scrapped. 

In order to generate the data please run :

```bash 
bash bash_scripts/generate_rag_dataset.sh
```

A patient and doctor dataset will be provided `rag_dataset_patient.json` and `rag_dataset_doctor.json`.


# To Launch the chatbot 

Before doing anything please make sure to specify the following API keys necessary for the chatbot : 

```bash 
"MISTRAL_API_KEY"
"TAVILY_API_KEY"
"HF_TOKEN"
"LANGCHAIN_API_KEY"
```

Launch `main.py` to get your personal oncology specialized assistant that will help you through your diagnosis ! 

```python 
python main.py
```