"""
Code adapted from HF https://huggingface.co/spaces/pdf2dataset/pdf2dataset/blob/main/app.py

"""

import json
import re
import urllib.request

import gradio as gr
from cleantext import clean as cl
from datasets import Dataset
from pypdf import PdfReader


# Function to download a PDF
def download_pdf(url, save_path):
    # Download the PDF from the URL
    urllib.request.urlretrieve(url, save_path + "/" + url.split("/")[-1])
    print(f"PDF downloaded successfully and saved as: {save_path}")


to_be_removed = ["ͳ", "•", "→", "□", "▪", "►", "�", "", "", "", ""]
to_be_replaced = {
    "½": "1/2",
    "–": "-",
    "‘": "'",
    "’": "'",
    "…": "...",
    "₋": "-",
    "−": "-",
    "⓫": "11.",
    "⓬": "12.",
    "⓭": "13.",
    "⓮": "14.",
    "◦": "°",
    "❶": "1.",
    "❷": "2.",
    "❸": "3.",
    "❹": "4.",
    "❺": "5.",
    "❻": "6.",
    "❼": "7.",
    "❽": "8.",
    "❾": "9.",
    "❿": "10.",
    "\n": " ",
}


def clean_text(text):
    # Remove all the unwanted characters
    for char in to_be_removed:
        text = text.replace(char, "")

    # Replace all the characters that need to be replaced
    for char, replacement in to_be_replaced.items():
        text = text.replace(char, replacement)

    # For all \n, if the next line doesn't start
    #  with a capital letter, remove the \n
    # text = re.sub(r"\n([^A-ZÀ-ÖØ-Þ])", r" \1", text)

    # Make sure that every "." is followed by a space
    text = re.sub(r"\.([^ ])", r". \1", text)

    # Add a space between a lowercase followed by
    # an uppercase "aA" -> "a A" (include accents)
    text = re.sub(r"([a-zà-öø-ÿ])([A-ZÀ-ÖØ-Þ])", r"\1 \2", text)

    # Make sure that there is no space before
    # a comma, a period, or a hyphen
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" -", "-")
    text = text.replace("- ", "-")

    while "  " in text:
        text = text.replace("  ", " ")

    return text


def pdf2dataset(path: str, progress=gr.Progress()):
    print("path", path)
    reader = PdfReader(path)
    # Convert the PDFs to text
    page_texts = []
    page_filenames = []
    progress(0, desc="Converting pages...")
    for page in reader.pages:
        page_text = page.extract_text()
        page_text = clean_text(page_text)
        page_texts.append(page_text)
        page_filenames.append(path)

    dataset = Dataset.from_dict({"text": page_texts, "source": page_filenames})
    return dataset


def post_process_scraped_pdf(data: Dataset, limit=20, urlprefix="") -> Dataset:
    """
    This function return a dictionary with the key
    'url_pdf' and the value being the pdf's content.
    """
    filtered_test = [
        cl(
            str(text_sample),
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_line_breaks=False,
            no_urls=False,
            no_emails=False,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
            replace_with_punct="",
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en",  # set to 'de' for German special handling
        )
        for text_sample in data["text"]
        if len(text_sample) >= limit
    ]
    return urlprefix + data["source"][0], " ".join(filtered_test)


def create_json_file_dataset(list_of_pdf, json_filename):
    list_of_texts, list_of_urls = [], []
    list_of_pdf = list_of_pdf.split(",")
    for pdf in list_of_pdf:
        print("pdf", pdf)
        dataset = pdf2dataset(pdf)
        url, text = post_process_scraped_pdf(dataset)
        list_of_texts.append(text)
        list_of_urls.append(url)

    # Create the data dictionary
    data = {}
    for url, text in zip(list_of_urls, list_of_texts):
        data[url.split("/")[-1]] = {"url": url, "text": list_of_texts}

    # Convert to JSON
    json_data = json.dumps(data, indent=2)

    # Optionally, save to a file
    with open(json_filename, "w") as f:
        f.write(json_data)
