import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import numpy as np
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

pdf_path = "NE-Syllabus.pdf"


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


extracted_text = extract_text_from_pdf(pdf_path)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})

db = FAISS.from_texts(extracted_text, embeddings)

db.save_local("NE-Syllabus")
