# import pickle
# import faiss
# from sklearn.feature_extraction.text import TfidfVectorizer
# from PyPDF2 import PdfReader
# import numpy as np
# import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

# pdf_path = "data/NE-Syllabus.pdf"
# pdf_path = "MusicLyrics.pdf"
pdf_path = "llama2-dataset.json"

file = open(pdf_path, encoding="utf8")
extracted_text = file.read()

# print(extracted_text)


# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, "rb") as f:
#         pdf_reader = PdfReader(f)
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text()
#     return text


# extracted_text = extract_text_from_pdf(pdf_path)
# print(extracted_text)

embeddings = HuggingFaceEmbeddings(model_name="imdeadinside410/Llama2-Syllabus",
                                   model_kwargs={'device': 'cpu'})

db = FAISS.from_texts(extracted_text, embeddings)

db.save_local("NE-Syllabus")
