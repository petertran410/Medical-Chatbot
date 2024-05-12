import pickle
import faiss
from openai import embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import numpy as np
import os

# pdf_path = "71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf"
pdf_path = "MusicLyrics.pdf"


# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, "rb") as f:
#         pdf_reader = PdfReader(f)
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             text += page.extractText()
#     return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


extracted_text = extract_text_from_pdf(pdf_path)


def vectorize_text(text):
    tfidf_vectorizer = TfidfVectorizer()
    vectorized_text = tfidf_vectorizer.fit_transform([text])
    return vectorized_text


vectorized_text = vectorize_text(extracted_text)


def build_and_save_faiss_index(vectorized_text):
    text = vectorized_text.toarray().astype("float32", casting="same_kind")
    index = faiss.IndexFlatL2(text.shape[1])
    index.add(text)
    faiss.write_index(index, "index.faiss")
    return index


index = build_and_save_faiss_index(vectorized_text)

index_2 = faiss.read_index(f'index.faiss')
print(index_2)

with open("index.pkl", "wb") as f:
    pickle.dump(vectorized_text, f)
    
with open("index.pkl", "rb") as f:
    loaded_vectorized_text = pickle.load(f)

# with open('index.pkl', 'rb') as file:
#     docstore, index_to_docstore_id = pickle.load(file)
# vectors = faiss(embeddings.embed_query, index, docstore,
#                 index_to_docstore_id, normalize_L2=True)
