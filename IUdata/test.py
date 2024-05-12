import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfFileReader
import numpy as np
import os

pdf_path = "MusicLyrics.pdf"


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfFileReader(f)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text


extracted_text = extract_text_from_pdf(pdf_path)


def vectorize_text(text):
    tfidf_vectorizer = TfidfVectorizer()
    vectorized_text = tfidf_vectorizer.fit_transform([text])
    return vectorized_text


print("\nText vectorized:")
vectorized_text = vectorize_text(extracted_text)
print(vectorize_text(extracted_text))


def build_faiss_index(text):
    text = text.toarray()
    text = text.astype("float32", casting="same_kind")
    index = faiss.IndexFlatL2(text.shape[1])
    index.add(text)
    return index

index = build_faiss_index(vectorized_text)

############################################
# writeIndex = faiss.write_index(index, "index.faiss")

# chunk = faiss.serialize_index(index)
faiss.write_index(index, "index.faiss")
# faiss.deserialize_index(np.load("index.faiss"))

with open("index.pkl", "wb") as f:
    pickle.dump(vectorized_text, f)

# with open("index.pkl", "wb") as f:
#     pickle.dump(writeIndex, f)
# with open("index.pkl", "rb") as f:
#     faiss.read_index(pickle.load(f))

############################################


# def save_index_and_text_representation(index, vectorized_text, index_file, text_file):
#     with open(index_file, "wb") as f:
#         pickle.dump(index, f)
#     with open(text_file, "wb") as f:
#         pickle.dump(vectorized_text, f)

# def convert_pdf_to_faiss_and_pkl(pdf_path, index_file, text_file):
#     text = extract_text_from_pdf(pdf_path)
#     vectorized_text = vectorize_text(text)
#     index = build_faiss_index(vectorized_text)
#     save_index_and_text_representation(index, vectorized_text, index_file, text_file)

# Usage

# index_file = "index.faiss"
# text_file = "index.pkl"
# convert_pdf_to_faiss_and_pkl(pdf_path, index_file, text_file)

# print(convert_pdf_to_faiss_and_pkl(pdf_path, index_file, text_file))
