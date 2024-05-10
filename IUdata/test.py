import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfFileReader

# Step 1: Extract text from PDF
pdf_path = "71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf"
# pdf_path = "NE-Syllabus.pdf"


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfFileReader(f)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

extracted_text = extract_text_from_pdf(pdf_path)


# def save_text_to_file(text, file_path):
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(text)

# extracted_text = extract_text_from_pdf(pdf_path)
# output_file_path = "extracted_text_2.txt"
# save_text_to_file(extracted_text, output_file_path)

# print("Text has been successfully saved to:", output_file_path)

# Step 2: Vectorize text


def vectorize_text(text):
    # Using TF-IDF for simplicity
    tfidf_vectorizer = TfidfVectorizer()
    vectorized_text = tfidf_vectorizer.fit_transform([text])
    return vectorized_text
  
print("\nStep 2: Text vectorized:")
vectorized_text = vectorize_text(extracted_text)
print(vectorized_text)

# Step 3: Build FAISS index


def build_faiss_index(vectorized_text):
    # Convert scipy sparse matrix to numpy array
    vectorized_text = vectorized_text.toarray()
    # Initialize FAISS index
    index = faiss.IndexFlatL2(vectorized_text.shape[1])  # L2 distance metric
    # Add vectors to the index
    index.add(vectorized_text)
    return index
  
print("\nStep 3: FAISS index built:")
index = build_faiss_index(vectorized_text)
print(index)

# Step 4: Save index and text representation in pickle file

def save_index_and_text_representation(index, vectorized_text, index_file, text_file):
    with open(index_file, "wb") as f:
        pickle.dump(index, f)
    with open(text_file, "wb") as f:
        pickle.dump(vectorized_text, f)

def convert_pdf_to_faiss_and_pkl(pdf_path, index_file, text_file):
    text = extract_text_from_pdf(pdf_path)
    vectorized_text = vectorize_text(text)
    index = build_faiss_index(vectorized_text)
    save_index_and_text_representation(index, vectorized_text, index_file, text_file)

# Usage

index_file = "index.faiss"
text_file = "index.pkl"
convert_pdf_to_faiss_and_pkl(pdf_path, index_file, text_file)
