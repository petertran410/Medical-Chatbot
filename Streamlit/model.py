import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
import torch
from torch import cuda
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# DB_FAISS_PATH = 'vectorstores/db_faiss/NE-Syllabus'

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# # Context: {answer}
# # Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """


# def set_custom_prompt():
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt


# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                            chain_type='stuff',
#                                            retriever=db.as_retriever(
#                                                search_kwargs={'k': 2}),
#                                            return_source_documents=True,
#                                            chain_type_kwargs={'prompt': prompt}
#                                            )
#     return qa_chain


# def load_llm():
#     llm = CTransformers(
#         model="TheBloke/Llama-2-7B-Chat-GGML",
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.5
#     )
#     return llm


# def qa_bot(query):
#     # sentence-transformers/all-MiniLM-L6-v2
#     embeddings = HuggingFaceEmbeddings(model_name="imdeadinside410/TestTrainedModel",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     # Implement the question-answering logic here
#     response = qa({'query': query})
#     return response['result']

peft_model_id = "imdeadinside410/Llama2-Syllabus"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map=device)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


pipe = pipeline(task="text-generation",
                model=model,
                tokenizer=tokenizer, max_length=300)

prompt = "What is the mission of the School of Computer Science and Engineering?"

result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'].split("[/INST]")[1])

############################################################
############################################################

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)

# chain = prompt | hf

# question = "What is IT ?"

# print(chain.invoke({"question": question}))

############################################################
############################################################

# def add_vertical_space(spaces=1):
#     for _ in range(spaces):
#         st.markdown("---")


# def main():
#     st.set_page_config(page_title="AIoTLab NE Syllabus")

#     with st.sidebar:
#         st.title('AIoTLab NE Syllabus')
#         st.markdown('''
#         Hi
#         ''')
#         add_vertical_space(1)  # Adjust the number of spaces as needed
#         st.write(
#             'AIoT Lab')

#     st.title("AIoTLab NE Syllabus")
#     st.markdown(
#         """
#         <style>
#             .chat-container {
#                 display: flex;
#                 flex-direction: column;
#                 height: 400px;
#                 overflow-y: auto;
#                 padding: 10px;
#                 color: white; /* Font color */
#             }
#             .user-bubble {
#                 background-color: #007bff; /* Blue color for user */
#                 align-self: flex-end;
#                 border-radius: 10px;
#                 padding: 8px;
#                 margin: 5px;
#                 max-width: 70%;
#                 word-wrap: break-word;
#             }
#             .bot-bubble {
#                 background-color: #363636; /* Slightly lighter background color */
#                 align-self: flex-start;
#                 border-radius: 10px;
#                 padding: 8px;
#                 margin: 5px;
#                 max-width: 70%;
#                 word-wrap: break-word;
#             }
#         </style>
#         """, unsafe_allow_html=True)

#     conversation = st.session_state.get("conversation", [])
   
#     query = st.text_input("Please input your question here:", key="user_input")
#     result = pipe(f"<s>[INST] {query} [/INST]")
#     if st.button("Get Answer"):
#         if query:
#             # Display the processing message
#             with st.spinner("Processing your question..."):
#                 conversation.append({"role": "user", "message": query})
#                 # Call your QA function
#                 answer = result[0]['generated_text'].split("[/INST]")[1]
#                 conversation.append({"role": "bot", "message": answer})
#                 st.session_state.conversation = conversation
#         else:
#             st.warning("Please input a question.")

#     chat_container = st.empty()
#     chat_bubbles = ''.join(
#         [f'<div class="{c["role"]}-bubble">{c["message"]}</div>' for c in conversation])
#     chat_container.markdown(
#         f'<div class="chat-container">{chat_bubbles}</div>', unsafe_allow_html=True)


# if __name__ == "__main__":
#     main()
