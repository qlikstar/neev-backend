import os
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv()
FAISS_VECTOR_STORE = "faiss_vector_store_ollama"
EMBEDDINGS = OllamaEmbeddings(model="gemma:2b")


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")


def extract_file_paths(data_files):
    file_paths = []
    if data_files is not None:
        for file in data_files:
            file_paths.append(file.name)
    return file_paths


def excel_to_csv(input_file):
    # Read the Excel file
    xls = pd.ExcelFile(input_file)
    output_filenames = []

    # Loop through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name)

        create_directory(f".datafiles/{input_file.split('.')[0]}")
        # Define the CSV file name (you can customize the file name as needed)
        csv_file = f".datafiles/{input_file.split('.')[0]}/{sheet_name}.csv"

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file, index=False)
        print(f"Converted sheet '{sheet_name}' to CSV: {csv_file}")
        output_filenames.append(csv_file)
    return output_filenames


def load_document(file) -> List[Document]:
    name, extension = os.path.splitext(file)
    if extension == '.csv':
        print(f'Loading {file}')
        loader = CSVLoader(file)
    elif extension == '.pdf':
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return []

    data = loader.load()
    return data


def chunk_data(data, chunk_size=1000, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(data_files):
    for data_file in data_files:
        name, extension = os.path.splitext(data_file)
        if extension == '.xls' or extension == '.xlsx':
            print(f'Loading {data_file}')
            output_csv_files = excel_to_csv(data_file)
            for file in output_csv_files[:1]:
                save_to_vector_store(file)
        else:
            save_to_vector_store(data_file)


def save_to_vector_store(doc):
    content = load_document(doc)
    chunks = chunk_data(content)
    vector_store = FAISS.from_documents(chunks, embedding=EMBEDDINGS)
    vector_store.save_local(FAISS_VECTOR_STORE)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    You are an expert AI assistant on finance domain. 
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in provided context 
    just say, "Answer is not available in the context" \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = Ollama(model="gemma:2b")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, prompt=prompt)
    # chain = prompt | model | StrOutputParser()

    return chain


def user_input(user_question):
    vector_db = FAISS.load_local(FAISS_VECTOR_STORE, EMBEDDINGS, allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Explore Neev's Fintelligence")
    st.header("OLLAMA:gemma:2b -> Explore Neev's Fintelligence💁")

    user_question = st.text_input("Ask a Question from the uploaded Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button",
                                      accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                data_file_paths = extract_file_paths(data_files)
                create_embeddings(data_file_paths)
                st.success("Done")


if __name__ == "__main__":
    main()
