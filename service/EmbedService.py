import os
import tempfile

import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from models.LargeLangModel import LargeLanguageModel


class EmbeddingCreatorService:
    def __init__(self, large_lang_model: LargeLanguageModel):
        self.large_lang_model = large_lang_model

    def excel_to_csv(self, input_file):
        xls = pd.ExcelFile(input_file)
        output_filenames = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)

            self.create_directory(f".datafiles/{input_file.split('.')[0]}")
            csv_file = f".datafiles/{input_file.split('.')[0]}/{sheet_name}.csv"

            df.to_csv(csv_file, index=False)
            output_filenames.append(csv_file)
        return output_filenames

    def save_to_vector_store(self, doc):
        content = self.load_document(doc)
        chunks = self.chunk_data(content)
        vector_store = FAISS.from_documents(chunks, embedding=self.large_lang_model.get_embeddings())
        vector_store.save_local(self.large_lang_model.get_vector_store_name())
        return vector_store

    def create_embeddings(self, data_files):
        for data_file in data_files:
            name, extension = os.path.splitext(data_file)
            if extension in ['.xls', '.xlsx']:
                output_csv_files = self.excel_to_csv(data_file)
                for file in output_csv_files[:1]:
                    self.save_to_vector_store(file)
            else:
                self.save_to_vector_store(data_file)

    @staticmethod
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def save_files_to_temp(data_files):
        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        file_paths = []  # Store paths of saved files

        if data_files is not None:
            for file in data_files:
                file_path = os.path.join(temp_dir, file.name)  # Create full file path in temporary directory
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())  # Write file contents to the temporary file
                file_paths.append(file_path)  # Store the file path

        return file_paths

    @staticmethod
    def load_document(file) -> List[Document]:
        loaders = {
            '.csv': CSVLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader
        }
        name, extension = os.path.splitext(file)
        if extension in loaders:
            loader = loaders[extension](file)
            return loader.load()
        else:
            print('Document format is not supported!')
            return []

    @staticmethod
    def chunk_data(data, chunk_size=1000, chunk_overlap=10):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks
