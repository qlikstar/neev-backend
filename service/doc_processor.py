import os
import tempfile

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_core.documents import Document


class DocProcessorService:

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

    def chunk_doc(self, doc) -> list[Document]:
        content = self.load_document(doc)
        return self.chunk_data(content)

    def get_chunked_documents(self, data_files) -> list[Document]:
        file_paths = self.save_files_to_temp(data_files)
        documents = []
        for file_path in file_paths:
            name, extension = os.path.splitext(file_path)
            if extension in ['.xls', '.xlsx']:
                output_csv_files = self.excel_to_csv(file_path)
                for file in output_csv_files[:1]:
                    documents.extend(self.chunk_doc(file))
            else:
                documents.extend(self.chunk_doc(file_path))
        return documents

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
    def load_document(file) -> list[Document]:
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
    def chunk_data(data, chunk_size=1000, chunk_overlap=10) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks
