import os
import shutil
from abc import ABC, abstractmethod

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

from models.large_lang_model import LargeLanguageModel

BASE_VECTOR_STORE = ".tmp/VSTORE"


class VectorStore(ABC):

    @abstractmethod
    def get_db(self):
        pass

    @abstractmethod
    def create_vector_embeddings(self, doc: Document):
        pass

    @abstractmethod
    def get_vector_store_name(self):
        pass

    def drop_vector_store(self):
        # Path to the directory under .tmp
        directory_path = os.path.join(self.get_vector_store_name())

        # Check if the directory exists
        if os.path.exists(directory_path):
            # Delete the directory and its contents
            shutil.rmtree(directory_path)
            print(f"Vector Store '{self.get_vector_store_name()}' deleted successfully.")
        else:
            print(f"Vector Store '{self.get_vector_store_name()}' does not exist.")


class FaissVectorStore(VectorStore, ABC):

    def __init__(self, model: LargeLanguageModel):
        self.llm = model
        self.embeddings = model.get_embeddings()
        self.vector_store = None
        self.vector_store_name = f"{BASE_VECTOR_STORE}-FAISS-{self.llm.get_model_name()}"

    def get_db(self):
        return FAISS.load_local(self.vector_store_name,
                                self.embeddings,
                                allow_dangerous_deserialization=True)

    def create_vector_embeddings(self, chunks):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_store_name)

    def get_vector_store_name(self):
        return self.vector_store_name


class ChromaVectorStore(VectorStore, ABC):

    def __init__(self, model: LargeLanguageModel):
        self.llm = model
        self.embeddings = model.get_embeddings()
        self.vector_store = None
        self.vector_store_name = f"{BASE_VECTOR_STORE}-CHROMA-{self.llm.get_model_name()}"

    def get_db(self):
        return Chroma(persist_directory=self.vector_store_name,
                      embedding_function=self.embeddings)

    def create_vector_embeddings(self, chunks):
        self.vector_store = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.vector_store_name)

    def get_vector_store_name(self):
        return self.vector_store_name