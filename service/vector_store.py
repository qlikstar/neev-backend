from abc import ABC, abstractmethod

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

from models.large_lang_model import LargeLanguageModel
from util.constants import VECTOR_STORE


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


class FaissVectorStore(VectorStore, ABC):

    def __init__(self, model: LargeLanguageModel):
        self.llm = model
        self.embeddings = model.get_embeddings()
        self.vector_store = None
        self.vector_store_name = f"FAISS-{self.llm.get_model_name()}"
        self.persist_directory = f'{VECTOR_STORE}/{self.vector_store_name}'

    def get_db(self):
        return FAISS.load_local(self.persist_directory,
                                self.embeddings,
                                allow_dangerous_deserialization=True)

    def create_vector_embeddings(self, chunks):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.persist_directory)

    def get_vector_store_name(self):
        return self.vector_store_name


class ChromaVectorStore(VectorStore, ABC):

    def __init__(self, model: LargeLanguageModel):
        self.llm = model
        self.embeddings = model.get_embeddings()
        self.vector_store = None
        self.vector_store_name = f"CHROMA-{self.llm.get_model_name()}"
        self.persist_directory = f'{VECTOR_STORE}/{self.vector_store_name}'

    def get_db(self):
        return Chroma(persist_directory=self.persist_directory,
                      embedding_function=self.embeddings)

    def create_vector_embeddings(self, chunks):
        self.vector_store = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_directory)

    def get_vector_store_name(self):
        return self.vector_store.save_local
