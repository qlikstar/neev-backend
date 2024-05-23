from abc import ABC, abstractmethod

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from models.large_lang_model import LargeLanguageModel
from util.constants import VECTOR_STORE


class BaseVectorStore(ABC):

    @abstractmethod
    def get_db(self):
        pass

    @abstractmethod
    def create_vector_embeddings(self, doc: Document):
        pass

    @abstractmethod
    def get_vector_store(self) -> VectorStore:
        pass


class FaissVectorStore(BaseVectorStore, ABC):

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

    def get_vector_store(self) -> VectorStore:
        if self.vector_store is None:
            raise ValueError("Vector store is None")
        return self.vector_store


class ChromaVectorStore(BaseVectorStore, ABC):
    _instance = None  # Static variable to hold the single instance
    store: Chroma = None  # Static variable for the vector store

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChromaVectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: LargeLanguageModel):
        if not hasattr(self, 'initialized'):  # Ensure the instance is only initialized once
            self.llm = model
            self.embeddings = model.get_embeddings()
            self.vector_store_name = f"CHROMA-{self.llm.get_model_name()}"
            self.persist_directory = f'{VECTOR_STORE}/{self.vector_store_name}'
            print(f"Initializing Vector Store ... {self.vector_store_name}")
            self.initialized = True

    def get_db(self):
        return Chroma(persist_directory=self.persist_directory,
                      embedding_function=self.embeddings)

    def create_vector_embeddings(self, chunks):
        ChromaVectorStore.store = Chroma.from_documents(chunks, self.embeddings,
                                                        persist_directory=self.persist_directory)
        print(f"Populated vector store .. {ChromaVectorStore.store}")

    def get_vector_store(self) -> Chroma:
        if ChromaVectorStore.store is None:
            raise ValueError("Vector store is None")
        return ChromaVectorStore.store
