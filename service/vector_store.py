from abc import ABC, abstractmethod

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.vectorstores import VectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from models.large_lang_model import LargeLanguageModel
from service.doc_processor import DocProcessorService
from util.constants import VECTOR_STORE


class BaseVectorStore(ABC):

    @abstractmethod
    def get_db(self):
        pass

    @abstractmethod
    def create_vector_embeddings(self, data_files: list[UploadedFile]):
        pass

    @abstractmethod
    def get_vector_store(self) -> VectorStore:
        pass


class FaissVectorStore(BaseVectorStore, ABC):

    def __init__(self, model: LargeLanguageModel, doc_processor_service: DocProcessorService):
        self.llm = model
        self.doc_processor_service = doc_processor_service
        self.embeddings = model.get_embeddings()
        self.vector_store = None
        self.vector_store_name = f"FAISS-{self.llm.get_model_name()}"
        self.persist_directory = f'{VECTOR_STORE}/{self.vector_store_name}'

    def get_db(self):
        return FAISS.load_local(self.persist_directory,
                                self.embeddings,
                                allow_dangerous_deserialization=True)

    def create_vector_embeddings(self, data_files: list[UploadedFile]):
        chunks = self.doc_processor_service.get_chunks_from_upload_documents(data_files)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.persist_directory)

    def get_vector_store(self) -> VectorStore:
        if self.vector_store is None:
            self.vector_store = FAISS.load_local(self.persist_directory, embeddings=self.embeddings,
                                                 allow_dangerous_deserialization=True)
        return self.vector_store


class ChromaVectorStore(BaseVectorStore, ABC):
    _instance = None  # Static variable to hold the single instance
    store: Chroma = None  # Static variable for the vector store

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChromaVectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: LargeLanguageModel, doc_processor_service: DocProcessorService):
        if not hasattr(self, 'initialized'):  # Ensure the instance is only initialized once
            self.llm = model
            self.doc_processor_service = doc_processor_service
            self.embeddings = model.get_embeddings()
            self.vector_store_name = f"CHROMA-{self.llm.get_model_name()}"
            self.persist_directory = f'{VECTOR_STORE}/{self.vector_store_name}'
            print(f"Initializing Vector Store ... {self.vector_store_name}")
            self.initialized = True

    def get_db(self):
        return Chroma(persist_directory=self.persist_directory,
                      embedding_function=self.embeddings)

    def create_vector_embeddings(self, data_files: list[UploadedFile]):
        chunks = self.doc_processor_service.get_chunks_from_upload_documents(data_files)
        ChromaVectorStore.store = Chroma.from_documents(chunks, self.embeddings,
                                                        persist_directory=self.persist_directory)
        print(f"Populated vector store from upload docs.. {ChromaVectorStore.store}")

    def get_vector_store(self) -> Chroma:
        if ChromaVectorStore.store is None:
            chunks = self.doc_processor_service.get_chunks_from_directory()
            ChromaVectorStore.store = Chroma.from_documents(chunks, self.embeddings,
                                                            persist_directory=self.persist_directory)
            print(f"Populated vector store from directory.. {ChromaVectorStore.store}")
        return ChromaVectorStore.store
