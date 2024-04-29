import os
import shutil
from abc import ABC, abstractmethod

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OllamaEmbeddings, VoyageEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAIEmbeddings, OpenAI

from models.EmbeddingModelEnum import VoyageEmbedIdentifier
from models.LLModelEnum import OllamaModelIdentifier, AnthropicModelIdentifier

BASE_VECTOR_STORE = ".tmp/VECTOR-STORE"


class LargeLanguageModel(ABC):

    def __init__(self):
        self.create_temp_directory()

    @staticmethod
    def create_temp_directory():
        temp_dir = ".tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def get_llm(self):
        pass

    @abstractmethod
    def get_vector_store_name(self):
        pass


class OpenAIModel(LargeLanguageModel):
    def __init__(self):

        super().__init__()
        self.key = os.getenv('OPENAI_API_KEY')
        if self.key is None:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.embedding_model = "text-embedding-3-large"
        self.llm_model = "gpt-3.5-turbo-instruct"
        self.temp = 0.5

    def get_model_name(self):
        return "OpenAI"

    def get_embeddings(self):
        return OpenAIEmbeddings(openai_api_key=self.key, model=self.embedding_model)

    def get_llm(self):
        return OpenAI(openai_api_key=self.key, model_name=self.llm_model, temperature=self.temp)

    def get_vector_store_name(self):
        return f"{BASE_VECTOR_STORE}-OPENAI"


class OllamaModel(LargeLanguageModel):
    def __init__(self, model_identifier: OllamaModelIdentifier):
        super().__init__()
        self.model_identifier = model_identifier

    def get_model_name(self):
        return f"Ollama:{self.model_identifier.name}"

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.model_identifier.value)

    def get_llm(self):
        return Ollama(model=self.model_identifier.value)

    def get_vector_store_name(self):
        return f"{BASE_VECTOR_STORE}-{self.model_identifier.name}"


class AnthropicModel(LargeLanguageModel):
    def __init__(self, model_identifier: AnthropicModelIdentifier, embed_identifier: VoyageEmbedIdentifier):

        super().__init__()
        self.model_identifier = model_identifier
        self.voyage_api_key = os.getenv('VOYAGE_API_KEY')
        self.embedding_identifier = embed_identifier

    def get_model_name(self):
        return f"Anthropic:{self.model_identifier.name}"

    def get_embeddings(self):
        return VoyageEmbeddings(voyage_api_key=self.voyage_api_key, model=self.embedding_identifier.value)

    def get_llm(self):
        return ChatAnthropic(model=self.model_identifier.value)

    def get_vector_store_name(self):
        return f"{BASE_VECTOR_STORE}-{self.model_identifier.name}"


def drop_vector_store(vector_store_name: str):
    # Path to the directory under .tmp
    directory_path = os.path.join(".tmp", vector_store_name)

    # Check if the directory exists
    if os.path.exists(directory_path):
        # Delete the directory and its contents
        shutil.rmtree(directory_path)
        print(f"Vector Store '{vector_store_name}' deleted successfully.")
    else:
        print(f"Vector Store '{vector_store_name}' does not exist.")
