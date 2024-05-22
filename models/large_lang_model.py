import os
from abc import ABC, abstractmethod

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FastEmbedEmbeddings, HuggingFaceHubEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_together import Together

from models.LLM_enum import OllamaModelIdentifier, AnthropicModelIdentifier, HuggingFaceModelIdentifier, \
    TogetherAiIdentifier
from models.embedding_enum import VoyageEmbedIdentifier


class LargeLanguageModel(ABC):
    """
    Abstract base class for large language models.
    """

    def __init__(self):
        """
        Initializes the language model and prints the model name and embeddings model.
        """
        print(f"Now using .. {self.get_model_name()} with {self.get_embeddings().model}")

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def get_llm(self):
        pass


class OpenAIModel(LargeLanguageModel):
    def __init__(self):
        self.key = os.getenv('OPENAI_API_KEY')
        if self.key is None:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.embedding_model = "text-embedding-ada-002"
        self.llm_model = "gpt-3.5-turbo-instruct"
        self.temp = 0.5
        super().__init__()

    def get_model_name(self):
        return "OpenAI"

    def get_embeddings(self):
        return OpenAIEmbeddings(openai_api_key=self.key, model=self.embedding_model)

    def get_llm(self):
        return OpenAI(openai_api_key=self.key, model_name=self.llm_model, temperature=self.temp)


class OllamaModel(LargeLanguageModel):
    def __init__(self, model_identifier: OllamaModelIdentifier):
        self.model_identifier = model_identifier
        self.key = os.getenv('OPENAI_API_KEY')
        self.embedding_model = "text-embedding-ada-002"
        self.llm_model = "gpt-3.5-turbo-instruct"
        super().__init__()

    def get_model_name(self):
        return f"Ollama:{self.model_identifier.name}"

    def get_embeddings(self):
        return OpenAIEmbeddings(openai_api_key=self.key, model=self.embedding_model)

    def get_llm(self):
        return Ollama(model=self.model_identifier.value)


class AnthropicModel(LargeLanguageModel):
    def __init__(self, model_identifier: AnthropicModelIdentifier, embed_identifier: VoyageEmbedIdentifier):
        self.model_identifier = model_identifier
        self.voyage_api_key = os.getenv('VOYAGE_API_KEY')
        self.embedding_identifier = embed_identifier
        super().__init__()

    def get_model_name(self):
        return f"Anthropic:{self.model_identifier.name}"

    def get_embeddings(self):
        return VoyageAIEmbeddings(voyage_api_key=self.voyage_api_key, model=self.embedding_identifier.value)

    def get_llm(self):
        return ChatAnthropic(model=self.model_identifier.value)


class HuggingFaceModel(LargeLanguageModel):

    def __init__(self, model_identifier: HuggingFaceModelIdentifier):
        self.key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if self.key is None:
            raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")

        self.llm_model = model_identifier
        self.temp = 0.5
        self.max_len = 64
        super().__init__()

    def get_model_name(self):
        return self.llm_model.name

    def get_embeddings(self):
        return FastEmbedEmbeddings()

    def get_llm(self):
        return HuggingFaceHub(repo_id=self.llm_model.value,
                              model_kwargs={"temperature": self.temp, "max_length": self.max_len})


class TogetherAiModel(LargeLanguageModel):

    def __init__(self, model_identifier: TogetherAiIdentifier):
        self.key = os.getenv('TOGETHER_API_KEY')
        if self.key is None:
            raise EnvironmentError("TOGETHER_API_KEY environment variable is not set.")

        self.llm_model = model_identifier
        self.temp = 0.5
        self.max_tokens = 128
        super().__init__()

    def get_model_name(self):
        return self.llm_model.name

    def get_embeddings(self):
        return HuggingFaceHubEmbeddings(model="BAAI/bge-small-en-v1.5")

    def get_llm(self):
        return Together(model=self.llm_model.value, temperature=self.temp, max_tokens=self.max_tokens, top_k=1)
