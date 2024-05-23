from abc import ABC, abstractmethod

from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore


class RetrieverTool(ABC):

    @abstractmethod
    def get_tool_name(self):
        pass

    @abstractmethod
    def get_tool(self):
        pass


class VectorStoreRetrieverTool(RetrieverTool):

    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore
        print(f"Initializing ... {self.get_tool_name()}")

    def get_tool_name(self):
        return "Vector Store Retriever Tool"

    def get_tool(self):
        return create_retriever_tool(
            self.vectorstore.as_retriever(), "data_retriever",
            "Fetch all the data related to the question asked from vector store"
        )


class DatabaseRetrieverTool(RetrieverTool):
    def get_tool_name(self):
        pass

    def get_tool(self):
        pass
