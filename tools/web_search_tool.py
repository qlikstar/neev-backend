import os
from abc import ABC, abstractmethod

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool, BaseTool


class WebSearchTool(ABC):

    @abstractmethod
    def get_tool_name(self):
        pass

    @abstractmethod
    def get_tool(self):
        pass


class TavilySearchTool(WebSearchTool):

    def __init__(self):
        self.key = os.getenv('TAVILY_API_KEY')
        if self.key is None:
            raise EnvironmentError("TAVILY_API_KEY environment variable is not set.")
        self.results_count = 3
        print(f"Initializing ... {self.get_tool_name()}")

    def get_tool_name(self):
        return "Tavily-Search-Tool"

    def get_tool(self) -> BaseTool:
        return TavilySearchResults(max_results=self.results_count)


class DuckDuckGoSearchTool(WebSearchTool):

    def __init__(self):
        self.results_count: int = 10
        self.desc: str = (
            "A search engine optimized for comprehensive, accurate, and trusted results. "
            "Useful for when you need to answer questions about current events. "
            "Input should be a search query."
        )
        self.duckduckgo = DuckDuckGoSearchRun(max_results=self.results_count)
        print(f"Initializing ... {self.get_tool_name()}")

    def get_tool_name(self):
        return "DuckDuckGo-Search-Tool"

    def get_tool(self) -> BaseTool:
        return Tool(name=self.get_tool_name(), func=self.duckduckgo.run, description="search current events")
