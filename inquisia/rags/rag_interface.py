from abc import ABC, abstractmethod

from langchain.chains.llm import LLMChain
from langchain.schema import BaseRetriever


class RAGInterface(ABC):
    @abstractmethod
    def get_question_generation_llm(self) -> LLMChain:
        raise NotImplementedError(
            "get_question_generation_llm is an abstract method and must be implemented"
        )

    @abstractmethod
    def get_retriever(self) -> BaseRetriever:
        raise NotImplementedError(
            "get_retriever is an abstract method and must be implemented"
        )
