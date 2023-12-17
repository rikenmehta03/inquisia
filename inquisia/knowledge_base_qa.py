from typing import Optional

from langchain.schema.runnable import RunnablePassthrough

from inquisia.rags.rag_interface import RAGInterface
from inquisia.rags.inquisia_rag import InquisiaRAG
from inquisia.qa_interface import QAInterface


class KnowledgeBaseQA(QAInterface):
    """
    Main class for query/answering system using custom knowledge base.
    """

    knowledge_qa: Optional[RAGInterface]

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.knowledge_qa = InquisiaRAG(model=model, knowledge_base="test_collection")

    def generate_answer(self, question: str) -> str:
        rag_chain = {
            "context": self.knowledge_qa.get_retriever(),
            "question": RunnablePassthrough(),
        } | self.knowledge_qa.get_question_generation_llm()
        return rag_chain.invoke(question)["text"]
