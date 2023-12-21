from typing import Optional

from inquisia.rags.rag_interface import RAGInterface
from inquisia.rags.inquisia_rag import InquisiaRAG
from inquisia.qa_interface import QAInterface

from langchain.chains import ConversationalRetrievalChain


class KnowledgeBaseQA(QAInterface):
    """
    Main class for query/answering system using custom knowledge base.
    """

    knowledge_qa: Optional[RAGInterface]

    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        knowledge_base: str = "test_kb",
    ):
        self.knowledge_qa = InquisiaRAG(model=model, knowledge_base=knowledge_base)

    def generate_answer(self, question: str) -> str:
        qa = ConversationalRetrievalChain(
            combine_docs_chain=self.knowledge_qa.get_doc_chain(),
            question_generator=self.knowledge_qa.get_question_generation_llm(),
            retriever=self.knowledge_qa.get_retriever(),
        )

        return qa({"question": question, "chat_history": []})
