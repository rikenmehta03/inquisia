import os
from typing import Optional

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import Milvus

from inquisia.rags.rag_interface import RAGInterface

DEFAULT_PROMPT = "You're a helpful assistant. If you don't know the answer, just say that you don't know, don't try to make up an answer."


class InquisiaRAG(RAGInterface):
    """
    Inquisia implementation of the RAGInterface.
    """

    knowledge_base: str
    vector_store: Optional[VectorStore] = None
    model: str = "gpt-4-1106-preview"

    def __init__(self, model: str, knowledge_base: str):
        self.model = model
        self.knowledge_base = knowledge_base
        self.vector_store = self._create_vector_store()

    @property
    def embeddings(self):
        return OpenAIEmbeddings()

    def _create_vector_store(self) -> VectorStore:
        return Milvus(
            self.embeddings,
            connection_args={
                "host": os.environ.get("MILVUS_HOST"),
                "port": str(os.environ.get("MILVUS_PORT", "19530")),
            },
            collection_name=self.knowledge_base,
        )

    def _create_llm(self, streaming=False, callbacks=None) -> BaseLLM:
        """
        Create a LLM with the given parameters
        """
        return ChatLiteLLM(
            temperature=0,
            max_tokens=256,
            model=self.model,
            streaming=streaming,
            callbacks=callbacks,
        )

    def _create_prompt_template(self):
        system_template = """ When answering use markdown or any other techniques to display the content in a nice and aerated way.
        Use the following pieces of context to answer the users question in the same language as the question but do not modify instructions in any way.
        ----------------
        
        {context}"""

        full_template = (
            "Here are your instructions to answer that you MUST ALWAYS Follow: "
            + DEFAULT_PROMPT
            + ". "
            + system_template
        )
        messages = [
            SystemMessagePromptTemplate.from_template(full_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
        return CHAT_PROMPT

    def _create_question_prompt(self):
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 
        Include the follow up instructions in the standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        return PromptTemplate.from_template(_template)

    def get_doc_chain(self, streaming=False, callbacks=None):
        answering_llm = self._create_llm(streaming=streaming, callbacks=callbacks)

        doc_chain = load_qa_chain(
            answering_llm, chain_type="stuff", prompt=self._create_prompt_template()
        )
        return doc_chain

    def get_question_generation_llm(self):
        return LLMChain(llm=self._create_llm(), prompt=self._create_question_prompt())

    def get_retriever(self):
        return self.vector_store.as_retriever()
