import asyncio
from typing import AsyncIterable, Awaitable, List, Optional

from inquisia.rags.rag_interface import RAGInterface
from inquisia.rags.inquisia_rag import InquisiaRAG
from inquisia.qa_interface import QAInterface

from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler


class KnowledgeBaseQA(QAInterface):
    """
    Main class for query/answering system using custom knowledge base.
    """

    knowledge_qa: Optional[RAGInterface]
    callbacks: List[AsyncIteratorCallbackHandler] = None

    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        knowledge_base: str = "test_kb",
    ):
        self.knowledge_qa = InquisiaRAG(model=model, knowledge_base=knowledge_base)

    def generate_answer(self, question: str) -> str:
        qa = ConversationalRetrievalChain(
            combine_docs_chain=self.knowledge_qa.get_doc_chain(streaming=False),
            question_generator=self.knowledge_qa.get_question_generation_llm(),
            retriever=self.knowledge_qa.get_retriever(),
        )

        return qa({"question": question, "chat_history": []})

    async def generate_stream_answer(self, question: str) -> AsyncIterable:
        callback = AsyncIteratorCallbackHandler()
        self.callbacks = [callback]

        qa = ConversationalRetrievalChain(
            combine_docs_chain=self.knowledge_qa.get_doc_chain(
                streaming=True, callbacks=self.callbacks
            ),
            question_generator=self.knowledge_qa.get_question_generation_llm(),
            retriever=self.knowledge_qa.get_retriever(),
        )

        async def wrap_done(fn: Awaitable, event: asyncio.Event):
            try:
                return await fn
            except Exception as e:
                return None  # Or some sentinel value that indicates failure
            finally:
                event.set()

        run = asyncio.create_task(
            wrap_done(
                qa.acall({"question": question, "chat_history": []}),
                callback.done,
            )
        )

        response_tokens = []

        async for token in callback.aiter():
            response_tokens.append(token)
            yield token

        await run
