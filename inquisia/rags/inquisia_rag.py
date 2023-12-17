from typing import Optional

import torch

from transformers import BitsAndBytesConfig

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import Milvus
from langchain.llms.base import BaseLLM
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from inquisia.rags.rag_interface import RAGInterface


class InquisiaRAG(RAGInterface):
    """
    Inquisia implementation of the RAGInterface.
    """

    knowledge_base: str = "test_collection"
    vector_store: Optional[VectorStore] = None
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"

    def __init__(self, model: str, knowledge_base: str):
        self.vector_store = self._create_vector_store()
        self.knowledge_base = knowledge_base

    @property
    def embeddings(self):
        return HuggingFaceEmbeddings()

    def _create_vector_store(self) -> VectorStore:
        return Milvus(
            self.embeddings,
            connection_args={"host": "127.0.0.1", "port": "19530"},
            collection_name="collection_1",
        )

    def _create_llm(self, model) -> BaseLLM:
        """
        Create a LLM with the given parameters
        """
        return self._get_mistral_model(model)

    def _get_quantization_config(self):
        compute_dtype = getattr(torch, "float16")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    def _get_mistral_model(self, model):
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        return HuggingFacePipeline.from_model_id(
            model_name,
            task="text-generation",
            device=0,
            model_kwargs={"quantization_config": self._get_quantization_config()},
            pipeline_kwargs={
                "temperature": 0.2,
                "repetition_penalty": 1.1,
                "return_full_text": True,
                "max_new_tokens": 1000,
            },
        )

    def _create_prompt_template(self):
        prompt_template = """
        ### [INST] Instruction: Answer the question based on your fantasy football knowledge. Here is context to help:
        
        {context}
        
        ### QUESTION:
        
        {question} [/INST]
        """
        return PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

    def get_question_generation_llm(self):
        return LLMChain(
            llm=self._create_llm(model=self.model),
            prompt=self._create_prompt_template(),
        )

    def get_retriever(self):
        return self.vector_store.as_retriever()
