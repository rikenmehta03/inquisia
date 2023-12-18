from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus

import nest_asyncio

nest_asyncio.apply()

# Articles to index
articles = ["https://huggingface.co/blog/overview-quantization-transformers"]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

# Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

vector_db = Milvus.from_documents(
    chunked_documents,
    OpenAIEmbeddings(),
    collection_name="test_kb1",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
