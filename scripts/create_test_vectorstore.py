from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

import nest_asyncio

nest_asyncio.apply()

# Articles to index
articles = [
    "https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/",
    "https://www.fantasypros.com/2023/11/5-stats-to-know-before-setting-your-fantasy-lineup-week-10/",
    "https://www.fantasypros.com/2023/11/nfl-week-10-sleeper-picks-player-predictions-2023/",
    "https://www.fantasypros.com/2023/11/nfl-dfs-week-10-stacking-advice-picks-2023-fantasy-football/",
    "https://www.fantasypros.com/2023/11/players-to-buy-low-sell-high-trade-advice-2023-fantasy-football/",
]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

# Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

vector_db = Milvus.from_documents(
    docs,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    collection_name="test_collection",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
