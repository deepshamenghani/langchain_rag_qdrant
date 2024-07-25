from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API"))

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0
)

loader = TextLoader("superhero_facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# Initialize Qdrant client for local storage
client = QdrantClient(path="./embeddings")

# Define collection name
collection_name = "my_documents"

# Create the collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Initialize Qdrant vector store
db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings
)

# Add documents to the vector store
db.add_documents(docs)

# Perform similarity search
results = db.similarity_search(
    "Give me a fact about Ant man"
)

# Print results
for result in results:
    print("\n")
    print(result.metadata)
    print(result.page_content)