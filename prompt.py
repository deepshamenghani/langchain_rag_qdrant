from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API"))

# Initialize Qdrant client for local storage
client = QdrantClient(path="./embeddings")

# Define collection name
collection_name = "my_documents"

# Initialize Qdrant vector store
db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings
)

# Create a retriever from the Qdrant vector store
retriever = db.as_retriever()

# Create the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.invoke("What is an interesting fact about Ant man?")

print(result)
