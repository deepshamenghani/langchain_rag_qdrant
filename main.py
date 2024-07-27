from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.chains import RetrievalQA
import os
import dotenv

dotenv.load_dotenv()
llmclient = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API"))

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("superhero_facts.txt")
documents = loader.load_and_split(text_splitter=text_splitter)

client = QdrantClient(path="./embeddings")
collection_name = "my_documents"

if collection_name not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

db = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)

db.add_documents(documents)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llmclient, retriever=retriever, chain_type="stuff"
)

while True:
    humaninput = input(">> ")
    result = chain.invoke(humaninput)
    print(result["result"])
