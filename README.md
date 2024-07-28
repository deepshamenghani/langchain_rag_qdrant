# Building a Superhero Facts RAG QA Bot Using LangChain, Qdrant, and OpenAI

This repository demonstrates the process of building a Retrieval-Augmented Generation (RAG) Question-Answering bot using LangChain, Qdrant, and OpenAI. It showcases the implementation of a local RAG system focused on superhero facts.

## Project Structure

* `main.py`: Implementation of the Superhero Facts QA bot
* `superhero_facts.txt`: Text file containing superhero facts
* `.gitignore`: Specifies intentionally untracked files to ignore

## Features

* Document loading and text splitting
* Embedding generation using OpenAI
* Vector storage with local Qdrant
* Retrieval QA chain implementation
* Integration with OpenAI's language models
* Utilization of LangChain for building the RAG system

## Installation

1. Clone this repository
2. Install required packages: `pip install langchain langchain_openai langchain_community langchain_qdrant python-dotenv qdrant-client`
3. Set up your OpenAI API key in a `.env` file: `OPENAI_API=your_api_key_here`

## Usage

Run the Superhero Facts QA bot:
`python main.py`

## How It Works

This project demonstrates the process of building a RAG system:

1. **Document Loading and Chunking**
   * Loads the superhero facts from a text file
   * Splits the text into manageable chunks using CharacterTextSplitter

2. **Embedding and Vector Storage**
   * Generates embeddings using OpenAI's embedding model
   * Stores the embeddings in a local Qdrant vector database

3. **Retrieval QA Chain**
   * Implements a retrieval mechanism using the Qdrant vector store
   * Creates a RetrievalQA chain to combine retrieval and question-answering

4. **Interactive QA Loop**
   * Provides an interactive interface for users to ask questions about superheroes
   * Retrieves relevant information and generates answers using the RAG system

## Key Components

* **ChatOpenAI**: Interface to OpenAI's chat models
* **OpenAIEmbeddings**: Generates embeddings for text
* **Qdrant**: Vector database for efficient similarity search
* **CharacterTextSplitter**: Splits text into manageable chunks
* **RetrievalQA**: Combines retrieval and question-answering capabilities

## Customization

Feel free to modify the code to experiment with different:
* Text splitting strategies
* Embedding models
* Vector databases
* Retrieval methods
* Question-answering chain types

This project is part of a tutorial on building RAG systems. For a detailed explanation of each step, please refer to the associated blog post: [How I Built a Superhero Facts RAG QA Bot Using LangChain, Qdrant, and OpenAI](https://medium.com/@menghani.deepsha)
