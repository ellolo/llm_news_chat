# LLM news chat 

This simple demo enables to chat with a locally installed AI agent to stay updated on the latest news on a specific topic.\
The demo uses _The Guardian_ as a news source.\
The user first enters a topic of interest and a start and end date. Based on this, the AI agent does the following:
- Retrieve articles from _The Guardian_ that match the topic and the start and end date
- Start a chat session with the user
- Provide a summary of the retrieved articles in the chat
- Allow the user to ask questions (QA) about the retrieved articles in the chat

In addition, the user can choose which language and which style the agent should speak

## Under the hood

### LLM
Interactions with the user (chat session, summaries) are provided by an LLM which runs on the local machine, i.e. there are
no API calls to external LLMs. The demo uses [Ollama](https://ollama.com/) as a local LLM framework. Three local-friendly LLM can be used
- *Gemma 3*: Google LLM with 4 billion parameters and max context window of 128K tokens. Requires 3.3GB of memory).
- *Gemma 3:1b*: Google LLM with 1 billion parameters and max context window of 128K tokens. Requires 0.8GB of memory).
- *Llama3.2*: Google LLM with 3.2 billion parameters and max context window of 128K tokens. Requires 2.0GB of memory).

### RAG
Articles are stored in a local in-memory vector store, and are used by the LLM to answer the user questions using a standard RAG
approach. The demo uses in-memory FAISS as vector store.

## Requirements

- A machine with at least 16GB of memory (to store the LLM models in memory) and a decent processor (otherwise it will take too long to generate LLM responses)
- Python 3.12

## Setup

### Python environment setup

Install pinned dependencies with pip: `pip install -r requirements.txt`\

Otherwise install as follows in a Conda environement:
- `conda install langchain -c conda-forge`
- `pip install langchain-ollama`
- `conda install langchain-text-splitters langchain-community -c conda-forge`
- `conda install faiss-cpu -c conda-forge`
- `conda install streamlit`
- `conda install bs4`


### The Guardian API key

- Get a _The Guardian_ API key for free: https://open-platform.theguardian.com/
- Add key to your environment: `GUARDIAN_KEY=<your key>` (for Conda look [here](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#setting-environment-variables))

### Ollama

Download and install Ollama as from [official docs](https://ollama.com/).

## Run

- Open a shell
  - Pull the following Ollama chat models:
    - ``ollama pull gemma3:1b``
    - ``ollama pull gemma3``
    - ``ollama pull llama3.2``
  - start Ollama server: ``ollama serve``
- Open another shell
  - Go to project ``src`` dir
  - Launch app: ``streamlit run app.py``