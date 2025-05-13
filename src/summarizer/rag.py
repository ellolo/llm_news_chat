import logging
from typing import List, Dict, Tuple
import bs4
import faiss
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    )
from langchain_core.prompt_values import PromptValue


class GuardianRAG:
    """
    A simple implementation of a RAG using The Guardian articles.
    This class is used to index articles from The Guardian and generate summaries and
    answer questions using the articles as context.
    It uses the Ollama chat model and the FAISS vector store for indexing.
    """

    # system prompt for summarizing articles
    system_prompt_template_summarization = (
        "You are a useful assistant that summarizes news articles on topic ``{topic}``. You speak "
        "in {language} only. You talk like a {style} person.\n"
        "Your task is to summarize the news articles reported below. Each article starts with"
        "<start> and ends with <end>. Build a summary made of bullet points. There must be one "
        "bullet point per article. After the summary, complete the message with a joke related "
        "to one of the news articles.{articles}"
    )

    # system prompt for QA with RAG
    system_prompt_template_rag = (
        "You are an assistant for question-answering tasks on {topic}. Use the following pieces "
        "of retrieved context to answer questions. If you cannot find the answer in the retrieve "
        "context, just say that you don't know. You can only use the retrieve context to answer "
        "the question. Use three sentences maximum and keep the answer concise. You speak in "
        " {language} only. You talk like a {style} person.\nContext:\n{rag_context}"
    )


    def __init__(
        self,
        topic: str,
        language: str,
        style: str,
        chat_model_name: str = "gemma3:1b",
        emb_model_name: str = "nomic-embed-text", # "all-minilm", "mxbai-embed-large"
        chat_ctx_length: int = 4096,
        chat_temperature: float = 0.7,
    ):
        """
        Args:
            chat_model_name: The name of the chat model to use.
            emb_model_name: The name of the embedding model to use for indexing articles.
            chat_ctx_length: The maximum context length for the chat model.
            chat_temperature: The temperature for the chat model.
        """
        self.topic = topic
        self.language = language
        self.style = style
        self.emb_model_name = emb_model_name
        # initialize indexing components
        embedding_model = OllamaEmbeddings(model=emb_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.faiss_store = FAISS(
            embedding_function=embedding_model,
            index=faiss.IndexFlatL2(len(embedding_model.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.urls =  []
        self.docs = []
        # initializing chat component
        self.chat_ctx_length = chat_ctx_length
        self.llm = ChatOllama(
            model=chat_model_name,
            temperature=chat_temperature,
            num_ctx=self.chat_ctx_length) # setting max context tokens' size


    # TODO
    @staticmethod
    def compute_number_of_tokens(prompt: PromptValue, response: BaseMessage) -> Tuple[int, int]:
        """
        Computes the number of tokens in the prompt and the actual number of tokens of the prompt
        that were taken as input by the chat model that generated the response.
        Args:
            prompt_text: The prompt text.
            response: The response from the chat model.
        Returns:   
            A tuple containing the number of tokens in the prompt and the number of tokens used."""
        num_tokens_prompt = count_tokens_approximately(prompt)
        num_tokens_prompt_used = -1
        if isinstance(response, AIMessage) and response.usage_metadata:
            num_tokens_prompt_used = response.usage_metadata["input_tokens"]
        logging.debug("Number of tokens in the prompt: %d", num_tokens_prompt)
        logging.debug("Number of tokens of the prompt considered by the model: %d", num_tokens_prompt_used)
        return (num_tokens_prompt, num_tokens_prompt_used)
    
    @staticmethod
    def convert_messages(messages:  List[Dict[str, str]]) -> List[HumanMessage | AIMessage]:
        """
        Convert the messages from streamlit format to LangChain format.
        Args:
            messages: The chat messages in streamlit format.
        Returns:
            A list of LangChain BaseMessages.
        """
        lc_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "human" or role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))
        return lc_messages
     

    def index_articles(self, urls: List[str]):
        """
        Index The Guardian articles from the given URLs into a FAISS in memory index.
        Args:
            urls: The list of URLs to index.
        """
        # load documents
        loader = WebBaseLoader(
            web_paths=(urls),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    # "the guardian" specific logic to get content
                    "div", attrs={"data-gu-name": ["headline", "standfirst", "body"]},
                )
            ),
        )
        docs =  loader.load()
        self.docs += docs
        self.urls += urls
        for idx, doc in enumerate(docs):
            logging.debug("Retrieved doc %d: %d characters", idx, len(doc.page_content))
        # Split the document in chunks. The search will be done at chunk level, not doc level.
        # This is because a whole document may be tool long to fit into the model context.
        all_splits = self.text_splitter.split_documents(docs)
        logging.debug("There are %d splits", len(all_splits))
        # embed all splits and store them in the vector store
        self.faiss_store.add_documents(all_splits)
        logging.debug("Number of indexed splits: %d", self.faiss_store.index.ntotal)


    def get_article_summary(self) -> Tuple[str, Tuple[int, int]]:
        '''
        Generate a summary of the indexed articles using the chat model.
        Args:
            topic: The topic of the articles.
            language: The language to use for the summary.
            style: The style to use for the summary.
        Returns:
            The summary of the articles.
            A tuple containing the number of tokens of the prompt passed to the chat model, and the
            actual number of tokens used to compute the response.
        '''
        if len(self.docs) == 0:
            raise ValueError("No documents indexed. Please index articles first.")
        news_content = ""
        for doc in self.docs:
            news_content += "\n\n<start>\n" + doc.page_content + "\n<end>"
        # create a prompt template that contains the system prompt template and the human instruction
        system_prompt_template = ChatPromptTemplate(
            [
                SystemMessagePromptTemplate.from_template(GuardianRAG.system_prompt_template_summarization),
            ])
        prompt = system_prompt_template.invoke(
            {
                "language": self.language,
                "topic": self.topic,
                "style": self.style,
                "articles": news_content,
            }
        )
        logging.debug("Summarization prompt:\n%s\n=================", str(prompt))
        llm_response = self.llm.invoke(prompt)
        token_counts = GuardianRAG.compute_number_of_tokens(prompt, llm_response)
        response = str(llm_response.content) + "\n\nRetrieved articles from The Guardian:\n"
        for url in self.urls:
            response += f"- {url}\n"
        response = response + "\n\n"
        return response, token_counts


    def generate_response_with_rag(
        self, messages: List[Dict[str, str]], k: int = 5
        ) -> Tuple[str, Tuple[int, int]]:
        """
        Generate a response to the last message in the chat history using RAG.
        Args:
            messages: The chat history.
            k: The number of top document splits to retrieve.
        Returns:
            The response to the last message in the chat history.
            A tuple containing the number of tokens of the prompt passed to the chat model, and the
            actual number of tokens used to compute the response."""
        if len(messages) == 0:
            raise ValueError("No messages in the chat history. Please provide a question.")
        if len(self.docs) == 0:
            raise ValueError("No documents indexed. Please index articles first.")
        logging.debug("number of messages in history: %d", len(messages))
        question = messages[-1]["content"]
        logging.debug("question is: %s", question)

        # get top document splits from the vector store matching the question
        top_splits = self.faiss_store.similarity_search_with_relevance_scores(question, k=k)
        top_split_content =  "\n\n".join(doc[0].page_content for doc in top_splits)
        # TODO
        # we could in theory set a threshold to the k splits, and only retain those above it,
        # to avoid rubbish retrieved results
        # see: https://meisinlee.medium.com/better-rag-retrieval-similarity-with-threshold-a6dbb535ef9e
        logging.debug("========== TOP SPLITS FOR QUERY: %s ==========", question)
        for spl, rel_score in top_splits:
            logging.debug("\n\nScore: %s\n%s\n================", rel_score, spl)

        # create a prompt template that contains the system prompt template and all previous
        # messages in the chat
        system_prompt_template = ChatPromptTemplate(
            [
                SystemMessagePromptTemplate.from_template(GuardianRAG.system_prompt_template_rag),
                MessagesPlaceholder("message_history")
            ])
        prompt = system_prompt_template.invoke({
            "language": self.language,
            "topic": self.topic,
            "style": self.style,
            "rag_context": top_split_content,
            "message_history": GuardianRAG.convert_messages(messages)})
        logging.debug("Rag prompt:\n%s\n=================", str(prompt))
        llm_response = self.llm.invoke(prompt)
        token_counts = GuardianRAG.compute_number_of_tokens(prompt, llm_response)
        return str(llm_response.content), token_counts
