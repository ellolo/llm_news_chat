import os
import logging
from datetime import datetime, timedelta, date
import streamlit as st

from summarizer.guardian_retrieval import GuardianRetriever
from summarizer.rag import GuardianRAG

st.set_page_config(page_title="News summarizer", page_icon="🔥", layout="wide", )
st.title("The Guardian news summarizer and QA")
st.write(
    "I am an exciting news summarizer and question answering agent powered by LangChain and The "
    " Guardian API. ")


@st.cache_data
def retrieve(query: str, from_date: datetime, to_date: datetime, guardian_key: str, top_k: int):
    """
    Retrieve articles from The Guardian API.
    Args:  
        query: The query to search for articles.
        from_date: The start date for the article search.
        to_date: The end date for the article search.
        guardian_key: The API key for The Guardian API.
        top_k: The number of articles to retrieve.
    Returns:
        The list of article URLs from The Guardian.
    """
    assert from_date <= to_date
    assert top_k <= 50
    gr = GuardianRetriever(guardian_key)
    return gr.get_articles(query, from_date, to_date, top_k=top_k)


def retrieve_and_summarize(
    rag_model: GuardianRAG,
    query: str,
    from_date: datetime,
    to_date: datetime,
    guardian_key: str,
    top_k: int = 5,
):
    """
    Retrieve articles from The Guardian and summarize them using RAG.
    Args:
        rag_model: RAG instance for summarization.
        query: The query to search for articles.
        from_date: The start date for the article search.
        to_date: The end date for the article search.
        guardian_key: The API key for The Guardian API.
        top_k: The number of articles to retrieve."""
    with st.status("Initializing...", expanded=True) as status:
        status.update(label="Retrieving articles...", state="running", expanded=False)
        st.session_state.article_urls = retrieve(query, from_date, to_date, guardian_key, top_k)
        status.update(label="Indexing articles...", state="running", expanded=False)
        rag_model.index_articles(st.session_state.article_urls)
        status.update(label="Summarizing articles...", state="running", expanded=False)
        summary, token_counts = rag_model.get_article_summary()
        if token_counts[0] > rag_model.chat_ctx_length:
            logging.warning(
                "The length of all articles (%d tokens) exceeds the maximum context length of the "
                "chat model (%d tokens).",
                token_counts[0], rag_model.chat_ctx_length)
            col2.warning(
                f"Warning: The length of all articles ({token_counts[0]} tokens) exceeds the "
                f"maximum context length of the chat model ({rag_model.chat_ctx_length} tokens). "
                f"The prompt was therefore truncated, which may produce an incorrect summary. "
                f"Please try to reduce the number of articles or increase the context length."
            )
        status.update(label="Preparation completed!", state="complete", expanded=False)
        st.session_state.messages.append({"role": "assistant", "content": summary})


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

RAG_TOP_K_SPLITS = 5

# get API key for The Guardian
if not os.environ.get("GUARDIAN_KEY"):
    st.warning("Please set the GUARDIAN_KEY environment variable to your Guardian API key.")
    GUARDIAN_KEY = st.text_input("Enter your Guardian API key", type="password")
    if not GUARDIAN_KEY:
        st.stop()
else:
    GUARDIAN_KEY = str(os.environ.get("GUARDIAN_KEY"))

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "article_urls" not in st.session_state:
    st.session_state.article_urls = []


col1, col2 = st.columns([1,3])

with col1:
    with st.form(key="setup_form", border=True):
        st.selectbox(
            "Model",
            ["gemma3:1b", "gemma3", "llama3.2"],
            index=0,
            placeholder="Select model...",
            key="model"
        )
        st.selectbox(
            "Max context token length",
            [4096, 8192, 32768, 128000],
            index=0,
            placeholder="Select num tokens...",
            key="chat_ctx_length"
        )
        st.text_input("Topic", value="soccer", key="topic")
        st.text_input("Language of the assistant", value="italian", key="language")
        st.selectbox(
            "Style of the assistant",
            ["formal", "informal", "street"],
            index=0,
            placeholder="Select style...",
            key="style"
        )
        today = datetime.today()
        last_year = today.year
        st.date_input("Period to retrieve articles",
                      (today - timedelta(days=7), today),
                      min_value=date(last_year, 1, 1),
                      max_value=datetime.date(today),
                      key="dates"
        )
        st.number_input(
            "Number of articles to retrieve",
            value=5,
            min_value=2,
            max_value=10,
            key="top_k"
        )
        submitted = st.form_submit_button("Start")
        if submitted:
            rag = GuardianRAG(
                st.session_state.topic,
                st.session_state.language,
                st.session_state.style,
                chat_model_name=st.session_state.model,
                chat_ctx_length=st.session_state.chat_ctx_length
            )
            retrieve_and_summarize(
                rag,
                st.session_state.topic,
                st.session_state.dates[0],
                st.session_state.dates[1],
                GUARDIAN_KEY,
                top_k=st.session_state.top_k,
            )
            st.session_state["rag"] = rag

with col2:

    # Display chat messages from history
    chat_container = st.container(border=True, height=400)
    ## see: https://discuss.streamlit.io/t/two-column-chat-with-pdf-interface/66612/3
    ## see: https://discuss.streamlit.io/t/when-using-st-chat-input-inside-st-columns-chat-box-moves-up-how-to-keep-it-stuck-to-the-bottom/61578/5
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input and generate assistant response
    if "rag" in st.session_state:
        
        if user_message := st.chat_input("Say something"):
            chat_container.chat_message("human").markdown(user_message)
            st.session_state.messages.append({"role": "human", "content": user_message})
            if user_message.lower() == "bye":
                bye = "Bye! See you next time!"
                chat_container.chat_message("assistant").markdown(bye)
                st.session_state.messages.append(
                    {"role": "assistant", "content": bye}
                )
                st.stop()
            response, token_counts = st.session_state.rag.generate_response_with_rag(
                st.session_state.messages,
                k = RAG_TOP_K_SPLITS
            )
            if token_counts[0] > st.session_state.rag.chat_ctx_length:
                logging.warning(
                    "The length of all articles (%d tokens) exceeds the maximum context length of "
                    "the chat model (%d tokens).",
                    token_counts[0], st.session_state.rag.chat_ctx_length)
                col2.warning(
                    f"Warning: The length of the RAG prompt ({token_counts[0]} tokens) exceeds the "
                    f"maximum context length of the chat model "
                    f"({st.session_state.rag.chat_ctx_length} tokens). The prompt was therefore "
                    f" truncated, which may produce an incorrect response. Please try to increase "
                    f" the context length."
                )
            chat_container.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            #st.write(st.session_state.messages)
