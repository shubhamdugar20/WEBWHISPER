import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from gtts import gTTS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

FAISS_INDEX_PATH = "./faiss_index"
AUDIO_PATH = "./audio_responses"  # Directory for storing audio responses

# ✅ Ensure directories exist
os.makedirs(AUDIO_PATH, exist_ok=True)

# ✅ Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "website_url" not in st.session_state:
    st.session_state.website_url = None

if "audio_files" not in st.session_state:
    st.session_state.audio_files = {}  # Store audio file paths for AI responses

# ✅ Function to create FAISS vectorstore from a website URL
def get_vectorstore_from_url(url):
    try:
        shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        loader = WebBaseLoader(url)
        document = loader.load()
        if not document:
            st.error("No content found on the provided URL.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        document_chunks = text_splitter.split_documents(document)

        vector_store = FAISS.from_documents(
            documents=document_chunks,
            embedding=OpenAIEmbeddings()
        )
        vector_store.save_local(FAISS_INDEX_PATH)  
        st.success("Website data successfully indexed!")

        return vector_store

    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        return None

# ✅ Function to create a context-aware retriever
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Use the above conversation history to generate a search query and fetch the most relevant context."),
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

# ✅ Function to create a conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# ✅ Function to generate speech from text (only on demand)
def generate_speech(text, index):
    filename = f"response_{index}.mp3"
    filepath = os.path.join(AUDIO_PATH, filename)

    # Generate TTS file only if it doesn't exist
    if not os.path.exists(filepath):
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)

    return filepath






def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


st.title("WEBWHISPER - Talk to Websites!")


with st.sidebar:
    website_url = st.text_input("Website URL")

if website_url:
    if st.session_state.website_url != website_url:
        st.session_state.website_url = website_url
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        if st.session_state.vector_store:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
            st.session_state.audio_files = {}  # Reset stored audio files

# ✅ Chat input and processing
user_query = st.chat_input("Type your message here...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Vector store not initialized. Please enter a valid URL first.")
    else:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# ✅ Display chat messages with Play Button for AI responses
for index, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)

            # "Play" button to generate and play TTS only when clicked
            if st.button(f"▶️ Play", key=f"play_{index}"):
                audio_file = generate_speech(message.content, index)
                st.audio(audio_file, format="audio/mp3")
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)