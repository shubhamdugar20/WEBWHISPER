import streamlit as st
import os
import shutil
import uuid
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

# Ensure directories exist
os.makedirs(AUDIO_PATH, exist_ok=True)






# ✅ Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "website_url" not in st.session_state:
    st.session_state.website_url = None

if "audio_files" not in st.session_state:
    st.session_state.audio_files = []  # Store generated audio file paths

# ✅ Function to delete only session-tracked audio files
def delete_audio_files():
    for file_path in st.session_state.audio_files:
        if os.path.isfile(file_path):
            os.remove(file_path)
    st.session_state.audio_files = []  # Clear the list after deletion

# ✅ Function to create FAISS vector store from a website URL
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

# ✅ Function to generate unique speech file
def generate_speech(text):
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    filename = f"response_{unique_id}.mp3"  # Unique filename
    filepath = os.path.join(AUDIO_PATH, filename)

    try:
        # Generate TTS file
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)
        st.session_state.audio_files.append(filepath)  # Store in session state
        return filepath
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None
    








# ✅ Function to get AI response
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


# ✅ Streamlit UI
st.title("WEBWHISPER - Talk to Websites! ")
st.markdown("<hr>", unsafe_allow_html=True)


with st.sidebar:
    website_url = st.text_input("🔗 Enter Website URL")

if website_url:
    # Delete only session audio files when the website URL changes
    if st.session_state.website_url != website_url:
        st.session_state.website_url = website_url
        delete_audio_files()  # Delete only session-tracked audio files
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        if st.session_state.vector_store:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# ✅ Chat input and processing
user_query = st.chat_input("💬 Type your message here...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Vector store not initialized. Please enter a valid URL first.")
    else:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# ✅ Display chat messages with "Play" button for AI responses
for index, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)

            # 🎯 "Play" button to generate and play TTS only when clicked
            if st.button(f"▶️ Play", key=f"play_{index}"):
                audio_file = generate_speech(message.content)  # Generate audio only on button click
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                else:
                    st.error("Failed to generate audio.")

    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
